from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import coptpy as cp
import numpy as np

from hourly_component_task import HourlyTask


_COPT_ENV = cp.Envr()

@dataclass(frozen=True)
class _TaskPattern:
    start_hour: int | None
    end_hour: int | None
    work_hours: tuple[int, ...]
    downtime_hours: tuple[int, ...]
    complete: bool
    visit: bool


def _first_access_hours(access: np.ndarray, start: int, required_hours: int) -> tuple[tuple[int, ...], bool]:
    hours = tuple(int(h) for h in range(start, len(access)) if access[h] > 0.5)
    if len(hours) >= required_hours:
        return hours[:required_hours], True
    return hours, False


def _build_continuous_outage_patterns(task: HourlyTask, access_forecast: np.ndarray) -> list[_TaskPattern]:
    """Enumerate start-time patterns.

    Once a task starts, the turbine is assumed to be shut down until the required
    effective work hours have been accumulated. Work can pause only in hours that
    are not accessible in ``access_forecast``. Optional tasks are represented only
    by complete patterns; mandatory/ongoing tasks may also use partial progress
    patterns and an idle pattern to avoid infeasibility under tight resources.
    """

    horizon = len(access_forecast)
    patterns: list[_TaskPattern] = []

    if task.ongoing:
        work_hours, complete = _first_access_hours(access_forecast, 0, task.hours)
        if work_hours:
            end_hour = work_hours[-1] if complete else horizon - 1
            patterns.append(
                _TaskPattern(
                    start_hour=0,
                    end_hour=end_hour,
                    work_hours=work_hours,
                    downtime_hours=tuple(range(0, end_hour + 1)),
                    complete=complete,
                    visit=True,
                )
            )
        patterns.append(
            _TaskPattern(
                start_hour=0,
                end_hour=horizon - 1,
                work_hours=tuple(),
                downtime_hours=(
                    tuple(range(horizon)) if task.idle_blocks_turbine else tuple()
                ),
                complete=False,
                visit=False,
            )
        )
        return patterns

    start_hours = [int(h) for h in range(horizon) if access_forecast[h] > 0.5]
    for start in start_hours:
        work_hours, complete = _first_access_hours(access_forecast, start, task.hours)
        if not work_hours:
            continue
        if complete:
            end_hour = work_hours[-1]
            patterns.append(
                _TaskPattern(
                    start_hour=start,
                    end_hour=end_hour,
                    work_hours=work_hours,
                    downtime_hours=tuple(range(start, end_hour + 1)),
                    complete=True,
                    visit=True,
                )
            )
        elif task.mandatory:
            patterns.append(
                _TaskPattern(
                    start_hour=start,
                    end_hour=horizon - 1,
                    work_hours=work_hours,
                    downtime_hours=tuple(range(start, horizon)),
                    complete=False,
                    visit=True,
                )
            )

    if task.mandatory:
        # Waiting is allowed for mandatory failed tasks when resources or weather
        # make progress impossible, but it carries no completion/progress benefit.
        patterns.append(
            _TaskPattern(
                start_hour=None,
                end_hour=None,
                work_hours=tuple(),
                downtime_hours=tuple(),
                complete=False,
                visit=False,
            )
        )

    return patterns


def solve_hourly_component_schedule(
    tasks: Iterable[HourlyTask],
    access_forecast: np.ndarray,
    loss_cost_yuan_per_hour: np.ndarray,
    *,
    max_parallel_turbines: int,
    standard_work_hours: int,
    crew_cost_yuan_per_hour: float,
    overtime_cost_yuan_per_hour: float,
    vessel_cost_yuan_per_day: float,
    solver_seed: int | None = None,
    solver_threads: int | None = None,
    solver_time_limit: float = 300.0,
    mip_gap: float = 0.001,
    enforce_contiguous: bool = False,
    visit_cost_yuan_per_turbine: np.ndarray | None = None,
    visit_time_hours_per_turbine: np.ndarray | None = None,
    vessel_technician_capacity: int | None = None,
    max_daily_turbine_visits: int | None = None,
) -> dict:
    task_list = list(tasks)
    horizon, n_turbines = access_forecast.shape
    n_tasks = len(task_list)

    if n_tasks == 0:
        return {
            "status": "EMPTY",
            "solver": "COPT",
            "solver_seed_applied": False,
            "objective": 0.0,
            "work_plan": np.zeros((horizon, 0), dtype=int),
            "downtime_plan": np.zeros((horizon, 0), dtype=int),
            "completed": np.zeros(0, dtype=int),
            "overtime_hours": 0.0,
            "vessel_used": 0,
            "visited_turbines": [],
            "task_ids": [],
            "runtime_sec": 0.0,
            "mip_gap": 0.0,
            "num_vars": 0,
            "num_constraints": 0,
        }

    model = _COPT_ENV.createModel("hourly_component_maintenance")
    complete = model.addVars(n_tasks, vtype=cp.COPT.BINARY, nameprefix="complete")
    vessel = model.addVar(vtype=cp.COPT.BINARY, name="vessel")
    visit = model.addVars(n_turbines, vtype=cp.COPT.BINARY, nameprefix="visit")
    overtime = model.addVar(lb=0.0, vtype=cp.COPT.CONTINUOUS, name="overtime")
    visit_cost_array = (
        np.zeros(n_turbines, dtype=float)
        if visit_cost_yuan_per_turbine is None
        else np.asarray(visit_cost_yuan_per_turbine, dtype=float)
    )
    visit_time_array = (
        np.zeros(n_turbines, dtype=float)
        if visit_time_hours_per_turbine is None
        else np.asarray(visit_time_hours_per_turbine, dtype=float)
    )
    if len(visit_cost_array) != n_turbines or len(visit_time_array) != n_turbines:
        raise ValueError("visit cost/time arrays must match the number of turbines.")

    task_patterns: list[list[_TaskPattern]] = [
        _build_continuous_outage_patterns(task, access_forecast[:, task.turbine])
        for task in task_list
    ]
    pattern_vars: list[list[cp.Var]] = []
    for j, task in enumerate(task_list):
        patterns = task_patterns[j]
        z = [
            model.addVar(vtype=cp.COPT.BINARY, name=f"pattern_{j}_{p}")
            for p in range(len(patterns))
        ]
        pattern_vars.append(z)
        if task.mandatory:
            model.addConstr(cp.quicksum(z) == 1, name=f"mandatory_select_one_{j}")
        else:
            model.addConstr(cp.quicksum(z) <= 1, name=f"optional_select_at_most_one_{j}")
        model.addConstr(
            complete[j] == cp.quicksum(z[p] for p, pattern in enumerate(patterns) if pattern.complete),
            name=f"complete_from_pattern_{j}",
        )

    for h in range(horizon):
        model.addConstr(
            cp.quicksum(
                pattern_vars[j][p]
                for j, patterns in enumerate(task_patterns)
                for p, pattern in enumerate(patterns)
                if h in pattern.work_hours
            )
            <= max_parallel_turbines,
            name=f"parallel_{h}",
        )
        if vessel_technician_capacity is not None:
            model.addConstr(
                cp.quicksum(
                    task_list[j].crew_size * pattern_vars[j][p]
                    for j, patterns in enumerate(task_patterns)
                    for p, pattern in enumerate(patterns)
                    if h in pattern.work_hours
                )
                <= vessel_technician_capacity,
                name=f"technician_capacity_{h}",
            )
        for turbine in range(n_turbines):
            same_turbine_tasks = [j for j, task in enumerate(task_list) if task.turbine == turbine]
            if same_turbine_tasks:
                model.addConstr(
                    cp.quicksum(
                        pattern_vars[j][p]
                        for j in same_turbine_tasks
                        for p, pattern in enumerate(task_patterns[j])
                        if h in pattern.work_hours
                    )
                    <= 1,
                    name=f"one_work_turbine_{h}_{turbine}",
                )

    total_work = cp.quicksum(
        len(pattern.work_hours) * pattern_vars[j][p]
        for j, patterns in enumerate(task_patterns)
        for p, pattern in enumerate(patterns)
    )
    proxy_travel_time = cp.quicksum(float(visit_time_array[turbine]) * visit[turbine] for turbine in range(n_turbines))
    model.addConstr(overtime >= total_work + proxy_travel_time - standard_work_hours, name="overtime_def")
    for j, patterns in enumerate(task_patterns):
        turbine = task_list[j].turbine
        for p, pattern in enumerate(patterns):
            if pattern.visit:
                model.addConstr(vessel >= pattern_vars[j][p], name=f"vessel_use_{j}_{p}")
                model.addConstr(visit[turbine] >= pattern_vars[j][p], name=f"visit_use_{j}_{p}")
    if max_daily_turbine_visits is not None:
        model.addConstr(
            cp.quicksum(visit[turbine] for turbine in range(n_turbines))
            <= max_daily_turbine_visits,
            name="max_daily_turbine_visits",
        )

    completion_benefit = cp.quicksum(
        (task.risk_benefit_yuan - task.direct_cost_yuan) * complete[j]
        for j, task in enumerate(task_list)
    )
    ongoing_progress_benefit = cp.quicksum(
        (50000.0 if task.ongoing else 0.0)
        * len(pattern.work_hours)
        * pattern_vars[j][p]
        for j, patterns in enumerate(task_patterns)
        for p, pattern in enumerate(patterns)
        for task in [task_list[j]]
    )
    corrective_partial_progress_benefit = cp.quicksum(
        (
            task.partial_progress_weight_yuan_per_hour
            if task.corrective and not task.ongoing and not pattern.complete
            else 0.0
        )
        * len(pattern.work_hours)
        * pattern_vars[j][p]
        for j, patterns in enumerate(task_patterns)
        for p, pattern in enumerate(patterns)
        for task in [task_list[j]]
    )
    energy_loss_cost = cp.quicksum(
        float(sum(loss_cost_yuan_per_hour[h, task_list[j].turbine] for h in pattern.downtime_hours))
        * pattern_vars[j][p]
        for j, patterns in enumerate(task_patterns)
        for p, pattern in enumerate(patterns)
    )
    crew_person_hours = cp.quicksum(
        task_list[j].crew_size * len(pattern.work_hours) * pattern_vars[j][p]
        for j, patterns in enumerate(task_patterns)
        for p, pattern in enumerate(patterns)
    )
    crew_cost = crew_cost_yuan_per_hour * crew_person_hours
    vessel_cost = vessel_cost_yuan_per_day * vessel
    routing_visit_cost = cp.quicksum(float(visit_cost_array[turbine]) * visit[turbine] for turbine in range(n_turbines))
    overtime_cost = overtime_cost_yuan_per_hour * overtime

    model.setObjective(
        completion_benefit
        + ongoing_progress_benefit
        + corrective_partial_progress_benefit
        - energy_loss_cost
        - crew_cost
        - vessel_cost
        - routing_visit_cost
        - overtime_cost,
        cp.COPT.MAXIMIZE,
    )
    model.setParam(cp.COPT.Param.Logging, 0)
    model.setParam(cp.COPT.Param.RelGap, mip_gap)
    model.setParam(cp.COPT.Param.TimeLimit, solver_time_limit)
    # COPT 8.0 has no public random-seed parameter. The argument is retained
    # to keep the backend signature identical to CBC/Gurobi.
    _ = solver_seed
    if solver_threads is not None:
        model.setParam(cp.COPT.Param.Threads, int(solver_threads))

    model.solve()

    status_name = {
        cp.COPT.OPTIMAL: "OPTIMAL",
        cp.COPT.TIMEOUT: "TIME_LIMIT",
        cp.COPT.INFEASIBLE: "INFEASIBLE",
        cp.COPT.INF_OR_UNB: "INF_OR_UNBD",
        cp.COPT.UNBOUNDED: "UNBOUNDED",
        cp.COPT.INTERRUPTED: "INTERRUPTED",
    }.get(model.status, str(model.status))

    if not model.hasmipsol:
        return {
            "status": status_name,
            "solver": "COPT",
            "solver_seed_applied": False,
            "objective": None,
            "work_plan": np.zeros((horizon, n_tasks), dtype=int),
            "downtime_plan": np.zeros((horizon, n_tasks), dtype=int),
            "completed": np.zeros(n_tasks, dtype=int),
            "overtime_hours": 0.0,
            "vessel_used": 0,
            "visited_turbines": [],
            "task_ids": [task.task_id for task in task_list],
            "runtime_sec": float(model.solvingtime),
            "mip_gap": None,
            "num_vars": int(model.cols),
            "num_constraints": int(model.rows),
        }

    work_plan = np.zeros((horizon, n_tasks), dtype=int)
    downtime_plan = np.zeros((horizon, n_tasks), dtype=int)
    selected_patterns: list[dict | None] = [None] * n_tasks
    for j, patterns in enumerate(task_patterns):
        for p, pattern in enumerate(patterns):
            if int(round(pattern_vars[j][p].x)) <= 0:
                continue
            for h in pattern.work_hours:
                work_plan[h, j] = 1
            for h in pattern.downtime_hours:
                downtime_plan[h, j] = 1
            selected_patterns[j] = {
                "start_hour": pattern.start_hour,
                "end_hour": pattern.end_hour,
                "work_hours": list(pattern.work_hours),
                "downtime_hours": list(pattern.downtime_hours),
                "complete": pattern.complete,
                "visit": pattern.visit,
            }

    completed = np.array([int(round(complete[j].x)) for j in range(n_tasks)], dtype=int)

    return {
        "status": status_name,
        "solver": "COPT",
        "solver_seed_applied": False,
        "objective": float(model.objval),
        "work_plan": work_plan,
        "downtime_plan": downtime_plan,
        "completed": completed,
        "overtime_hours": float(overtime.x),
        "vessel_used": int(round(vessel.x)),
        "visited_turbines": [i for i in range(n_turbines) if round(visit[i].x) > 0],
        "task_ids": [task.task_id for task in task_list],
        "task_hours": [task.hours for task in task_list],
        "selected_patterns": selected_patterns,
        "runtime_sec": float(model.solvingtime),
        "mip_gap": float(model.bestgap) if model.hasmipsol else None,
        "num_vars": int(model.cols),
        "num_constraints": int(model.rows),
    }
