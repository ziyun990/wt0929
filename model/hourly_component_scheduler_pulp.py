from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np
import pulp

from hourly_component_task import HourlyTask


@dataclass(frozen=True)
class _TaskPattern:
    start_hour: int | None
    end_hour: int | None
    work_hours: tuple[int, ...]
    downtime_hours: tuple[int, ...]
    complete: bool
    visit: bool


def _first_access_hours(
    access: np.ndarray, start: int, required_hours: int
) -> tuple[tuple[int, ...], bool]:
    hours = tuple(int(h) for h in range(start, len(access)) if access[h] > 0.5)
    if len(hours) >= required_hours:
        return hours[:required_hours], True
    return hours, False


def _build_continuous_outage_patterns(
    task: HourlyTask, access_forecast: np.ndarray
) -> list[_TaskPattern]:
    """Enumerate the same continuous-outage patterns used by the Gurobi model."""
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
                downtime_hours=tuple(range(horizon)) if task.idle_blocks_turbine else tuple(),
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
    del enforce_contiguous
    task_list = list(tasks)
    horizon, n_turbines = access_forecast.shape
    n_tasks = len(task_list)

    if n_tasks == 0:
        return {
            "status": "EMPTY",
            "solver": "CBC",
            "solver_seed_applied": solver_seed is not None,
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

    model = pulp.LpProblem("hourly_component_maintenance", pulp.LpMaximize)
    complete = [pulp.LpVariable(f"complete_{j}", cat="Binary") for j in range(n_tasks)]
    vessel = pulp.LpVariable("vessel", cat="Binary")
    visit = [pulp.LpVariable(f"visit_{i}", cat="Binary") for i in range(n_turbines)]
    overtime = pulp.LpVariable("overtime", lowBound=0.0, cat="Continuous")

    task_patterns = [
        _build_continuous_outage_patterns(task, access_forecast[:, task.turbine])
        for task in task_list
    ]
    pattern_vars: list[list[pulp.LpVariable]] = []
    for j, task in enumerate(task_list):
        patterns = task_patterns[j]
        z = [pulp.LpVariable(f"pattern_{j}_{p}", cat="Binary") for p in range(len(patterns))]
        pattern_vars.append(z)
        if task.mandatory:
            model += pulp.lpSum(z) == 1, f"mandatory_select_one_{j}"
        else:
            model += pulp.lpSum(z) <= 1, f"optional_select_at_most_one_{j}"
        model += (
            complete[j]
            == pulp.lpSum(z[p] for p, pattern in enumerate(patterns) if pattern.complete)
        ), f"complete_from_pattern_{j}"

    for hour in range(horizon):
        model += (
            pulp.lpSum(
                pattern_vars[j][p]
                for j, patterns in enumerate(task_patterns)
                for p, pattern in enumerate(patterns)
                if hour in pattern.work_hours
            )
            <= max_parallel_turbines
        ), f"parallel_{hour}"
        if vessel_technician_capacity is not None:
            model += (
                pulp.lpSum(
                    task_list[j].crew_size * pattern_vars[j][p]
                    for j, patterns in enumerate(task_patterns)
                    for p, pattern in enumerate(patterns)
                    if hour in pattern.work_hours
                )
                <= vessel_technician_capacity
            ), f"technician_capacity_{hour}"
        for turbine in range(n_turbines):
            same_turbine_tasks = [
                j for j, task in enumerate(task_list) if task.turbine == turbine
            ]
            if same_turbine_tasks:
                model += (
                    pulp.lpSum(
                        pattern_vars[j][p]
                        for j in same_turbine_tasks
                        for p, pattern in enumerate(task_patterns[j])
                        if hour in pattern.work_hours
                    )
                    <= 1
                ), f"one_work_turbine_{hour}_{turbine}"

    total_work = pulp.lpSum(
        len(pattern.work_hours) * pattern_vars[j][p]
        for j, patterns in enumerate(task_patterns)
        for p, pattern in enumerate(patterns)
    )
    proxy_travel_time = pulp.lpSum(
        float(visit_time_array[turbine]) * visit[turbine]
        for turbine in range(n_turbines)
    )
    model += overtime >= total_work + proxy_travel_time - standard_work_hours, "overtime_def"

    for j, patterns in enumerate(task_patterns):
        turbine = task_list[j].turbine
        for p, pattern in enumerate(patterns):
            if pattern.visit:
                model += vessel >= pattern_vars[j][p], f"vessel_use_{j}_{p}"
                model += visit[turbine] >= pattern_vars[j][p], f"visit_use_{j}_{p}"
    if max_daily_turbine_visits is not None:
        model += pulp.lpSum(visit) <= max_daily_turbine_visits, "max_daily_turbine_visits"

    completion_benefit = pulp.lpSum(
        (task.risk_benefit_yuan - task.direct_cost_yuan) * complete[j]
        for j, task in enumerate(task_list)
    )
    ongoing_progress_benefit = pulp.lpSum(
        (50000.0 if task_list[j].ongoing else 0.0)
        * len(pattern.work_hours)
        * pattern_vars[j][p]
        for j, patterns in enumerate(task_patterns)
        for p, pattern in enumerate(patterns)
    )
    corrective_partial_progress_benefit = pulp.lpSum(
        (
            task_list[j].partial_progress_weight_yuan_per_hour
            if task_list[j].corrective
            and not task_list[j].ongoing
            and not pattern.complete
            else 0.0
        )
        * len(pattern.work_hours)
        * pattern_vars[j][p]
        for j, patterns in enumerate(task_patterns)
        for p, pattern in enumerate(patterns)
    )
    energy_loss_cost = pulp.lpSum(
        float(
            sum(
                loss_cost_yuan_per_hour[hour, task_list[j].turbine]
                for hour in pattern.downtime_hours
            )
        )
        * pattern_vars[j][p]
        for j, patterns in enumerate(task_patterns)
        for p, pattern in enumerate(patterns)
    )
    crew_person_hours = pulp.lpSum(
        task_list[j].crew_size * len(pattern.work_hours) * pattern_vars[j][p]
        for j, patterns in enumerate(task_patterns)
        for p, pattern in enumerate(patterns)
    )
    crew_cost = crew_cost_yuan_per_hour * crew_person_hours
    vessel_cost = vessel_cost_yuan_per_day * vessel
    routing_visit_cost = pulp.lpSum(
        float(visit_cost_array[turbine]) * visit[turbine]
        for turbine in range(n_turbines)
    )
    overtime_cost = overtime_cost_yuan_per_hour * overtime
    model += (
        completion_benefit
        + ongoing_progress_benefit
        + corrective_partial_progress_benefit
        - energy_loss_cost
        - crew_cost
        - vessel_cost
        - routing_visit_cost
        - overtime_cost
    )

    options: list[str] = []
    if solver_seed is not None:
        options.append(f"randomSeed {int(solver_seed)}")
    solver = pulp.PULP_CBC_CMD(
        msg=False,
        timeLimit=solver_time_limit,
        gapRel=mip_gap,
        threads=solver_threads,
        options=options,
        timeMode="elapsed",
    )
    started = perf_counter()
    model.solve(solver)
    runtime_sec = perf_counter() - started

    pulp_status = pulp.LpStatus.get(model.status, str(model.status))
    status_name = {
        "Optimal": "OPTIMAL",
        "Infeasible": "INFEASIBLE",
        "Unbounded": "UNBOUNDED",
        "Not Solved": "TIME_LIMIT_OR_NOT_SOLVED",
        "Undefined": "UNDEFINED",
    }.get(pulp_status, pulp_status.upper().replace(" ", "_"))
    objective_value = pulp.value(model.objective)
    has_solution = objective_value is not None and all(
        variable.varValue is not None for variable in complete
    )

    if not has_solution:
        return {
            "status": status_name,
            "solver": "CBC",
            "solver_seed_applied": solver_seed is not None,
            "objective": None,
            "work_plan": np.zeros((horizon, n_tasks), dtype=int),
            "downtime_plan": np.zeros((horizon, n_tasks), dtype=int),
            "completed": np.zeros(n_tasks, dtype=int),
            "overtime_hours": 0.0,
            "vessel_used": 0,
            "visited_turbines": [],
            "task_ids": [task.task_id for task in task_list],
            "runtime_sec": runtime_sec,
            "mip_gap": None,
            "num_vars": len(model.variables()),
            "num_constraints": len(model.constraints),
        }

    work_plan = np.zeros((horizon, n_tasks), dtype=int)
    downtime_plan = np.zeros((horizon, n_tasks), dtype=int)
    selected_patterns: list[dict | None] = [None] * n_tasks
    for j, patterns in enumerate(task_patterns):
        for p, pattern in enumerate(patterns):
            if int(round(float(pattern_vars[j][p].varValue or 0.0))) <= 0:
                continue
            for hour in pattern.work_hours:
                work_plan[hour, j] = 1
            for hour in pattern.downtime_hours:
                downtime_plan[hour, j] = 1
            selected_patterns[j] = {
                "start_hour": pattern.start_hour,
                "end_hour": pattern.end_hour,
                "work_hours": list(pattern.work_hours),
                "downtime_hours": list(pattern.downtime_hours),
                "complete": pattern.complete,
                "visit": pattern.visit,
            }

    completed = np.array(
        [int(round(float(variable.varValue or 0.0))) for variable in complete], dtype=int
    )
    return {
        "status": status_name,
        "solver": "CBC",
        "solver_seed_applied": solver_seed is not None,
        "objective": float(objective_value),
        "work_plan": work_plan,
        "downtime_plan": downtime_plan,
        "completed": completed,
        "overtime_hours": float(overtime.varValue or 0.0),
        "vessel_used": int(round(float(vessel.varValue or 0.0))),
        "visited_turbines": [
            i for i, variable in enumerate(visit) if round(float(variable.varValue or 0.0)) > 0
        ],
        "task_ids": [task.task_id for task in task_list],
        "task_hours": [task.hours for task in task_list],
        "selected_patterns": selected_patterns,
        "runtime_sec": runtime_sec,
        "mip_gap": 0.0 if status_name == "OPTIMAL" else None,
        "num_vars": len(model.variables()),
        "num_constraints": len(model.constraints),
    }
