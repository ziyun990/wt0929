from __future__ import annotations

import numpy as np

from case_8x8_multicomponent_hourly_milp import (
    CORRECTIVE_COMPLETION_PREMIUM_YUAN,
    _candidate_tasks_to_hourly,
    _simulate_real_execution,
)
from hourly_component_scheduler_copt import solve_hourly_component_schedule
from hourly_component_task import HourlyTask
from multi_component_params import CM_COSTS_YUAN


def _failed_task(
    component: int,
    *,
    ongoing: bool = False,
    shared_outage: bool = False,
) -> HourlyTask:
    return HourlyTask(
        task_id=f"WT1_component_{component}",
        turbine=0,
        component=component,
        component_name=f"component_{component}",
        component_cn=f"component_{component}",
        hours=8,
        direct_cost_yuan=1000.0,
        risk_benefit_yuan=10000.0,
        net_benefit_yuan=9000.0,
        reasons=(("ONGOING",) if ongoing else ("FAILURE",)),
        mandatory=True,
        ongoing=ongoing,
        corrective=True,
        partial_progress_weight_yuan_per_hour=(
            0.0 if ongoing else CORRECTIVE_COMPLETION_PREMIUM_YUAN / 8.0
        ),
        idle_blocks_turbine=not shared_outage,
    )


def main() -> None:
    tasks = [_failed_task(0), _failed_task(1)]
    work_plan = np.zeros((24, 2), dtype=int)
    downtime_plan = np.zeros_like(work_plan)
    access_actual = np.ones((24, 1), dtype=int)
    loss_actual = np.zeros((24, 1), dtype=float)

    _, pending, _ = _simulate_real_execution(
        tasks,
        work_plan,
        downtime_plan,
        access_actual,
        loss_actual,
        keep_unfinished_mandatory=False,
    )
    assert pending == {}, "Unstarted failed tasks must remain candidates, not ongoing tasks."

    waiting_task = [_failed_task(0, ongoing=True)]
    waiting_work = np.zeros((24, 1), dtype=int)
    waiting_downtime = np.ones_like(waiting_work)
    _, waiting, _ = _simulate_real_execution(
        waiting_task,
        waiting_work,
        waiting_downtime,
        access_actual,
        loss_actual,
        keep_unfinished_mandatory=False,
    )
    assert waiting == {(0, 0): 8.0}, (
        "An existing ongoing task must retain its remaining hours on a zero-work day."
    )

    work_plan[5, 0] = 1
    downtime_plan[5:, 0] = 1
    _, ongoing, _ = _simulate_real_execution(
        tasks,
        work_plan,
        downtime_plan,
        access_actual,
        loss_actual,
        keep_unfinished_mandatory=False,
    )
    assert ongoing == {(0, 0): 7.0}, "A partially executed task must retain its remaining hours."

    shared_tasks = [
        _failed_task(0, ongoing=True, shared_outage=True),
        _failed_task(1, ongoing=True, shared_outage=True),
    ]
    shared_result = solve_hourly_component_schedule(
        shared_tasks,
        np.ones((24, 1), dtype=int),
        np.zeros((24, 1), dtype=float),
        max_parallel_turbines=1,
        standard_work_hours=24,
        crew_cost_yuan_per_hour=0.0,
        overtime_cost_yuan_per_hour=0.0,
        vessel_cost_yuan_per_day=0.0,
        solver_time_limit=30.0,
        mip_gap=0.001,
    )
    assert shared_result["status"] == "OPTIMAL"
    assert sorted(shared_result["work_plan"].sum(axis=0).tolist()) == [0, 8]

    corrective_ongoing = _candidate_tasks_to_hourly(
        type("TaskSet", (), {"tasks": []})(),
        np.zeros((1, len(CM_COSTS_YUAN)), dtype=float),
        {(0, 1): 6.0},
        set(),
        shared_failure_outage=True,
    )[0]
    assert corrective_ongoing.risk_benefit_yuan - corrective_ongoing.direct_cost_yuan == (
        CORRECTIVE_COMPLETION_PREMIUM_YUAN
    ), "An ongoing corrective task must retain the original completion premium."

    new_failure = HourlyTask(
        task_id="WT1_new_failure",
        turbine=0,
        component=0,
        component_name="component_0",
        component_cn="component_0",
        hours=22,
        direct_cost_yuan=float(CM_COSTS_YUAN[0]),
        risk_benefit_yuan=float(
            CM_COSTS_YUAN[0] + CORRECTIVE_COMPLETION_PREMIUM_YUAN
        ),
        net_benefit_yuan=CORRECTIVE_COMPLETION_PREMIUM_YUAN,
        reasons=("FAILURE",),
        mandatory=True,
        ongoing=False,
        corrective=True,
        partial_progress_weight_yuan_per_hour=(
            CORRECTIVE_COMPLETION_PREMIUM_YUAN / 22.0
        ),
        idle_blocks_turbine=False,
    )
    priority_result = solve_hourly_component_schedule(
        [corrective_ongoing, new_failure],
        np.ones((16, 1), dtype=int),
        np.zeros((16, 1), dtype=float),
        max_parallel_turbines=1,
        standard_work_hours=16,
        crew_cost_yuan_per_hour=0.0,
        overtime_cost_yuan_per_hour=0.0,
        vessel_cost_yuan_per_day=0.0,
        solver_time_limit=30.0,
        mip_gap=0.001,
    )
    assert priority_result["status"] == "OPTIMAL"
    assert priority_result["completed"].tolist() == [1, 0], (
        "A finishable ongoing corrective task must be completed before capacity is "
        "spent only on partial progress of a new failure task."
    )
    assert priority_result["work_plan"][:, 1].sum() == 10, (
        "A new corrective task must use residual capacity for partial progress after "
        "the existing ongoing task is completed."
    )

    print("S1 ongoing-task, shared-outage, and completion-reward validation passed")


if __name__ == "__main__":
    main()
