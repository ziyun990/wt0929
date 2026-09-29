from __future__ import annotations

import numpy as np

from case_8x8_multicomponent_hourly_milp import (
    VESSEL_COST_YUAN_PER_DAY,
    _actual_route_windows,
    _simulate_real_execution,
)
from hourly_component_task import HourlyTask
from routing_costs import RoutingConfig, evaluate_dropoff_pickup_route


def _task() -> HourlyTask:
    return HourlyTask(
        task_id="WT1_test",
        turbine=0,
        component=0,
        component_name="test",
        component_cn="test",
        hours=3,
        direct_cost_yuan=1000.0,
        risk_benefit_yuan=0.0,
        net_benefit_yuan=-1000.0,
        reasons=("TEST",),
        mandatory=True,
        ongoing=False,
    )


def _route(rows: list[dict]) -> dict:
    layout = np.array([[0.0, 0.0]])
    return evaluate_dropoff_pickup_route(
        _actual_route_windows(rows),
        layout,
        RoutingConfig(base_distance_km=10.0),
    )


def main() -> None:
    work_plan = np.zeros((24, 1), dtype=int)
    work_plan[[5, 6, 8], 0] = 1
    downtime_plan = np.zeros_like(work_plan)
    downtime_plan[5:9, 0] = 1
    loss_actual = np.zeros((24, 1), dtype=float)

    blocked_rows, _, _ = _simulate_real_execution(
        [_task()],
        work_plan,
        downtime_plan,
        np.zeros((24, 1), dtype=int),
        loss_actual,
    )
    assert blocked_rows[0]["executed_hours"] == 0
    assert blocked_rows[0]["first_actual_hour"] is None
    assert blocked_rows[0]["last_actual_hour"] is None
    assert _actual_route_windows(blocked_rows) == []
    blocked_route = _route(blocked_rows)
    assert blocked_route["visited_turbines"] == []
    assert blocked_route["sailing_distance_km"] == 0.0
    assert blocked_route["fuel_cost_yuan"] == 0.0
    blocked_vessel_cost = (
        VESSEL_COST_YUAN_PER_DAY if blocked_route["visited_turbines"] else 0.0
    )
    assert blocked_vessel_cost == 0.0

    partial_access = np.zeros((24, 1), dtype=int)
    partial_access[[5, 8], 0] = 1
    partial_rows, _, _ = _simulate_real_execution(
        [_task()],
        work_plan,
        downtime_plan,
        partial_access,
        loss_actual,
    )
    assert partial_rows[0]["executed_hours"] == 2
    assert partial_rows[0]["first_actual_hour"] == 5
    assert partial_rows[0]["last_actual_hour"] == 8
    assert _actual_route_windows(partial_rows) == [(0, 5, 9)]
    partial_route = _route(partial_rows)
    assert partial_route["visited_turbines"] == [1]
    assert partial_route["sailing_distance_km"] > 0.0
    assert partial_route["fuel_cost_yuan"] > 0.0

    print("Actual-execution route and vessel-cost validation passed")


if __name__ == "__main__":
    main()
