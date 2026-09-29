from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cl_stochos_mu import CandidateTaskSet, TriggerConfig, WakeConfig, default_grid_layout, generate_candidate_tasks
from cl_stochos_mu import load_power_curve_bins, power_curve_mw, wake_marginal_loss
from hourly_component_task import HourlyTask
from input_scenarios import generate_rul_scenarios, read_model_inputs
from multi_component_params import (
    BASE_DISTANCE_KM,
    CM_CREW_SIZES,
    CM_COSTS_YUAN,
    CM_HOURS,
    COMPONENT_CHINESE_NAMES,
    COMPONENT_NAMES,
    CREW_COST_YUAN_PER_PERSON_HOUR,
    DAY_END_HOUR,
    DAY_START_HOUR,
    FORECAST_STD_DAYS,
    FUEL_COST_YUAN_PER_SAILING_HOUR,
    LIFETIME_DAYS,
    MAX_DAILY_TURBINE_VISITS,
    MAX_MAINTENANCE_WAVE_M,
    MAX_MAINTENANCE_WIND_M_S,
    MAX_PARALLEL_TURBINES,
    MATERIAL_COST_MULTIPLIER,
    OVERTIME_COST_YUAN_PER_HOUR,
    PM_COSTS_YUAN,
    PM_CREW_SIZES,
    PM_HOURS,
    RATED_POWER_MW,
    STANDARD_WORK_HOURS_PER_DAY,
    VESSEL_COST_YUAN_PER_DAY,
    VESSEL_SPEED_KNOTS,
    VESSEL_TECHNICIAN_CAPACITY,
)
from rul_profiles import initial_component_state
from routing_costs import RoutingConfig, evaluate_dropoff_pickup_route, proxy_roundtrip_visit_costs


EXPERIMENT_CONFIG_VERSION = "64turbine_8x8_carroll_dalgic_cl_material5x_cm_partial_progress_20260928"
CORRECTIVE_COMPLETION_PREMIUM_YUAN = 300000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the hourly component-level 8x8 (64-turbine) maintenance MILP."
    )
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=8,
        help="Number of rows and columns in the square wind-farm layout.",
    )
    parser.add_argument("--horizon-days", type=int, default=20)
    parser.add_argument(
        "--max-parallel-turbines",
        type=int,
        default=MAX_PARALLEL_TURBINES,
        help=(
            "Maximum number of turbine/component jobs that can be maintained in the same hour. "
            "This is used as the maintenance crew/resource capacity in sensitivity analysis."
        ),
    )
    parser.add_argument("--n-scenarios", type=int, default=10)
    parser.add_argument(
        "--rul-scenario-mode",
        choices=["legacy_scale", "mean_corrected", "deterministic_mean"],
        default="mean_corrected",
        help=(
            "Weibull RUL scenario parameterization. mean_corrected preserves the supplied "
            "forecast mean; deterministic_mean supplies a no-uncertainty baseline; "
            "legacy_scale reproduces the original manuscript results."
        ),
    )
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--solver-time-limit", type=float, default=300.0)
    parser.add_argument("--solver-seed", type=int, default=2026)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument(
        "--solver",
        choices=["copt", "cbc"],
        default="copt",
        help=(
            "MILP backend. copt requires the stochos Conda environment; "
            "cbc uses the solver bundled with PuLP."
        ),
    )
    parser.add_argument(
        "--task-continuity",
        choices=["continuous_outage", "interruptible", "contiguous"],
        default="continuous_outage",
        help=(
            "Maintenance execution logic. continuous_outage is the current model: after a task starts, "
            "the turbine remains stopped until the required effective work hours are completed; work "
            "can pause only in inaccessible or non-working-window hours. interruptible/contiguous are "
            "kept as legacy labels for old experiment metadata."
        ),
    )
    parser.add_argument("--input-seed", type=int, default=2026)
    parser.add_argument("--rul-seed", type=int, default=37)
    parser.add_argument(
        "--rul-profile",
        choices=["balanced", "pressure"],
        default="balanced",
        help="balanced is the literature-inspired main case; pressure preserves the dense stress-test RULs.",
    )
    parser.add_argument("--scenario-seed", type=int, default=11)
    parser.add_argument(
        "--loss-scenario-count",
        type=int,
        default=10,
        help=(
            "Number of empirical forecast-error scenarios used to form the decision-stage expected "
            "generation-loss cost. A value of 1 preserves the original point-forecast model."
        ),
    )
    parser.add_argument(
        "--loss-scenario-seed",
        type=int,
        default=2027,
        help="Random seed for joint wind-direction-price forecast-error block sampling.",
    )
    parser.add_argument(
        "--loss-error-pool",
        choices=["past_only", "out_of_window"],
        default="past_only",
        help=(
            "Residual blocks used for decision-stage loss scenarios. past_only uses only "
            "blocks fully observed before the current rolling decision; out_of_window is "
            "retained solely to reproduce the earlier audit."
        ),
    )
    parser.add_argument("--spatial-wind-std", type=float, default=0.15)
    parser.add_argument("--lead-time-days", type=float, default=5.0)
    parser.add_argument("--rul-prob-threshold", type=float, default=0.4)
    parser.add_argument("--age-threshold", type=float, default=0.75)
    parser.add_argument("--aged-count-threshold", type=int, default=32)
    parser.add_argument("--risk-exposure-cost-per-day", type=float, default=1000.0)
    parser.add_argument("--downtime-loss-per-hour", type=float, default=812.0)
    parser.add_argument("--min-net-benefit", type=float, default=2000.0)
    parser.add_argument("--wake", action="store_true", help="Use wake-aware field-level marginal loss.")
    parser.add_argument(
        "--strategy",
        choices=[
            "failure_repair",
            "predictive_rul",
            "weather_opportunity",
            "age_based_preventive",
            "multi_opportunity",
            "wake_multi_opportunity",
            "ab_static_rul_weather",
            "ab_closed_rul_weather",
            "ab_closed_rul_age_no_exec_feedback",
            "ab_closed_rul_age",
            "ab_closed_rul_age_econ",
            "full",
            "rul_only",
            "static_initial",
        ],
        default="wake_multi_opportunity",
        help=(
            "Literature-style strategies: failure_repair, predictive_rul, weather_opportunity, "
            "multi_opportunity, wake_multi_opportunity. Legacy aliases are also kept: full, rul_only, "
            "static_initial."
        ),
    )
    parser.add_argument("--output-tag", type=str, default="hourly")
    parser.add_argument(
        "--routing-cost-in-objective",
        action="store_true",
        help=(
            "Use base-turbine-base proxy sailing cost and time in the hourly MILP objective. "
            "When disabled, routing and fuel are still evaluated after execution."
        ),
    )
    parser.add_argument("--vessel-speed-knots", type=float, default=VESSEL_SPEED_KNOTS)
    parser.add_argument(
        "--fuel-cost-yuan-per-sailing-hour",
        type=float,
        default=FUEL_COST_YUAN_PER_SAILING_HOUR,
    )
    parser.add_argument("--base-distance-km", type=float, default=BASE_DISTANCE_KM)
    parser.add_argument(
        "--base-direction",
        choices=["west", "east", "south", "north"],
        default="west",
        help="Direction of the O&M base relative to the wind-farm boundary.",
    )
    return parser.parse_args()


def _hourly_access(ws: np.ndarray, wh: np.ndarray) -> np.ndarray:
    access = (
        (ws < MAX_MAINTENANCE_WIND_M_S)
        & (wh.reshape(-1, 1) < MAX_MAINTENANCE_WAVE_M)
    ).astype(int)
    access[:DAY_START_HOUR, :] = 0
    access[DAY_END_HOUR:, :] = 0
    return access


def _loss_cost_per_hour(
    ws: np.ndarray,
    wd: np.ndarray,
    price: np.ndarray,
    *,
    wake: bool,
) -> np.ndarray:
    if wake:
        side = int(round(np.sqrt(ws.shape[1])))
        if side * side != ws.shape[1]:
            raise ValueError("Wake evaluation currently requires a square turbine layout.")
        layout = default_grid_layout(side, side)
        bins = load_power_curve_bins(REPO_ROOT / "method_of_bins.csv")
        wake_loss_mw, _ = wake_marginal_loss(
            ws.mean(axis=1)[:, np.newaxis],
            wd,
            layout,
            bins,
            config=WakeConfig(),
        )
        loss_mw = wake_loss_mw[:, :, 0]
    else:
        bins = load_power_curve_bins(REPO_ROOT / "method_of_bins.csv")
        loss_mw = power_curve_mw(ws, bins, RATED_POWER_MW)
    return loss_mw * price.reshape(-1, 1)


def _circular_direction_error(actual_deg: np.ndarray, forecast_deg: np.ndarray) -> np.ndarray:
    return (np.asarray(actual_deg) - np.asarray(forecast_deg) + 180.0) % 360.0 - 180.0


def _expected_loss_from_empirical_error_scenarios(
    inputs: dict[str, np.ndarray],
    time_slice: slice,
    *,
    wake: bool,
    n_scenarios: int,
    seed: int,
    error_pool: str = "past_only",
) -> np.ndarray:
    """Estimate decision-stage loss using joint 24 h forecast-error block scenarios.

    The current evaluation window is excluded from the residual block pool. The first
    scenario is the unperturbed point forecast, and the remaining scenarios preserve
    within-day and cross-variable error dependence by sampling whole 24 h blocks.
    """

    if n_scenarios <= 1:
        return _loss_cost_per_hour(
            inputs["ws_nwp"][time_slice],
            inputs["wd_nwp"][time_slice],
            inputs["ep_fcst"][time_slice],
            wake=wake,
        )

    start = int(time_slice.start or 0)
    stop = int(time_slice.stop or len(inputs["ep_fcst"]))
    horizon = stop - start
    if horizon != 24:
        raise ValueError("Empirical loss scenarios currently require a 24 h rolling window.")

    total_hours = len(inputs["ep_fcst"])
    possible_starts = np.arange(0, total_hours - horizon + 1, dtype=int)
    if error_pool == "past_only":
        eligible_starts = possible_starts[possible_starts + horizon <= start]
    elif error_pool == "out_of_window":
        eligible_starts = possible_starts[
            (possible_starts + horizon <= start) | (possible_starts >= stop)
        ]
    else:
        raise ValueError(f"Unknown loss-error pool: {error_pool}")
    if len(eligible_starts) == 0:
        raise ValueError(
            "No eligible forecast-error blocks are available. For past_only scenarios, "
            "start the evaluation at least 24 h after the beginning of the stored series."
        )

    ws_error = inputs["ws_true"] - inputs["ws_nwp"]
    wd_error = _circular_direction_error(inputs["wd_true"], inputs["wd_nwp"])
    price_error = inputs["ep_true"] - inputs["ep_fcst"]
    ws_base = inputs["ws_nwp"][time_slice]
    wd_base = inputs["wd_nwp"][time_slice]
    price_base = inputs["ep_fcst"][time_slice]

    scenario_costs = [
        _loss_cost_per_hour(ws_base, wd_base, price_base, wake=wake)
    ]
    rng = np.random.default_rng(seed)
    sampled_starts = rng.choice(eligible_starts, size=n_scenarios - 1, replace=True)
    for block_start in sampled_starts:
        block = slice(int(block_start), int(block_start) + horizon)
        ws_scenario = np.maximum(ws_base + ws_error[block], 0.0)
        wd_scenario = (wd_base + wd_error[block]) % 360.0
        price_scenario = price_base + price_error[block]
        scenario_costs.append(
            _loss_cost_per_hour(ws_scenario, wd_scenario, price_scenario, wake=wake)
        )
    return np.mean(np.stack(scenario_costs, axis=0), axis=0)


def _task_key(turbine: int, component: int) -> tuple[int, int]:
    return int(turbine), int(component)


def _strategy_options(strategy: str, wake_flag: bool) -> dict[str, bool | str]:
    """Return model switches for the selected literature-style strategy."""

    if strategy == "failure_repair":
        return {
            "trigger_mode": "failure_only",
            "disable_age_trigger": True,
            "static_initial": False,
            "optimize_energy_loss": False,
            "evaluation_wake": False,
            "label": "故障后维修策略",
        }
    if strategy == "predictive_rul":
        return {
            "trigger_mode": "rul_only",
            "disable_age_trigger": True,
            "static_initial": False,
            "optimize_energy_loss": False,
            "evaluation_wake": False,
            "label": "预测RUL状态维护策略",
        }
    if strategy == "weather_opportunity":
        return {
            "trigger_mode": "rul_only",
            "disable_age_trigger": True,
            "static_initial": False,
            "optimize_energy_loss": True,
            "evaluation_wake": False,
            "label": "风速可达性机会维护策略",
        }
    if strategy == "age_based_preventive":
        return {
            "trigger_mode": "multi",
            "disable_age_trigger": False,
            "static_initial": False,
            "optimize_energy_loss": True,
            "evaluation_wake": False,
            "rul_probability_threshold": 1.1,
            "aged_count_threshold": 1,
            "benefit_cost_filter": False,
            "force_candidate_mandatory": True,
            "label": "年龄阈值预防维护策略",
        }
    if strategy == "multi_opportunity":
        return {
            "trigger_mode": "multi",
            "disable_age_trigger": False,
            "static_initial": False,
            "optimize_energy_loss": True,
            "evaluation_wake": False,
            "label": "闭环多机会预测性维护策略",
        }
    if strategy == "wake_multi_opportunity":
        return {
            "trigger_mode": "multi",
            "disable_age_trigger": False,
            "static_initial": False,
            "optimize_energy_loss": True,
            "evaluation_wake": True,
            "label": "尾流感知闭环多机会维护策略",
        }
    if strategy == "ab_static_rul_weather":
        return {
            "trigger_mode": "rul_only",
            "disable_age_trigger": True,
            "static_initial": True,
            "optimize_energy_loss": True,
            "evaluation_wake": False,
            "benefit_cost_filter": False,
            "benefit_cost_selection": "soft_priority",
            "benefit_cost_filter_scope": "all_preventive",
            "preserve_age_triggered_tasks": True,
            "label": "B0 static RUL task set + weather",
        }
    if strategy == "ab_closed_rul_weather":
        return {
            "trigger_mode": "rul_only",
            "disable_age_trigger": True,
            "static_initial": False,
            "optimize_energy_loss": True,
            "evaluation_wake": False,
            "benefit_cost_filter": False,
            "benefit_cost_selection": "soft_priority",
            "benefit_cost_filter_scope": "all_preventive",
            "preserve_age_triggered_tasks": True,
            "label": "B1 closed-loop RUL task update + weather",
        }
    if strategy == "ab_closed_rul_age":
        return {
            "trigger_mode": "multi",
            "disable_age_trigger": False,
            "static_initial": False,
            "optimize_energy_loss": True,
            "evaluation_wake": False,
            "benefit_cost_filter": False,
            "benefit_cost_selection": "soft_priority",
            "benefit_cost_filter_scope": "all_preventive",
            "preserve_age_triggered_tasks": True,
            "execution_feedback": True,
            "label": "B2 closed-loop RUL + age opportunity",
        }
    if strategy == "ab_closed_rul_age_no_exec_feedback":
        return {
            "trigger_mode": "multi",
            "disable_age_trigger": False,
            "static_initial": False,
            "optimize_energy_loss": True,
            "evaluation_wake": False,
            "benefit_cost_filter": False,
            "benefit_cost_selection": "soft_priority",
            "benefit_cost_filter_scope": "all_preventive",
            "preserve_age_triggered_tasks": True,
            "execution_feedback": False,
            "label": "B2a closed-loop RUL + age without realized-access feedback",
        }
    if strategy == "ab_closed_rul_age_econ":
        return {
            "trigger_mode": "multi",
            "disable_age_trigger": False,
            "static_initial": False,
            "optimize_energy_loss": True,
            "evaluation_wake": False,
            "benefit_cost_filter": True,
            "benefit_cost_selection": "soft_priority",
            "benefit_cost_filter_scope": "all_preventive",
            "preserve_age_triggered_tasks": True,
            "label": "B3 closed-loop RUL + age + economic marker",
        }
    if strategy == "rul_only":
        return {
            "trigger_mode": "rul_only",
            "disable_age_trigger": True,
            "static_initial": False,
            "optimize_energy_loss": True,
            "evaluation_wake": wake_flag,
            "label": "仅RUL闭环策略",
        }
    if strategy == "static_initial":
        return {
            "trigger_mode": "multi",
            "disable_age_trigger": False,
            "static_initial": True,
            "optimize_energy_loss": True,
            "evaluation_wake": wake_flag,
            "label": "静态初始任务策略",
        }
    return {
        "trigger_mode": "multi",
        "disable_age_trigger": False,
        "static_initial": False,
        "optimize_energy_loss": True,
        "evaluation_wake": wake_flag,
        "label": "完整闭环多机会策略",
    }


def _failure_repair_task_set(
    true_rul: np.ndarray,
    ages: np.ndarray,
    lifetimes: np.ndarray,
) -> CandidateTaskSet:
    """Generate only corrective tasks for components that have already failed."""

    tasks = []
    for turbine in range(true_rul.shape[0]):
        for component in range(true_rul.shape[1]):
            if true_rul[turbine, component] > 0.0:
                continue
            risk_benefit = float(
                CM_COSTS_YUAN[component] + CORRECTIVE_COMPLETION_PREMIUM_YUAN
            )
            task = SimpleNamespace(
                turbine=turbine,
                component=component,
                component_name=COMPONENT_NAMES[component],
                reasons=("FAILURE_REPAIR",),
                probability_rul_within_lead=1.0,
                average_effective_cost=float(CM_COSTS_YUAN[component]),
                age_ratio=float(ages[turbine, component] / max(lifetimes[turbine, component], 1e-9)),
                task_hours=float(CM_HOURS[component]),
                corrective=True,
                priority=100.0,
                risk_reduction_benefit=risk_benefit,
                risk_exposure_benefit=0.0,
                screening_cost=float(CM_COSTS_YUAN[component]),
                net_benefit=risk_benefit - float(CM_COSTS_YUAN[component]),
                soft_priority_reward=0.0,
            )
            tasks.append(task)
    return CandidateTaskSet(tasks=tasks)


def _candidate_tasks_to_hourly(
    task_set,
    true_rul: np.ndarray,
    ongoing: dict[tuple[int, int], float],
    completed_today_skip: set[tuple[int, int]],
    force_preventive_mandatory: bool = False,
    shared_failure_outage: bool = False,
) -> list[HourlyTask]:
    tasks: list[HourlyTask] = []

    for (turbine, component), remaining in sorted(ongoing.items()):
        corrective_ongoing = bool(true_rul[turbine, component] <= 0.0)
        ongoing_direct_cost = CM_COSTS_YUAN[component] if corrective_ongoing else PM_COSTS_YUAN[component]
        ongoing_risk_benefit = (
            ongoing_direct_cost + CORRECTIVE_COMPLETION_PREMIUM_YUAN
            if corrective_ongoing
            else 100000.0
        )
        tasks.append(
            HourlyTask(
                task_id=f"WT{turbine + 1}_{COMPONENT_NAMES[component]}_ONGOING",
                turbine=turbine,
                component=component,
                component_name=COMPONENT_NAMES[component],
                component_cn=COMPONENT_CHINESE_NAMES[component],
                hours=int(np.ceil(remaining)),
                direct_cost_yuan=float(ongoing_direct_cost),
                risk_benefit_yuan=float(ongoing_risk_benefit),
                net_benefit_yuan=float(ongoing_risk_benefit - ongoing_direct_cost),
                reasons=("ONGOING",),
                crew_size=int(
                    CM_CREW_SIZES[component]
                    if corrective_ongoing
                    else PM_CREW_SIZES[component]
                ),
                mandatory=True,
                ongoing=True,
                corrective=corrective_ongoing,
                idle_blocks_turbine=not shared_failure_outage,
            )
        )

    for task in task_set.tasks:
        key = _task_key(task.turbine, task.component)
        if key in ongoing or key in completed_today_skip:
            continue

        corrective = bool(true_rul[task.turbine, task.component] <= 0.0 or task.corrective)
        hours = CM_HOURS[task.component] if corrective else PM_HOURS[task.component]
        direct_cost = CM_COSTS_YUAN[task.component] if corrective else PM_COSTS_YUAN[task.component]
        tasks.append(
            HourlyTask(
                task_id=f"WT{task.turbine + 1}_{COMPONENT_NAMES[task.component]}",
                turbine=int(task.turbine),
                component=int(task.component),
                component_name=COMPONENT_NAMES[task.component],
                component_cn=COMPONENT_CHINESE_NAMES[task.component],
                hours=int(np.ceil(hours)),
                direct_cost_yuan=float(direct_cost),
                risk_benefit_yuan=float(task.risk_reduction_benefit),
                net_benefit_yuan=float(task.net_benefit),
                reasons=tuple(task.reasons),
                crew_size=int(
                    CM_CREW_SIZES[task.component]
                    if corrective
                    else PM_CREW_SIZES[task.component]
                ),
                mandatory=bool(corrective or force_preventive_mandatory),
                ongoing=False,
                corrective=corrective,
                partial_progress_weight_yuan_per_hour=(
                    float(CORRECTIVE_COMPLETION_PREMIUM_YUAN / max(float(hours), 1.0))
                    if corrective
                    else 0.0
                ),
                idle_blocks_turbine=not shared_failure_outage,
            )
        )

    return tasks


def _simulate_real_execution(
    tasks: list[HourlyTask],
    work_plan: np.ndarray,
    downtime_plan: np.ndarray,
    access_actual: np.ndarray,
    loss_cost_actual: np.ndarray,
    *,
    keep_unfinished_mandatory: bool = False,
    failed_component_keys: set[tuple[int, int]] | None = None,
    use_failure_outage_loss: bool = False,
) -> tuple[list[dict], dict[tuple[int, int], float], float]:
    rows: list[dict] = []
    next_ongoing: dict[tuple[int, int], float] = {}
    repair_hour_loss_cost = 0.0
    completion_hour_by_key: dict[tuple[int, int], int] = {}

    for j, task in enumerate(tasks):
        planned_hours = np.where(work_plan[:, j] > 0)[0]
        downtime_hours = (
            np.where(downtime_plan[:, j] > 0)[0]
            if downtime_plan.shape == work_plan.shape
            else planned_hours
        )
        actual_hours = [h for h in planned_hours if access_actual[h, task.turbine] > 0.5]
        executed = len(actual_hours)
        complete = executed >= task.hours
        remaining = max(0.0, float(task.hours - executed))
        key = _task_key(task.turbine, task.component)
        if not complete and (
            task.ongoing
            or executed > 0
            or (keep_unfinished_mandatory and task.mandatory)
        ):
            next_ongoing[key] = remaining
        if complete and actual_hours:
            completion_hour_by_key[key] = max(actual_hours)
        if downtime_hours.size:
            if complete and actual_hours:
                outage_start = int(downtime_hours.min())
                outage_end = int(max(actual_hours))
                outage_hours = range(outage_start, outage_end + 1)
            else:
                outage_hours = range(int(downtime_hours.min()), loss_cost_actual.shape[0])
        else:
            outage_hours = []
        loss_cost = float(sum(loss_cost_actual[h, task.turbine] for h in outage_hours))
        repair_hour_loss_cost += loss_cost
        if planned_hours.size > 0 or downtime_hours.size > 0 or task.ongoing:
            rows.append(
                {
                    "task_id": task.task_id,
                    "turbine": task.turbine + 1,
                    "component": task.component_name,
                    "component_cn": task.component_cn,
                    "task_type": (
                        "ONGOING_CM"
                        if task.ongoing and task.corrective
                        else "ONGOING_PM"
                        if task.ongoing
                        else "CM"
                        if task.corrective
                        else "PM"
                    ),
                    "corrective": int(task.corrective),
                    "ongoing": int(task.ongoing),
                    "partial_progress_weight_yuan_per_hour": (
                        task.partial_progress_weight_yuan_per_hour
                    ),
                    "reasons": "+".join(task.reasons),
                    "required_hours": task.hours,
                    "planned_hours": int(planned_hours.size),
                    "executed_hours": executed,
                    "crew_size": task.crew_size,
                    "crew_person_hours": executed * task.crew_size,
                    "complete": int(complete),
                    "remaining_hours": remaining,
                    "first_planned_hour": None if planned_hours.size == 0 else int(planned_hours.min()),
                    "last_planned_hour": None if planned_hours.size == 0 else int(planned_hours.max()),
                    "first_actual_hour": None if not actual_hours else int(min(actual_hours)),
                    "last_actual_hour": None if not actual_hours else int(max(actual_hours)),
                    "first_downtime_hour": None if downtime_hours.size == 0 else int(downtime_hours.min()),
                    "last_downtime_hour": None if downtime_hours.size == 0 else int(downtime_hours.max()),
                    "repair_hour_loss_cost_yuan": loss_cost,
                    "loss_cost_yuan": loss_cost,
                    "direct_cost_yuan": task.direct_cost_yuan if complete else 0.0,
                    "risk_benefit_yuan": task.risk_benefit_yuan if complete else 0.0,
                }
            )

    if not use_failure_outage_loss:
        return rows, next_ongoing, repair_hour_loss_cost

    failed_keys = failed_component_keys or set()
    outage_loss_cost = 0.0
    for turbine in sorted({turbine for turbine, _ in failed_keys}):
        turbine_failed_keys = {key for key in failed_keys if key[0] == turbine}
        if turbine_failed_keys and turbine_failed_keys <= set(completion_hour_by_key):
            # The turbine is unavailable through the final repair hour.
            outage_hours = range(max(completion_hour_by_key[key] for key in turbine_failed_keys) + 1)
        else:
            outage_hours = range(loss_cost_actual.shape[0])
        outage_loss_cost += float(sum(loss_cost_actual[h, turbine] for h in outage_hours))

    return rows, next_ongoing, outage_loss_cost


def _actual_route_windows(execution_rows: list[dict]) -> list[tuple[int, int, int]]:
    """Build zero-based route windows from realized maintenance work only."""

    windows: list[tuple[int, int, int]] = []
    for row in execution_rows:
        first_hour = row.get("first_actual_hour")
        last_hour = row.get("last_actual_hour")
        if row.get("executed_hours", 0) <= 0 or first_hour is None or last_hour is None:
            continue
        windows.append(
            (
                int(row["turbine"]) - 1,
                int(first_hour),
                int(last_hour) + 1,
            )
        )
    return windows


def main() -> None:
    args = parse_args()
    if args.solver == "copt":
        from hourly_component_scheduler_copt import (
            solve_hourly_component_schedule as schedule_solver,
        )
    else:
        from hourly_component_scheduler_pulp import (
            solve_hourly_component_schedule as schedule_solver,
        )
    if args.loss_scenario_count < 1:
        raise ValueError("--loss-scenario-count must be at least 1.")
    if args.grid_size < 1:
        raise ValueError("--grid-size must be positive.")
    n_turbines = args.grid_size ** 2
    n_components = len(COMPONENT_NAMES)
    layout = default_grid_layout(args.grid_size, args.grid_size)
    routing_config = RoutingConfig(
        vessel_speed_knots=args.vessel_speed_knots,
        fuel_cost_yuan_per_sailing_hour=args.fuel_cost_yuan_per_sailing_hour,
        base_distance_km=args.base_distance_km,
        base_direction=args.base_direction,
    )
    visit_cost_proxy, visit_time_proxy = proxy_roundtrip_visit_costs(layout, routing_config)

    inputs = read_model_inputs(n_turbines, args.spatial_wind_std, args.input_seed)
    true_rul, forecast_mean, forecast_std, lifetimes = initial_component_state(
        args.rul_profile,
        n_turbines=n_turbines,
        seed=args.rul_seed,
    )
    expected_shape = (n_turbines, n_components)
    if true_rul.shape != expected_shape:
        raise ValueError(
            f"RUL profile returned {true_rul.shape}; expected {expected_shape}. "
            "The 64-turbine case must not repeat a smaller state matrix."
        )
    ages = np.maximum(lifetimes - forecast_mean, 0.0)

    strategy_options = _strategy_options(args.strategy, args.wake)
    disable_age_trigger = bool(strategy_options["disable_age_trigger"])
    trigger_config = TriggerConfig(
        lead_time_days=args.lead_time_days,
        rul_probability_threshold=float(
            strategy_options.get("rul_probability_threshold", args.rul_prob_threshold)
        ),
        age_threshold=args.age_threshold,
        aged_count_threshold=(
            10**9
            if disable_age_trigger
            else int(strategy_options.get("aged_count_threshold", args.aged_count_threshold))
        ),
        benefit_cost_filter=bool(strategy_options.get("benefit_cost_filter", True)),
        benefit_cost_selection=str(strategy_options.get("benefit_cost_selection", "soft_priority")),
        benefit_cost_filter_scope=str(strategy_options.get("benefit_cost_filter_scope", "all_preventive")),
        downtime_loss_per_hour=args.downtime_loss_per_hour,
        min_net_benefit=args.min_net_benefit,
        risk_exposure_cost_per_day=args.risk_exposure_cost_per_day,
        soft_priority_weight=0.25,
        soft_priority_cap=3000.0,
        preserve_age_triggered_tasks=bool(strategy_options.get("preserve_age_triggered_tasks", True)),
    )

    ongoing: dict[tuple[int, int], float] = {}
    roll_rows: list[dict] = []
    task_rows: list[dict] = []
    hourly_rows: list[dict] = []
    completed_components: set[tuple[int, int]] = set()
    solution_records: list[dict] = []
    static_initial_tasks = None

    start_hour = args.sample * 24
    print(f"8x8 hourly component MILP ({n_turbines} turbines)")
    print("RUL profile:", args.rul_profile)
    print("Strategy:", args.strategy, strategy_options["label"])
    print("Wake evaluation:", strategy_options["evaluation_wake"], "Optimize energy loss:", strategy_options["optimize_energy_loss"])
    print("time window:", inputs["time"][start_hour], "to", inputs["time"][start_hour + args.horizon_days * 24 - 1])

    initial_rows = []
    for turbine in range(n_turbines):
        for component in range(n_components):
            initial_rows.append(
                {
                    "turbine": turbine + 1,
                    "component": COMPONENT_NAMES[component],
                    "component_cn": COMPONENT_CHINESE_NAMES[component],
                    "true_rul_days": true_rul[turbine, component],
                    "forecast_mean_days": forecast_mean[turbine, component],
                    "rul_weibull_shape": forecast_std[turbine, component],
                    "age_days": ages[turbine, component],
                    "lifetime_days": lifetimes[turbine, component],
                    "age_ratio": ages[turbine, component] / max(lifetimes[turbine, component], 1e-9),
                }
            )

    for roll in range(args.horizon_days):
        day = args.sample + roll
        sl = slice(day * 24, (day + 1) * 24)
        scenarios = generate_rul_scenarios(
            forecast_mean,
            forecast_std,
            args.n_scenarios,
            seed=args.scenario_seed + roll,
            calibration_mode=args.rul_scenario_mode,
        )
        if strategy_options["trigger_mode"] == "failure_only":
            generated_task_set = _failure_repair_task_set(true_rul, ages, lifetimes)
        else:
            generated_task_set = generate_candidate_tasks(
                scenarios,
                ages_days=ages,
                lifetime_estimates_days=lifetimes,
                task_hours=PM_HOURS,
                preventive_costs=PM_COSTS_YUAN,
                corrective_costs=CM_COSTS_YUAN,
                component_names=COMPONENT_NAMES,
                config=trigger_config,
            )
        if strategy_options["static_initial"]:
            if static_initial_tasks is None:
                static_initial_tasks = list(generated_task_set.tasks)
            active_static_tasks = [
                task for task in static_initial_tasks
                if _task_key(task.turbine, task.component) not in completed_components
            ]
            task_set = SimpleNamespace(
                tasks=active_static_tasks,
                rejected_tasks=[],
                theta=None,
                tau=None,
                reason_labels=[],
                rejected_reason_labels=[],
            )
        else:
            task_set = generated_task_set
        tasks = _candidate_tasks_to_hourly(
            task_set,
            true_rul,
            ongoing,
            completed_components,
            force_preventive_mandatory=bool(
                strategy_options.get("force_candidate_mandatory", False)
            ),
            shared_failure_outage=args.strategy == "failure_repair",
        )

        access_forecast = _hourly_access(inputs["ws_nwp"][sl], inputs["wh_nwp"][sl])
        access_actual = _hourly_access(inputs["ws_true"][sl], inputs["wh_true"][sl])
        loss_forecast_raw = _expected_loss_from_empirical_error_scenarios(
            inputs,
            sl,
            wake=bool(strategy_options["evaluation_wake"]),
            n_scenarios=args.loss_scenario_count,
            seed=args.loss_scenario_seed + roll,
            error_pool=args.loss_error_pool,
        )
        loss_forecast = loss_forecast_raw if strategy_options["optimize_energy_loss"] else np.zeros_like(loss_forecast_raw)
        loss_actual = _loss_cost_per_hour(
            inputs["ws_true"][sl],
            inputs["wd_true"][sl],
            inputs["ep_true"][sl],
            wake=bool(strategy_options["evaluation_wake"]),
        )

        result = schedule_solver(
            tasks,
            access_forecast,
            loss_forecast,
            max_parallel_turbines=args.max_parallel_turbines,
            standard_work_hours=args.max_parallel_turbines * STANDARD_WORK_HOURS_PER_DAY,
            crew_cost_yuan_per_hour=CREW_COST_YUAN_PER_PERSON_HOUR,
            overtime_cost_yuan_per_hour=OVERTIME_COST_YUAN_PER_HOUR,
            vessel_cost_yuan_per_day=VESSEL_COST_YUAN_PER_DAY,
            solver_seed=args.solver_seed,
            solver_threads=args.solver_threads,
            solver_time_limit=args.solver_time_limit,
            mip_gap=args.mip_gap,
            enforce_contiguous=args.task_continuity == "contiguous",
            visit_cost_yuan_per_turbine=visit_cost_proxy if args.routing_cost_in_objective else None,
            visit_time_hours_per_turbine=visit_time_proxy if args.routing_cost_in_objective else None,
            vessel_technician_capacity=VESSEL_TECHNICIAN_CAPACITY,
            max_daily_turbine_visits=MAX_DAILY_TURBINE_VISITS,
        )
        if result["objective"] is None and result["status"] != "EMPTY":
            raise RuntimeError(
                f"No feasible schedule for sample={args.sample}, roll={roll}, "
                f"strategy={args.strategy}, status={result['status']}."
            )

        failed_component_keys = {
            (turbine, component)
            for turbine in range(n_turbines)
            for component in range(n_components)
            if true_rul[turbine, component] <= 0.0
        }
        execution_access = (
            access_actual
            if bool(strategy_options.get("execution_feedback", True))
            else access_forecast
        )
        exec_rows, next_ongoing, loss_cost = _simulate_real_execution(
            tasks,
            result["work_plan"],
            result.get("downtime_plan", np.zeros_like(result["work_plan"])),
            execution_access,
            loss_actual,
            # A failed task that has not started remains a mandatory candidate next day.
            # Only a task with realized work is carried as an ongoing task.
            keep_unfinished_mandatory=False,
            failed_component_keys=failed_component_keys,
            use_failure_outage_loss=args.strategy == "failure_repair",
        )

        completed_this_roll: set[tuple[int, int]] = set()
        for row in exec_rows:
            row["roll"] = roll
            row["date"] = inputs["time"][day * 24][:10]
            task_rows.append(row)
            if row["complete"]:
                key = (row["turbine"] - 1, COMPONENT_NAMES.index(row["component"]))
                completed_this_roll.add(key)
                completed_components.add(key)

        for h in range(24):
            for j, task in enumerate(tasks):
                downtime_plan = result.get("downtime_plan", np.zeros_like(result["work_plan"]))
                if result["work_plan"].shape[1] and (
                    result["work_plan"][h, j] > 0 or downtime_plan[h, j] > 0
                ):
                    hourly_rows.append(
                        {
                            "roll": roll,
                            "timestamp": inputs["time"][day * 24 + h],
                            "hour": h,
                            "turbine": task.turbine + 1,
                            "component": task.component_name,
                            "component_cn": task.component_cn,
                            "planned_work": int(result["work_plan"][h, j] > 0),
                            "planned_downtime": int(downtime_plan[h, j] > 0),
                            "forecast_access": int(access_forecast[h, task.turbine]),
                            "actual_access": int(access_actual[h, task.turbine]),
                            "forecast_loss_cost_yuan": float(loss_forecast[h, task.turbine]),
                            "actual_loss_cost_yuan": float(loss_actual[h, task.turbine]),
                        }
                    )

        direct_cost = sum(row["direct_cost_yuan"] for row in exec_rows)
        crew_hours = sum(row["executed_hours"] for row in exec_rows)
        crew_person_hours = sum(row["crew_person_hours"] for row in exec_rows)
        crew_cost = crew_person_hours * CREW_COST_YUAN_PER_PERSON_HOUR
        task_windows = _actual_route_windows(exec_rows)
        route_eval = evaluate_dropoff_pickup_route(task_windows, layout, routing_config)
        vessel_cost = (
            VESSEL_COST_YUAN_PER_DAY if route_eval["visited_turbines"] else 0.0
        )
        fuel_cost = float(route_eval["fuel_cost_yuan"])
        sailing_time = float(route_eval["sailing_time_hours"])
        sailing_distance = float(route_eval["sailing_distance_km"])
        overtime_hours = max(
            0.0,
            crew_hours + sailing_time - args.max_parallel_turbines * STANDARD_WORK_HOURS_PER_DAY,
        )
        overtime_cost = overtime_hours * OVERTIME_COST_YUAN_PER_HOUR
        total_actual_cost = direct_cost + crew_cost + vessel_cost + fuel_cost + overtime_cost + loss_cost

        roll_rows.append(
            {
                "roll": roll,
                "date": inputs["time"][day * 24][:10],
                "candidate_component_tasks": len(task_set.tasks),
                "scheduled_tasks": int(np.sum(np.asarray(result["work_plan"]).sum(axis=0) > 0)),
                "completed_tasks": len(completed_this_roll),
                "ongoing_in": len(ongoing),
                "ongoing_out": len(next_ongoing),
                "executed_hours": crew_hours,
                "crew_person_hours": crew_person_hours,
                "direct_cost_yuan": direct_cost,
                "crew_cost_yuan": crew_cost,
                "vessel_cost_yuan": vessel_cost,
                "fuel_cost_yuan": fuel_cost,
                "sailing_distance_km": sailing_distance,
                "sailing_time_hours": sailing_time,
                "visited_turbines": ";".join(map(str, route_eval["visited_turbines"])),
                "route_nodes": "-".join(map(str, route_eval["route_nodes"])),
                "route_events": route_eval["route_events"],
                "overtime_cost_yuan": overtime_cost,
                "loss_cost_yuan": loss_cost,
                "total_actual_cost_yuan": total_actual_cost,
                "solver_status": result["status"],
                "solver": args.solver,
                "solver_seed_applied": int(bool(result.get("solver_seed_applied", False))),
                "objective": result["objective"],
                "solver_runtime_sec": result["runtime_sec"],
                "solver_mip_gap": result["mip_gap"],
                "solver_num_vars": result["num_vars"],
                "solver_num_constraints": result["num_constraints"],
                "strategy": args.strategy,
                "strategy_label": strategy_options["label"],
                "wake": int(bool(strategy_options["evaluation_wake"])),
                "optimize_energy_loss": int(bool(strategy_options["optimize_energy_loss"])),
                "rul_profile": args.rul_profile,
                "max_parallel_turbines": args.max_parallel_turbines,
                "horizon_days": args.horizon_days,
                "grid_size": args.grid_size,
                "n_turbines": n_turbines,
                "n_components_total": n_turbines * n_components,
                "loss_scenario_count": args.loss_scenario_count,
                "loss_error_pool": args.loss_error_pool,
                "routing_cost_in_objective": int(bool(args.routing_cost_in_objective)),
                "vessel_speed_knots": args.vessel_speed_knots,
                "fuel_cost_yuan_per_sailing_hour": args.fuel_cost_yuan_per_sailing_hour,
                "base_distance_km": args.base_distance_km,
                "base_direction": args.base_direction,
                "rul_scenario_mode": args.rul_scenario_mode,
                "task_continuity": args.task_continuity,
                "execution_feedback": int(bool(strategy_options.get("execution_feedback", True))),
                "n_rul_scenarios": args.n_scenarios,
                "rul_seed": args.rul_seed,
                "scenario_seed": args.scenario_seed,
                "input_seed": args.input_seed,
                "loss_scenario_seed": args.loss_scenario_seed,
                "solver_seed": args.solver_seed,
                "lead_time_days": args.lead_time_days,
                "rul_probability_threshold": args.rul_prob_threshold,
                "age_threshold": args.age_threshold,
                "aged_count_threshold": args.aged_count_threshold,
                "spatial_wind_std": args.spatial_wind_std,
                "material_cost_multiplier": MATERIAL_COST_MULTIPLIER,
                "experiment_config_version": EXPERIMENT_CONFIG_VERSION,
            }
        )
        solution_records.append(
            {
                "roll": roll,
                "tasks": tasks,
                "task_set": task_set,
                "result": result,
                "access_forecast": access_forecast,
                "access_actual": access_actual,
                "loss_forecast": loss_forecast,
                "loss_actual": loss_actual,
                "true_rul": true_rul.copy(),
                "forecast_mean": forecast_mean.copy(),
                "ages": ages.copy(),
            }
        )

        print(
            f"roll {roll + 1}/{args.horizon_days}: candidates={len(task_set.tasks)}, "
            f"scheduled={roll_rows[-1]['scheduled_tasks']}, completed={len(completed_this_roll)}, "
            f"ongoing_out={len(next_ongoing)}, cost={total_actual_cost:.1f}"
        )

        # Daily state update after realized execution.
        true_rul = np.maximum(true_rul - 1.0, 0.0)
        forecast_mean = np.maximum(forecast_mean - 1.0, 0.0)
        ages = ages + 1.0
        for turbine, component in completed_this_roll:
            true_rul[turbine, component] = lifetimes[turbine, component]
            forecast_mean[turbine, component] = lifetimes[turbine, component]
            ages[turbine, component] = 0.0

        ongoing = next_ongoing

    suffix = f"sample{args.sample}_{args.output_tag}"
    initial_csv = CASE_DIR / f"hourly_milp_initial_rul_{suffix}.csv"
    roll_csv = CASE_DIR / f"hourly_milp_roll_summary_{suffix}.csv"
    task_csv = CASE_DIR / f"hourly_milp_task_execution_{suffix}.csv"
    hourly_csv = CASE_DIR / f"hourly_milp_hourly_schedule_{suffix}.csv"
    pkl_path = CASE_DIR / f"CASE_8X8_MULTICOMP_HOURLY_MILP_{args.n_scenarios}S_{suffix}.pkl"

    pd.DataFrame(initial_rows).to_csv(initial_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(roll_rows).to_csv(roll_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(task_rows).to_csv(task_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(hourly_rows).to_csv(hourly_csv, index=False, encoding="utf-8-sig")
    with pkl_path.open("wb") as f:
        pickle.dump(
            {
                "case": "8x8_64turbine_multicomponent_hourly_milp",
                "args": vars(args),
                "roll_summary_csv": str(roll_csv),
                "initial_rul_csv": str(initial_csv),
                "task_execution_csv": str(task_csv),
                "hourly_schedule_csv": str(hourly_csv),
                "solution_records": solution_records,
            },
            f,
        )

    print("saved:", roll_csv)
    print("saved:", initial_csv)
    print("saved:", task_csv)
    print("saved:", hourly_csv)
    print("saved:", pkl_path)


if __name__ == "__main__":
    main()
