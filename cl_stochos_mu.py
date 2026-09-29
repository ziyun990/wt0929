"""Helper models for the CL-STOCHOS-MU maintenance strategy.

This module intentionally does not modify STOCHOS.py. It provides the
state-trigger and wake-loss preprocessing pieces that can be called from a
rolling solver before invoking STOCHOS().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import ceil, log, sqrt
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TriggerConfig:
    """Thresholds for dynamic candidate task generation."""

    lead_time_days: float = 3.0
    rul_probability_threshold: float = 0.6
    effective_cost_threshold: float | None = None
    opportunistic_cost_threshold: float | None = None
    age_threshold: float = 0.8
    aged_fraction_threshold: float = 0.15
    aged_count_threshold: int | None = None
    influential_age_jump_days: float = 2.0
    minor_power_derate: float = 0.05
    default_task_hours: float = 8.0
    benefit_cost_filter: bool = False
    downtime_loss_per_hour: float = 0.0
    min_net_benefit: float = 0.0
    benefit_cost_filter_scope: str = "all_preventive"
    risk_exposure_cost_per_day: float = 0.0
    benefit_cost_selection: str = "hard_filter"
    soft_priority_weight: float = 0.25
    soft_priority_cap: float = 3000.0
    preserve_age_triggered_tasks: bool = False


@dataclass(frozen=True)
class CandidateTask:
    """A component-level maintenance candidate."""

    turbine: int
    component: int
    component_name: str
    reasons: tuple[str, ...]
    probability_rul_within_lead: float
    average_effective_cost: float
    age_ratio: float
    task_hours: float
    corrective: bool
    priority: float
    risk_reduction_benefit: float = 0.0
    risk_exposure_benefit: float = 0.0
    screening_cost: float = 0.0
    net_benefit: float = 0.0
    soft_priority_reward: float = 0.0


@dataclass(frozen=True)
class WakeConfig:
    """Parameters for a compact Jensen wake-loss calculation."""

    rated_power_mw: float = 10.0
    rotor_diameter_m: float = 178.3
    hub_height_m: float = 119.0
    roughness_length_m: float = 0.0002
    thrust_coefficient: float = 0.8
    wake_decay: float | None = None
    wind_direction_is_from: bool = True

    @property
    def rotor_radius_m(self) -> float:
        return self.rotor_diameter_m / 2.0

    @property
    def jensen_decay(self) -> float:
        if self.wake_decay is not None:
            return self.wake_decay
        return 1.0 / (2.0 * log(self.hub_height_m / self.roughness_length_m))


@dataclass
class CandidateTaskSet:
    """STOCHOS-ready turbine-level task parameters."""

    tasks: list[CandidateTask] = field(default_factory=list)
    rejected_tasks: list[CandidateTask] = field(default_factory=list)
    theta: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    tau: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    turbine_priority: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    soft_priority_reward: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    reason_labels: list[list[str]] = field(default_factory=list)
    rejected_reason_labels: list[list[str]] = field(default_factory=list)


def _as_3d_rul(rul_scenarios: np.ndarray) -> np.ndarray:
    """Return RUL scenarios as (n_turbines, n_components, n_scenarios)."""

    arr = np.asarray(rul_scenarios, dtype=float)
    if arr.ndim == 2:
        return arr[:, np.newaxis, :]
    if arr.ndim == 3:
        return arr
    raise ValueError("rul_scenarios must have shape (N, S) or (N, M, S)")


def _broadcast_component_param(
    value: np.ndarray | Sequence[float] | float | None,
    shape: tuple[int, int],
    default: float,
) -> np.ndarray:
    """Broadcast scalar, turbine, or component parameters to (N, M)."""

    if value is None:
        return np.full(shape, default, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(shape, float(arr), dtype=float)
    if arr.shape == shape:
        return arr.astype(float)
    if arr.shape == (shape[0],):
        return np.repeat(arr[:, np.newaxis], shape[1], axis=1)
    if arr.shape == (shape[1],):
        return np.repeat(arr[np.newaxis, :], shape[0], axis=0)
    raise ValueError(f"cannot broadcast parameter with shape {arr.shape} to {shape}")


def lead_time_failure_probability(
    rul_scenarios: np.ndarray,
    lead_time_days: float | np.ndarray,
) -> np.ndarray:
    """Estimate P(RUL <= lead time) from scenarios."""

    rul = _as_3d_rul(rul_scenarios)
    n_turbines, n_components, _ = rul.shape
    lead = _broadcast_component_param(lead_time_days, (n_turbines, n_components), 0.0)
    return np.mean(rul <= lead[:, :, np.newaxis], axis=2)


def average_effective_maintenance_cost(
    rul_scenarios: np.ndarray,
    age_days: np.ndarray | Sequence[float] | float,
    lead_time_days: float | np.ndarray,
    preventive_cost: np.ndarray | Sequence[float] | float,
    corrective_cost: np.ndarray | Sequence[float] | float,
) -> np.ndarray:
    """Compute Zhou-style lead-time-aware average effective cost."""

    rul = _as_3d_rul(rul_scenarios)
    shape = rul.shape[:2]
    ages = _broadcast_component_param(age_days, shape, 0.0)
    lead = _broadcast_component_param(lead_time_days, shape, 0.0)
    cp = _broadcast_component_param(preventive_cost, shape, 4000.0)
    cf = _broadcast_component_param(corrective_cost, shape, 10000.0)

    fail_prob = lead_time_failure_probability(rul, lead)
    numerator = cp * (1.0 - fail_prob) + cf * fail_prob
    denominator = np.maximum(ages + lead, 1e-6)
    return numerator / denominator


def generate_candidate_tasks(
    rul_scenarios: np.ndarray,
    ages_days: np.ndarray | Sequence[float] | float | None = None,
    lifetime_estimates_days: np.ndarray | Sequence[float] | float | None = None,
    task_hours: np.ndarray | Sequence[float] | float | None = None,
    preventive_costs: np.ndarray | Sequence[float] | float | None = None,
    corrective_costs: np.ndarray | Sequence[float] | float | None = None,
    component_names: Sequence[str] | None = None,
    shock_state: np.ndarray | Sequence[str] | None = None,
    ongoing_theta: np.ndarray | Sequence[int] | None = None,
    ongoing_rho: np.ndarray | Sequence[int] | None = None,
    remaining_hours: np.ndarray | Sequence[float] | None = None,
    config: TriggerConfig | None = None,
) -> CandidateTaskSet:
    """Generate dynamic theta/tau inputs for STOCHOS."""

    cfg = config or TriggerConfig()
    rul = _as_3d_rul(rul_scenarios)
    n_turbines, n_components, _ = rul.shape
    shape = (n_turbines, n_components)

    ages = _broadcast_component_param(ages_days, shape, 0.0)
    lifetimes = _broadcast_component_param(
        lifetime_estimates_days,
        shape,
        np.maximum(np.mean(rul), 1.0),
    )
    hours = _broadcast_component_param(task_hours, shape, cfg.default_task_hours)
    cp = _broadcast_component_param(preventive_costs, shape, 4000.0)
    cf = _broadcast_component_param(corrective_costs, shape, 10000.0)

    names = list(component_names or [f"component_{j + 1}" for j in range(n_components)])
    if len(names) != n_components:
        raise ValueError("component_names length must match the component dimension")

    if shock_state is None:
        shocks = np.full(shape, "none", dtype=object)
    else:
        shocks = np.asarray(shock_state, dtype=object)
        if shocks.ndim == 1 and n_components == 1:
            shocks = shocks[:, np.newaxis]
        if shocks.shape != shape:
            raise ValueError(f"shock_state must have shape {shape}")
        shocks = np.char.lower(shocks.astype(str))

    if ongoing_theta is None:
        base_theta = np.zeros(n_turbines, dtype=int)
    else:
        base_theta = np.asarray(ongoing_theta, dtype=int).copy()

    if ongoing_rho is None:
        base_rho = np.zeros(n_turbines, dtype=int)
    else:
        base_rho = np.asarray(ongoing_rho, dtype=int)

    if remaining_hours is None:
        remaining = np.zeros(n_turbines, dtype=float)
    else:
        remaining = np.asarray(remaining_hours, dtype=float)

    fail_prob = lead_time_failure_probability(rul, cfg.lead_time_days)
    current_failure_prob = np.mean(rul <= 0.0, axis=2)
    effective_cost = average_effective_maintenance_cost(
        rul,
        ages,
        cfg.lead_time_days,
        cp,
        cf,
    )
    age_ratio = ages / np.maximum(lifetimes, 1e-6)

    aged_mask = age_ratio >= cfg.age_threshold
    if cfg.aged_count_threshold is None:
        aged_limit = max(1, ceil(cfg.aged_fraction_threshold * n_turbines * n_components))
    else:
        aged_limit = cfg.aged_count_threshold
    age_opportunity = int(np.sum(aged_mask) >= aged_limit)

    tasks: list[CandidateTask] = []
    rejected_tasks: list[CandidateTask] = []
    theta = base_theta.copy()
    tau = np.zeros(n_turbines, dtype=float)
    turbine_priority = np.zeros(n_turbines, dtype=float)
    soft_priority_reward_by_turbine = np.zeros(n_turbines, dtype=float)
    reason_labels: list[list[str]] = [[] for _ in range(n_turbines)]
    rejected_reason_labels: list[list[str]] = [[] for _ in range(n_turbines)]

    for i in range(n_turbines):
        if base_rho[i] == 1:
            theta[i] = 1
            tau[i] = max(float(remaining[i]), 1.0)
            reason_labels[i].append("ONGOING")
            turbine_priority[i] = max(turbine_priority[i], 100.0)
            continue

        for c in range(n_components):
            reasons: list[str] = []
            corrective = False
            if current_failure_prob[i, c] >= 0.5:
                corrective = True

            if fail_prob[i, c] >= cfg.rul_probability_threshold:
                reasons.append("RUL_LEADTIME")

            if (
                cfg.effective_cost_threshold is not None
                and effective_cost[i, c] <= cfg.effective_cost_threshold
            ):
                reasons.append("EFFECTIVE_COST")

            if (
                cfg.opportunistic_cost_threshold is not None
                and cfg.effective_cost_threshold is not None
                and cfg.effective_cost_threshold < effective_cost[i, c] <= cfg.opportunistic_cost_threshold
            ):
                reasons.append("OPPORTUNISTIC_COST")

            shock = str(shocks[i, c]).lower()
            minor_monitor = False
            if shock == "critical":
                reasons.append("CRITICAL_SHOCK")
                corrective = True
            elif shock == "influential":
                reasons.append("INFLUENTIAL_SHOCK")
            elif shock == "minor":
                minor_monitor = True

            if age_opportunity and aged_mask[i, c]:
                reasons.append("AGE_PROPORTION")

            if not reasons:
                continue
            if minor_monitor:
                reasons.append("MINOR_SHOCK_MONITOR")

            priority = (
                100.0 * corrective
                + 40.0 * fail_prob[i, c]
                + 25.0 * min(age_ratio[i, c], 1.5)
                + 10.0 * ("INFLUENTIAL_SHOCK" in reasons)
                + 5.0 * ("OPPORTUNISTIC_COST" in reasons)
            )
            lead_time_shortfall_days = float(np.mean(np.maximum(cfg.lead_time_days - rul[i, c, :], 0.0)))
            risk_exposure_benefit = float(cfg.risk_exposure_cost_per_day * lead_time_shortfall_days)
            risk_reduction_benefit = float(fail_prob[i, c] * cf[i, c] + risk_exposure_benefit)
            screening_cost = float(cp[i, c] + cfg.downtime_loss_per_hour * hours[i, c])
            net_benefit = float(risk_reduction_benefit - screening_cost)

            strong_reasons = {"RUL_LEADTIME", "EFFECTIVE_COST"}
            weak_reasons = {"AGE_PROPORTION", "OPPORTUNISTIC_COST", "MINOR_SHOCK_MONITOR"}
            reason_set = set(dict.fromkeys(reasons))
            scope = cfg.benefit_cost_filter_scope
            if scope == "weak_triggers":
                filter_in_scope = bool(reason_set & weak_reasons) and not bool(reason_set & strong_reasons)
            elif scope == "age_only":
                filter_in_scope = "AGE_PROPORTION" in reason_set and not bool(reason_set & strong_reasons)
            else:
                filter_in_scope = True

            must_keep = (
                corrective
                or "INFLUENTIAL_SHOCK" in reasons
                or "CRITICAL_SHOCK" in reasons
                or (cfg.preserve_age_triggered_tasks and "AGE_PROPORTION" in reasons)
            )

            soft_priority_reward = 0.0
            if cfg.benefit_cost_filter and cfg.benefit_cost_selection == "soft_priority" and filter_in_scope:
                raw_reward = (net_benefit - cfg.min_net_benefit) * cfg.soft_priority_weight
                if must_keep:
                    raw_reward = max(raw_reward, 0.5 * cfg.soft_priority_cap)
                soft_priority_reward = float(np.clip(raw_reward, -cfg.soft_priority_cap, cfg.soft_priority_cap))

            task = CandidateTask(
                turbine=i,
                component=c,
                component_name=names[c],
                reasons=tuple(dict.fromkeys(reasons)),
                probability_rul_within_lead=float(fail_prob[i, c]),
                average_effective_cost=float(effective_cost[i, c]),
                age_ratio=float(age_ratio[i, c]),
                task_hours=float(hours[i, c]),
                corrective=corrective,
                priority=float(priority),
                risk_reduction_benefit=risk_reduction_benefit,
                risk_exposure_benefit=risk_exposure_benefit,
                screening_cost=screening_cost,
                net_benefit=net_benefit,
                soft_priority_reward=soft_priority_reward,
            )

            if (
                cfg.benefit_cost_filter
                and cfg.benefit_cost_selection == "hard_filter"
                and filter_in_scope
                and not must_keep
                and net_benefit < cfg.min_net_benefit
            ):
                rejected_tasks.append(task)
                rejected_reason_labels[i].extend(task.reasons)
                continue

            tasks.append(task)
            theta[i] = 1
            tau[i] += float(hours[i, c])
            turbine_priority[i] = max(turbine_priority[i], priority)
            soft_priority_reward_by_turbine[i] = float(
                np.clip(
                    soft_priority_reward_by_turbine[i] + soft_priority_reward,
                    -cfg.soft_priority_cap,
                    cfg.soft_priority_cap,
                )
            )
            reason_labels[i].extend(task.reasons)

    for i in range(n_turbines):
        if theta[i] and tau[i] <= 0:
            tau[i] = cfg.default_task_hours
        reason_labels[i] = list(dict.fromkeys(reason_labels[i]))

    return CandidateTaskSet(
        tasks=tasks,
        rejected_tasks=rejected_tasks,
        theta=theta.astype(int),
        tau=tau,
        turbine_priority=turbine_priority,
        soft_priority_reward=soft_priority_reward_by_turbine,
        reason_labels=reason_labels,
        rejected_reason_labels=rejected_reason_labels,
    )


@lru_cache(maxsize=8)
def load_power_curve_bins(path: str | Path = "method_of_bins.csv") -> np.ndarray:
    """Load the wind-speed-to-normalized-power bins used by STOCHOS."""

    return pd.read_csv(path, usecols=[0, 1]).to_numpy(dtype=float)


def power_curve_mw(
    wind_speed: np.ndarray | float,
    bins: np.ndarray,
    rated_power_mw: float,
) -> np.ndarray:
    """Convert wind speed to MW using the binning method from STOCHOS."""

    ws = np.asarray(wind_speed, dtype=float)
    out = np.zeros_like(ws, dtype=float)
    for j in range(bins.shape[0] - 1):
        mask = (ws >= bins[j, 0]) & (ws < bins[j + 1, 0])
        out[mask] = bins[j, 1] * rated_power_mw
    return out


def default_line_layout(
    n_turbines: int,
    spacing_rotor_diameters: float = 7.0,
    rotor_diameter_m: float = 178.3,
) -> np.ndarray:
    """Create a simple line layout when no farm coordinates are available."""

    spacing = spacing_rotor_diameters * rotor_diameter_m
    x = np.arange(n_turbines, dtype=float) * spacing
    y = np.zeros(n_turbines, dtype=float)
    return np.column_stack([x, y])


def default_grid_layout(
    rows: int,
    cols: int,
    spacing_rotor_diameters: float = 7.0,
    rotor_diameter_m: float = 178.3,
    row_spacing_rotor_diameters: float = 8.0,
) -> np.ndarray:
    """Create the adapted NORCOWE-style 7D by 8D rectangular layout."""

    column_spacing = spacing_rotor_diameters * rotor_diameter_m
    row_spacing = row_spacing_rotor_diameters * rotor_diameter_m
    xs = np.arange(cols, dtype=float) * column_spacing
    ys = np.arange(rows, dtype=float) * row_spacing
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.reshape(-1), yy.reshape(-1)])


def _flow_unit_vector(wind_direction_deg: float, direction_is_from: bool) -> np.ndarray:
    """Return the horizontal unit vector in the downwind flow direction."""

    deg = wind_direction_deg + (180.0 if direction_is_from else 0.0)
    rad = np.deg2rad(90.0 - deg)
    return np.array([np.cos(rad), np.sin(rad)], dtype=float)


def _circle_overlap_fraction(distance: float, wake_radius: float, rotor_radius: float) -> float:
    """Fraction of a rotor disk covered by a circular wake."""

    d = abs(float(distance))
    rw = float(wake_radius)
    rr = float(rotor_radius)
    if d >= rw + rr:
        return 0.0
    if d <= abs(rw - rr):
        return 1.0 if rw >= rr else (rw * rw) / (rr * rr)

    part1 = rw * rw * np.arccos((d * d + rw * rw - rr * rr) / (2.0 * d * rw))
    part2 = rr * rr * np.arccos((d * d + rr * rr - rw * rw) / (2.0 * d * rr))
    part3 = 0.5 * sqrt(
        max(
            0.0,
            (-d + rw + rr) * (d + rw - rr) * (d - rw + rr) * (d + rw + rr),
        )
    )
    overlap_area = part1 + part2 - part3
    rotor_area = np.pi * rr * rr
    return float(np.clip(overlap_area / rotor_area, 0.0, 1.0))


def _jensen_deficit_matrix(
    wind_direction_deg: float,
    layout_xy: np.ndarray,
    config: WakeConfig,
) -> np.ndarray:
    """Return source-to-target Jensen deficits for one wind direction.

    Entry ``[i, j]`` is the speed deficit imposed by source turbine ``i`` on
    target turbine ``j``. Geometry is evaluated once and reused for the
    baseline and every single-turbine outage state.
    """

    layout = np.asarray(layout_xy, dtype=float)
    flow = _flow_unit_vector(wind_direction_deg, config.wind_direction_is_from)
    cross = np.array([-flow[1], flow[0]])
    downwind_pos = layout @ flow
    crosswind_pos = layout @ cross

    # Rows are wake sources and columns are downstream target turbines.
    dx = downwind_pos[np.newaxis, :] - downwind_pos[:, np.newaxis]
    dy = np.abs(crosswind_pos[np.newaxis, :] - crosswind_pos[:, np.newaxis])
    valid = dx > 0.0

    rotor_radius = float(config.rotor_radius_m)
    wake_radius = rotor_radius + config.jensen_decay * np.maximum(dx, 0.0)
    overlap = np.zeros_like(dx, dtype=float)

    contained = valid & (dy <= np.abs(wake_radius - rotor_radius))
    if np.any(contained):
        overlap[contained] = np.where(
            wake_radius[contained] >= rotor_radius,
            1.0,
            np.square(wake_radius[contained] / rotor_radius),
        )

    partial = valid & ~contained & (dy < wake_radius + rotor_radius)
    if np.any(partial):
        d = dy[partial]
        rw = wake_radius[partial]
        rr = rotor_radius
        arg1 = np.clip((d * d + rw * rw - rr * rr) / (2.0 * d * rw), -1.0, 1.0)
        arg2 = np.clip((d * d + rr * rr - rw * rw) / (2.0 * d * rr), -1.0, 1.0)
        radical = np.maximum(
            0.0,
            (-d + rw + rr) * (d + rw - rr) * (d - rw + rr) * (d + rw + rr),
        )
        overlap_area = (
            rw * rw * np.arccos(arg1)
            + rr * rr * np.arccos(arg2)
            - 0.5 * np.sqrt(radical)
        )
        overlap[partial] = np.clip(overlap_area / (np.pi * rr * rr), 0.0, 1.0)

    base_deficit = 1.0 - sqrt(max(0.0, 1.0 - config.thrust_coefficient))
    denominator = np.square(1.0 + config.jensen_decay * np.maximum(dx, 0.0) / rotor_radius)
    return np.where(valid, base_deficit * overlap / denominator, 0.0)


def jensen_wake_power(
    wind_speed: float,
    wind_direction_deg: float,
    layout_xy: np.ndarray,
    power_bins: np.ndarray,
    active: Iterable[bool] | None = None,
    config: WakeConfig | None = None,
) -> np.ndarray:
    """Compute turbine power under a compact Jensen wake model."""

    cfg = config or WakeConfig()
    layout = np.asarray(layout_xy, dtype=float)
    n_turbines = layout.shape[0]
    if active is None:
        active_mask = np.ones(n_turbines, dtype=bool)
    else:
        active_mask = np.asarray(list(active), dtype=bool)
    if active_mask.shape != (n_turbines,):
        raise ValueError("active must have length equal to layout rows")

    flow = _flow_unit_vector(wind_direction_deg, cfg.wind_direction_is_from)
    cross = np.array([-flow[1], flow[0]])
    downwind_pos = layout @ flow
    crosswind_pos = layout @ cross

    local_speed = np.full(n_turbines, float(wind_speed), dtype=float)
    base_deficit = 1.0 - sqrt(max(0.0, 1.0 - cfg.thrust_coefficient))

    for j in range(n_turbines):
        if not active_mask[j]:
            local_speed[j] = 0.0
            continue

        deficits: list[float] = []
        for i in range(n_turbines):
            if i == j or not active_mask[i]:
                continue
            dx = downwind_pos[j] - downwind_pos[i]
            if dx <= 0:
                continue
            dy = crosswind_pos[j] - crosswind_pos[i]
            wake_radius = cfg.rotor_radius_m + cfg.jensen_decay * dx
            overlap = _circle_overlap_fraction(dy, wake_radius, cfg.rotor_radius_m)
            if overlap <= 0:
                continue
            deficit = base_deficit / (1.0 + cfg.jensen_decay * dx / cfg.rotor_radius_m) ** 2
            deficits.append(deficit * overlap)

        if deficits:
            combined = sqrt(float(np.sum(np.square(deficits))))
            local_speed[j] = max(0.0, float(wind_speed) * (1.0 - combined))

    power = power_curve_mw(local_speed, power_bins, cfg.rated_power_mw)
    power[~active_mask] = 0.0
    return power


def wake_marginal_loss(
    wind_speed: np.ndarray,
    wind_direction_deg: np.ndarray,
    layout_xy: np.ndarray,
    power_bins: np.ndarray,
    config: WakeConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute field-level marginal power loss from each turbine outage."""

    ws = np.asarray(wind_speed, dtype=float)
    wd = np.asarray(wind_direction_deg, dtype=float)
    n_turbines = np.asarray(layout_xy).shape[0]

    if ws.ndim == 2:
        free_ws = ws
    elif ws.ndim == 3:
        free_ws = np.mean(ws, axis=1)
    else:
        raise ValueError("wind_speed must have shape (T, S) or (T, N, S)")

    if wd.ndim == 1:
        free_wd = np.repeat(wd[:, np.newaxis], free_ws.shape[1], axis=1)
    elif wd.ndim == 2:
        free_wd = wd
    elif wd.ndim == 3:
        free_wd = np.mean(wd, axis=1)
    else:
        raise ValueError("wind_direction_deg must have shape (T,), (T, S), or (T, N, S)")

    if free_wd.shape != free_ws.shape:
        raise ValueError("wind direction shape is not compatible with wind speed")

    horizon, n_scenarios = free_ws.shape
    baseline = np.zeros((horizon, n_scenarios), dtype=float)
    marginal = np.zeros((horizon, n_turbines, n_scenarios), dtype=float)

    cfg = config or WakeConfig()
    outage_index = np.arange(n_turbines)

    for t in range(horizon):
        for s in range(n_scenarios):
            free_speed = float(free_ws[t, s])
            deficits = _jensen_deficit_matrix(
                float(free_wd[t, s]),
                layout_xy,
                cfg,
            )
            deficit_sq = np.square(deficits)
            total_deficit_sq = np.sum(deficit_sq, axis=0)

            baseline_speed = np.maximum(
                0.0,
                free_speed * (1.0 - np.sqrt(total_deficit_sq)),
            )
            baseline_power = power_curve_mw(
                baseline_speed,
                power_bins,
                cfg.rated_power_mw,
            )
            base_power = float(np.sum(baseline_power))
            baseline[t, s] = base_power

            # Removing source turbine i subtracts its squared wake deficit from
            # every target. The stopped turbine's own output is then set to zero.
            outage_deficit_sq = np.maximum(
                total_deficit_sq[np.newaxis, :] - deficit_sq,
                0.0,
            )
            outage_speed = np.maximum(
                0.0,
                free_speed * (1.0 - np.sqrt(outage_deficit_sq)),
            )
            outage_power = power_curve_mw(
                outage_speed,
                power_bins,
                cfg.rated_power_mw,
            )
            outage_power[outage_index, outage_index] = 0.0
            marginal[t, :, s] = base_power - np.sum(outage_power, axis=1)

    return marginal, baseline


def sample_environment_shocks(
    shape: tuple[int, int],
    critical_probability: float = 0.001,
    influential_probability: float = 0.005,
    shock_arrival_probability: float = 0.02,
    random_seed: int | None = None,
) -> np.ndarray:
    """Sample critical/influential/minor/none environmental shock labels."""

    rng = np.random.default_rng(random_seed)
    arrival = rng.random(shape) < shock_arrival_probability
    labels = np.full(shape, "none", dtype=object)
    if not np.any(arrival):
        return labels

    u = rng.random(shape)
    p_c = critical_probability
    p_i = critical_probability + influential_probability
    labels[arrival & (u < p_c)] = "critical"
    labels[arrival & (u >= p_c) & (u < p_i)] = "influential"
    labels[arrival & (u >= p_i)] = "minor"
    return labels


def update_rul_after_roll(
    rul_mean: np.ndarray,
    completed_theta: np.ndarray,
    roll_days: float = 1.0,
    reset_rul_days: float | None = None,
) -> np.ndarray:
    """Simple closed-loop RUL mean update after executing one roll."""

    updated = np.asarray(rul_mean, dtype=float).copy()
    updated = np.maximum(updated - roll_days, 0.0)
    completed = np.asarray(completed_theta, dtype=bool)
    if reset_rul_days is not None:
        updated[completed] = reset_rul_days
    return updated


__all__ = [
    "CandidateTask",
    "CandidateTaskSet",
    "TriggerConfig",
    "WakeConfig",
    "average_effective_maintenance_cost",
    "default_grid_layout",
    "default_line_layout",
    "generate_candidate_tasks",
    "jensen_wake_power",
    "lead_time_failure_probability",
    "load_power_curve_bins",
    "power_curve_mw",
    "sample_environment_shocks",
    "update_rul_after_roll",
    "wake_marginal_loss",
]
