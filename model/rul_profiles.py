from __future__ import annotations

import numpy as np

from multi_component_params import LIFETIME_DAYS


MAIN_CASE_TURBINES = 64
SHORT_RUL_FRACTION = 0.20
MEDIUM_RUL_FRACTION = 0.30
MAX_SHORT_RUL_COMPONENTS_PER_TURBINE = 2
WEIBULL_SHAPE_BY_COMPONENT = np.array([5.0, 5.0, 6.0, 5.0], dtype=float)


def _stratified_values(
    rng: np.random.Generator,
    count: int,
    low: float,
    high: float,
) -> np.ndarray:
    """Return shuffled values that cover the full interval without clustering."""
    if count <= 0:
        return np.empty(0, dtype=float)
    if high <= low:
        return np.full(count, low, dtype=float)

    edges = np.linspace(low, high, count + 1)
    values = rng.uniform(edges[:-1], edges[1:])
    rng.shuffle(values)
    return values


def _short_rul_cells(
    rng: np.random.Generator,
    n_turbines: int,
    counts_by_component: np.ndarray,
) -> list[tuple[int, int]]:
    """Select component-balanced short-RUL cells with at most two per turbine."""
    selected: list[tuple[int, int]] = []
    turbine_load = np.zeros(n_turbines, dtype=int)

    component_order = rng.permutation(len(counts_by_component))
    for component in component_order:
        needed = int(counts_by_component[component])
        candidates = rng.permutation(n_turbines)
        chosen = 0
        for turbine in candidates:
            if turbine_load[turbine] >= MAX_SHORT_RUL_COMPONENTS_PER_TURBINE:
                continue
            selected.append((int(turbine), int(component)))
            turbine_load[turbine] += 1
            chosen += 1
            if chosen == needed:
                break
        if chosen != needed:
            raise RuntimeError("Unable to distribute short-RUL components under the turbine cap.")
    return selected


def _component_state(
    n_turbines: int,
    seed: int,
    *,
    short_fraction: float,
    medium_fraction: float,
    short_range: tuple[float, float],
    medium_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_turbines < 1:
        raise ValueError("n_turbines must be positive")
    if short_fraction + medium_fraction >= 1.0:
        raise ValueError("short and medium RUL fractions must sum to less than one")

    rng = np.random.default_rng(seed)
    n_components = len(LIFETIME_DAYS)
    n_short_per_component = int(round(short_fraction * n_turbines))
    n_medium_per_component = int(round(medium_fraction * n_turbines))
    n_long_per_component = n_turbines - n_short_per_component - n_medium_per_component
    if n_long_per_component < 1:
        raise ValueError("RUL fractions leave no long-RUL components")

    short_counts = np.full(n_components, n_short_per_component, dtype=int)
    medium_counts = np.full(n_components, n_medium_per_component, dtype=int)
    long_counts = np.full(n_components, n_long_per_component, dtype=int)
    short_cells = set(_short_rul_cells(rng, n_turbines, short_counts))

    forecast_mean = np.empty((n_turbines, n_components), dtype=float)
    for component in range(n_components):
        short_turbines = [t for t in range(n_turbines) if (t, component) in short_cells]
        remaining = [t for t in range(n_turbines) if (t, component) not in short_cells]
        rng.shuffle(remaining)

        medium_count = int(medium_counts[component])
        long_count = int(long_counts[component])
        if len(remaining) != medium_count + long_count:
            raise RuntimeError("RUL stratum counts do not match the component population.")

        medium_turbines = remaining[:medium_count]
        long_turbines = remaining[medium_count:]
        lifetime = float(LIFETIME_DAYS[component])

        forecast_mean[short_turbines, component] = _stratified_values(
            rng, len(short_turbines), short_range[0], min(short_range[1], lifetime - 1.0)
        )
        forecast_mean[medium_turbines, component] = _stratified_values(
            rng, len(medium_turbines), medium_range[0], min(medium_range[1], lifetime - 1.0)
        )
        forecast_mean[long_turbines, component] = _stratified_values(
            rng, len(long_turbines), 41.0, max(41.0, lifetime - 1.0)
        )

    error_scale = np.array([1.0, 1.0, 1.2, 1.0], dtype=float)
    true_error = rng.normal(0.0, error_scale.reshape(1, -1), size=forecast_mean.shape)
    true_error = np.clip(true_error, -2.5, 2.5)
    true_rul = np.clip(forecast_mean + true_error, 1.0, LIFETIME_DAYS.reshape(1, -1))

    forecast_shape = np.repeat(
        WEIBULL_SHAPE_BY_COMPONENT.reshape(1, -1), n_turbines, axis=0
    )
    lifetimes = np.repeat(LIFETIME_DAYS.reshape(1, -1), n_turbines, axis=0)
    return true_rul, forecast_mean, forecast_shape, lifetimes


def balanced_component_state(
    n_turbines: int = MAIN_CASE_TURBINES,
    seed: int = 37,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate the main 64-turbine case with 20/30/50 percent RUL strata."""
    return _component_state(
        n_turbines,
        seed,
        short_fraction=SHORT_RUL_FRACTION,
        medium_fraction=MEDIUM_RUL_FRACTION,
        short_range=(1.5, 20.0),
        medium_range=(21.0, 40.0),
    )


def pressure_component_state(
    n_turbines: int = MAIN_CASE_TURBINES,
    seed: int = 37,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a denser stress-test profile without repeating 16-turbine blocks."""
    return _component_state(
        n_turbines,
        seed,
        short_fraction=0.35,
        medium_fraction=0.35,
        short_range=(1.0, 15.0),
        medium_range=(16.0, 35.0),
    )


def initial_component_state(
    profile: str,
    n_turbines: int = MAIN_CASE_TURBINES,
    seed: int = 37,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if profile == "balanced":
        return balanced_component_state(n_turbines=n_turbines, seed=seed)
    if profile == "pressure":
        return pressure_component_state(n_turbines=n_turbines, seed=seed)
    raise ValueError(f"unknown RUL profile: {profile}")
