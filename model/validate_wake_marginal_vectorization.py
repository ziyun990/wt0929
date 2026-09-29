from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from cl_stochos_mu import (  # noqa: E402
    WakeConfig,
    default_grid_layout,
    jensen_wake_power,
    load_power_curve_bins,
    wake_marginal_loss,
)


def legacy_marginal_loss(
    wind_speed: np.ndarray,
    wind_direction: np.ndarray,
    layout: np.ndarray,
    bins: np.ndarray,
    config: WakeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation that rebuilds the farm for every outage."""

    horizon = len(wind_speed)
    n_turbines = len(layout)
    marginal = np.zeros((horizon, n_turbines, 1), dtype=float)
    baseline = np.zeros((horizon, 1), dtype=float)
    active = np.ones(n_turbines, dtype=bool)

    for hour in range(horizon):
        base_power = jensen_wake_power(
            float(wind_speed[hour]),
            float(wind_direction[hour]),
            layout,
            bins,
            active=active,
            config=config,
        ).sum()
        baseline[hour, 0] = base_power
        for turbine in range(n_turbines):
            outage_active = active.copy()
            outage_active[turbine] = False
            outage_power = jensen_wake_power(
                float(wind_speed[hour]),
                float(wind_direction[hour]),
                layout,
                bins,
                active=outage_active,
                config=config,
            ).sum()
            marginal[hour, turbine, 0] = base_power - outage_power

    return marginal, baseline


def main() -> None:
    config = WakeConfig()
    layout = default_grid_layout(8, 8)
    bins = load_power_curve_bins(REPO_ROOT / "method_of_bins.csv")
    wind_speed = np.array([3.9, 5.2, 8.0, 11.4, 14.0, 24.9, 25.0])
    wind_direction = np.array([0.0, 17.0, 45.0, 91.0, 180.0, 271.0, 359.0])

    expected_marginal, expected_baseline = legacy_marginal_loss(
        wind_speed,
        wind_direction,
        layout,
        bins,
        config,
    )
    actual_marginal, actual_baseline = wake_marginal_loss(
        wind_speed[:, np.newaxis],
        wind_direction,
        layout,
        bins,
        config,
    )

    np.testing.assert_allclose(actual_baseline, expected_baseline, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual_marginal, expected_marginal, rtol=0.0, atol=1e-12)
    print("PASS: vectorized wake marginal loss matches the legacy outage enumeration.")
    print("max baseline abs error:", float(np.max(np.abs(actual_baseline - expected_baseline))))
    print("max marginal abs error:", float(np.max(np.abs(actual_marginal - expected_marginal))))


if __name__ == "__main__":
    main()
