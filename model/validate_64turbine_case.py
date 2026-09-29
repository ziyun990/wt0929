from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


CASE_DIR = Path(__file__).resolve().parent
ROOT = CASE_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cl_stochos_mu import WakeConfig, default_grid_layout
from multi_component_params import BASE_DISTANCE_KM, RATED_POWER_MW
from routing_costs import RoutingConfig, infer_base_coordinate
from rul_profiles import balanced_component_state


N_TURBINES = 64
GRID_SIZE = 8
RUL_SEED = 37


def main() -> None:
    true_rul, forecast_mean, forecast_shape, lifetimes = balanced_component_state(
        n_turbines=N_TURBINES,
        seed=RUL_SEED,
    )
    repeated = balanced_component_state(n_turbines=N_TURBINES, seed=RUL_SEED)

    expected_shape = (N_TURBINES, 4)
    assert true_rul.shape == expected_shape
    assert forecast_mean.shape == expected_shape
    assert forecast_shape.shape == expected_shape
    assert lifetimes.shape == expected_shape
    assert all(np.array_equal(a, b) for a, b in zip(repeated, (true_rul, forecast_mean, forecast_shape, lifetimes)))

    short = forecast_mean <= 20.0
    medium = (forecast_mean >= 21.0) & (forecast_mean <= 40.0)
    long = forecast_mean >= 41.0
    assert int(short.sum()) == 52
    assert int(medium.sum()) == 76
    assert int(long.sum()) == 128
    assert int(short.sum(axis=1).max()) <= 2

    wake = WakeConfig()
    layout = default_grid_layout(
        GRID_SIZE,
        GRID_SIZE,
        spacing_rotor_diameters=7.0,
        rotor_diameter_m=wake.rotor_diameter_m,
    )
    assert layout.shape == (N_TURBINES, 2)
    expected_x_span = 7.0 * wake.rotor_diameter_m * (GRID_SIZE - 1)
    expected_y_span = 8.0 * wake.rotor_diameter_m * (GRID_SIZE - 1)
    assert np.isclose(layout[:, 0].max() - layout[:, 0].min(), expected_x_span)
    assert np.isclose(layout[:, 1].max() - layout[:, 1].min(), expected_y_span)
    assert np.isclose(wake.rated_power_mw, RATED_POWER_MW)

    route_config = RoutingConfig(base_distance_km=BASE_DISTANCE_KM, base_direction="west")
    base = infer_base_coordinate(layout, route_config)
    assert np.isclose(layout[:, 0].min() / 1000.0 - base[0], BASE_DISTANCE_KM)

    weather_path = (
        ROOT
        / "data"
        / "openmeteo_liaoning_wind_wave_2025_gfs"
        / "openmeteo_liaoning_wind_wave_2025_gfs.csv"
    )
    price_path = (
        ROOT
        / "data"
        / "liaoning_spot_prices_api_2025"
        / "liaoning_spot_hourly_2025.csv"
    )
    weather = pd.read_csv(weather_path)
    prices = pd.read_csv(price_path)
    assert len(weather) == 7344
    assert len(prices) == 7344

    print("64-turbine case validation passed")
    print(
        f"layout: {GRID_SIZE}x{GRID_SIZE}, "
        f"span={expected_x_span / 1000.0:.3f}x{expected_y_span / 1000.0:.3f} km"
    )
    print(f"component units: {forecast_mean.size}")
    print(f"RUL strata: short={short.sum()}, medium={medium.sum()}, long={long.sum()}")
    print(f"max short-RUL components on one turbine: {short.sum(axis=1).max()}")
    print(f"weather rows: {len(weather)}, price rows: {len(prices)}")


if __name__ == "__main__":
    main()
