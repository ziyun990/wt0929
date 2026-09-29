from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parent


def read_model_inputs(
    n_turbines: int,
    spatial_std: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Load aligned weather, wave, direction, and electricity-price inputs."""
    weather_path = (
        REPO_ROOT
        / "data"
        / "openmeteo_liaoning_wind_wave_2025_gfs"
        / "openmeteo_liaoning_wind_wave_2025_gfs.csv"
    )
    price_path = (
        REPO_ROOT
        / "data"
        / "liaoning_spot_prices_api_2025"
        / "liaoning_spot_hourly_2025.csv"
    )

    weather = pd.read_csv(weather_path, parse_dates=["timestamp"])
    prices = pd.read_csv(price_path, parse_dates=["timestamp"])
    merged = pd.merge(weather, prices, on="timestamp", how="inner", validate="one_to_one")
    if merged.empty:
        raise RuntimeError("No overlapping timestamp rows between weather/wave and price data.")

    rng = np.random.default_rng(seed)

    def turbine_matrix(column: str) -> np.ndarray:
        base = merged[column].to_numpy(dtype=float).reshape(-1, 1)
        if spatial_std <= 0:
            return np.repeat(base, n_turbines, axis=1)
        noise = rng.normal(0.0, spatial_std, size=(len(merged), n_turbines))
        return np.maximum(base + noise, 0.0)

    return {
        "time": merged["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "ws_true": turbine_matrix("wind_speed_actual_m_s"),
        "ws_nwp": turbine_matrix("wind_speed_forecast_m_s"),
        "wh_true": merged["wave_height_actual_m"].to_numpy(dtype=float),
        "wh_nwp": merged["wave_height_forecast_m"].to_numpy(dtype=float),
        "wd_true": merged["wind_direction_actual_deg"].to_numpy(dtype=float),
        "wd_nwp": merged["wind_direction_forecast_deg"].to_numpy(dtype=float),
        "ep_true": merged["real_time_price_yuan_per_mwh"].to_numpy(dtype=float),
        "ep_fcst": merged["day_ahead_price_yuan_per_mwh"].to_numpy(dtype=float),
    }


def generate_rul_scenarios(
    mean_days: np.ndarray,
    shape_parameter: np.ndarray,
    n_scenarios: int,
    seed: int,
    calibration_mode: str = "legacy_scale",
) -> np.ndarray:
    """Generate finite Weibull-factor RUL scenarios for candidate triggering."""
    rng = np.random.default_rng(seed)
    if calibration_mode not in {"legacy_scale", "mean_corrected", "deterministic_mean"}:
        raise ValueError(f"unknown RUL scenario calibration mode: {calibration_mode}")

    if calibration_mode == "deterministic_mean":
        return np.repeat(mean_days[:, :, np.newaxis], n_scenarios, axis=2)

    shape = np.maximum(shape_parameter, 1.0)[:, :, np.newaxis]
    scale = mean_days[:, :, np.newaxis]
    if calibration_mode == "mean_corrected":
        from scipy.special import gamma

        scale = scale / gamma(1.0 + 1.0 / shape)
    factors = rng.weibull(shape, size=mean_days.shape + (n_scenarios,))
    scenarios = scale * factors
    scenarios[scenarios < 1.0] = 1.0
    failed = np.repeat(mean_days[:, :, np.newaxis] == 0.0, n_scenarios, axis=2)
    scenarios[failed] = 0.0
    return scenarios
