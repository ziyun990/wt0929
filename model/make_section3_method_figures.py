from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, TwoSlopeNorm


ROOT = Path(__file__).resolve().parent.parent
CASE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cl_stochos_mu import (
    WakeConfig,
    default_grid_layout,
    load_power_curve_bins,
    power_curve_mw,
    wake_marginal_loss,
)
from multi_component_params import (
    DAY_END_HOUR,
    DAY_START_HOUR,
    MAX_MAINTENANCE_WAVE_M,
    MAX_MAINTENANCE_WIND_M_S,
    RATED_POWER_MW,
)


WEATHER = ROOT / "data" / "openmeteo_liaoning_wind_wave_2025_gfs" / "openmeteo_liaoning_wind_wave_2025_gfs.csv"
PRICE = ROOT / "data" / "liaoning_spot_prices_api_2025" / "liaoning_spot_hourly_2025.csv"
SAMPLE = 31
N_TURBINES = 64
SPATIAL_STD = 0.15
INPUT_SEED = 2026


def set_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9.5


def load_inputs() -> dict[str, np.ndarray]:
    weather = pd.read_csv(WEATHER, parse_dates=["timestamp"])
    prices = pd.read_csv(PRICE, parse_dates=["timestamp"])
    merged = weather.merge(prices, on="timestamp", how="inner", validate="one_to_one")
    rng = np.random.default_rng(INPUT_SEED)

    def turbine_matrix(column: str) -> np.ndarray:
        base = merged[column].to_numpy(dtype=float).reshape(-1, 1)
        noise = rng.normal(0.0, SPATIAL_STD, size=(len(merged), N_TURBINES))
        return np.maximum(base + noise, 0.0)

    return {
        "time": merged["timestamp"].to_numpy(),
        # Preserve the exact random-number call order used by the simulation.
        "ws_true": turbine_matrix("wind_speed_actual_m_s"),
        "ws_forecast": turbine_matrix("wind_speed_forecast_m_s"),
        "wave_true": merged["wave_height_actual_m"].to_numpy(dtype=float),
        "wave_forecast": merged["wave_height_forecast_m"].to_numpy(dtype=float),
        "direction_forecast": merged["wind_direction_forecast_deg"].to_numpy(dtype=float),
        "price_forecast": merged["day_ahead_price_yuan_per_mwh"].to_numpy(dtype=float),
    }


def hourly_access(ws: np.ndarray, wave: np.ndarray) -> np.ndarray:
    access = (
        (ws < MAX_MAINTENANCE_WIND_M_S)
        & (wave.reshape(-1, 1) < MAX_MAINTENANCE_WAVE_M)
    ).astype(int)
    access[:DAY_START_HOUR, :] = 0
    access[DAY_END_HOUR:, :] = 0
    return access


def draw_access_formation(inputs: dict[str, np.ndarray]) -> None:
    start = SAMPLE * 24
    sl = slice(start, start + 24)
    ws_f = inputs["ws_forecast"][sl]
    ws_a = inputs["ws_true"][sl]
    wave_f = inputs["wave_forecast"][sl]
    wave_a = inputs["wave_true"][sl]
    access_f = hourly_access(ws_f, wave_f)
    access_a = hourly_access(ws_a, wave_a)
    hours = np.arange(24)
    hour_centers = hours + 0.5

    fig = plt.figure(figsize=(11.2, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 1])
    ax_wind = fig.add_subplot(gs[0, 0])
    ax_wave = fig.add_subplot(gs[0, 1])
    ax_fcst = fig.add_subplot(gs[1, :])
    ax_actual = fig.add_subplot(gs[2, :])

    ax_wind.fill_between(
        hour_centers,
        ws_f.min(axis=1),
        ws_f.max(axis=1),
        color="#9ecae1",
        alpha=0.55,
        label=f"{N_TURBINES}台风机预测范围",
    )
    ax_wind.plot(hour_centers, ws_f.mean(axis=1), color="#1f5a91", marker="o", ms=3, lw=1.2, label="预测均值")
    ax_wind.plot(hour_centers, ws_a.mean(axis=1), color="#397d54", lw=1.0, label="实际均值")
    ax_wind.axhline(15, color="#b23a48", ls="--", lw=1.1, label="风速阈值15 m/s")
    ax_wind.set_title("(a) 风速条件")
    ax_wind.set_xlabel("日内小时")
    ax_wind.set_ylabel("风速/(m/s)")
    ax_wind.set_xlim(0, 24)
    ax_wind.set_xticks(np.arange(0, 25, 3))
    ax_wind.grid(axis="y", alpha=0.25)
    ax_wind.legend(fontsize=7.5, ncol=2, loc="upper right")

    ax_wave.plot(hour_centers, wave_f, color="#2b8a6e", marker="o", ms=3, lw=1.2, label="预测浪高")
    ax_wave.plot(hour_centers, wave_a, color="#7b6d3b", lw=1.0, label="实际浪高")
    ax_wave.axhline(1.5, color="#b23a48", ls="--", lw=1.1, label="浪高阈值1.5 m")
    ax_wave.axvspan(0, 5, color="#d9d9d9", alpha=0.6)
    ax_wave.axvspan(21, 24, color="#d9d9d9", alpha=0.6, label="非日间窗口")
    ax_wave.set_title("(b) 浪高与日间作业窗口")
    ax_wave.set_xlabel("日内小时")
    ax_wave.set_ylabel("浪高/m")
    ax_wave.set_xlim(0, 24)
    ax_wave.set_xticks(np.arange(0, 25, 3))
    ax_wave.grid(axis="y", alpha=0.25)
    ax_wave.legend(fontsize=7.5, ncol=2, loc="upper right")

    cmap = ListedColormap(["#d95f5f", "#4f9d69"])
    hour_edges = np.arange(25)
    turbine_edges = np.arange(17)
    for ax, matrix, title in [
        (ax_fcst, access_f.T, "(c) 优化模型使用的预测可达性矩阵"),
        (ax_actual, access_a.T, "(d) 执行反馈使用的实际可达性矩阵"),
    ]:
        ax.pcolormesh(
            hour_edges,
            turbine_edges,
            matrix,
            cmap=cmap,
            vmin=0,
            vmax=1,
            shading="flat",
        )
        ax.set_title(title)
        ax.set_ylabel("风机编号")
        ax.set_xlim(0, 24)
        tick_rows = np.arange(0, N_TURBINES, 8)
        ax.set_ylim(N_TURBINES, 0)
        ax.set_yticks(tick_rows + 0.5)
        ax.set_yticklabels([f"WT{i + 1}" for i in tick_rows], fontsize=7)
        ax.set_xticks(np.arange(25))
        ax.set_xticklabels([str(i) for i in range(25)], fontsize=7)
        ax.axvline(5, color="black", linewidth=0.8)
        ax.axvline(21, color="black", linewidth=0.8)
    ax_actual.set_xlabel("日内小时边界（色块表示区间 [t, t+1)）")

    diff = int(np.sum(access_f != access_a))
    fig.suptitle(
        f"sample31第1滚动周期可达性矩阵形成过程（预测与实际差异单元数：{diff}）",
        fontsize=13,
        weight="bold",
    )
    fig.savefig(CASE_DIR / "fig_journal_access_matrix_method.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_loss_parameters(inputs: dict[str, np.ndarray]) -> None:
    start = SAMPLE * 24
    sl = slice(start, start + 24)
    ws = inputs["ws_forecast"][sl]
    wd = inputs["direction_forecast"][sl]
    price = inputs["price_forecast"][sl]

    bins = load_power_curve_bins(ROOT / "method_of_bins.csv")
    single_power = power_curve_mw(ws, bins, RATED_POWER_MW)
    single_cost = single_power * price.reshape(-1, 1)
    layout = default_grid_layout(8, 8)
    wake_power, _ = wake_marginal_loss(
        ws.mean(axis=1)[:, np.newaxis],
        wd,
        layout,
        bins,
        config=WakeConfig(),
    )
    wake_cost = wake_power[:, :, 0] * price.reshape(-1, 1)
    delta = wake_cost - single_cost

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.9), constrained_layout=True)
    vmax = max(float(np.nanmax(single_cost)), float(np.nanmax(wake_cost)), 1.0)
    common = dict(aspect="auto", interpolation="nearest", cmap="YlOrRd", vmin=0, vmax=vmax)
    im0 = axes[0].imshow(single_cost.T, **common)
    im1 = axes[1].imshow(wake_cost.T, **common)
    dmax = max(float(np.nanmax(np.abs(delta))), 1.0)
    im2 = axes[2].imshow(delta.T, aspect="auto", interpolation="nearest", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax))
    titles = ["(a) 单机停机损失参数", "(b) 尾流感知场级边际损失参数", "(c) 尾流修正增量"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("日内小时")
        ax.set_xticks(np.arange(0, 24, 3))
        tick_rows = np.arange(0, N_TURBINES, 8)
        ax.set_yticks(tick_rows)
        ax.set_yticklabels([f"WT{i + 1}" for i in tick_rows], fontsize=7)
    axes[0].set_ylabel("风机编号")
    fig.colorbar(im1, ax=axes[:2], shrink=0.86, label="发电损失参数/(元/h)")
    fig.colorbar(im2, ax=axes[2], shrink=0.86, label="尾流修正增量/(元/h)")
    fig.suptitle("sample31第1滚动周期发电损失参数预计算结果", fontsize=13, weight="bold")
    fig.savefig(CASE_DIR / "fig_journal_loss_parameter_method.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_style()
    inputs = load_inputs()
    draw_access_formation(inputs)
    draw_loss_parameters(inputs)
    print(CASE_DIR / "fig_journal_access_matrix_method.png")
    print(CASE_DIR / "fig_journal_loss_parameter_method.png")


if __name__ == "__main__":
    main()
