from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ORDER = [
    "S1_failure_repair_route_obj",
    "S2_predictive_rul_route_obj",
    "S3_weather_route_obj",
    "S4_multi_route_obj",
    "S5_wake_multi_route_obj",
]
LABELS = ["S1", "S2", "S3", "S4", "S5"]
COLORS = ["#7A7F87", "#4C78A8", "#3E9B8F", "#D49B33", "#C44E52"]
TCRIT_95_DF11 = 2.200985


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.3,
            "axes.titlesize": 8.3,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for suffix, kwargs in [
        ("png", {"dpi": 450}),
        ("tiff", {"dpi": 600}),
        ("pdf", {}),
        ("svg", {}),
    ]:
        fig.savefig(out_dir / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.05, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    lo = matrix.min(axis=1, keepdims=True)
    hi = matrix.max(axis=1, keepdims=True)
    return (matrix - lo) / np.where(hi > lo, hi - lo, 1.0)


def build_roll_figure(roll_csv: Path, task_csv: Path, fig_dir: Path, source_dir: Path) -> None:
    roll = pd.read_csv(roll_csv)
    task = pd.read_csv(task_csv)
    roll["day"] = roll["roll"] + 1
    metrics = [
        ("candidate_component_tasks", "候选记录"),
        ("scheduled_tasks", "排程任务"),
        ("completed_tasks", "完成任务"),
        ("ongoing_out", "持续任务"),
        ("executed_hours", "执行工时"),
        ("loss_cost_yuan", "发电损失"),
    ]
    raw = np.vstack([roll[col].to_numpy(float) for col, _ in metrics])
    normalized = normalize_rows(raw)

    roll[["day"] + [col for col, _ in metrics] + ["total_actual_cost_yuan", "solver_runtime_sec"]].to_csv(
        source_dir / "fig5_roll_metrics.csv", index=False, encoding="utf-8-sig"
    )

    task["day"] = task["roll"] + 1
    type_map = {"PM": "新PM", "CM": "新CM", "ONGOING_PM": "持续PM", "ONGOING_CM": "持续CM"}
    task["任务状态"] = task["task_type"].map(type_map).fillna(task["task_type"])
    agg = (
        task.groupby(["day", "component_cn", "任务状态"], as_index=False)
        .agg(executed_hours=("executed_hours", "sum"), task_rows=("task_id", "size"))
    )
    agg.to_csv(source_dir / "fig5_task_events.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.15, 5.65))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.42)
    ax0 = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list("state", ["#F3F4F6", "#B8D5E5", "#326B8A"])
    im = ax0.imshow(normalized, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax0.set_yticks(np.arange(len(metrics)), [name for _, name in metrics])
    ax0.set_xticks(np.arange(20), [f"D{i}" for i in range(1, 21)], rotation=0)
    ax0.set_title("滚动周期状态与执行强度")
    ax0.axvline(15.5, color="#C44E52", lw=1.1, ls="--")
    ax0.text(16, -0.83, "D17无有效维护工时", color="#9B3438", fontsize=6.5, ha="center")
    for r in range(raw.shape[0]):
        for c in range(raw.shape[1]):
            value = raw[r, c]
            text = f"{value/10000:.1f}" if r == 5 else f"{value:.0f}"
            ax0.text(c, r, text, ha="center", va="center", fontsize=4.9,
                     color="white" if normalized[r, c] > 0.58 else "#111827")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.018, pad=0.012)
    cbar.set_label("行内归一化强度", fontsize=6.4)
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[1])
    components = ["齿轮箱", "发电机", "叶片", "主轴承"]
    ymap = {name: idx for idx, name in enumerate(components)}
    markers = {"新PM": "o", "新CM": "^", "持续PM": "s", "持续CM": "D"}
    component_colors = {"齿轮箱": "#4C78A8", "发电机": "#3E9B8F", "叶片": "#D49B33", "主轴承": "#C44E52"}
    for state, marker in markers.items():
        sub = agg[agg["任务状态"] == state]
        if sub.empty:
            continue
        ax1.scatter(
            sub["day"], sub["component_cn"].map(ymap),
            s=12 + 6.0 * sub["executed_hours"], marker=marker,
            c=sub["component_cn"].map(component_colors), edgecolors="white", linewidths=0.45, alpha=0.88,
        )
    ax1.set_xlim(0.3, 20.7)
    ax1.set_xticks(np.arange(1, 21), [f"D{i}" for i in range(1, 21)])
    ax1.set_yticks(np.arange(4), components)
    ax1.invert_yaxis()
    ax1.set_xlabel("滚动周期")
    ax1.set_title("部件任务的跨周期推进轨迹（气泡面积表示执行工时）")
    ax1.grid(axis="x", color="#D1D5DB", lw=0.4, alpha=0.6)
    legend_handles = [
        Line2D([0], [0], marker=m, color="none", markerfacecolor="#6B7280", markeredgecolor="white",
               markersize=6, label=s) for s, m in markers.items()
    ]
    ax1.legend(handles=legend_handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.23))
    panel_label(ax1, "b")
    fig.suptitle("典型窗口S5策略的闭环滚动执行过程", fontsize=10.2, fontweight="bold", y=0.995)
    save_figure(fig, fig_dir, "fig5_closed_loop_execution")


def build_overall_figure(data: pd.DataFrame, fig_dir: Path, source_dir: Path) -> None:
    wide = data.pivot(index="sample", columns="case", values="total_actual_cost_yuan")[ORDER] / 10000.0
    ranks = wide.rank(axis=1, method="min")
    means = data.groupby("case").agg(
        cost=("total_actual_cost_yuan", "mean"),
        completed=("completed_tasks", "mean"),
        loss=("loss_cost_yuan", "mean"),
        ongoing=("ongoing_out_final", "mean"),
    ).reindex(ORDER)
    wide.to_csv(source_dir / "fig6_cost_by_sample_wan.csv", encoding="utf-8-sig")
    ranks.to_csv(source_dir / "fig6_rank_by_sample.csv", encoding="utf-8-sig")
    means.to_csv(source_dir / "fig6_strategy_means.csv", encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.15, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95], width_ratios=[1.25, 1.0], hspace=0.44, wspace=0.37)
    ax0 = fig.add_subplot(gs[0, 0])
    values = [wide[c].to_numpy() for c in ORDER]
    violin = ax0.violinplot(values, positions=np.arange(5), widths=0.75, showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(violin["bodies"]):
        body.set_facecolor(COLORS[i]); body.set_edgecolor("none"); body.set_alpha(0.20 if i < 4 else 0.34)
    for i, vals in enumerate(values):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax0.plot([i, i], [q1, q3], color=COLORS[i], lw=5.0, solid_capstyle="butt")
        ax0.scatter([i], [med], s=18, color="white", edgecolor=COLORS[i], linewidth=1.0, zorder=4)
        jitter = np.linspace(-0.13, 0.13, len(vals))
        ax0.scatter(i + jitter, vals, s=10, color=COLORS[i], alpha=0.75, edgecolors="white", linewidths=0.25, zorder=3)
    ax0.set_xticks(np.arange(5), LABELS)
    ax0.set_ylabel("实际执行成本/万元")
    ax0.set_title("12个配对窗口的成本分布")
    ax0.grid(axis="y", alpha=0.18)
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    label_offsets = {
        "S1": (0.55, 0.0),
        "S2": (-1.45, 7.0),
        "S3": (0.45, 4.0),
        "S4": (-1.35, -8.0),
        "S5": (0.45, -7.0),
    }
    for i, (case, label, color) in enumerate(zip(ORDER, LABELS, COLORS)):
        row = means.loc[case]
        size = 50 + row["loss"] / 2500.0
        ax1.scatter(row["completed"], row["cost"] / 10000.0, s=size, color=color, alpha=0.86,
                    edgecolor="white", linewidth=0.7)
        dx, dy = label_offsets[label]
        ax1.text(row["completed"] + dx, row["cost"] / 10000.0 + dy, label,
                 color=color, fontweight="bold", va="center")
    ax1.annotate("成本降低", xy=(46.5, 375), xytext=(40.0, 535),
                 arrowprops=dict(arrowstyle="->", color="#6B7280", lw=0.8), color="#4B5563")
    ax1.set_xlabel("平均完成任务数/项")
    ax1.set_ylabel("平均实际执行成本/万元")
    ax1.set_title("任务完成量—成本—发电损失权衡")
    ax1.grid(alpha=0.18)
    panel_label(ax1, "b")

    ax2 = fig.add_subplot(gs[1, :])
    rank_cmap = LinearSegmentedColormap.from_list("rank", ["#C44E52", "#F5D8D8", "#F3F4F6", "#CBD5E1", "#64748B"])
    im = ax2.imshow(ranks.to_numpy(), aspect="auto", cmap=rank_cmap, vmin=1, vmax=5)
    ax2.set_xticks(np.arange(5), LABELS)
    ax2.set_yticks(np.arange(len(ranks)), [f"sample{s}" for s in ranks.index])
    ax2.set_title("样本内实际执行成本排序（1为最低）")
    for r in range(ranks.shape[0]):
        for c in range(ranks.shape[1]):
            value = int(ranks.iloc[r, c])
            ax2.text(c, r, str(value), ha="center", va="center", fontsize=6.1,
                     color="white" if value in (1, 5) else "#111827", fontweight="bold" if value == 1 else "normal")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.016, pad=0.012, ticks=[1, 2, 3, 4, 5])
    cbar.set_label("成本排序", fontsize=6.4)
    panel_label(ax2, "c")
    fig.suptitle("五种策略的综合执行表现与样本稳健性", fontsize=10.2, fontweight="bold", y=0.995)
    save_figure(fig, fig_dir, "fig6_strategy_performance")


def ci95(diff: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    half = TCRIT_95_DF11 * sd / np.sqrt(len(diff))
    return mean, mean - half, mean + half, sd


def build_increment_figure(data: pd.DataFrame, fig_dir: Path, source_dir: Path) -> None:
    wide_cost = data.pivot(index="sample", columns="case", values="total_actual_cost_yuan") / 10000.0
    comparisons = [(ORDER[i], ORDER[i + 1], f"S{i+1}→S{i+2}") for i in range(4)]
    forest_rows = []
    for a, b, label in comparisons:
        diff = (wide_cost[a] - wide_cost[b]).to_numpy()
        mean, lo, hi, sd = ci95(diff)
        forest_rows.append({"comparison": label, "mean": mean, "ci_low": lo, "ci_high": hi, "dz": mean / sd,
                            "wins": int(np.sum(diff > 0))})
    forest = pd.DataFrame(forest_rows)
    forest.to_csv(source_dir / "fig7_paired_forest.csv", index=False, encoding="utf-8-sig")

    s4, s5 = ORDER[3], ORDER[4]
    paired = pd.DataFrame({"sample": wide_cost.index, "S4": wide_cost[s4].values, "S5": wide_cost[s5].values})
    paired["saving"] = paired["S4"] - paired["S5"]
    paired.to_csv(source_dir / "fig7_s4_s5_paired.csv", index=False, encoding="utf-8-sig")

    components = [
        ("direct_cost_yuan", "直接维护"),
        ("crew_cost_yuan", "人工"),
        ("vessel_cost_yuan", "船舶固定"),
        ("fuel_cost_yuan", "燃油"),
        ("loss_cost_yuan", "发电损失"),
    ]
    changes = []
    for col, name in components:
        w = data.pivot(index="sample", columns="case", values=col) / 10000.0
        changes.append((name, float((w[s4] - w[s5]).mean())))
    decomposition = pd.DataFrame(changes, columns=["cost_item", "S4_minus_S5_wan"])
    decomposition.to_csv(source_dir / "fig7_s4_s5_decomposition.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.15, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.1], hspace=0.49, wspace=0.38)
    top = gs[0, :].subgridspec(1, 2, width_ratios=[1.55, 0.9], wspace=0.10)
    ax0 = fig.add_subplot(top[0])
    ax0_far = fig.add_subplot(top[1])
    colors = ["#4C78A8", "#3E9B8F", "#D49B33", "#C44E52"]
    ax0.axvline(0, color="#6B7280", lw=0.8, ls="--")
    local = forest.iloc[1:].reset_index(drop=True)
    local_y = np.arange(len(local))[::-1]
    for yi, row, color in zip(local_y, local.itertuples(), colors[1:]):
        m, lo, hi = row.mean, row.ci_low, row.ci_high
        ax0.plot([lo, hi], [yi, yi], color=color, lw=2.0)
        ax0.scatter([m], [yi], s=38, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        ax0.text(hi + 0.7, yi, f"d$_z$={row.dz:.2f}，{row.wins}/12", va="center", fontsize=6.3)
    ax0.set_yticks(local_y, local["comparison"])
    ax0.set_xlim(-10, 27)
    ax0.set_xlabel("前一策略−后一策略的实际执行成本/万元（正值表示后一策略更低）")
    ax0.set_title("局部模块效应")
    ax0.grid(axis="x", alpha=0.18)
    panel_label(ax0, "a")

    row = forest.iloc[0]
    ax0_far.plot([row.ci_low, row.ci_high], [0, 0], color=colors[0], lw=2.0)
    ax0_far.scatter([row["mean"]], [0], s=38, color=colors[0], edgecolor="white", linewidth=0.7, zorder=3)
    ax0_far.text(row.ci_high + 2.0, 0, f"d$_z$={row.dz:.2f}，{int(row.wins)}/12", va="center", fontsize=6.3)
    ax0_far.set_yticks([0], [row["comparison"]])
    ax0_far.set_xlim(120, 255)
    ax0_far.set_title("RUL提前维护效应")
    ax0_far.grid(axis="x", alpha=0.18)
    ax0_far.spines["left"].set_visible(False)
    fig.text(0.50, 0.925, "递进模块的配对效应及95%置信区间（横轴采用分段尺度）",
             ha="center", fontsize=8.3)

    ax1 = fig.add_subplot(gs[1, 0])
    ys = np.arange(len(paired))
    for yi, row in zip(ys, paired.itertuples()):
        color = "#C44E52" if row.saving > 0 else "#6B7280"
        ax1.plot([row.S4, row.S5], [yi, yi], color="#CBD5E1", lw=0.9)
        ax1.scatter(row.S4, yi, color="#D49B33", s=19, edgecolor="white", linewidth=0.4, zorder=3)
        ax1.scatter(row.S5, yi, color=color, s=19, edgecolor="white", linewidth=0.4, zorder=3)
    ax1.set_yticks(ys, [f"sample{s}" for s in paired["sample"]])
    ax1.invert_yaxis()
    ax1.set_xlabel("实际执行成本/万元")
    ax1.set_title("S4与S5的样本内配对变化")
    ax1.grid(axis="x", alpha=0.18)
    ax1.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D49B33", markersize=5, label="S4"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#C44E52", markersize=5, label="S5"),
    ], ncol=2, loc="lower right")
    panel_label(ax1, "b")

    ax2 = fig.add_subplot(gs[1, 1])
    names = decomposition["cost_item"].tolist()
    vals = decomposition["S4_minus_S5_wan"].to_numpy()
    cumulative = np.r_[0, np.cumsum(vals)]
    for i, (name, val) in enumerate(zip(names, vals)):
        start = cumulative[i]
        color = "#C44E52" if val > 0 else "#4C78A8"
        ax2.bar(i, val, bottom=start, color=color, width=0.68, edgecolor="white", linewidth=0.6)
        ax2.plot([i - 0.34, i + 0.34], [start, start], color="#6B7280", lw=0.45)
        ax2.text(i, start + val + (0.35 if val >= 0 else -0.55), f"{val:+.2f}", ha="center",
                 va="bottom" if val >= 0 else "top", fontsize=6.2, color=color)
    total = float((wide_cost[s4] - wide_cost[s5]).mean())
    ax2.bar(len(names), total, color="#7A7F87", width=0.68, edgecolor="white", linewidth=0.6)
    ax2.text(len(names), total + 0.35, f"{total:+.2f}", ha="center", va="bottom", fontsize=6.4, fontweight="bold")
    ax2.axhline(0, color="#6B7280", lw=0.7)
    ax2.set_xticks(np.arange(len(names) + 1), names + ["净节省"], rotation=28, ha="right")
    ax2.set_ylabel("S4−S5平均成本/万元")
    ax2.set_title("尾流修正的平均成本贡献分解")
    ax2.grid(axis="y", alpha=0.18)
    panel_label(ax2, "c")
    fig.suptitle("递进模块贡献与尾流感知修正的配对证据", fontsize=10.2, fontweight="bold", y=0.995)
    save_figure(fig, fig_dir, "fig7_increment_and_wake")


def sensitivity_row(root: Path, data: pd.DataFrame, k: int) -> pd.Series:
    if k == 3:
        return data[(data["sample"] == 26) & (data["case"] == ORDER[4])].iloc[0]
    path = root / f"five_strategy_routing_objective_summary_sensitivity_K{k}_sample26_copt.csv"
    frame = pd.read_csv(path)
    return frame[frame["case"] == ORDER[4]].iloc[0]


def build_sensitivity_figure(root: Path, data: pd.DataFrame, fig_dir: Path, source_dir: Path) -> None:
    rows = []
    for k in [2, 3, 4]:
        row = sensitivity_row(root, data, k)
        rows.append(
            {
                "K": k,
                "completed_tasks": row["completed_tasks"],
                "executed_hours": row["executed_hours"],
                "ongoing_out_final": row["ongoing_out_final"],
                "direct_cost_wan": row["direct_cost_yuan"] / 10000,
                "loss_cost_wan": row["loss_cost_yuan"] / 10000,
                "total_cost_wan": row["total_actual_cost_yuan"] / 10000,
                "distance_km": row["sailing_distance_km"],
            }
        )
    sens = pd.DataFrame(rows)
    sens.to_csv(source_dir / "fig8_sensitivity.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.15, 3.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.38)
    ax0 = fig.add_subplot(gs[0])
    colors = ["#4C78A8", "#D49B33", "#C44E52"]
    for row, color in zip(sens.itertuples(), colors):
        size = 130 + row.loss_cost_wan * 3.4
        ax0.scatter(row.completed_tasks, row.total_cost_wan, s=size, color=color, alpha=0.76,
                    edgecolor="white", linewidth=0.8)
        ax0.text(row.completed_tasks + 0.65, row.total_cost_wan, f"K={row.K}\n持续={row.ongoing_out_final:.0f}",
                 color=color, va="center", fontweight="bold")
    ax0.set_xlabel("完成任务数/项")
    ax0.set_ylabel("实际执行成本/万元")
    ax0.set_title("资源能力下的任务量—成本权衡")
    ax0.grid(alpha=0.18)
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[1])
    metric_cols = ["completed_tasks", "executed_hours", "distance_km", "direct_cost_wan", "loss_cost_wan", "total_cost_wan"]
    metric_names = ["完成任务", "执行工时", "航行距离", "直接维护", "发电损失", "实际成本"]
    matrix = sens[metric_cols].to_numpy(float).T
    norm = normalize_rows(matrix)
    cmap = LinearSegmentedColormap.from_list("sens", ["#F3F4F6", "#C7DDEB", "#326B8A"])
    im = ax1.imshow(norm, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax1.set_xticks(np.arange(3), ["K=2", "K=3", "K=4"])
    ax1.set_yticks(np.arange(len(metric_names)), metric_names)
    ax1.set_title("指标变化及任务组合差异")
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            if r == 0: text = f"{val:.0f}"
            elif r == 1: text = f"{val:.0f} h"
            elif r == 2: text = f"{val:.0f} km"
            else: text = f"{val:.1f} 万"
            ax1.text(c, r, text, ha="center", va="center", fontsize=6.0,
                     color="white" if norm[r, c] > 0.60 else "#111827")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.035, pad=0.025)
    cbar.set_label("行内归一化强度", fontsize=6.4)
    panel_label(ax1, "b")
    fig.suptitle("典型窗口中并行维护能力的敏感性", fontsize=10.2, fontweight="bold", y=0.995)
    save_figure(fig, fig_dir, "fig8_parallel_sensitivity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    fig_dir = args.output_dir / "figures"
    source_dir = args.output_dir / "source_data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    summary_path = args.model_root / "five_strategy_routing_objective_summary_material5x_12samples_five_strategies_copt.csv"
    data = pd.read_csv(summary_path)
    data.to_csv(source_dir / "section5_strategy_results.csv", index=False, encoding="utf-8-sig")
    build_roll_figure(
        args.model_root / "hourly_milp_roll_summary_sample26_material5x_12samples_five_strategies_copt_S5_wake_multi_route_obj.csv",
        args.model_root / "hourly_milp_task_execution_sample26_material5x_12samples_five_strategies_copt_S5_wake_multi_route_obj.csv",
        fig_dir,
        source_dir,
    )
    build_overall_figure(data, fig_dir, source_dir)
    build_increment_figure(data, fig_dir, source_dir)
    build_sensitivity_figure(args.model_root, data, fig_dir, source_dir)


if __name__ == "__main__":
    main()
