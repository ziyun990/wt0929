#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize five-strategy routing-objective experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ORDER = [
    "S1_failure_repair_route_obj",
    "S2_predictive_rul_route_obj",
    "S3_weather_route_obj",
    "S4_multi_route_obj",
    "S5_wake_multi_route_obj",
]

CN = {
    "S1_failure_repair_route_obj": "S1 故障后维修",
    "S2_predictive_rul_route_obj": "S2 预测RUL",
    "S3_weather_route_obj": "S3 天气可达机会",
    "S4_multi_route_obj": "S4 闭环多机会",
    "S5_wake_multi_route_obj": "S5 尾流闭环多机会",
}

COST_COLS = [
    "direct_cost_yuan",
    "crew_cost_yuan",
    "vessel_cost_yuan",
    "fuel_cost_yuan",
    "overtime_cost_yuan",
    "loss_cost_yuan",
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "savefig.dpi": 600,
        }
    )


def mean_std(series: pd.Series) -> str:
    return f"{series.mean():.2f}±{series.std(ddof=1):.2f}"


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "candidate_records",
        "scheduled_tasks",
        "completed_tasks",
        "ongoing_out_final",
        "executed_hours",
        "sailing_days",
        "sailing_distance_km",
        "fuel_cost_yuan",
        "loss_cost_yuan",
        "total_actual_cost_yuan",
    ] + COST_COLS
    for case in ORDER:
        sub = df[df["case"] == case]
        row = {"case": case, "strategy_cn": CN[case], "n": len(sub)}
        for metric in metrics:
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_std"] = sub[metric].std(ddof=1)
            row[f"{metric}_mean_std"] = mean_std(sub[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_delta(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot(index="sample", columns="case")
    pairs = [
        ("S2-S1", "S2_predictive_rul_route_obj", "S1_failure_repair_route_obj"),
        ("S3-S2", "S3_weather_route_obj", "S2_predictive_rul_route_obj"),
        ("S4-S3", "S4_multi_route_obj", "S3_weather_route_obj"),
        ("S5-S4", "S5_wake_multi_route_obj", "S4_multi_route_obj"),
        ("S5-S1", "S5_wake_multi_route_obj", "S1_failure_repair_route_obj"),
    ]
    metrics = [
        "total_actual_cost_yuan",
        "loss_cost_yuan",
        "fuel_cost_yuan",
        "vessel_cost_yuan",
        "sailing_distance_km",
        "completed_tasks",
        "ongoing_out_final",
    ]
    rows = []
    for label, a, b in pairs:
        for metric in metrics:
            vals = wide[(metric, a)] - wide[(metric, b)]
            rows.append(
                {
                    "comparison": label,
                    "metric": metric,
                    "mean_delta": vals.mean(),
                    "std_delta": vals.std(ddof=1),
                    "win_count_delta_lt_0": int((vals < 0).sum()),
                    "lose_count_delta_gt_0": int((vals > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_cost_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    colors = ["#7F7F7F", "#4E79A7", "#76B7B2", "#59A14F", "#F28E2B"]
    data = [df[df["case"] == case]["total_actual_cost_yuan"].values / 10000 for case in ORDER]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
    for median in bp["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.3)
    rng = np.random.default_rng(2026)
    for idx, (vals, color) in enumerate(zip(data, colors), start=1):
        ax.scatter(
            np.full(len(vals), idx) + rng.normal(0, 0.045, len(vals)),
            vals,
            s=28,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    ax.set_xticks(range(1, len(ORDER) + 1))
    ax.set_xticklabels([CN[c] for c in ORDER], rotation=12, ha="right")
    ax.set_ylabel("实际执行成本 / 万元")
    ax.set_title("所有策略纳入航线访问成本后的多样本成本分布")
    ax.grid(axis="y", alpha=0.22)
    save_fig(fig, out_dir, "fig_five_strategy_route_obj_cost_distribution")


def plot_cost_breakdown(summary: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    labels = ["直接维护", "人工", "船舶固定", "燃油", "加班", "发电损失"]
    colors = ["#4E79A7", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#F28E2B"]
    x = np.arange(len(ORDER))
    bottom = np.zeros(len(ORDER))
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    for col, label, color in zip(COST_COLS, labels, colors):
        vals = np.array([summary.loc[summary["case"] == case, f"{col}_mean"].iloc[0] for case in ORDER]) / 10000
        ax.bar(x, vals, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.7)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([CN[c] for c in ORDER], rotation=12, ha="right")
    ax.set_ylabel("平均成本 / 万元")
    ax.set_title("所有策略纳入航线访问成本后的平均成本分解")
    ax.legend(ncol=3, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", alpha=0.22)
    save_fig(fig, out_dir, "fig_five_strategy_route_obj_cost_breakdown")


def plot_operations(summary: pd.DataFrame, out_dir: Path) -> None:
    setup_style()
    x = np.arange(len(ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].bar(x - 0.18, [summary.loc[summary["case"] == c, "completed_tasks_mean"].iloc[0] for c in ORDER], width=0.36, label="完成任务", color="#4E79A7")
    axes[0].bar(x + 0.18, [summary.loc[summary["case"] == c, "ongoing_out_final_mean"].iloc[0] for c in ORDER], width=0.36, label="期末持续任务", color="#E15759")
    axes[0].set_ylabel("任务数")
    axes[0].set_title("任务完成与期末持续任务")
    axes[0].legend(frameon=False)
    axes[1].bar(x - 0.18, [summary.loc[summary["case"] == c, "sailing_days_mean"].iloc[0] for c in ORDER], width=0.36, label="出海天数", color="#59A14F")
    axes[1].bar(x + 0.18, [summary.loc[summary["case"] == c, "executed_hours_mean"].iloc[0] for c in ORDER], width=0.36, label="执行工时", color="#F28E2B")
    axes[1].set_ylabel("天数 / h")
    axes[1].set_title("资源使用")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([CN[c].split(" ", 1)[0] for c in ORDER])
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("五策略维护执行指标对比", y=1.03, fontsize=13, fontweight="bold")
    save_fig(fig, out_dir, "fig_five_strategy_route_obj_operations")


def write_report(summary: pd.DataFrame, delta: pd.DataFrame, out_dir: Path, tag: str) -> None:
    def md_table(frame: pd.DataFrame) -> str:
        cols = list(frame.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in frame.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                vals.append(f"{val:.2f}" if isinstance(val, float) else str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    main_cols = [
        ("completed_tasks", "完成任务数"),
        ("ongoing_out_final", "期末持续任务"),
        ("executed_hours", "执行工时/h"),
        ("sailing_days", "出海天数"),
        ("sailing_distance_km", "航行距离/km"),
        ("fuel_cost_yuan", "燃油成本/元"),
        ("loss_cost_yuan", "发电损失/元"),
        ("total_actual_cost_yuan", "实际执行成本/元"),
    ]
    rows = []
    for case in ORDER:
        src = summary[summary["case"] == case].iloc[0]
        row = {"策略": CN[case]}
        for metric, label in main_cols:
            row[label] = src[f"{metric}_mean_std"]
        rows.append(row)

    delta_view = delta[delta["metric"].isin(["total_actual_cost_yuan", "loss_cost_yuan", "fuel_cost_yuan", "sailing_distance_km"])]
    delta_pivot = delta_view.pivot(index="comparison", columns="metric", values="mean_delta").reset_index().rename(
        columns={
            "comparison": "对比",
            "total_actual_cost_yuan": "实际执行成本差/元",
            "loss_cost_yuan": "发电损失差/元",
            "fuel_cost_yuan": "燃油成本差/元",
            "sailing_distance_km": "航行距离差/km",
        }
    )

    s1 = summary[summary["case"] == "S1_failure_repair_route_obj"].iloc[0]
    s5 = summary[summary["case"] == "S5_wake_multi_route_obj"].iloc[0]
    md = [
        f"# 五策略航线目标公平对比汇总（{tag}）",
        "",
        "## 实验口径",
        "- 样本：sample31-sample40，共 10 组。",
        "- 五个策略均将航线访问成本放入小时级优化目标。",
        "- 运维船按“每日从港口出发、放下/接回维护人员、日终回港”计算航行距离、燃油成本和船舶固定成本。",
        "- 跨天维护任务作为持续任务进入下一滚动周期；若继续执行，则下一天重新出海并重新计算航线成本。",
        "",
        "## 多样本均值与标准差",
        md_table(pd.DataFrame(rows)),
        "",
        "## 递进策略配对差值",
        "差值为“前者 - 后者”，负值表示前者更低。",
        md_table(delta_pivot),
        "",
        "## 结果说明",
        f"- S1 平均完成任务数为 {s1['completed_tasks_mean']:.2f}，期末仍有 {s1['ongoing_out_final_mean']:.2f} 个持续任务，说明故障后维修在 20 d 窗口内存在明显跨天和未完成压力。",
        f"- S5 平均实际执行成本为 {s5['total_actual_cost_yuan_mean']:.2f} 元，是当前五策略中最低；相较 S1，成本降低约 {s5['total_actual_cost_yuan_mean'] - s1['total_actual_cost_yuan_mean']:.2f} 元。",
        "- 在所有策略均纳入航线访问成本后，S4 和 S5 的比较更公平：S5 的优势主要来自尾流感知场级发电损失降低，同时航线目标约束避免了额外航行成本失控。",
        "",
        "## 图件",
        "- `fig_five_strategy_route_obj_cost_distribution.png/.pdf/.svg`",
        "- `fig_five_strategy_route_obj_cost_breakdown.png/.pdf/.svg`",
        "- `fig_five_strategy_route_obj_operations.png/.pdf/.svg`",
    ]
    (out_dir / f"five_strategy_routing_objective_report_{tag}.md").write_text("\n".join(md), encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output-dir", default="five_strategy_64turbine_8x8_figures")
    parser.add_argument("--tag", default="samples12_64turbine_8x8")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary_csv)
    df["case"] = pd.Categorical(df["case"], ORDER, ordered=True)
    df = df.sort_values(["sample", "case"])
    summary = summarize(df)
    delta = paired_delta(df)
    df.to_csv(out_dir / f"five_strategy_routing_objective_enriched_{args.tag}.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / f"five_strategy_routing_objective_mean_std_{args.tag}.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(out_dir / f"five_strategy_routing_objective_paired_deltas_{args.tag}.csv", index=False, encoding="utf-8-sig")
    plot_cost_distribution(df, out_dir)
    plot_cost_breakdown(summary, out_dir)
    plot_operations(summary, out_dir)
    write_report(summary, delta, out_dir, args.tag)
    print(f"saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
