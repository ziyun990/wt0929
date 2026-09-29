#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize routing/fuel multi-sample experiments and draw paper figures."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STRATEGY_ORDER = [
    "S4_no_wake_routing_post",
    "S5_wake_routing_post",
    "S5_wake_routing_objective",
]

STRATEGY_CN = {
    "S4_no_wake_routing_post": "S4 无尾流闭环",
    "S5_wake_routing_post": "S5 尾流闭环",
    "S5_wake_routing_objective": "S5_obj 尾流+航线目标",
}

COMPONENT_COLS = [
    "direct_cost_yuan",
    "crew_cost_yuan",
    "vessel_cost_yuan",
    "fuel_cost_yuan",
    "overtime_cost_yuan",
    "loss_cost_yuan",
]

METRIC_COLS = [
    "scheduled_tasks",
    "completed_tasks",
    "executed_hours",
    "sailing_distance_km",
    "sailing_time_hours",
    "fuel_cost_yuan",
    "vessel_cost_yuan",
    "loss_cost_yuan",
    "total_actual_cost_yuan",
]


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#111111",
            "axes.labelcolor": "#111111",
            "axes.titleweight": "bold",
            "savefig.dpi": 600,
        }
    )


def fmt_mean_std(mean: float, std: float, decimals: int = 2) -> str:
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


def paired_stats(values: np.ndarray) -> dict[str, float | int | str]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    mean = float(np.mean(values)) if n else math.nan
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    median = float(np.median(values)) if n else math.nan
    win_count = int(np.sum(values < 0))
    lose_count = int(np.sum(values > 0))
    tie_count = int(np.sum(np.isclose(values, 0.0)))
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else math.nan
    p_value = math.nan
    test_name = "not_available"
    try:
        from scipy import stats

        if n > 1 and np.any(~np.isclose(values, values[0])):
            res = stats.ttest_1samp(values, 0.0)
            p_value = float(res.pvalue)
            test_name = "paired_delta_ttest"
    except Exception:
        pass
    return {
        "n": n,
        "mean_delta": mean,
        "std_delta": std,
        "median_delta": median,
        "ci95_delta": ci95,
        "win_count_delta_lt_0": win_count,
        "lose_count_delta_gt_0": lose_count,
        "tie_count": tie_count,
        "p_value": p_value,
        "test": test_name,
    }


def load_and_enrich(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    rows = []
    for row in df.to_dict("records"):
        roll_path = Path(str(row["roll_summary_csv"]))
        if not roll_path.exists():
            roll_path = summary_csv.parent / roll_path.name
        roll = pd.read_csv(roll_path)
        enriched = dict(row)
        for col in COMPONENT_COLS:
            enriched[col] = float(roll[col].sum()) if col in roll.columns else float(row.get(col, 0.0))
        enriched["total_actual_cost_yuan"] = float(roll["total_actual_cost_yuan"].sum())
        enriched["sailing_days"] = int((roll.get("vessel_cost_yuan", pd.Series(dtype=float)) > 0).sum())
        rows.append(enriched)
    out = pd.DataFrame(rows)
    out["strategy_cn"] = out["case"].map(STRATEGY_CN)
    out["case"] = pd.Categorical(out["case"], STRATEGY_ORDER, ordered=True)
    return out.sort_values(["sample", "case"]).reset_index(drop=True)


def build_summary_tables(df: pd.DataFrame, out_dir: Path, tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped_rows = []
    for case in STRATEGY_ORDER:
        sub = df[df["case"].astype(str) == case]
        row = {"case": case, "strategy_cn": STRATEGY_CN[case], "n": len(sub)}
        for col in METRIC_COLS + ["sailing_days"]:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_std"] = sub[col].std(ddof=1)
            row[f"{col}_mean_std"] = fmt_mean_std(sub[col].mean(), sub[col].std(ddof=1), 2)
        for col in COMPONENT_COLS:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_std"] = sub[col].std(ddof=1)
            row[f"{col}_mean_std"] = fmt_mean_std(sub[col].mean(), sub[col].std(ddof=1), 2)
        grouped_rows.append(row)
    summary = pd.DataFrame(grouped_rows)
    summary.to_csv(out_dir / f"routing_fuel_multisample_mean_std_{tag}.csv", index=False, encoding="utf-8-sig")

    wide = df.pivot(index="sample", columns="case")
    comparisons = [
        ("S5-S4", "S5_wake_routing_post", "S4_no_wake_routing_post"),
        ("S5_obj-S5", "S5_wake_routing_objective", "S5_wake_routing_post"),
        ("S5_obj-S4", "S5_wake_routing_objective", "S4_no_wake_routing_post"),
    ]
    delta_rows = []
    for label, a, b in comparisons:
        for metric in [
            "total_actual_cost_yuan",
            "loss_cost_yuan",
            "vessel_cost_yuan",
            "fuel_cost_yuan",
            "overtime_cost_yuan",
            "sailing_distance_km",
            "sailing_days",
            "completed_tasks",
        ]:
            vals = wide[(metric, a)] - wide[(metric, b)]
            stat = paired_stats(vals.values)
            stat.update({"comparison": label, "metric": metric})
            delta_rows.append(stat)
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(out_dir / f"routing_fuel_multisample_paired_deltas_{tag}.csv", index=False, encoding="utf-8-sig")
    return summary, deltas


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_total_cost(df: pd.DataFrame, out_dir: Path) -> None:
    setup_plot_style()
    colors = ["#4E79A7", "#59A14F", "#F28E2B"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    data = [df[df["case"].astype(str) == case]["total_actual_cost_yuan"].values / 10000 for case in STRATEGY_ORDER]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
    for median in bp["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.4)
    rng = np.random.default_rng(2026)
    for i, (vals, color) in enumerate(zip(data, colors), start=1):
        jitter = rng.normal(0, 0.045, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=32, color=color, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([STRATEGY_CN[c] for c in STRATEGY_ORDER], rotation=0)
    ax.set_ylabel("实际执行成本 / 万元")
    ax.set_title("航线燃油扩展下多样本实际执行成本分布")
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    save_fig(fig, out_dir, "fig_routing_fuel_total_cost_distribution")


def plot_cost_breakdown(summary: pd.DataFrame, out_dir: Path) -> None:
    setup_plot_style()
    comp_labels = {
        "direct_cost_yuan": "直接维护",
        "crew_cost_yuan": "人工",
        "vessel_cost_yuan": "船舶固定",
        "fuel_cost_yuan": "燃油",
        "overtime_cost_yuan": "加班",
        "loss_cost_yuan": "发电损失",
    }
    colors = ["#4E79A7", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#F28E2B"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(STRATEGY_ORDER))
    bottom = np.zeros(len(STRATEGY_ORDER))
    for col, color in zip(COMPONENT_COLS, colors):
        vals = np.array([summary.loc[summary["case"] == case, f"{col}_mean"].iloc[0] for case in STRATEGY_ORDER]) / 10000
        ax.bar(x, vals, bottom=bottom, label=comp_labels[col], color=color, edgecolor="white", linewidth=0.7)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_CN[c] for c in STRATEGY_ORDER])
    ax.set_ylabel("平均成本 / 万元")
    ax.set_title("多样本平均实际执行成本分解")
    ax.legend(ncol=3, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    save_fig(fig, out_dir, "fig_routing_fuel_cost_breakdown")


def plot_distance_loss(df: pd.DataFrame, out_dir: Path) -> None:
    setup_plot_style()
    colors = {
        "S4_no_wake_routing_post": "#4E79A7",
        "S5_wake_routing_post": "#59A14F",
        "S5_wake_routing_objective": "#F28E2B",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for metric, ylabel, ax in [
        ("sailing_distance_km", "航行距离 / km", axes[0]),
        ("loss_cost_yuan", "发电损失成本 / 万元", axes[1]),
    ]:
        for case in STRATEGY_ORDER:
            sub = df[df["case"].astype(str) == case].sort_values("sample")
            y = sub[metric].values / (10000 if metric.endswith("_yuan") else 1)
            ax.plot(sub["sample"], y, marker="o", linewidth=1.6, markersize=4.5, label=STRATEGY_CN[case], color=colors[case])
        ax.set_xlabel("样本编号")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22, linewidth=0.7)
    axes[0].set_title("航行距离")
    axes[1].set_title("发电损失")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("航线与尾流损失指标的样本变化", y=1.03, fontsize=13, fontweight="bold")
    save_fig(fig, out_dir, "fig_routing_fuel_distance_loss_by_sample")


def plot_delta_breakdown(df: pd.DataFrame, out_dir: Path) -> None:
    setup_plot_style()
    comparisons = [
        ("S5-S4", "S5_wake_routing_post", "S4_no_wake_routing_post"),
        ("S5_obj-S5", "S5_wake_routing_objective", "S5_wake_routing_post"),
    ]
    comp_cols = COMPONENT_COLS
    comp_labels = ["直接维护", "人工", "船舶固定", "燃油", "加班", "发电损失"]
    colors = ["#4E79A7", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#F28E2B"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    wide = df.pivot(index="sample", columns="case")
    for ax, (title, a, b) in zip(axes, comparisons):
        vals = np.array([(wide[(col, a)] - wide[(col, b)]).mean() for col in comp_cols]) / 10000
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.bar(np.arange(len(vals)), vals, color=colors, edgecolor="white", linewidth=0.7)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(comp_labels, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    axes[0].set_ylabel("平均成本差值 / 万元")
    fig.suptitle("策略差异的成本分项来源", y=1.03, fontsize=13, fontweight="bold")
    save_fig(fig, out_dir, "fig_routing_fuel_delta_breakdown")


def write_markdown(summary: pd.DataFrame, deltas: pd.DataFrame, df: pd.DataFrame, out_dir: Path, tag: str) -> None:
    def df_to_md(frame: pd.DataFrame) -> str:
        cols = list(frame.columns)
        lines = [
            "| " + " | ".join(str(c) for c in cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in frame.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.2f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    key_metrics = [
        ("completed_tasks", "完成任务数"),
        ("executed_hours", "执行工时/h"),
        ("sailing_days", "出海天数"),
        ("sailing_distance_km", "航行距离/km"),
        ("fuel_cost_yuan", "燃油成本/元"),
        ("loss_cost_yuan", "发电损失/元"),
        ("total_actual_cost_yuan", "实际执行成本/元"),
    ]
    table_rows = []
    for case in STRATEGY_ORDER:
        row = {"策略": STRATEGY_CN[case]}
        line = summary[summary["case"] == case].iloc[0]
        for col, label in key_metrics:
            row[label] = line[f"{col}_mean_std"]
        table_rows.append(row)
    table = pd.DataFrame(table_rows)

    delta_key = deltas[deltas["metric"].isin(["total_actual_cost_yuan", "loss_cost_yuan", "sailing_distance_km", "fuel_cost_yuan", "vessel_cost_yuan"])]
    delta_table = delta_key.pivot(index="comparison", columns="metric", values="mean_delta").reset_index()
    rename = {
        "comparison": "对比",
        "total_actual_cost_yuan": "实际执行成本差/元",
        "loss_cost_yuan": "发电损失差/元",
        "sailing_distance_km": "航行距离差/km",
        "fuel_cost_yuan": "燃油成本差/元",
        "vessel_cost_yuan": "船舶固定成本差/元",
    }
    delta_table = delta_table.rename(columns=rename)

    s4 = df[df["case"].astype(str) == "S4_no_wake_routing_post"].set_index("sample")
    s5 = df[df["case"].astype(str) == "S5_wake_routing_post"].set_index("sample")
    s5o = df[df["case"].astype(str) == "S5_wake_routing_objective"].set_index("sample")

    s5_vs_s4 = (s5["total_actual_cost_yuan"] - s4["total_actual_cost_yuan"]).mean()
    s5o_vs_s5 = (s5o["total_actual_cost_yuan"] - s5["total_actual_cost_yuan"]).mean()
    s5o_vs_s4 = (s5o["total_actual_cost_yuan"] - s4["total_actual_cost_yuan"]).mean()
    loss_s5_vs_s4 = (s5["loss_cost_yuan"] - s4["loss_cost_yuan"]).mean()
    loss_s5o_vs_s5 = (s5o["loss_cost_yuan"] - s5["loss_cost_yuan"]).mean()

    md = []
    md.append(f"# 航线燃油扩展模型多样本实验汇总（{tag}）\n")
    md.append("## 实验口径\n")
    md.append("- 样本：sample31-sample40，共 10 组。")
    md.append("- 策略：S4 无尾流闭环、S5 尾流闭环、S5_obj 尾流+航线目标。")
    md.append("- 评价口径：每日运维船从港口出发，按放下/接回维护人员事件形成航线，日终回港；实际执行成本包含直接维护、人工、船舶固定、燃油、加班和发电损失成本。")
    md.append("- 说明：S5_obj 在优化目标中加入航线/燃油代理成本；S4 与 S5 仍按后处理航线口径统一评价。\n")
    md.append("## 多样本均值与标准差\n")
    md.append(df_to_md(table))
    md.append("\n## 配对差值均值\n")
    md.append("差值均按“前者 - 后者”计算，负值表示前者成本或指标更低。")
    md.append(df_to_md(delta_table))
    md.append("\n## 主要结论\n")
    md.append(f"- S5 相比 S4 的平均发电损失变化为 {loss_s5_vs_s4:.2f} 元，实际执行成本变化为 {s5_vs_s4:.2f} 元。")
    md.append(f"- S5_obj 相比 S5 的平均发电损失变化为 {loss_s5o_vs_s5:.2f} 元，实际执行成本变化为 {s5o_vs_s5:.2f} 元。")
    md.append(f"- S5_obj 相比 S4 的平均实际执行成本变化为 {s5o_vs_s4:.2f} 元。")
    md.append("- 该结果说明：只考虑尾流损失可能降低发电损失，但若导致额外出海或航程增加，总执行成本未必同步下降；将航线燃油因素纳入优化目标后，可以在尾流损失和航行成本之间形成更稳的折中。")
    md.append("\n## 生成图件\n")
    for name in [
        "fig_routing_fuel_total_cost_distribution",
        "fig_routing_fuel_cost_breakdown",
        "fig_routing_fuel_distance_loss_by_sample",
        "fig_routing_fuel_delta_breakdown",
    ]:
        md.append(f"- `{name}.png/.pdf/.svg`")
    (out_dir / f"routing_fuel_multisample_report_{tag}.md").write_text("\n".join(md), encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True, help="routing_fuel_comparison_summary_*.csv")
    parser.add_argument("--output-dir", default="routing_fuel_64turbine_8x8_figures")
    parser.add_argument("--tag", default="samples12_64turbine_8x8")
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_enrich(summary_csv)
    df.to_csv(out_dir / f"routing_fuel_multisample_enriched_{args.tag}.csv", index=False, encoding="utf-8-sig")
    summary, deltas = build_summary_tables(df, out_dir, args.tag)
    plot_total_cost(df, out_dir)
    plot_cost_breakdown(summary, out_dir)
    plot_distance_loss(df, out_dir)
    plot_delta_breakdown(df, out_dir)
    write_markdown(summary, deltas, df, out_dir, args.tag)
    print(f"saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
