#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run five strategies for the 8x8 case with routing cost in every objective."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from multi_component_params import BASE_DISTANCE_KM, MATERIAL_COST_MULTIPLIER


CASE_DIR = Path(__file__).resolve().parent
RUNNER = CASE_DIR / "case_8x8_multicomponent_hourly_milp.py"


CASES = [
    ("failure_repair", "S1_failure_repair_route_obj", False, "S1 故障后维修"),
    ("predictive_rul", "S2_predictive_rul_route_obj", False, "S2 预测RUL"),
    ("weather_opportunity", "S3_weather_route_obj", False, "S3 天气可达机会"),
    ("multi_opportunity", "S4_multi_route_obj", False, "S4 闭环多机会"),
    ("wake_multi_opportunity", "S5_wake_multi_route_obj", True, "S5 尾流闭环多机会"),
]


def run_case(sample: int, strategy: str, tag: str, wake: bool, args: argparse.Namespace) -> Path:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--sample",
        str(sample),
        "--grid-size",
        str(args.grid_size),
        "--horizon-days",
        str(args.horizon_days),
        "--n-scenarios",
        str(args.n_scenarios),
        "--loss-scenario-count",
        str(args.loss_scenario_count),
        "--strategy",
        strategy,
        "--max-parallel-turbines",
        str(args.max_parallel_turbines),
        "--aged-count-threshold",
        str(args.aged_count_threshold),
        "--base-distance-km",
        str(args.base_distance_km),
        "--rul-seed",
        str(args.rul_seed_base + sample),
        "--output-tag",
        tag,
        "--solver",
        args.solver,
        "--routing-cost-in-objective",
    ]
    if wake:
        cmd.append("--wake")
    subprocess.run(cmd, cwd=str(CASE_DIR), check=True)
    return CASE_DIR / f"hourly_milp_roll_summary_sample{sample}_{tag}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run S1-S5 for the 8x8 64-turbine case."
    )
    parser.add_argument("--samples", type=int, nargs="+", default=[31])
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--horizon-days", type=int, default=20)
    parser.add_argument("--n-scenarios", type=int, default=10)
    parser.add_argument("--loss-scenario-count", type=int, default=10)
    parser.add_argument("--max-parallel-turbines", type=int, default=3)
    parser.add_argument("--aged-count-threshold", type=int, default=32)
    parser.add_argument("--base-distance-km", type=float, default=BASE_DISTANCE_KM)
    parser.add_argument("--rul-seed-base", type=int, default=37)
    parser.add_argument("--output-tag", type=str, default="64turbine_8x8_five_route_obj")
    parser.add_argument("--solver", choices=["copt", "cbc"], default="copt")
    args = parser.parse_args()

    rows = []
    for sample in args.samples:
        for strategy, label, wake, label_cn in CASES:
            tag = f"{args.output_tag}_{label}"
            roll_csv = run_case(sample, strategy, tag, wake, args)
            df = pd.read_csv(roll_csv)
            rows.append(
                {
                    "sample": sample,
                    "n_turbines": args.grid_size**2,
                    "case": label,
                    "strategy": strategy,
                    "strategy_cn": label_cn,
                    "routing_cost_in_objective": 1,
                    "wake": int(wake),
                    "rul_seed": args.rul_seed_base + sample,
                    "base_distance_km": args.base_distance_km,
                    "material_cost_multiplier": MATERIAL_COST_MULTIPLIER,
                    "experiment_config_version": df["experiment_config_version"].iloc[0],
                    "rolls": len(df),
                    "candidate_records": df["candidate_component_tasks"].sum(),
                    "executed_hours": df["executed_hours"].sum(),
                    "scheduled_tasks": df["scheduled_tasks"].sum(),
                    "completed_tasks": df["completed_tasks"].sum(),
                    "ongoing_out_final": df["ongoing_out"].iloc[-1] if len(df) else 0,
                    "sailing_days": int((df["vessel_cost_yuan"] > 0).sum()),
                    "sailing_distance_km": df["sailing_distance_km"].sum(),
                    "sailing_time_hours": df["sailing_time_hours"].sum(),
                    "direct_cost_yuan": df["direct_cost_yuan"].sum(),
                    "crew_cost_yuan": df["crew_cost_yuan"].sum(),
                    "vessel_cost_yuan": df["vessel_cost_yuan"].sum(),
                    "fuel_cost_yuan": df["fuel_cost_yuan"].sum(),
                    "overtime_cost_yuan": df["overtime_cost_yuan"].sum(),
                    "loss_cost_yuan": df["loss_cost_yuan"].sum(),
                    "total_actual_cost_yuan": df["total_actual_cost_yuan"].sum(),
                    "roll_summary_csv": str(roll_csv),
                }
            )

    out = CASE_DIR / f"five_strategy_routing_objective_summary_{args.output_tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print("saved:", out)


if __name__ == "__main__":
    main()
