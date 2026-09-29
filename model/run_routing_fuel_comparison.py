from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from multi_component_params import BASE_DISTANCE_KM


CASE_DIR = Path(__file__).resolve().parent
RUNNER = CASE_DIR / "case_8x8_multicomponent_hourly_milp.py"


def run_case(sample: int, strategy: str, tag: str, routing_objective: bool, args: argparse.Namespace) -> Path:
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
    ]
    if strategy == "wake_multi_opportunity":
        cmd.append("--wake")
    if routing_objective:
        cmd.append("--routing-cost-in-objective")
    subprocess.run(cmd, cwd=str(CASE_DIR), check=True)
    return CASE_DIR / f"hourly_milp_roll_summary_sample{sample}_{tag}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 64-turbine S4/S5 routing-fuel cases.")
    parser.add_argument("--samples", type=int, nargs="+", default=[31])
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--horizon-days", type=int, default=20)
    parser.add_argument("--n-scenarios", type=int, default=10)
    parser.add_argument("--loss-scenario-count", type=int, default=10)
    parser.add_argument("--max-parallel-turbines", type=int, default=3)
    parser.add_argument("--aged-count-threshold", type=int, default=32)
    parser.add_argument("--base-distance-km", type=float, default=BASE_DISTANCE_KM)
    parser.add_argument("--rul-seed-base", type=int, default=37)
    parser.add_argument("--output-tag", type=str, default="64turbine_8x8_routing_fuel_cmp")
    args = parser.parse_args()

    rows = []
    for sample in args.samples:
        cases = [
            ("multi_opportunity", "S4_no_wake_routing_post", False),
            ("wake_multi_opportunity", "S5_wake_routing_post", False),
            ("wake_multi_opportunity", "S5_wake_routing_objective", True),
        ]
        for strategy, label, routing_obj in cases:
            tag = f"{args.output_tag}_{label}"
            roll_csv = run_case(sample, strategy, tag, routing_obj, args)
            df = pd.read_csv(roll_csv)
            rows.append(
                {
                    "sample": sample,
                    "n_turbines": args.grid_size**2,
                    "case": label,
                    "rul_seed": args.rul_seed_base + sample,
                    "base_distance_km": args.base_distance_km,
                    "routing_cost_in_objective": int(routing_obj),
                    "rolls": len(df),
                    "executed_hours": df["executed_hours"].sum(),
                    "scheduled_tasks": df["scheduled_tasks"].sum(),
                    "completed_tasks": df["completed_tasks"].sum(),
                    "sailing_distance_km": df["sailing_distance_km"].sum(),
                    "sailing_time_hours": df["sailing_time_hours"].sum(),
                    "fuel_cost_yuan": df["fuel_cost_yuan"].sum(),
                    "vessel_cost_yuan": df["vessel_cost_yuan"].sum(),
                    "loss_cost_yuan": df["loss_cost_yuan"].sum(),
                    "total_actual_cost_yuan": df["total_actual_cost_yuan"].sum(),
                    "roll_summary_csv": str(roll_csv),
                }
            )

    out = CASE_DIR / f"routing_fuel_comparison_summary_{args.output_tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print("saved:", out)


if __name__ == "__main__":
    main()
