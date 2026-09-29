from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HourlyTask:
    task_id: str
    turbine: int
    component: int
    component_name: str
    component_cn: str
    hours: int
    direct_cost_yuan: float
    risk_benefit_yuan: float
    net_benefit_yuan: float
    reasons: tuple[str, ...]
    crew_size: int = 1
    mandatory: bool = False
    ongoing: bool = False
    corrective: bool = False
    partial_progress_weight_yuan_per_hour: float = 0.0
    idle_blocks_turbine: bool = True
