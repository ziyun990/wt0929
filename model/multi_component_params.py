from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Scenario scaling applied only to component material costs. Labour, vessel,
# fuel, overtime and generation-loss costs are calculated separately.
MATERIAL_COST_MULTIPLIER = 5.0


def _scaled_material_cost(base_cost_yuan: float) -> float:
    return round(float(base_cost_yuan) * MATERIAL_COST_MULTIPLIER, 2)


@dataclass(frozen=True)
class ComponentParam:
    name: str
    chinese_name: str
    pm_cost_yuan: float
    cm_cost_yuan: float
    pm_hours: float
    cm_hours: float
    pm_crew_size: int
    cm_crew_size: int
    lifetime_days: float
    forecast_std_days: float


COMPONENTS: tuple[ComponentParam, ...] = (
    ComponentParam(
        name="gearbox",
        chinese_name="齿轮箱",
        pm_cost_yuan=_scaled_material_cost(965.60),
        cm_cost_yuan=_scaled_material_cost(19312.00),
        pm_hours=8.0,
        cm_hours=22.0,
        pm_crew_size=3,
        cm_crew_size=4,
        lifetime_days=60.0,
        forecast_std_days=3.0,
    ),
    ComponentParam(
        name="generator",
        chinese_name="发电机",
        pm_cost_yuan=_scaled_material_cost(1235.97),
        cm_cost_yuan=_scaled_material_cost(27036.80),
        pm_hours=7.0,
        cm_hours=24.0,
        pm_crew_size=3,
        cm_crew_size=3,
        lifetime_days=58.0,
        forecast_std_days=3.0,
    ),
    ComponentParam(
        name="blade",
        chinese_name="叶片",
        pm_cost_yuan=_scaled_material_cost(1313.22),
        cm_cost_yuan=_scaled_material_cost(11587.20),
        pm_hours=9.0,
        cm_hours=21.0,
        pm_crew_size=3,
        cm_crew_size=4,
        lifetime_days=70.0,
        forecast_std_days=4.0,
    ),
    ComponentParam(
        name="main_bearing",
        chinese_name="主轴承",
        pm_cost_yuan=_scaled_material_cost(7724.80),
        cm_cost_yuan=_scaled_material_cost(108147.20),
        pm_hours=10.0,
        cm_hours=36.0,
        pm_crew_size=2,
        cm_crew_size=3,
        lifetime_days=62.0,
        forecast_std_days=3.0,
    ),
)


COMPONENT_NAMES = [item.name for item in COMPONENTS]
COMPONENT_CHINESE_NAMES = [item.chinese_name for item in COMPONENTS]
PM_COSTS_YUAN = np.array([item.pm_cost_yuan for item in COMPONENTS], dtype=float)
CM_COSTS_YUAN = np.array([item.cm_cost_yuan for item in COMPONENTS], dtype=float)
PM_HOURS = np.array([item.pm_hours for item in COMPONENTS], dtype=float)
CM_HOURS = np.array([item.cm_hours for item in COMPONENTS], dtype=float)
PM_CREW_SIZES = np.array([item.pm_crew_size for item in COMPONENTS], dtype=int)
CM_CREW_SIZES = np.array([item.cm_crew_size for item in COMPONENTS], dtype=int)
LIFETIME_DAYS = np.array([item.lifetime_days for item in COMPONENTS], dtype=float)
FORECAST_STD_DAYS = np.array([item.forecast_std_days for item in COMPONENTS], dtype=float)


RATED_POWER_MW = 10.0
MAX_MAINTENANCE_WIND_M_S = 15.0
MAX_MAINTENANCE_WAVE_M = 1.5
CUT_IN_WIND_M_S = 4.0
RATED_WIND_M_S = 11.4
CUT_OUT_WIND_M_S = 25.0

DAY_START_HOUR = 8
DAY_END_HOUR = 20
STANDARD_WORK_HOURS_PER_DAY = 12
MAX_PARALLEL_TURBINES = 3
MAX_COMPONENTS_PER_TURBINE_AT_A_TIME = 1

CREW_COST_YUAN_PER_PERSON_DAY = 3862.40
CREW_COST_YUAN_PER_PERSON_HOUR = CREW_COST_YUAN_PER_PERSON_DAY / STANDARD_WORK_HOURS_PER_DAY
# Compatibility values for the legacy aggregate runner. The hourly MILP uses
# component-specific crew sizes instead of this representative three-person team.
CREW_SIZE = 3
CREW_TEAM_COST_YUAN_PER_HOUR = CREW_SIZE * CREW_COST_YUAN_PER_PERSON_HOUR
OVERTIME_COST_YUAN_PER_HOUR = 0.0
VESSEL_COST_YUAN_PER_DAY = 32241.30
VESSEL_TECHNICIAN_CAPACITY = 12
MAX_DAILY_TURBINE_VISITS = 6

VESSEL_SPEED_KNOTS = 26.0
VESSEL_FUEL_CONSUMPTION_L_PER_HOUR = 384.0
FUEL_PRICE_YUAN_PER_LITRE = 8.29062
FUEL_COST_YUAN_PER_SAILING_HOUR = (
    VESSEL_FUEL_CONSUMPTION_L_PER_HOUR * FUEL_PRICE_YUAN_PER_LITRE
)
BASE_DISTANCE_KM = 37.04
REFERENCE_PRICE_YUAN_PER_MWH = 812.0

UNFINISHED_TASK_PENALTY_YUAN_PER_HOUR = 1500.0
TERMINAL_HIGH_RISK_PENALTY_YUAN_PER_COMPONENT = 10000.0


def component_param_table() -> list[dict[str, float | str]]:
    return [
        {
            "component": item.name,
            "component_cn": item.chinese_name,
            "pm_cost_yuan": item.pm_cost_yuan,
            "cm_cost_yuan": item.cm_cost_yuan,
            "pm_hours": item.pm_hours,
            "cm_hours": item.cm_hours,
            "pm_crew_size": item.pm_crew_size,
            "cm_crew_size": item.cm_crew_size,
            "lifetime_days": item.lifetime_days,
            "forecast_std_days": item.forecast_std_days,
        }
        for item in COMPONENTS
    ]
