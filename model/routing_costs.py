from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


KNOT_TO_KM_PER_HOUR = 1.852


@dataclass(frozen=True)
class RoutingConfig:
    """Dalgic CTV-1 routing and fuel parameters in CNY units."""

    vessel_speed_knots: float = 26.0
    fuel_cost_yuan_per_sailing_hour: float = 3183.59808
    base_distance_km: float = 37.04
    base_direction: str = "west"

    @property
    def vessel_speed_km_per_hour(self) -> float:
        return self.vessel_speed_knots * KNOT_TO_KM_PER_HOUR


def infer_base_coordinate(layout_m: np.ndarray, config: RoutingConfig) -> np.ndarray:
    """Place the maintenance base at a fixed distance from the wind-farm boundary."""

    layout_km = np.asarray(layout_m, dtype=float) / 1000.0
    min_x, min_y = layout_km.min(axis=0)
    max_x, max_y = layout_km.max(axis=0)
    mid_x = (min_x + max_x) / 2.0
    mid_y = (min_y + max_y) / 2.0

    direction = config.base_direction.lower()
    if direction == "west":
        return np.array([min_x - config.base_distance_km, mid_y], dtype=float)
    if direction == "east":
        return np.array([max_x + config.base_distance_km, mid_y], dtype=float)
    if direction == "south":
        return np.array([mid_x, min_y - config.base_distance_km], dtype=float)
    if direction == "north":
        return np.array([mid_x, max_y + config.base_distance_km], dtype=float)
    raise ValueError(f"Unsupported base direction: {config.base_direction!r}")


def distance_matrix_km(layout_m: np.ndarray, base_coord_km: np.ndarray) -> np.ndarray:
    """Return pairwise distances for node 0 as base and nodes 1..N as turbines."""

    turbine_coords_km = np.asarray(layout_m, dtype=float) / 1000.0
    coords = np.vstack([np.asarray(base_coord_km, dtype=float), turbine_coords_km])
    delta = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def nearest_neighbor_route(
    visited_turbines_zero_based: Iterable[int],
    dist_km: np.ndarray,
) -> list[int]:
    """Build a simple base-turbine-...-base route using nearest-neighbor insertion."""

    remaining = {int(i) + 1 for i in visited_turbines_zero_based}
    route = [0]
    current = 0
    while remaining:
        nxt = min(remaining, key=lambda node: (dist_km[current, node], node))
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt
    route.append(0)
    return route


def route_distance_km(route: list[int], dist_km: np.ndarray) -> float:
    if len(route) < 2:
        return 0.0
    return float(sum(dist_km[a, b] for a, b in zip(route[:-1], route[1:])))


def evaluate_daily_route(
    visited_turbines_zero_based: Iterable[int],
    layout_m: np.ndarray,
    config: RoutingConfig,
) -> dict:
    """Evaluate distance, sailing time and fuel cost for one rolling day."""

    visited = sorted({int(i) for i in visited_turbines_zero_based})
    if not visited:
        return {
            "visited_turbines": [],
            "route_nodes": [0],
            "route_turbines": [],
            "sailing_distance_km": 0.0,
            "sailing_time_hours": 0.0,
            "fuel_cost_yuan": 0.0,
        }

    base = infer_base_coordinate(layout_m, config)
    dist = distance_matrix_km(layout_m, base)
    route = nearest_neighbor_route(visited, dist)
    distance = route_distance_km(route, dist)
    sailing_time = distance / config.vessel_speed_km_per_hour
    fuel_cost = sailing_time * config.fuel_cost_yuan_per_sailing_hour
    return {
        "visited_turbines": [i + 1 for i in visited],
        "route_nodes": route,
        "route_turbines": [node if node == 0 else node for node in route],
        "route_events": "",
        "sailing_distance_km": distance,
        "sailing_time_hours": sailing_time,
        "fuel_cost_yuan": fuel_cost,
    }


def _nearest_order_from_current(current: int, targets: set[int], dist_km: np.ndarray) -> list[int]:
    ordered: list[int] = []
    remaining = set(targets)
    while remaining:
        nxt = min(remaining, key=lambda node: (dist_km[current, node], node))
        ordered.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return ordered


def evaluate_dropoff_pickup_route(
    task_windows: Iterable[tuple[int, int, int]],
    layout_m: np.ndarray,
    config: RoutingConfig,
) -> dict:
    """Evaluate a daily route with crew drop-off and pick-up events.

    Each task window is ``(turbine_zero_based, first_work_hour, end_hour_exclusive)``.
    When multiple tasks are assigned to the same turbine in one day, they are merged
    into one crew-occupancy window from the earliest effective work hour to the latest
    effective work hour. The vessel starts from the base, visits all drop-off/pick-up
    event turbines using a nearest-neighbor sequence, and returns to the base at the
    end of the rolling day.
    """

    merged: dict[int, tuple[int, int]] = {}
    for turbine, start, end in task_windows:
        turbine = int(turbine)
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        if turbine in merged:
            old_start, old_end = merged[turbine]
            merged[turbine] = (min(old_start, start), max(old_end, end))
        else:
            merged[turbine] = (start, end)

    if not merged:
        return {
            "visited_turbines": [],
            "route_nodes": [0],
            "route_turbines": [],
            "route_events": "",
            "sailing_distance_km": 0.0,
            "sailing_time_hours": 0.0,
            "fuel_cost_yuan": 0.0,
        }

    base = infer_base_coordinate(layout_m, config)
    dist = distance_matrix_km(layout_m, base)

    events: dict[tuple[int, int], set[int]] = {}
    for turbine, (start, end) in merged.items():
        node = turbine + 1
        # Pick-up events are processed before drop-off events when times tie.
        events.setdefault((start, 1), set()).add(node)
        events.setdefault((end, 0), set()).add(node)

    route = [0]
    event_labels: list[str] = []
    current = 0
    for (hour, kind), targets in sorted(events.items()):
        ordered = _nearest_order_from_current(current, targets, dist)
        action = "PICK" if kind == 0 else "DROP"
        for node in ordered:
            if current != node:
                route.append(node)
            event_labels.append(f"h{hour}:{action}:WT{node}")
            current = node

    if current != 0:
        route.append(0)

    distance = route_distance_km(route, dist)
    sailing_time = distance / config.vessel_speed_km_per_hour
    fuel_cost = sailing_time * config.fuel_cost_yuan_per_sailing_hour
    return {
        "visited_turbines": [i + 1 for i in sorted(merged)],
        "route_nodes": route,
        "route_turbines": [node if node == 0 else node for node in route],
        "route_events": ";".join(event_labels),
        "sailing_distance_km": distance,
        "sailing_time_hours": sailing_time,
        "fuel_cost_yuan": fuel_cost,
    }


def proxy_roundtrip_visit_costs(
    layout_m: np.ndarray,
    config: RoutingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate per-turbine visit cost/time by base-turbine-base round trip."""

    base = infer_base_coordinate(layout_m, config)
    dist = distance_matrix_km(layout_m, base)
    roundtrip_km = 2.0 * dist[0, 1:]
    time_hours = roundtrip_km / config.vessel_speed_km_per_hour
    costs = time_hours * config.fuel_cost_yuan_per_sailing_hour
    return costs.astype(float), time_hours.astype(float)
