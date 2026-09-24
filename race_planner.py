"""Offline trail-race pacing and fuelling planner.

The implementation is intentionally independent of trail-race-planner.  It uses
the public modelling ideas documented by that project (FED/Riegel, grade-adjusted
pace, fatigue and altitude factors), but does not copy its source code.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from typing import Iterable, Mapping
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class CourseSummary:
    distance_km: float
    elevation_gain_m: float
    elevation_loss_m: float
    min_elevation_m: float
    max_elevation_m: float


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_gpx(gpx: bytes | str, resample_m: float = 50.0) -> pd.DataFrame:
    """Parse a GPX track/route and return a smoothed, distance-resampled course."""
    if isinstance(gpx, str):
        payload = gpx.encode("utf-8")
    else:
        payload = gpx
    try:
        root = ET.parse(BytesIO(payload)).getroot()
    except (ET.ParseError, ValueError) as exc:
        raise ValueError("无法解析 GPX 文件，请确认文件格式完整。") from exc

    points: list[tuple[float, float, float]] = []
    for node in root.iter():
        if _local_name(node.tag) not in {"trkpt", "rtept"}:
            continue
        try:
            lat = float(node.attrib["lat"])
            lon = float(node.attrib["lon"])
        except (KeyError, ValueError):
            continue
        elevation = None
        for child in node:
            if _local_name(child.tag) == "ele" and child.text:
                try:
                    elevation = float(child.text)
                except ValueError:
                    pass
                break
        if elevation is not None and math.isfinite(elevation):
            points.append((lat, lon, elevation))

    if len(points) < 2:
        raise ValueError("GPX 至少需要两个带海拔的轨迹点。")

    raw = pd.DataFrame(points, columns=["latitude", "longitude", "elevation_m"])
    if len(raw) >= 9:
        raw["elevation_m"] = (
            raw["elevation_m"]
            .rolling(7, center=True, min_periods=1)
            .median()
            .rolling(3, center=True, min_periods=1)
            .mean()
        )
    elif len(raw) >= 5:
        raw["elevation_m"] = raw["elevation_m"].rolling(
            3, center=True, min_periods=1
        ).median()
    raw_segment_km = _haversine_segments(raw["latitude"], raw["longitude"])
    raw_distance_km = np.cumsum(raw_segment_km)
    total_km = float(raw_distance_km[-1])
    if total_km <= 0:
        raise ValueError("GPX 轨迹没有可计算的距离。")

    step_km = max(float(resample_m), 10.0) / 1000.0
    sample_distance = np.arange(0.0, total_km, step_km)
    if len(sample_distance) == 0 or sample_distance[-1] < total_km:
        sample_distance = np.append(sample_distance, total_km)

    course = pd.DataFrame(
        {
            "distance_km": sample_distance,
            "latitude": np.interp(sample_distance, raw_distance_km, raw["latitude"]),
            "longitude": np.interp(sample_distance, raw_distance_km, raw["longitude"]),
            "elevation_m": np.interp(sample_distance, raw_distance_km, raw["elevation_m"]),
        }
    )
    course["segment_distance_km"] = course["distance_km"].diff().fillna(0.0)
    course["elevation_change_m"] = course["elevation_m"].diff().fillna(0.0)

    # Grade is measured over a 100 m centred window to reduce GPS elevation spikes.
    half_window_km = 0.05
    left = np.maximum(course["distance_km"].to_numpy() - half_window_km, 0.0)
    right = np.minimum(course["distance_km"].to_numpy() + half_window_km, total_km)
    left_ele = np.interp(left, sample_distance, course["elevation_m"])
    right_ele = np.interp(right, sample_distance, course["elevation_m"])
    span_m = np.maximum((right - left) * 1000.0, 1.0)
    course["grade"] = (right_ele - left_ele) / span_m
    course.loc[0, "grade"] = course.loc[1, "grade"]
    return course


def _haversine_segments(latitudes: Iterable[float], longitudes: Iterable[float]) -> np.ndarray:
    lat = np.radians(np.asarray(latitudes, dtype=float))
    lon = np.radians(np.asarray(longitudes, dtype=float))
    dlat = np.diff(lat, prepend=lat[0])
    dlon = np.diff(lon, prepend=lon[0])
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(np.roll(lat, 1)) * np.sin(dlon / 2.0) ** 2
    a[0] = 0.0
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def summarize_course(course: pd.DataFrame) -> CourseSummary:
    elevation_change = course["elevation_change_m"].to_numpy()
    return CourseSummary(
        distance_km=float(course["distance_km"].iloc[-1]),
        elevation_gain_m=float(elevation_change[elevation_change > 0].sum()),
        elevation_loss_m=float(-elevation_change[elevation_change < 0].sum()),
        min_elevation_m=float(course["elevation_m"].min()),
        max_elevation_m=float(course["elevation_m"].max()),
    )


def grade_cost(grade: float | np.ndarray) -> float | np.ndarray:
    """Convert decimal grade to a pace multiplier using a public GAP curve."""
    values = np.asarray(grade, dtype=float)
    knots = np.array([-0.20, -0.06, -0.05, -0.01, 0.0, 0.01, 0.20])
    costs = np.array([1.60, 0.90, 0.85, 0.97, 1.00, 1.08, 2.60])
    result = np.interp(np.clip(values, -0.20, 0.20), knots, costs)
    result = np.where(values > 0.20, 2.60 * values / 0.20, result)
    result = np.where(values < -0.20, 1.60 * np.abs(values) / 0.20, result)
    if np.isscalar(grade):
        return float(result)
    return result


def fed_distance_km(distance_km: float, elevation_gain_m: float) -> float:
    """Flat-equivalent distance: 100 m of climbing is treated as 1 km flat."""
    return float(distance_km) + float(elevation_gain_m) / 100.0


def riegel_time_seconds(
    reference_distance_km: float,
    reference_time_seconds: float,
    distance_km: float,
    elevation_gain_m: float,
) -> float:
    if min(reference_distance_km, reference_time_seconds, distance_km) <= 0:
        raise ValueError("参考距离、参考时间和比赛距离必须大于 0。")
    effective_distance = fed_distance_km(distance_km, elevation_gain_m)
    exponent = 1.06 + 0.013422 * math.sqrt(max(effective_distance - reference_distance_km, 0.0))
    return float(reference_time_seconds) * (effective_distance / float(reference_distance_km)) ** exponent


def _normalise_checkpoints(
    checkpoints: Iterable[Mapping[str, object]], total_km: float
) -> list[dict[str, float | str]]:
    cleaned: list[dict[str, float | str]] = []
    for checkpoint in checkpoints:
        try:
            distance = float(checkpoint.get("distance_km", 0.0))
            stop = max(float(checkpoint.get("stop_min", 0.0)), 0.0)
        except (TypeError, ValueError):
            continue
        if 0.0 < distance < total_km:
            cleaned.append(
                {
                    "name": str(checkpoint.get("name") or f"CP {len(cleaned) + 1}"),
                    "distance_km": distance,
                    "stop_min": stop,
                }
            )
    cleaned.sort(key=lambda row: float(row["distance_km"]))
    result: list[dict[str, float | str]] = [
        {"name": "Start", "distance_km": 0.0, "stop_min": 0.0}
    ]
    for row in cleaned:
        if float(row["distance_km"]) - float(result[-1]["distance_km"]) >= 0.05:
            result.append(row)
    result.append({"name": "Finish", "distance_km": total_km, "stop_min": 0.0})
    return result


def build_race_plan(
    course: pd.DataFrame,
    checkpoints: Iterable[Mapping[str, object]],
    *,
    target_elapsed_seconds: float | None = None,
    reference_distance_km: float | None = None,
    reference_time_seconds: float | None = None,
    fatigue_decay_pct: float = 12.0,
    altitude_effect: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Create checkpoint-to-checkpoint pacing from a course and timing target."""
    summary = summarize_course(course)
    cp = _normalise_checkpoints(checkpoints, summary.distance_km)
    total_stop_seconds = sum(float(row["stop_min"]) * 60.0 for row in cp)

    if target_elapsed_seconds is not None:
        if target_elapsed_seconds <= total_stop_seconds:
            raise ValueError("目标完赛时间必须长于计划停留时间。")
        moving_seconds = float(target_elapsed_seconds) - total_stop_seconds
    else:
        if reference_distance_km is None or reference_time_seconds is None:
            raise ValueError("请提供目标完赛时间，或参考距离与参考成绩。")
        moving_seconds = riegel_time_seconds(
            reference_distance_km,
            reference_time_seconds,
            summary.distance_km,
            summary.elevation_gain_m,
        )
        target_elapsed_seconds = moving_seconds + total_stop_seconds

    progress = course["distance_km"].to_numpy() / max(summary.distance_km, 0.001)
    fatigue = 1.0 + max(float(fatigue_decay_pct), 0.0) / 100.0 * progress
    altitude = np.ones(len(course))
    if altitude_effect:
        altitude += 0.063 * np.maximum(course["elevation_m"].to_numpy() - 1000.0, 0.0) / 1000.0
    weight = (
        course["segment_distance_km"].to_numpy()
        * grade_cost(course["grade"].to_numpy())
        * fatigue
        * altitude
    )
    if float(weight.sum()) <= 0:
        raise ValueError("赛道没有可用于配速计算的有效距离。")
    point_seconds = moving_seconds * weight / weight.sum()

    rows: list[dict[str, float | str]] = []
    cumulative_elapsed = 0.0
    for index in range(1, len(cp)):
        start = cp[index - 1]
        end = cp[index]
        start_km = float(start["distance_km"])
        end_km = float(end["distance_km"])
        mask = (course["distance_km"].to_numpy() > start_km) & (
            course["distance_km"].to_numpy() <= end_km + 1e-9
        )
        if index == len(cp) - 1:
            mask = course["distance_km"].to_numpy() > start_km
        leg_seconds = float(point_seconds[mask].sum())
        changes = course.loc[mask, "elevation_change_m"].to_numpy()
        gain = float(changes[changes > 0].sum())
        loss = float(-changes[changes < 0].sum())
        leg_km = end_km - start_km
        arrival = cumulative_elapsed + leg_seconds
        stop_seconds = float(end["stop_min"]) * 60.0
        departure = arrival + stop_seconds
        rows.append(
            {
                "segment": f'{start["name"]} → {end["name"]}',
                "from": str(start["name"]),
                "to": str(end["name"]),
                "start_km": start_km,
                "end_km": end_km,
                "distance_km": leg_km,
                "gain_m": gain,
                "loss_m": loss,
                "moving_seconds": leg_seconds,
                "avg_pace_min_km": leg_seconds / 60.0 / max(leg_km, 0.001),
                "arrival_seconds": arrival,
                "stop_min": float(end["stop_min"]),
                "departure_seconds": departure,
            }
        )
        cumulative_elapsed = departure

    metadata = {
        "distance_km": summary.distance_km,
        "elevation_gain_m": summary.elevation_gain_m,
        "elevation_loss_m": summary.elevation_loss_m,
        "moving_seconds": moving_seconds,
        "stop_seconds": total_stop_seconds,
        "elapsed_seconds": float(target_elapsed_seconds),
    }
    return pd.DataFrame(rows), metadata


def build_nutrition_plan(
    race_plan: pd.DataFrame,
    *,
    carbs_per_hour: float = 60.0,
    fluid_ml_per_hour: float = 500.0,
    flask_ml: float = 500.0,
    drink_carbs_per_flask: float = 30.0,
    gel_carbs: float = 25.0,
    solid_carbs: float = 40.0,
    gel_share: float = 0.7,
) -> pd.DataFrame:
    """Allocate practical half-servings of gels/solids for every race segment."""
    if carbs_per_hour <= 0 or fluid_ml_per_hour <= 0 or flask_ml <= 0:
        raise ValueError("碳水、饮水和水壶容量目标必须大于 0。")
    if gel_carbs <= 0 or solid_carbs <= 0:
        raise ValueError("能量胶和固体食物的碳水含量必须大于 0。")
    gel_share = float(np.clip(gel_share, 0.0, 1.0))
    output: list[dict[str, float | str]] = []
    for _, leg in race_plan.iterrows():
        hours = float(leg["moving_seconds"]) / 3600.0
        target_carbs = hours * float(carbs_per_hour)
        target_fluid = hours * float(fluid_ml_per_hour)
        drink_carbs = target_fluid / float(flask_ml) * float(drink_carbs_per_flask)
        remaining = max(target_carbs - drink_carbs, 0.0)

        max_gel_half = int(math.ceil(remaining / gel_carbs * 2.0)) + 4
        max_solid_half = int(math.ceil(remaining / solid_carbs * 2.0)) + 4
        best: tuple[float, float, float, float] | None = None
        for gel_half in range(max_gel_half + 1):
            for solid_half in range(max_solid_half + 1):
                gels = gel_half / 2.0
                solids = solid_half / 2.0
                food_carbs = gels * gel_carbs + solids * solid_carbs
                error = abs(food_carbs - remaining)
                if food_carbs > 0:
                    mix_error = abs(gels * gel_carbs / food_carbs - gel_share)
                else:
                    mix_error = 0.0 if remaining == 0 else 1.0
                objective = error + 2.0 * mix_error + 0.01 * (gels + solids)
                candidate = (objective, gels, solids, food_carbs)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        _, gels, solids, food_carbs = best
        actual_carbs = drink_carbs + food_carbs
        output.append(
            {
                "segment": str(leg["segment"]),
                "duration_hours": hours,
                "target_carbs_g": target_carbs,
                "planned_carbs_g": actual_carbs,
                "target_fluid_ml": target_fluid,
                "flasks_to_carry": math.ceil(target_fluid / flask_ml - 1e-9),
                "drink_flask_equiv": target_fluid / flask_ml,
                "gel_servings": gels,
                "solid_servings": solids,
            }
        )
    return pd.DataFrame(output)
