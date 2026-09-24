"""Offline ITRA Race Score estimator.

The official ITRA formula is not public.  This module models the two pieces that
can be checked from public data:

* Within one race, ``race_score * finish_time`` is almost constant.  A known
  score/time pair is therefore a very strong race-specific calibration anchor.
* Without an anchor, the race constant is estimated from distance and ascent.
  Its scaling prior is learned from the large, unlabelled TRAP result corpus and
  its level is calibrated with modern public ITRA score/time pairs.

Outputs are estimates, never official ITRA scores.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def time_to_hours(value: str | float | int) -> float:
    """Convert H:MM, H:MM:SS, or a numeric hour value to hours."""
    if isinstance(value, (float, int, np.floating, np.integer)):
        hours = float(value)
    else:
        parts = str(value).strip().split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Unsupported time value: {value!r}")
        hours = float(parts[0]) + float(parts[1]) / 60
        if len(parts) == 3:
            hours += float(parts[2]) / 3600
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError("Finish time must be a positive finite value")
    return hours


def course_features(distance_km: float, elevation_gain_m: float) -> np.ndarray:
    """Return the compact course feature vector used by the prototype."""
    distance_km = float(distance_km)
    elevation_gain_m = float(elevation_gain_m)
    if distance_km <= 0 or elevation_gain_m < 0:
        raise ValueError("Distance must be positive and elevation gain non-negative")
    km_effort = distance_km + elevation_gain_m / 100.0
    vertical_share = (elevation_gain_m / 100.0) / km_effort
    return np.array([1.0, math.log(km_effort), vertical_share], dtype=float)


def robust_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_iterations: int = 40,
    huber_k: float = 1.345,
) -> np.ndarray:
    """Small dependency-free Huber iteratively reweighted least-squares fit."""
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    for _ in range(max_iterations):
        residual = y - x @ beta
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        if not np.isfinite(scale) or scale < 1e-10:
            break
        cutoff = huber_k * scale
        weights = np.ones_like(residual)
        large = np.abs(residual) > cutoff
        weights[large] = cutoff / np.abs(residual[large])
        root_w = np.sqrt(weights)
        updated = np.linalg.lstsq(x * root_w[:, None], y * root_w, rcond=None)[0]
        if np.max(np.abs(updated - beta)) < 1e-9:
            beta = updated
            break
        beta = updated
    return beta


def ridge_to_prior_fit(
    x: np.ndarray,
    y: np.ndarray,
    prior: np.ndarray,
    penalty: float,
) -> np.ndarray:
    """Fit log race constants while shrinking course slopes to the TRAP prior."""
    regularizer = np.diag([0.0, penalty, penalty])
    return np.linalg.solve(x.T @ x + regularizer, x.T @ y + regularizer @ prior)


@dataclass(frozen=True)
class RaceScoreEstimate:
    score: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float
    confidence: str
    method: str


@dataclass(frozen=True)
class AnchorCalibration:
    race_constant: float
    anchor_count: int
    used_count: int
    rejected_count: int
    relative_mad: float


class RaceScoreModel:
    """Load and apply a trained offline model artifact."""

    def __init__(self, artifact: dict):
        self.artifact = artifact

    @classmethod
    def load(cls, path: str | Path) -> "RaceScoreModel":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def predict_race_constant(self, distance_km: float, elevation_gain_m: float) -> float:
        features = course_features(distance_km, elevation_gain_m)
        coefficients = np.asarray(self.artifact["course_model"]["coefficients"], dtype=float)
        return float(math.exp(features @ coefficients))

    def calibrate_anchors(
        self,
        anchors: Iterable[tuple[str | float, float]],
    ) -> AnchorCalibration:
        """Robustly combine same-race score/time anchors into one race constant."""
        constants = []
        for anchor_time, anchor_score in anchors:
            score = float(anchor_score)
            if not math.isfinite(score) or not 0 < score <= 1000:
                raise ValueError("Anchor scores must be between 0 and 1000")
            constants.append(time_to_hours(anchor_time) * score)
        if not constants:
            raise ValueError("At least one valid score/time anchor is required")

        values = np.asarray(constants, dtype=float)
        center = float(np.median(values))
        deviations = np.abs(values - center)
        mad = float(np.median(deviations))
        keep = np.ones(len(values), dtype=bool)
        if len(values) >= 3:
            if mad > 1e-12:
                keep = deviations <= 3.5 * 1.4826 * mad
            else:
                close = np.isclose(values, center, rtol=0.01, atol=1e-9)
                if int(close.sum()) >= 2:
                    keep = close
        used = values[keep]
        race_constant = float(np.median(used))
        used_mad = float(np.median(np.abs(used - race_constant)))
        relative_mad = 1.4826 * used_mad / race_constant if race_constant > 0 else 0.0
        return AnchorCalibration(
            race_constant=race_constant,
            anchor_count=len(values),
            used_count=len(used),
            rejected_count=len(values) - len(used),
            relative_mad=relative_mad,
        )

    def estimate_with_anchors(
        self,
        *,
        finish_time: str | float,
        anchors: Iterable[tuple[str | float, float]],
    ) -> RaceScoreEstimate:
        """Estimate from multiple anchors using robust aggregation and uncertainty."""
        calibration = self.calibrate_anchors(anchors)
        hours = time_to_hours(finish_time)
        score = float(np.clip(calibration.race_constant / hours, 0, 1000))
        root_n = math.sqrt(calibration.used_count)
        base_80 = float(self.artifact["within_race_model"]["anchor_error_80"]) / root_n
        base_95 = float(self.artifact["within_race_model"]["anchor_error_95"]) / root_n
        disagreement = score * calibration.relative_mad
        error_80 = math.sqrt(base_80**2 + disagreement**2)
        error_95 = math.sqrt(base_95**2 + (1.96 * disagreement) ** 2)
        return RaceScoreEstimate(
            score=score,
            lower_80=max(0.0, score - error_80),
            upper_80=min(1000.0, score + error_80),
            lower_95=max(0.0, score - error_95),
            upper_95=min(1000.0, score + error_95),
            confidence="high",
            method=(
                "multi_race_anchor_robust_median"
                if calibration.anchor_count > 1
                else "race_anchor_inverse_time"
            ),
        )

    def estimate(
        self,
        *,
        distance_km: float,
        elevation_gain_m: float,
        finish_time: str | float,
        anchor_time: str | float | None = None,
        anchor_score: float | None = None,
    ) -> RaceScoreEstimate:
        hours = time_to_hours(finish_time)
        if (anchor_time is None) != (anchor_score is None):
            raise ValueError("anchor_time and anchor_score must be supplied together")

        if anchor_time is not None and anchor_score is not None:
            race_constant = time_to_hours(anchor_time) * float(anchor_score)
            score = race_constant / hours
            error_80 = float(self.artifact["within_race_model"]["anchor_error_80"])
            error_95 = float(self.artifact["within_race_model"]["anchor_error_95"])
            confidence = "high"
            method = "race_anchor_inverse_time"
        else:
            race_constant = self.predict_race_constant(distance_km, elevation_gain_m)
            score = race_constant / hours
            error_80 = float(self.artifact["course_model"]["loo_error_80"])
            error_95 = float(self.artifact["course_model"]["loo_error_95"])
            confidence = "low"
            method = "course_only_transfer_calibration"

        score = float(np.clip(score, 0, 1000))
        return RaceScoreEstimate(
            score=score,
            lower_80=max(0.0, score - error_80),
            upper_80=min(1000.0, score + error_80),
            lower_95=max(0.0, score - error_95),
            upper_95=min(1000.0, score + error_95),
            confidence=confidence,
            method=method,
        )


def metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    actual_values = np.asarray(list(actual), dtype=float)
    predicted_values = np.asarray(list(predicted), dtype=float)
    errors = predicted_values - actual_values
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "median_absolute_error": float(np.median(np.abs(errors))),
        "max_absolute_error": float(np.max(np.abs(errors))),
        "bias": float(np.mean(errors)),
    }


def parse_trap_race_info(path: str | Path) -> pd.DataFrame:
    races = pd.read_csv(path)
    races["distance_km"] = pd.to_numeric(
        races["Distance"].astype(str).str.extract(r"([0-9.]+)")[0], errors="coerce"
    )
    races["elevation_gain_m"] = pd.to_numeric(
        races["Elevation"].astype(str).str.extract(r"([0-9.]+)")[0], errors="coerce"
    )
    races["key"] = pd.to_numeric(races["key"], errors="coerce")
    return races[["key", "Race", "Date", "distance_km", "elevation_gain_m"]]


def aggregate_trap_results(paths: Iterable[str | Path], chunksize: int = 250_000) -> pd.DataFrame:
    """Stream the two large TRAP TSV files into one compact race-level table."""
    counts: list[pd.Series] = []
    winners: list[pd.Series] = []
    top_three_rows: list[pd.DataFrame] = []

    for path in paths:
        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=["Time", "Ranking", "key_race"],
            chunksize=chunksize,
            on_bad_lines="skip",
            low_memory=False,
        ):
            chunk["key_race"] = pd.to_numeric(chunk["key_race"], errors="coerce")
            chunk["Ranking"] = pd.to_numeric(chunk["Ranking"], errors="coerce")
            chunk["hours"] = pd.to_timedelta(chunk["Time"], errors="coerce").dt.total_seconds() / 3600
            valid = chunk[
                chunk["key_race"].notna()
                & chunk["hours"].notna()
                & (chunk["hours"] > 0)
            ]
            counts.append(valid.groupby("key_race").size())
            winners.append(valid.groupby("key_race")["hours"].min())
            top_three_rows.append(valid.loc[valid["Ranking"].between(1, 3), ["key_race", "hours"]])

    count_series = pd.concat(counts).groupby(level=0).sum().rename("finishers")
    winner_series = pd.concat(winners).groupby(level=0).min().rename("winner_hours")
    top_three = pd.concat(top_three_rows, ignore_index=True)
    top_three_series = top_three.groupby("key_race")["hours"].median().rename("top3_median_hours")
    return pd.concat([count_series, winner_series, top_three_series], axis=1).reset_index()
