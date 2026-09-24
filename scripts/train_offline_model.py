"""Train and evaluate the first offline ITRA Race Score prototype."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from race_score_model import (  # noqa: E402
    aggregate_trap_results,
    course_features,
    metrics,
    parse_trap_race_info,
    ridge_to_prior_fit,
    robust_linear_fit,
    time_to_hours,
)


def load_or_build_trap_features(trap_dir: Path, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)

    race_info = parse_trap_race_info(trap_dir / "Race_info.csv")
    result_summary = aggregate_trap_results(
        [trap_dir / "Results1.csv", trap_dir / "Results2.csv"]
    )
    data = race_info.merge(result_summary, left_on="key", right_on="key_race", how="inner")
    data["km_effort"] = data["distance_km"] + data["elevation_gain_m"] / 100
    data["vertical_share"] = (data["elevation_gain_m"] / 100) / data["km_effort"]
    data["elite_equivalent_speed"] = data["km_effort"] / data["top3_median_hours"]
    data = data[
        data["distance_km"].between(5, 500)
        & data["elevation_gain_m"].between(0, 30_000)
        & data["finishers"].ge(5)
        & data["elite_equivalent_speed"].between(1.0, 30.0)
    ].copy()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(cache_path, index=False)
    return data


def fit_trap_prior(trap: pd.DataFrame) -> tuple[np.ndarray, dict]:
    x = np.column_stack(
        [
            np.ones(len(trap)),
            np.log(trap["km_effort"].to_numpy()),
            trap["vertical_share"].to_numpy(),
        ]
    )
    y = np.log(trap["top3_median_hours"].to_numpy())
    coefficients = robust_linear_fit(x, y)
    residual = y - x @ coefficients
    diagnostics = {
        "race_count": int(len(trap)),
        "result_count": int(trap["finishers"].sum()),
        "median_absolute_log_residual": float(np.median(np.abs(residual))),
        "coefficients": coefficients.tolist(),
    }
    return coefficients, diagnostics


def load_modern_labels(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(path)
    labels["hours"] = labels["time"].map(time_to_hours)
    labels["race_constant"] = labels["hours"] * labels["race_score"]
    labels["km_effort"] = labels["distance_km"] + labels["elevation_gain_m"] / 100
    labels["vertical_share"] = (labels["elevation_gain_m"] / 100) / labels["km_effort"]

    events = (
        labels.groupby("event_id", as_index=False)
        .agg(
            event=("event", "first"),
            year=("year", "first"),
            distance_km=("distance_km", "first"),
            elevation_gain_m=("elevation_gain_m", "first"),
            km_effort=("km_effort", "first"),
            vertical_share=("vertical_share", "first"),
            race_constant=("race_constant", "median"),
            constant_min=("race_constant", "min"),
            constant_max=("race_constant", "max"),
            label_count=("race_score", "size"),
        )
    )
    events["constant_relative_spread"] = (
        events["constant_max"] - events["constant_min"]
    ) / events["race_constant"]
    return labels, events


def event_matrix(events: pd.DataFrame) -> np.ndarray:
    return np.vstack(
        [course_features(row.distance_km, row.elevation_gain_m) for row in events.itertuples()]
    )


def fit_course_model(
    events: pd.DataFrame,
    trap_prior: np.ndarray,
    penalty: float,
) -> np.ndarray:
    prior = np.array([0.0, trap_prior[1], trap_prior[2]], dtype=float)
    return ridge_to_prior_fit(
        event_matrix(events),
        np.log(events["race_constant"].to_numpy()),
        prior,
        penalty,
    )


def leave_one_event_out(
    labels: pd.DataFrame,
    events: pd.DataFrame,
    trap_prior: np.ndarray,
    penalty: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for event_id in events["event_id"]:
        train_events = events[events["event_id"] != event_id]
        coefficients = fit_course_model(train_events, trap_prior, penalty)
        held_out = labels[labels["event_id"] == event_id].copy()
        features = course_features(
            held_out["distance_km"].iloc[0], held_out["elevation_gain_m"].iloc[0]
        )
        held_out["predicted_race_constant"] = math.exp(features @ coefficients)
        held_out["predicted_score"] = np.clip(
            held_out["predicted_race_constant"] / held_out["hours"], 0, 1000
        )
        held_out["error"] = held_out["predicted_score"] - held_out["race_score"]
        rows.append(held_out)
    return pd.concat(rows, ignore_index=True)


def tune_penalty(
    labels: pd.DataFrame,
    events: pd.DataFrame,
    trap_prior: np.ndarray,
) -> tuple[float, pd.DataFrame, list[dict]]:
    candidates = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    trials = []
    best: tuple[float, float, pd.DataFrame] | None = None
    for penalty in candidates:
        predictions = leave_one_event_out(labels, events, trap_prior, penalty)
        score_metrics = metrics(predictions["race_score"], predictions["predicted_score"])
        trials.append({"penalty": penalty, **score_metrics})
        candidate = (score_metrics["mae"], penalty, predictions)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2], trials


def evaluate_within_race(labels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Evaluate one-anchor inverse-time interpolation by event, without leakage."""
    predictions = []
    for _, event_rows in labels.groupby("event_id"):
        if len(event_rows) < 2:
            continue
        anchor = event_rows.sort_values("race_score", ascending=False).iloc[0]
        held_out = event_rows.drop(anchor.name).copy()
        held_out["predicted_score"] = anchor["race_score"] * anchor["hours"] / held_out["hours"]
        held_out["anchor_runner"] = anchor["runner"]
        predictions.append(held_out)
    combined = pd.concat(predictions, ignore_index=True)
    return combined, metrics(combined["race_score"], combined["predicted_score"])


def evaluate_community_log_baseline(labels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Benchmark the public UTMB-log approach from trail-race-planner.

    Its published 1000-to-400 time ratio is 2.4945403784, which implies this B
    in score = A - B*ln(time).  We reimplement the equation independently.
    """
    b = 600.0 / math.log(2.4945403784)
    predictions = []
    for _, event_rows in labels.groupby("event_id"):
        if len(event_rows) < 2:
            continue
        anchor = event_rows.sort_values("race_score", ascending=False).iloc[0]
        held_out = event_rows.drop(anchor.name).copy()
        held_out["predicted_score"] = anchor["race_score"] - b * np.log(
            held_out["hours"] / anchor["hours"]
        )
        predictions.append(held_out)
    combined = pd.concat(predictions, ignore_index=True)
    result = metrics(combined["race_score"], combined["predicted_score"])
    result["B"] = b
    return combined, result


def build_report(
    artifact: dict,
    events: pd.DataFrame,
    course_predictions: pd.DataFrame,
    within_metrics: dict,
    community_metrics: dict,
) -> str:
    course_metrics = artifact["course_model"]["loo_metrics"]
    event_lines = []
    for row in events.sort_values(["year", "event_id"]).itertuples():
        event_lines.append(
            f"| {row.year} | {row.event} | {row.label_count} | "
            f"{row.race_constant:.1f} | {row.constant_relative_spread * 100:.2f}% |"
        )
    return f"""# ITRA Race Score 离线估算原型 v1

生成时间：{artifact['trained_at']}  
定位：**非官方估算器**；用于验证公开数据能否支持 Race Score 近似计算。

## 数据审计

- TRAP：{artifact['trap_prior']['race_count']:,} 场有效比赛、{artifact['trap_prior']['result_count']:,} 条完赛记录。公开 CSV **不含 Race Score 标签**，因此只用于学习赛程长度/爬升与精英完赛时间的尺度先验。
- 现代标签：{artifact['label_data']['label_count']} 条公开的“时间 + Race Score”，覆盖 {artifact['label_data']['event_count']} 场比赛。
- 现代标签明显偏向精英（最低 {artifact['label_data']['min_score']:.0f} 分），普通跑者区间仍缺少直接校准数据。

## 最重要的经验结果

同一场比赛中，`Race Score × 完赛小时数` 几乎是常数。各场标签的一致性如下：

| 年份 | 比赛 | 标签数 | 比赛常数 | 常数相对极差 |
|---:|---|---:|---:|---:|
{chr(10).join(event_lines)}

因此有同场一个官方锚点时，推荐：

`估算分 = 锚点分 × 锚点时间 ÷ 目标时间`

## 验证结果

### 同场有一个锚点

- 反比例时间模型：MAE {within_metrics['mae']:.2f} 分，RMSE {within_metrics['rmse']:.2f} 分，最大绝对误差 {within_metrics['max_absolute_error']:.2f} 分。
- GitHub `yama-asobi-lab/trail-race-planner` 的 UTMB 对数曲线复现基线：MAE {community_metrics['mae']:.2f} 分，RMSE {community_metrics['rmse']:.2f} 分。
- 结论：当前官方公开样本更支持反比例关系；社区项目的对数拟合可作为参考，但不应替代逐场验证。

### 完全没有锚点，仅有距离和爬升

- 按比赛分组的留一法：MAE {course_metrics['mae']:.2f} 分，RMSE {course_metrics['rmse']:.2f} 分，最大绝对误差 {course_metrics['max_absolute_error']:.2f} 分。
- 80% 经验绝对误差：±{artifact['course_model']['loo_error_80']:.1f} 分；95%：±{artifact['course_model']['loo_error_95']:.1f} 分。
- 这个模式没有 GPX 海拔、技术难度、天气和 ITRA 的参赛者历史修正，必须显示为低置信度。

## 当前结论

1. **有同场 Race Score 锚点时已经可用**，适合把一个公开/用户自有分数扩展到整场选手。
2. **无锚点时只是粗估**；距离和 D+ 无法恢复 ITRA 的逐场环境修正系数。
3. 下一轮最有价值的数据不是更多精英冠军，而是 2021 年以后、覆盖 400–750 分的同场成组 Race Score。

## 运行

```powershell
.\\.venv\\Scripts\\python.exe scripts\\train_offline_model.py --trap-dir ..\\TRAP-data\\ITRA
.\\.venv\\Scripts\\python.exe scripts\\predict_race_score.py --distance 61 --elevation 3400 --time 07:30:00
.\\.venv\\Scripts\\python.exe scripts\\predict_race_score.py --distance 61 --elevation 3400 --time 07:30:00 --anchor-time 05:35:13 --anchor-score 853
```

## 数据和方法限制

- ITRA 未公开完整公式，本模型不声称重现官方算法。
- TRAP 仓库没有明确许可证；当前仅在本地用于研究验证，不随模型分发原始 CSV。
- 社区参考项目未声明许可证，本项目只独立复现其公开描述的数学基线，不复制其代码。

## 主要来源

- ITRA Race Score 说明：https://itra.run/FAQ/ItraScore
- ITRA 2021 算法更新说明：https://itra.run/content/news/EN-TRAIL_RUNNING_REPORT_2021.pdf
- TRAP 数据：https://github.com/ricfog/TRAP-data
- ScrapITRA：https://github.com/ricfog/ScrapITRA
- 社区对数曲线参考：https://github.com/yama-asobi-lab/trail-race-planner
- 逐条现代标签的来源保存在 `data/modern_race_score_labels.csv`。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trap-dir",
        type=Path,
        default=PROJECT_ROOT.parent / "TRAP-data" / "ITRA",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "data" / "modern_race_score_labels.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "model_artifacts",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trap = load_or_build_trap_features(
        args.trap_dir, args.output_dir / "trap_race_features.csv"
    )
    trap_prior, trap_diagnostics = fit_trap_prior(trap)
    labels, events = load_modern_labels(args.labels)

    penalty, course_predictions, tuning = tune_penalty(labels, events, trap_prior)
    course_coefficients = fit_course_model(events, trap_prior, penalty)
    course_metrics = metrics(
        course_predictions["race_score"], course_predictions["predicted_score"]
    )
    course_abs_error = np.abs(course_predictions["error"].to_numpy())

    within_predictions, within_metrics = evaluate_within_race(labels)
    within_abs_error = np.abs(
        within_predictions["predicted_score"].to_numpy()
        - within_predictions["race_score"].to_numpy()
    )
    community_predictions, community_metrics = evaluate_community_log_baseline(labels)

    artifact = {
        "model_version": "offline-v1",
        "trained_at": datetime.now(UTC).isoformat(),
        "disclaimer": "Estimated Race Score; not an official ITRA calculation.",
        "feature_definition": {
            "km_effort": "distance_km + elevation_gain_m / 100",
            "vertical_share": "(elevation_gain_m / 100) / km_effort",
        },
        "trap_prior": trap_diagnostics,
        "label_data": {
            "label_count": int(len(labels)),
            "event_count": int(len(events)),
            "min_score": float(labels["race_score"].min()),
            "max_score": float(labels["race_score"].max()),
        },
        "within_race_model": {
            "formula": "score = anchor_score * anchor_time_hours / finish_time_hours",
            "metrics": within_metrics,
            "anchor_error_80": float(np.quantile(within_abs_error, 0.80)),
            "anchor_error_95": float(np.quantile(within_abs_error, 0.95)),
            "community_log_baseline": community_metrics,
        },
        "course_model": {
            "formula": "race_constant = exp(b0 + b1*log(km_effort) + b2*vertical_share)",
            "coefficients": course_coefficients.tolist(),
            "ridge_penalty": penalty,
            "loo_metrics": course_metrics,
            "loo_error_80": float(np.quantile(course_abs_error, 0.80)),
            "loo_error_95": float(np.quantile(course_abs_error, 0.95)),
            "penalty_trials": tuning,
        },
    }

    model_path = args.output_dir / "race_score_model_v1.json"
    model_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    events.to_csv(args.output_dir / "modern_event_constants.csv", index=False)
    course_predictions.to_csv(args.output_dir / "course_loo_predictions.csv", index=False)
    within_predictions.to_csv(args.output_dir / "anchor_predictions.csv", index=False)
    community_predictions.to_csv(
        args.output_dir / "community_log_baseline_predictions.csv", index=False
    )
    report = build_report(
        artifact, events, course_predictions, within_metrics, community_metrics
    )
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(artifact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
