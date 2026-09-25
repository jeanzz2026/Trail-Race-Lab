import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from streamlit_ui import (
    add_filtered_ranking,
    age_distribution_figure,
    apply_result_filters,
    default_checkpoints,
    estimate_race_scores,
    format_duration,
    load_race_score_model,
    normalize_age_group,
    normalize_gender,
    pace_figure,
    parse_anchor_table,
    parse_browser_capture,
    parse_duration,
    prepare_results_frame,
    render_metric_grid,
)
from race_score_model import RaceScoreModel


class StreamlitUiTests(unittest.TestCase):
    def test_browser_capture_is_validated_and_normalized(self):
        results, course, capture_id = parse_browser_capture({
            "schema_version": 1,
            "capture_id": "capture-123",
            "source_url": "https://itra.run/Races/RaceResults/example/2026/123",
            "course_info": {"distance_km": "30.5", "elevation_gain_m": 1400},
            "results": [{
                "position": 1,
                "name": "Runner One",
                "profile_link": "https://itra.run/RunnerSpace/runner.one",
                "time": "03:12:34",
                "age": "35-39",
                "gender": "M",
                "nationality": "CHN",
            }],
        })

        self.assertEqual(capture_id, "capture-123")
        self.assertEqual(course, {"distance_km": 30.5, "elevation_gain_m": 1400.0})
        self.assertEqual(results[0]["position"], "1")
        self.assertEqual(results[0]["performance_index"], "N/A")

    def test_browser_capture_rejects_non_itra_source(self):
        with self.assertRaises(ValueError):
            parse_browser_capture({
                "schema_version": 1,
                "capture_id": "capture-123",
                "source_url": "https://example.com/Races/RaceResults/test",
                "results": [{"name": "Runner"}],
            })

    def test_plan_metric_grid_keeps_full_values_and_escapes_tooltips(self):
        with patch("streamlit_ui.st.markdown") as markdown:
            render_metric_grid([
                ("预计完赛", "12:00:00", None),
                ("预估 Race Score", "138.9", '区间 < 200 & "非官方"'),
            ])

        markup = markdown.call_args.args[0]
        self.assertIn('class="trl-metric-grid"', markup)
        self.assertIn("12:00:00", markup)
        self.assertIn("138.9", markup)
        self.assertIn("&lt; 200 &amp; &quot;非官方&quot;", markup)
        self.assertTrue(markdown.call_args.kwargs["unsafe_allow_html"])

    def test_race_score_model_is_not_stale_cached_across_hot_reload(self):
        first = load_race_score_model()
        second = load_race_score_model()
        self.assertIsNot(first, second)
        self.assertTrue(hasattr(second, "estimate_with_anchors"))

    def test_duration_supports_ultra_times(self):
        self.assertEqual(parse_duration("30:15:20"), 108920)
        self.assertEqual(format_duration(108920), "30:15:20")

    def test_invalid_duration_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_duration("12:75:00")

    def test_default_checkpoints_are_inside_course(self):
        checkpoints = default_checkpoints(82.0)
        self.assertGreater(len(checkpoints), 1)
        self.assertTrue((checkpoints["distance_km"] > 0).all())
        self.assertTrue((checkpoints["distance_km"] < 82.0).all())

    def test_pace_bar_chart_has_two_decimal_labels(self):
        plan = pd.DataFrame({
            "segment": ["Start → CP1", "CP1 → Finish"],
            "avg_pace_min_km": [7.126, 9.875],
        })
        figure = pace_figure(plan, "柱状图")
        self.assertIsNone(figure.layout.title.text)
        self.assertEqual(list(figure.data[0].text), ["7.13", "9.88"])

    def test_pace_line_chart_is_smoothed(self):
        plan = pd.DataFrame({
            "segment": ["Start → CP1", "CP1 → Finish"],
            "avg_pace_min_km": [7.126, 9.875],
        })
        figure = pace_figure(plan, "折线图")
        self.assertEqual(figure.data[0].line.shape, "spline")
        self.assertEqual(list(figure.data[0].text), ["7.13", "9.88"])

    def test_result_labels_preserve_itra_age_groups(self):
        self.assertEqual(normalize_age_group("23-34"), "23-34")
        self.assertIsNone(normalize_age_group("-"))
        self.assertIsNone(normalize_age_group("Other"))
        self.assertEqual(normalize_gender("M"), "男子")
        self.assertEqual(normalize_gender("F"), "女子")
        self.assertIsNone(normalize_gender("O"))
        self.assertIsNone(normalize_gender("-"))

    def test_result_frame_converts_seconds_to_hours(self):
        frame = pd.DataFrame([{
            "position": "3", "time": "10:30:00", "performance_index": "N/A",
            "gender": "F", "age": "35-39",
        }])
        prepared = prepare_results_frame(frame)
        self.assertEqual(prepared.iloc[0]["finish_hours"], 10.5)
        self.assertEqual(prepared.iloc[0]["position_numeric"], 3)
        self.assertEqual(prepared.iloc[0]["age_group"], "35-39")

    def test_empty_result_filters_show_all_rows(self):
        frame = pd.DataFrame([
            {"gender_label": "男子", "nationality": "CHN", "age_group": "35-39"},
            {"gender_label": None, "nationality": "N/A", "age_group": None},
        ])
        self.assertEqual(len(apply_result_filters(frame, [], [], [])), 2)
        self.assertEqual(len(apply_result_filters(frame, ["男子"], [], [])), 1)

    def test_age_distribution_stacks_male_and_female_with_distinct_colors(self):
        frame = pd.DataFrame([
            {"gender_label": "男子", "age_group": "35-39"},
            {"gender_label": "女子", "age_group": "35-39"},
            {"gender_label": None, "age_group": "35-39"},
            {"gender_label": "男子", "age_group": None},
        ])
        figure = age_distribution_figure(frame, ["35-39"])
        traces = {trace.name: trace for trace in figure.data}
        self.assertEqual(set(traces), {"男子", "女子"})
        self.assertEqual(traces["男子"].marker.color, "#2F6BFF")
        self.assertEqual(traces["女子"].marker.color, "#E0528D")
        self.assertEqual(figure.layout.barmode, "stack")

    def test_filtered_ranking_is_continuous_and_keeps_overall_position(self):
        frame = pd.DataFrame([
            {"position": "8", "position_numeric": 8, "time_seconds": 4200},
            {"position": "2", "position_numeric": 2, "time_seconds": 3600},
            {"position": "15", "position_numeric": 15, "time_seconds": 5000},
        ])
        ranked = add_filtered_ranking(frame)
        self.assertEqual(ranked["filtered_rank"].tolist(), [1, 2, 3])
        self.assertEqual(ranked["position"].tolist(), ["2", "8", "15"])

    def test_race_score_estimation_uses_same_race_anchor(self):
        model = RaceScoreModel.load(
            Path(__file__).resolve().parents[1]
            / "model_artifacts"
            / "race_score_model_v1.json"
        )
        frame = pd.DataFrame({"finish_hours": [10.0, 20.0, None]})
        result = estimate_race_scores(
            frame,
            model,
            distance_km=50.0,
            elevation_gain_m=2500.0,
            anchor_time="10:00:00",
            anchor_score=800.0,
        )
        self.assertAlmostEqual(result.iloc[0]["estimated_race_score"], 800.0)
        self.assertAlmostEqual(result.iloc[1]["estimated_race_score"], 400.0)
        self.assertTrue(pd.isna(result.iloc[2]["estimated_race_score"]))
        self.assertEqual(result.iloc[0]["score_confidence"], "high")
        self.assertLess(result.iloc[0]["score_lower_80"], 800.0)
        self.assertGreater(result.iloc[0]["score_upper_80"], 800.0)

    def test_multiple_anchor_table_and_estimation(self):
        anchors = parse_anchor_table(pd.DataFrame([
            {"finish_time": "10:00:00", "race_score": 800.0},
            {"finish_time": "20:00:00", "race_score": 400.0},
        ]))
        model = RaceScoreModel.load(
            Path(__file__).resolve().parents[1]
            / "model_artifacts"
            / "race_score_model_v1.json"
        )
        result = estimate_race_scores(
            pd.DataFrame({"finish_hours": [10.0, 20.0]}),
            model,
            distance_km=1.0,
            elevation_gain_m=0.0,
            anchors=anchors,
        )
        self.assertAlmostEqual(result.iloc[0]["estimated_race_score"], 800.0)
        self.assertAlmostEqual(result.iloc[1]["estimated_race_score"], 400.0)
        self.assertEqual(
            result.iloc[0]["score_method"], "multi_race_anchor_robust_median"
        )

if __name__ == "__main__":
    unittest.main()
