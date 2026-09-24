import unittest

import numpy as np

from race_planner import (
    build_nutrition_plan,
    build_race_plan,
    fed_distance_km,
    grade_cost,
    parse_gpx,
    riegel_time_seconds,
    summarize_course,
)


SAMPLE_GPX = b"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <trk><trkseg>
    <trkpt lat="30.0000" lon="120.0000"><ele>100</ele></trkpt>
    <trkpt lat="30.0010" lon="120.0000"><ele>110</ele></trkpt>
    <trkpt lat="30.0020" lon="120.0000"><ele>125</ele></trkpt>
    <trkpt lat="30.0030" lon="120.0000"><ele>115</ele></trkpt>
  </trkseg></trk>
</gpx>"""


class RacePlannerTests(unittest.TestCase):
    def test_parse_gpx_and_summary(self):
        course = parse_gpx(SAMPLE_GPX, resample_m=25)
        summary = summarize_course(course)
        self.assertGreater(len(course), 10)
        self.assertAlmostEqual(summary.distance_km, 0.334, places=2)
        self.assertGreater(summary.elevation_gain_m, 10)
        self.assertTrue(np.isfinite(course["grade"]).all())

    def test_grade_curve_and_fed(self):
        self.assertAlmostEqual(grade_cost(0.0), 1.0)
        self.assertAlmostEqual(grade_cost(0.20), 2.6)
        self.assertAlmostEqual(grade_cost(-0.05), 0.85)
        self.assertGreater(grade_cost(0.30), grade_cost(0.20))
        self.assertEqual(fed_distance_km(50, 2500), 75)

    def test_riegel_is_slower_for_harder_course(self):
        flat = riegel_time_seconds(10, 3600, 20, 0)
        hilly = riegel_time_seconds(10, 3600, 20, 1000)
        self.assertGreater(flat, 7200)
        self.assertGreater(hilly, flat)

    def test_plan_respects_elapsed_target_and_stops(self):
        course = parse_gpx(SAMPLE_GPX, resample_m=20)
        total_km = float(course["distance_km"].iloc[-1])
        plan, metadata = build_race_plan(
            course,
            [{"name": "Aid", "distance_km": total_km / 2, "stop_min": 5}],
            target_elapsed_seconds=3600,
            fatigue_decay_pct=10,
        )
        self.assertEqual(len(plan), 2)
        self.assertAlmostEqual(plan["moving_seconds"].sum(), 3300, places=5)
        self.assertAlmostEqual(plan.iloc[-1]["departure_seconds"], 3600, places=5)
        self.assertAlmostEqual(metadata["stop_seconds"], 300)

    def test_nutrition_plan_has_executable_quantities(self):
        course = parse_gpx(SAMPLE_GPX, resample_m=20)
        plan, _ = build_race_plan(course, [], target_elapsed_seconds=7200)
        nutrition = build_nutrition_plan(
            plan,
            carbs_per_hour=60,
            fluid_ml_per_hour=500,
        )
        self.assertEqual(len(nutrition), 1)
        self.assertAlmostEqual(nutrition.iloc[0]["target_carbs_g"], 120, places=3)
        self.assertEqual(nutrition.iloc[0]["flasks_to_carry"], 2)
        self.assertLess(
            abs(
                nutrition.iloc[0]["planned_carbs_g"]
                - nutrition.iloc[0]["target_carbs_g"]
            ),
            15,
        )


if __name__ == "__main__":
    unittest.main()
