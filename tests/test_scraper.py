import unittest
from unittest.mock import Mock, patch

from scraper import (
    PerformanceIndexRateLimited,
    extract_itra_course_info,
    scrape_itra_results,
)


class ScrapeItraResultsTests(unittest.TestCase):
    def test_extracts_public_course_distance_and_gain(self):
        page = """
        <section><h3>Course details</h3>
        <div>Distance: 40.01</div><div>Elevation Gain: +1687</div></section>
        """
        self.assertEqual(
            extract_itra_course_info(page),
            {"distance_km": 40.01, "elevation_gain_m": 1687.0},
        )

    def test_results_can_include_course_info_from_linked_details_page(self):
        results_page = """
        <a href="/Races/RaceDetails/test-race/2026/123">RACE DETAILS</a>
        <table id="RunnerRaceResults"><tr>
          <td>1</td><td><a href="/Runner/1">Runner One</a></td>
          <td>05:00:00</td><td>35-39</td><td>M</td><td>CHN</td>
        </tr></table>
        """
        details_page = """
        <div>Distance: 61.5 km</div><div>Elevation Gain: +3,400 m</div>
        """
        results_response = Mock(text=results_page, status_code=200)
        results_response.raise_for_status.return_value = None
        details_response = Mock(text=details_page, status_code=200)
        details_response.raise_for_status.return_value = None

        with patch(
            "scraper.requests.get",
            side_effect=[results_response, details_response],
        ):
            results, course = scrape_itra_results(
                "https://itra.run/Races/RaceResults/test-race/2026/123",
                return_course_info=True,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(course["distance_km"], 61.5)
        self.assertEqual(course["elevation_gain_m"], 3400.0)

    def test_returns_every_runner_in_the_results_table(self):
        rows = []
        for position in range(1, 7):
            race_score_cell = (
                '<td rowspan="6"><span style="display:none">6</span></td>'
                if position == 1
                else ''
            )
            rows.append(
                f"""
                <tr>
                    <td>{position}</td>
                    <td><a href="/RunnerSpace/Runner{position}/{position}">
                        Runner {position}
                    </a></td>
                    <td>0{position}:00:00</td>
                    {race_score_cell}
                    <td>{20 + position}</td>
                    <td>{'M' if position % 2 else 'F'}</td>
                    <td>{'<img src="/flags/test.svg"> TST' if position < 6 else ''}</td>
                </tr>
                """
            )

        page = """
        <table id="RunnerRaceResults">
            <thead><tr><th>Position</th><th>Runner</th></tr></thead>
            {rows}
        </table>
        """.format(rows="".join(rows))

        response = Mock(text=page, status_code=200)
        response.raise_for_status.return_value = None

        with patch("scraper.requests.get", return_value=response), patch(
            "scraper.get_performance_index", return_value="N/A"
        ) as performance_index_mock:
            results = scrape_itra_results("https://itra.run/Races/example")

        self.assertEqual(len(results), 6)
        self.assertEqual(
            [result["position"] for result in results],
            ["1", "2", "3", "4", "5", "6"],
        )
        self.assertEqual(results[0]["age"], "21")
        self.assertEqual(results[1]["age"], "22")
        self.assertEqual(results[-1]["name"], "Runner 6")
        self.assertEqual(results[-2]["nationality"], "TST")
        self.assertEqual(results[-1]["nationality"], "N/A")
        performance_index_mock.assert_not_called()

        with patch("scraper.requests.get", return_value=response), patch(
            "scraper.get_performance_index", return_value="500"
        ) as limited_index_mock:
            indexed_results = scrape_itra_results(
                "https://itra.run/Races/example",
                include_performance_index=True,
                performance_index_limit=2,
            )

        self.assertEqual(limited_index_mock.call_count, 2)
        self.assertEqual(indexed_results[0]["performance_index"], "500")
        self.assertEqual(indexed_results[1]["performance_index"], "500")
        self.assertTrue(
            all(result["performance_index"] == "N/A" for result in indexed_results[2:])
        )

        with patch("scraper.requests.get", return_value=response), patch(
            "scraper.get_performance_index",
            side_effect=PerformanceIndexRateLimited("blocked"),
        ) as blocked_index_mock:
            blocked_results = scrape_itra_results(
                "https://itra.run/Races/example",
                include_performance_index=True,
                performance_index_limit=6,
            )

        blocked_index_mock.assert_called_once()
        self.assertTrue(
            all(result["performance_index"] == "N/A" for result in blocked_results)
        )


if __name__ == "__main__":
    unittest.main()
