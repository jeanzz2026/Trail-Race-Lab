import requests
from bs4 import BeautifulSoup
from typing import Dict, List
from urllib.parse import urljoin
import logging
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceIndexRateLimited(Exception):
    """Raised when ITRA refuses further runner-profile requests."""


def _parse_number(value: str, *, integer_like: bool = False) -> float:
    cleaned = value.strip().replace(" ", "")
    if integer_like:
        cleaned = cleaned.replace(",", "")
    elif "," in cleaned and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")
    return float(cleaned)


def extract_itra_course_info(html: str) -> Dict[str, float]:
    """Extract public distance and elevation-gain fields from an ITRA page."""
    soup = BeautifulSoup(html or "", "html.parser")
    text = soup.get_text(" ", strip=True).replace("\xa0", " ")
    info: Dict[str, float] = {}
    distance_match = re.search(
        r"\bDistance\s*:\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:km)?\b",
        text,
        re.IGNORECASE,
    )
    gain_match = re.search(
        r"\bElevation\s+Gain\s*:\s*\+?\s*([0-9][0-9,.]*)\s*(?:m)?\b",
        text,
        re.IGNORECASE,
    )
    if distance_match:
        info["distance_km"] = _parse_number(distance_match.group(1))
    if gain_match:
        info["elevation_gain_m"] = _parse_number(
            gain_match.group(1), integer_like=True
        )
    return info


def _race_details_url(url: str, soup: BeautifulSoup) -> str | None:
    details_link = soup.find("a", href=re.compile(r"/Races/RaceDetails/", re.I))
    if details_link and details_link.get("href"):
        return urljoin(url, details_link["href"])
    if re.search(r"/Races/RaceResults/", url, re.I):
        return re.sub(r"/Races/RaceResults/", "/Races/RaceDetails/", url, flags=re.I)
    return None


def fetch_itra_course_info(
    url: str,
    results_html: str,
    headers: Dict,
) -> Dict[str, float]:
    """Read course fields from results HTML, then its public details page."""
    info = extract_itra_course_info(results_html)
    if {"distance_km", "elevation_gain_m"}.issubset(info):
        return info

    details_url = _race_details_url(url, BeautifulSoup(results_html, "html.parser"))
    if not details_url:
        return info
    try:
        response = requests.get(details_url, headers=headers, timeout=10)
        response.raise_for_status()
        info.update(extract_itra_course_info(response.text))
    except requests.RequestException as exc:
        logger.warning("Could not fetch ITRA race details: %s", exc)
    return info


def get_performance_index(profile_url: str, headers: Dict) -> str:
    """
    Extracts the ITRA Performance Index from a runner's profile page.
    Returns 'N/A' if the index is not available or if an error occurs.
    """
    try:
        logger.debug(f"Fetching performance index from profile URL: {profile_url}")
        
        # Ensure proper URL joining
        if not profile_url.startswith(('http://', 'https://')):
            profile_url = urljoin('https://itra.run', profile_url)
            logger.debug(f"Updated profile URL to absolute URL: {profile_url}")

        response = requests.get(profile_url, headers=headers, timeout=10)
        if response.status_code in (403, 405, 429):
            raise PerformanceIndexRateLimited(
                f"ITRA returned HTTP {response.status_code} for runner profiles"
            )
        response.raise_for_status()
        logger.debug(f"Profile page response status code: {response.status_code}")

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try multiple selectors to find the performance index
        index_span = soup.find('span', class_='level-count')
        if not index_span:
            logger.debug("Performance index not found with class 'level-count', trying alternative selectors")
            # Try alternative selectors
            index_span = soup.find('span', class_='itra-index') or \
                        soup.find('div', class_='performance-index') or \
                        soup.find('div', text=lambda t: t and 'Performance Index' in t)

        if index_span:
            index_value = index_span.text.strip()
            logger.debug(f"Found performance index value: {index_value}")
            return index_value
        else:
            # Check if the content might be loaded via JavaScript
            logger.debug("Performance index not found in static HTML, checking for JavaScript data")
            script_data = soup.find('script', type='application/json')
            if script_data:
                try:
                    json_data = json.loads(script_data.string)
                    if 'performanceIndex' in json_data:
                        return str(json_data['performanceIndex'])
                except json.JSONDecodeError:
                    logger.error("Failed to parse JavaScript data")
            
            logger.debug(f"No performance index found for profile: {profile_url}")
            return 'N/A'

    except PerformanceIndexRateLimited:
        raise
    except requests.Timeout:
        logger.error(f"Timeout while fetching profile: {profile_url}")
        return 'N/A'
    except requests.ConnectionError:
        logger.error(f"Connection error while fetching profile: {profile_url}")
        return 'N/A'
    except requests.RequestException as e:
        logger.error(f"Request error while fetching profile: {profile_url}, error: {str(e)}")
        return 'N/A'
    except Exception as e:
        logger.error(f"Unexpected error while fetching performance index: {str(e)}")
        return 'N/A'

def scrape_itra_results(
    url: str,
    include_performance_index: bool = False,
    performance_index_limit: int = 100,
    return_course_info: bool = False,
) -> List[Dict] | tuple[List[Dict], Dict[str, float]]:
    """
    Scrapes all runners from an ITRA race results table.

    Performance indexes require an additional profile-page request per runner,
    so they are opt-in and capped to keep large race downloads practical.
    Returns a list of dictionaries containing runner information.
    """
    try:
        # Validate URL
        if not url or not url.startswith(('http://', 'https://')):
            raise ValueError("Invalid URL format")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }

        # Make request with timeout and error handling
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.Timeout:
            raise Exception("Request timed out. Please try again.")
        except requests.ConnectionError:
            raise Exception("Failed to connect to the server. Please check your internet connection.")
        except requests.RequestException as e:
            raise Exception(f"Network error occurred: {str(e)}")

        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        performance_index_requests = 0
        performance_index_blocked = False

        # Find the results table using the specific ID
        runners_table = soup.select_one('#RunnerRaceResults')
        if not runners_table:
            logger.debug("HTML content: %s", soup.prettify())
            raise Exception("Could not find results table on the page")

        # Process every data row. Filtering on ``td`` is more robust than
        # slicing off the first row because the header may be wrapped in a
        # ``thead`` or otherwise differ between ITRA pages.
        runners = [
            row for row in runners_table.select('tr')
            if row.find_all('td')
        ]
        
        for runner in runners:
            try:
                # Get all td elements
                columns = runner.find_all('td')
                logger.debug("Processing runner row with %d columns", len(columns))
                logger.debug("Row HTML structure: %s", runner.prettify())

                result = {}

                # Extract position (index 0)
                try:
                    result['position'] = columns[0].get_text(strip=True)
                except (IndexError, AttributeError):
                    result['position'] = 'N/A'
                logger.debug("Extracted position: %s", result['position'])

                # Extract name and profile link (index 1)
                try:
                    name_cell = columns[1]
                    profile_link = name_cell.find('a')
                    if profile_link and profile_link.get('href'):
                        result['profile_link'] = urljoin('https://itra.run', profile_link['href'])
                        result['name'] = profile_link.get_text(strip=True).strip()
                        should_fetch_index = (
                            include_performance_index
                            and not performance_index_blocked
                            and performance_index_requests < max(0, performance_index_limit)
                        )
                        if should_fetch_index:
                            performance_index_requests += 1
                            try:
                                result['performance_index'] = get_performance_index(
                                    result['profile_link'], headers
                                )
                            except PerformanceIndexRateLimited as e:
                                performance_index_blocked = True
                                result['performance_index'] = 'N/A'
                                logger.warning(
                                    "%s; stopping remaining performance-index requests",
                                    e,
                                )
                        else:
                            result['performance_index'] = 'N/A'
                        logger.debug("Extracted performance index: %s", result['performance_index'])
                    else:
                        result['profile_link'] = 'N/A'
                        result['name'] = 'N/A'
                        result['performance_index'] = 'N/A'
                except (IndexError, AttributeError):
                    result['profile_link'] = 'N/A'
                    result['name'] = 'N/A'
                    result['performance_index'] = 'N/A'
                logger.debug("Extracted name: %s, profile_link: %s", result['name'], result['profile_link'])

                # Extract time (index 2)
                try:
                    result['time'] = columns[2].get_text(strip=True)
                except (IndexError, AttributeError):
                    result['time'] = 'N/A'
                logger.debug("Extracted time: %s", result['time'])

                # After extracting time, check for and skip race score column
                race_score_cell = next((col for col in columns if 'rowspan' in col.attrs), None)
                if race_score_cell:
                    # Adjust column indices for remaining fields when race score cell is present
                    age_index = 4  # Skip the race score cell
                    gender_index = 5
                    nationality_index = 6
                else:
                    # Normal column indices when no race score cell
                    age_index = 3
                    gender_index = 4
                    nationality_index = 5

                # Use these dynamic indices for remaining extractions
                result['age'] = columns[age_index].get_text(strip=True) if len(columns) > age_index else 'N/A'
                result['gender'] = columns[gender_index].get_text(strip=True) if len(columns) > gender_index else 'N/A'
                nationality = (
                    columns[nationality_index].get_text(strip=True)
                    if len(columns) > nationality_index
                    else ''
                )
                result['nationality'] = nationality.split()[-1] if nationality else 'N/A'

                logger.debug("Extracted age: %s", result['age'])
                logger.debug("Extracted gender: %s", result['gender'])
                logger.debug("Extracted nationality: %s", result['nationality'])

                results.append(result)
                logger.debug("Successfully added result for runner: %s", result['name'])

            except Exception as e:
                logger.error("Error processing runner row: %s", str(e))
                logger.debug("Row HTML structure: %s", runner.prettify())
                continue

        if not results:
            logger.debug("Table HTML structure: %s", runners_table.prettify())
            raise Exception("No valid results found on the page")

        logger.debug("Successfully extracted %d results", len(results))
        if return_course_info:
            return results, fetch_itra_course_info(url, response.text, headers)
        return results

    except Exception as e:
        logger.error("Scraping error: %s", str(e))
        raise Exception(f"Failed to scrape results: {str(e)}")
