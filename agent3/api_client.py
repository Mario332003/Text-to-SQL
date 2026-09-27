import requests
from requests.adapters import HTTPAdapter, Retry

from . import config


class ApiClientError(Exception):
    pass


class ApiClient:
    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def _get(self, url: str, params: dict) -> dict:
        try:
            response = self.session.get(url, params=params, timeout=config.API_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            raise ApiClientError(f"agent3 API call failed: {error}") from error

    def geocode(self, place_name: str) -> dict | None:
        """Resolve a place name to lat/long. Returns None if no match."""
        if not place_name or not place_name.strip():                # <-- ADD THIS
            raise ApiClientError("geocode() called with an empty place name")  # <-- ADD THIS
        data = self._get(config.GEOCODING_URL, {"name": place_name, "count": 1})
        results = data.get("results") or []
        return results[0] if results else None

    def forecast(self, latitude: float, longitude: float) -> dict:
        return self._get(config.FORECAST_URL, {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,wind_speed_10m,relative_humidity_2m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "timezone": "auto",
        })