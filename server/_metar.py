from typing import TypedDict, Any, Optional, Final, cast
from requests import Response, get
import asyncio
import dotenv
from os import environ
from threading import Lock, Thread
from time import time, sleep

dotenv.load_dotenv()
METAR_KEY: Final[Optional[str]] = environ.get("METAR-TAF", None)


class Airport(TypedDict, total=False):
    id: str
    iata: str
    name: str
    name_translated: str
    city_name: str
    admin1: str
    admin2: str
    country_id: str
    country_name: str
    lat: float
    lng: float
    metar: bool
    taf: bool
    timezone: int
    fir: str
    elevation: int
    type: int
    last_notam: int


class MetarCloud(TypedDict, total=False):
    id: int
    height: int
    report: str
    amount: str


class RunwayVisibility(TypedDict, total=False):
    id: str
    value: int
    prefix: Optional[str]
    min: Optional[int]
    max: Optional[int]
    trend: str


class Metar(TypedDict, total=False):
    cavok: bool
    ceiling: int
    ceiling_color: str
    clouds: list[MetarCloud]
    code: str
    code_colour: str
    colour_state: Optional[str]
    dewpoint: int
    dewpoint_exact: Optional[float]
    humidity: int
    is_day: bool
    observed: int
    qnh: int
    raw: str
    recent_weather_report: Optional[str]
    remarks: Optional[str]
    runway_condition: Optional[list[Any]]
    runway_visibility: list[RunwayVisibility]
    snoclo: bool
    station_id: str
    sunrise: int
    sunset: int
    temperature: int
    temperature_exact: Optional[float]
    trends: list[Any]
    vertical_visibility: Optional[int]
    visibility: int
    visibility_sign: Optional[str]
    visibility_color: str
    visibility_min: Optional[int]
    warnings: list[str]
    weather: Optional[str]
    weather_image: str
    weather_report: Optional[str]
    wind_color: str
    wind_dir: int
    wind_dir_max: Optional[int]
    wind_dir_min: Optional[int]
    wind_gust: Optional[int]
    wind_speed: int
    ws_all: str
    ws_runways: Optional[str]
    id: int


class Runway(TypedDict, total=False):
    id_l: str
    id_h: str
    hdg_l: int
    hdg_h: int
    in_use: int
    xwnd: int
    hwnd: float


class Station(TypedDict, total=False):
    id: str
    name: str
    taf: bool


class MetarResponse(TypedDict, total=False):
    status: bool
    credits: int
    airport: Airport
    metar: Metar
    runways: list[Runway]
    stations: list[Station]


class UnregisteredAirport(Exception):
    pass


class MetarCacheLayer:

    __slots__ = ("_QNH_CACHE", "_CACHE_LOCK", "update_thread", "_updated", "_interval")

    def __init__(self, airports: list[str], update_interval: int = 600) -> None:
        self._CACHE_LOCK = Lock()
        self._QNH_CACHE: dict[str, Optional[int]] = {port: None for port in airports}
        self._updated = int(time())
        self._interval = update_interval
        self.update_thread: Optional[Thread] = Thread(
            target=self.schedule_refresh, daemon=True
        )
        self.update_thread.start()

    async def request_metar_info(self, airport: str) -> Response:
        """
        Asynchronously makes a request to the Metar-Taf API using the provided function and URL.
        Args:
            airport (str): The airport code for which to fetch METAR data.
        Returns:
            Response: The response object returned by the request function.

        Raises:
            ValueError: If the METAR_KEY is not set in environment variables.
        """

        if METAR_KEY is None:
            raise ValueError("METAR-TAF API key is not set in environment variables.")

        kwargs: dict[str, Any] = {}
        kwargs["timeout"] = 30
        kwargs["headers"] = {
            "User-Agent": "skyDevisionImager/1.0.0",
            "Accept": "application/vnd.github.v3+json",
        }
        r = await asyncio.to_thread(
            get,
            f"https://api.metar-taf.com/metar?api_key={METAR_KEY}&v=2.3&locale=en-US&id={airport}",
            **kwargs,
        )
        await asyncio.sleep(0.10)
        return r

    async def fetch_qnh(self, airport: str) -> Optional[int]:
        """
        Fetches the QNH from the METAR data in hPa
        Args:
            airport (str): The airport code for which to fetch the QNH value.
        Returns:
            Optional[int]: The QNH value in hPa if available, otherwise None.
        """
        try:
            response = await self.request_metar_info(airport)
            if response.status_code != 200:
                print(f"Error fetching METAR data: {response.status_code}")
                return None

            data: MetarResponse = cast(MetarResponse, response.json())
            qnh = data.get("metar", {}).get("qnh", None)
            return qnh

        except Exception as e:
            print(f"Exception occurred while fetching METAR data: {e}")
            return None

    async def refresh_cache(self) -> None:
        """
        Refreshes the QNH cache by fetching the latest QNH value from the METAR data.
        If the fetch is successful, updates the cache with the new QNH value.

        Returns:
            None
        """
        awaitables = [self.fetch_qnh(airport) for airport in self._QNH_CACHE.keys()]
        responses = await asyncio.gather(*awaitables)

        for airport, response in zip(self._QNH_CACHE.keys(), responses):
            if response is None:
                continue

            self._QNH_CACHE[airport] = response

        self._updated = int(time())

    async def get_qnh(self, airport: str) -> Optional[int]:
        """
        Retrieves the QNH value for the specified airport from the cache.
        If the airport is not in the cache, it initializes it with a default value of 1013 hPa.

        Args:
            airport (str): The airport code for which to retrieve the QNH value.

        Returns:
            Optional[int]: The QNH value in hPa if available, otherwise None.

        Raises:
            UnregisteredAirport: If the airport is not registered in the cache.
        """

        if airport not in self._QNH_CACHE:
            raise UnregisteredAirport(
                f"Airport {airport} is not registered in the cache."
            )

        return self._QNH_CACHE.get(airport, None)

    def schedule_refresh(self) -> None:
        print(
            "Scheduling QNH cache refresh :: Starting background refresh every 10 minutes ..."
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while True:
                loop.run_until_complete(self.refresh_cache())
                print(
                    f"QNH cache refreshed at {self._updated}. Next refresh in {self._interval} seconds."
                )
                sleep(self._interval)
        except Exception as err:
            print(f"Error during scheduled refresh: {err}")
        finally:
            loop.close()

        print(" ")
