from typing import TypedDict, Any, Optional, Final, cast
from requests import Response, get
import asyncio
import dotenv
from os import environ

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


async def request_metar_info() -> Response:
    """
    Asynchronously makes a request to the Metar-Taf API using the provided function and URL.
    Args:
        fn (Callable): The function to use for making the request (e.g., requests.get).
        url (str): The endpoint URL to which the request will be made.
        **kwargs: Additional keyword arguments to pass to the request function.
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
    r = await asyncio.to_thread(get, f"https://api.metar-taf.com/metar?api_key={METAR_KEY}&v=2.3&locale=en-US&id=ESMX", **kwargs)
    await asyncio.sleep(0.10)
    return r

async def fetch_qnh() -> Optional[int]:
    """
    Fetches the QNH (altimeter setting) from the METAR data in hPa

    Returns:
        Optional[int]: The QNH value in hPa if available, otherwise None.
    """
    try:
        response = await request_metar_info()
        if response.status_code != 200:
            print(f"Error fetching METAR data: {response.status_code}")
            return None
        
        data: MetarResponse = cast(MetarResponse, response.json())
        qnh = data.get("metar", {}).get("qnh", None)
        return qnh

    except Exception as e:
        print(f"Exception occurred while fetching METAR data: {e}")
        return None
