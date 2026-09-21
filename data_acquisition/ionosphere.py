import json
import logging
import time
import zipfile

import pandas as pd
from dotenv import load_dotenv

from config.settings import settings

from . import helper

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fetch_regular_weather(lat: float = settings.site.lat, lon: float = settings.site.lon):
    """Current OpenWeather conditions for the site (temp, pressure, wind, etc.)."""
    api_key = settings.secrets.openweather_api_key
    if not api_key:
        logger.warning("OPENWEATHER_API_KEY not set; skipping regular weather.")
        return None
    url = settings.external_apis.openweather_url_template.format(lat=lat, lon=lon, api_key=api_key)
    return helper.safe_get_json(url)


def is_thunderstorm(weather: dict | None):
    """True if the current conditions include a thunderstorm (equipment-safety check)."""
    if not weather:
        return False
    return any(w.get("main") == "Thunderstorm" for w in weather.get("weather", []))


def fetch_vtec_local(lat: float = settings.site.lat, lon: float = settings.site.lon):
    """DLR global VTEC nowcast, reduced to points near the site + a global summary."""
    raw = helper.safe_get_json(settings.external_apis.dlr_vtec_global_url)
    if raw is None:
        return None

    try:
        features = raw["data"]["grid"]["features"]
    except KeyError, TypeError:
        logger.warning("Unexpected DLR VTEC payload shape; skipping.")
        return None

    rows = []
    for feature in features:
        lon_pt, lat_pt = feature["geometry"]["coordinates"]
        props = feature["properties"]
        rows.append(
            {
                "longitude": lon_pt,
                "latitude": lat_pt,
                "vtec_assimilated_tecu": props.get("vtec_assimilated_tecu"),
                "vtec_model_tecu": props.get("vtec_model_tecu"),
                "vtec_rms_tecu": props.get("vtec_rms_tecu"),
            }
        )
    grid = pd.DataFrame(rows)
    if grid.empty:
        return None

    local = helper.filter_local(
        grid, lat, lon, settings.data_acquisition.local_lat_pad_deg, settings.data_acquisition.local_lon_pad_deg
    )
    stats_cols = ["vtec_assimilated_tecu", "vtec_model_tecu", "vtec_rms_tecu"]

    return {
        "local_grid_points": helper.dataframe_to_records(local),
        "global_summary": grid[stats_cols].describe().to_dict(),
        "global_grid_point_count": len(grid),
    }


def fetch_esa_ism_local(url: str, lat: float = settings.site.lat, lon: float = settings.site.lon):
    """Fetch one ESA SWE ISM nowcast grid (S4 / sigma-phi / TEC) near the site."""
    xml_text = helper.safe_get_text(url)
    if xml_text is None:
        return None, None

    try:
        df, timestamp = helper.parse_esa_ism_xml(xml_text)
    except Exception as exc:
        logger.warning("Failed to parse ESA ISM XML from %s: %s", url, exc)
        return None, None

    local = helper.filter_local(
        df, lat, lon, settings.data_acquisition.local_lat_pad_deg, settings.data_acquisition.local_lon_pad_deg
    )
    return helper.dataframe_to_records(local), timestamp


# Geomagnetic conditions


def fetch_gfz_kp():
    """GFZ Kp nowcast: list of [unix_ms_timestamp, kp]."""
    return helper.safe_get_json(settings.external_apis.gfz_kp_url)


def fetch_gfz_hp30():
    """GFZ Hp30 nowcast: list of [unix_ms_timestamp, hp30]."""
    return helper.safe_get_json(settings.external_apis.gfz_hp30_url)


def fetch_kyoto_dst():
    """Kyoto/NOAA Dst index time series."""
    return helper.safe_get_json(settings.external_apis.noaa_kyoto_dst_url)


# Solar wind, interplanetary magnetic field, and solar activity


def fetch_solar_wind_speed():
    """NOAA solar wind speed summary, km/s."""
    return helper.safe_get_json(settings.external_apis.noaa_solar_wind_speed_url)


def fetch_solar_wind_mag_field():
    """NOAA solar wind magnetic field summary, nT (bt, bz_gsm)."""
    return helper.safe_get_json(settings.external_apis.noaa_solar_wind_mag_field_url)


def fetch_10cm_flux():
    """NOAA 10 cm solar flux, sfu."""
    return helper.safe_get_json(settings.external_apis.noaa_10cm_flux_url)


def fetch_noaa_scales():
    """NOAA space-weather scales (R/S/G), current + forecast."""
    return helper.safe_get_json(settings.external_apis.noaa_scales_url)


# Snapshot assembly + storage


def build_observation_snapshot(lat: float = settings.site.lat, lon: float = settings.site.lon):
    """Fetch every data source and assemble one observation snapshot dict."""
    started_at = time.time()

    weather = fetch_regular_weather(lat, lon)

    esa_s4, esa_ts = fetch_esa_ism_local(settings.external_apis.esa_s4_url, lat, lon)
    esa_sigma_phi, esa_ts_sigma_phi = fetch_esa_ism_local(settings.external_apis.esa_sigma_phi_url, lat, lon)
    esa_tec, esa_ts_tec = fetch_esa_ism_local(settings.external_apis.esa_tec_url, lat, lon)

    esa_timestamp = esa_ts or esa_ts_sigma_phi or esa_ts_tec

    snapshot = {
        "metadata": {
            "created_at_unix": int(started_at),
            "created_at_utc": helper.serialize_timestamp(pd.Timestamp.now(tz="UTC")),
            "collection_duration_s": None,  # filled in below
            "location": {"name": settings.site.name, "latitude": lat, "longitude": lon},
        },
        "regular_weather": weather,
        "ionospheric_conditions": {
            "dlr_vtec": fetch_vtec_local(lat, lon),
            "esa_s4_local": esa_s4,
            "esa_sigma_phi_local": esa_sigma_phi,
            "esa_tec_local": esa_tec,
            "esa_data_timestamp": helper.serialize_timestamp(esa_timestamp),
        },
        "geomagnetic_conditions": {
            "gfz_kp": fetch_gfz_kp(),
            "gfz_hp30": fetch_gfz_hp30(),
            "kyoto_dst": fetch_kyoto_dst(),
        },
        "solar_wind_and_magnetic_field": {
            "solar_wind_speed": fetch_solar_wind_speed(),
            "solar_wind_magnetic_field": fetch_solar_wind_mag_field(),
        },
        "solar_activity": {
            "noaa_10cm_flux": fetch_10cm_flux(),
            "noaa_space_weather_scales": fetch_noaa_scales(),
        },
    }

    snapshot["metadata"]["collection_duration_s"] = round(time.time() - started_at, 3)
    return snapshot


def save_snapshot(snapshot, output_dir):
    """Write a snapshot to <output_dir>/<unix_timestamp>.json.zip and return the path."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = snapshot["metadata"]["created_at_unix"]
    json_filename = f"{timestamp}.json"
    zip_path = output_dir / f"{json_filename}.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            json_filename,
            json.dumps(
                snapshot,
                indent=2,
                ensure_ascii=False,
                default=helper.serialize_timestamp,
            ),
        )

    logger.info("Saved observation snapshot to %s", zip_path)
    return zip_path


def collect_observation(output_dir, lat: float = settings.site.lat, lon: float = settings.site.lon):
    """Collect and store one full observation snapshot.

    Call this once at the start of an observation session. Skips saving (and
    returns None) if the current conditions include a thunderstorm, as a
    safety check for outdoor/antenna equipment - mirrors the original
    notebook's behavior, but without hard-exiting the process.
    """
    snapshot = build_observation_snapshot(lat, lon)

    if is_thunderstorm(snapshot["regular_weather"]):
        logger.critical("Thunderstorm detected at site. Shut down the device as soon as possible")

    return save_snapshot(snapshot, output_dir)


if __name__ == "__main__":
    if settings.debug:
        collect_observation(settings.output_dir)
