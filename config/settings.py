from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoaderSettings(BaseSettings):
    """Settings used in config/loader.py"""

    min_frequency_hz: float = 135e6
    max_frequency_hz: float = 139e6
    valid_decimation_factors: list[int] = [1, 2, 4, 8, 16, 32, 64]
    poll_interval_s: float = 60.0


class HelperSettings(BaseSettings):
    """Settings used for data_acquisition/helper"""

    default_timeout_s: int = 15


class DataAcquisitionSettings(BaseSettings):
    helper: HelperSettings = HelperSettings()
    # ionosphere map paddings
    local_lat_pad_deg: float = 2.0
    local_lon_pad_deg: float = 3.0


class SiteSettings(BaseSettings):
    """Ground station location and local ionosphere-grid padding."""

    name: str = "Moscow"
    lat: float = 55.5269
    lon: float = 37.0888


class ExternalApiSettings(BaseSettings):
    """External APIs, mostly for ionosphere and space weather data."""

    openweather_url_template: str = (
        "https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    )

    dlr_vtec_global_url: str = (
        "https://data.impc.dlr.de/tec-nowcast/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_GLOBAL"
        "/latest/DLR_GNSS_GCG_L4_VTEC-NTCM-SCM_NC_GLOBAL_latest_D.json"
    )

    esa_s4_url: str = "https://swe.ssa.esa.int/assets/ism/ism_nowcast/s4_nowcast_grid.xml"
    esa_sigma_phi_url: str = "https://swe.ssa.esa.int/assets/ism/ism_nowcast/sigma_phi_nowcast_grid.xml"
    esa_tec_url: str = "https://swe.ssa.esa.int/assets/ism/ism_nowcast/tec_nowcast_grid.xml"

    noaa_solar_wind_speed_url: str = "https://services.swpc.noaa.gov/products/summary/solar-wind-speed.json"
    noaa_10cm_flux_url: str = "https://services.swpc.noaa.gov/products/summary/10cm-flux.json"
    noaa_solar_wind_mag_field_url: str = "https://services.swpc.noaa.gov/products/summary/solar-wind-mag-field.json"
    noaa_scales_url: str = "https://services.swpc.noaa.gov/products/noaa-scales.json"
    noaa_kyoto_dst_url: str = "https://services.swpc.noaa.gov/products/kyoto-dst.json"

    gfz_kp_url: str = "https://kp.gfz.de/app/json/kpnowcast.json"
    gfz_hp30_url: str = "https://kp.gfz.de/app/json/hpo30nowcast.json"


class RTLTCPSettings(BaseSettings):
    """rtl_tcp connection and sampling parameters. (run this behind a multiplexer)"""

    host: str = "localhost"
    port: int = 1234
    sample_rate: int = 2_400_000
    f_center: float = 137e6
    frame_samples: int = 32768


class DatabaseSettings(BaseSettings):
    """Settings used in db.py."""

    sqlite_path: Path = Field(default=Path("satcap.db"))


class Secrets(BaseSettings):
    """API keys, passwords, etc"""

    openweather_api_key: str = Field(default="", env="OPENWEATHER_API_KEY")
    space_track_username: str = Field(default="", env="SPACE_TRACK_USERNAME")
    space_track_password: str = Field(default="", env="SPACE_TRACK_PASSWORD")


class SatdumpSettings(BaseSettings):
    """Settings for the Satdump application."""

    satdump_path_str: str = Field(default="~/gobdump", env="SATDUMP_PATH")
    satdump_run_path_str: str = Field(default="~/gobdump/build/", env="SATDUMP_RUN_PATH")
    satdump_database_path_str: str = Field(default="~/.config/gobdump/main.db", env="SATDUMP_DATABASE_PATH")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CONFIG__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    loader: LoaderSettings = LoaderSettings()
    data_acquisition: DataAcquisitionSettings = DataAcquisitionSettings()
    site: SiteSettings = SiteSettings()
    external_apis: ExternalApiSettings = ExternalApiSettings()
    rtl_tcp: RTLTCPSettings = RTLTCPSettings()
    database: DatabaseSettings = DatabaseSettings()
    secrets: Secrets = Secrets()
    satdump: SatdumpSettings = SatdumpSettings()

    output_dir: Path = Field(default=Path("observations"))
    debug: bool = Field(default=False, env="DEBUG")
    max_concurrent_recordings: int = 5


settings = Settings()


if __name__ == "__main__":
    print(settings.model_dump_json(indent=2))
