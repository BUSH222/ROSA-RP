from dataclasses import dataclass, field
from datetime import UTC, datetime

# Constants
OMM_FIELDS: tuple[str, ...] = (
    "ccsds_omm_vers",
    "comment",
    "creation_date",
    "originator",
    "object_name",
    "object_id",
    "center_name",
    "ref_frame",
    "time_system",
    "mean_element_theory",
    "mean_motion",
    "eccentricity",
    "inclination",
    "ra_of_asc_node",
    "arg_of_pericenter",
    "mean_anomaly",
    "ephemeris_type",
    "classification_type",
    "norad_cat_id",
    "element_set_no",
    "rev_at_epoch",
    "bstar",
    "mean_motion_dot",
    "mean_motion_ddot",
    "semimajor_axis",
    "period",
    "apoapsis",
    "periapsis",
    "object_type",
    "rcs_size",
    "country_code",
    "launch_date",
    "site",
    "decay_date",
    "file",
    "gp_id",
    "tle_line0",
    "tle_line1",
    "tle_line2",
)

CSV_TO_COLUMN: dict[str, str] = {name.upper(): name for name in OMM_FIELDS}

_INT_FIELDS = {
    "ephemeris_type",
    "norad_cat_id",
    "element_set_no",
    "rev_at_epoch",
    "gp_id",
}
_FLOAT_FIELDS = {
    "mean_motion",
    "eccentricity",
    "inclination",
    "ra_of_asc_node",
    "arg_of_pericenter",
    "mean_anomaly",
    "bstar",
    "mean_motion_dot",
    "mean_motion_ddot",
    "semimajor_axis",
    "period",
    "apoapsis",
    "periapsis",
}


def _coerce(column: str, value):
    if value is None or value == "":
        return None
    if column in _INT_FIELDS:
        return int(value)
    if column in _FLOAT_FIELDS:
        return float(value)
    return value


# helper functions


def utcnow_iso() -> str:
    """Current UTC time as an ISO8601 string, e.g. 2026-09-17T20:14:00.123456Z."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


# objects


@dataclass
class OmmSnapshot:
    norad_id: int
    epoch: str
    fetched_at: str = field(default_factory=utcnow_iso)
    id: int | None = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_csv_row(cls, row: dict, fetched_at=None):
        """Build a snapshot from one row of a Space-Track OMM CSV export,
        i.e. a dict keyed by the CSV's upper-case headers (NORAD_CAT_ID,
        EPOCH, MEAN_MOTION, TLE_LINE1, ...).
        """
        norad_id = int(row["NORAD_CAT_ID"])
        epoch = row["EPOCH"]
        raw = {}
        for csv_key, column in CSV_TO_COLUMN.items():
            if csv_key in row:
                raw[column] = _coerce(column, row[csv_key])
        return cls(norad_id=norad_id, epoch=epoch, fetched_at=fetched_at or utcnow_iso(), raw=raw)

    @classmethod
    def from_row(cls, row):
        """Rebuild a snapshot from a row fetched out of omm_snapshots."""
        raw = {f: row[f] for f in OMM_FIELDS}
        return cls(
            id=row["id"],
            norad_id=row["norad_id"],
            fetched_at=row["fetched_at"],
            epoch=row["epoch"],
            raw=raw,
        )

    @property
    def tle_line1(self):
        return self.raw.get("tle_line1")

    @property
    def tle_line2(self):
        return self.raw.get("tle_line2")


@dataclass
class Job:
    id: str
    norad_id: int
    aos: str
    los: str
    f_center: float
    decimation: int
    state: str = "pending"
    omm_snapshot_id: int | None = None
    observation_dir: str | None = None
    pid: int | None = None
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)
    failure_reason: str | None = None

    @classmethod
    def from_row(cls, row):
        return cls(**{key: row[key] for key in row.keys()})  # noqa: SIM118


class Satellite:
    def __init__(self, id: int, name: str, frequency: float, decimation_factor: int):
        self.id = id
        self.name = name
        self.frequency = frequency
        self.decimation_factor = decimation_factor

    def __repr__(self):
        return (
            f"Satellite(id={self.id}, name={self.name!r}, "
            f"frequency={self.frequency}, decimation_factor={self.decimation_factor})"
        )

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "frequency": self.frequency,
            "decimation_factor": self.decimation_factor,
        }
