import json
from pathlib import Path

from config.settings import settings
from core.events import Event, registry
from data_acquisition.ionosphere import collect_observation


@registry.on(Event.OBSERVATION_START)
def save_ionosphere_data(job, stop_event):
    """Collect ionosphere data at the start of an observation."""
    collect_observation(Path(job.observation_dir))


@registry.on(Event.OBSERVATION_START)
def write_metadata(job, stop_event):
    """Write metadata to a file at the start of an observation."""
    metadata_path = Path(job.observation_dir) / "metadata.json"
    metadata = {
        "job_id": job.id,
        "norad_id": job.norad_id,
        "aos": job.aos,
        "los": job.los,
        "f_center": job.f_center,
        "decimation": job.decimation,
        "f_sample": settings.rtl_tcp.sample_rate_hz / job.decimation,
        "omm_snapshot_id": job.omm_snapshot_id,  # for doppler correction
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
