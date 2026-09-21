from pathlib import Path

from config.settings import settings
from core.events import Event, registry
from data_acquisition.satdump_utils import run_satdump, update_satdump_kepler


@registry.on(Event.POST_OBSERVATION, applies_to={59051, 57166})
def process_with_satdump(job, stop_event):
    """process recorded data with satdump"""
    obs_dir = Path(job.observation_dir).expanduser().resolve()

    update_satdump_kepler(job.norad_id)
    sample_rate = settings.rtl_tcp.sample_rate / job.decimation

    input_file = (obs_dir / f"{job.norad_id}_baseband.iq").resolve()
    output_dir = (obs_dir / "satdump_output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_satdump(
        input_file=str(input_file),
        output_dir=str(output_dir),
        f_s=sample_rate,
    )
