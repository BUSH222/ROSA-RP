from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from config.settings import settings

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class ObservationPaths:
    root: Path
    norad_id: int

    @property
    def baseband(self) -> Path:
        return self.root / f"{self.norad_id}_baseband.iq"

    @property
    def baseband_compressed(self) -> Path:
        return self.root / f"{self.norad_id}_baseband.iq.zst"

    @property
    def spectrogram(self) -> Path:
        return self.root / "spectrogram.png"

    @property
    def decoded_dir(self) -> Path:
        return self.root / "decoded"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"


def new_observation_dir(norad_id: int, started_at: datetime | None = None, output_dir: Path | None = None) -> Path:
    """Create and return a fresh observation directory:
    observations/<unix_ts>_<YYYYmmddTHHMMSSZ>_<norad_id>/
    """
    started_at = started_at or datetime.now(UTC)
    output_dir = output_dir or settings.output_dir
    unix_ts = int(started_at.timestamp())
    human_ts = started_at.strftime("%Y%m%dT%H%M%SZ")
    root = output_dir / f"{unix_ts}_{human_ts}_{norad_id}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def paths_for(root: Path, norad_id: int) -> ObservationPaths:
    return ObservationPaths(root=root, norad_id=norad_id)
