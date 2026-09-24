import json
import logging
import threading
import time
from pathlib import Path

from config.settings import settings
from core.models import Satellite

logger = logging.getLogger(__name__)


class SatelliteValidationError(Exception):
    pass


def validate_satellites(raw: list[dict]):
    satellites = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SatelliteValidationError(f"Item {i} is not a dict")

        try:
            id = item["id"]
            name = item["name"]
            frequency = item["frequency"]
            decimation_factor = item["decimation_factor"]
        except KeyError as e:
            raise SatelliteValidationError(f"Item {i} missing key: {e}") from e

        if not isinstance(id, int) or id < 0:
            raise SatelliteValidationError(f"Item {i} has invalid id: {id}")
        if not isinstance(name, str) or not name:
            raise SatelliteValidationError(f"Item {i} has invalid name: {name}")
        if not isinstance(frequency, (int, float)) or not (
            settings.loader.min_frequency_hz <= frequency <= settings.loader.max_frequency_hz
        ):
            raise SatelliteValidationError(f"Item {i} has invalid frequency: {frequency}")
        if decimation_factor not in settings.loader.valid_decimation_factors:
            raise SatelliteValidationError(f"Item {i} has invalid decimation_factor: {decimation_factor}")

        satellites.append(Satellite(id=id, name=name, frequency=frequency, decimation_factor=decimation_factor))
    return satellites


class SatelliteLoader:
    def __init__(self, path: str, poll_interval: float = settings.loader.poll_interval_s):
        self.path = Path(path)
        self.poll_interval = poll_interval

        self._lock = threading.RLock()
        self._satellites = []
        self._mtime = None
        self._callbacks = []

        self._stop_event = threading.Event()
        self._thread = None

        self._load(initial=True)

    def get_satellites(self):
        with self._lock:
            return list(self._satellites)

    def on_reload(self, callback):
        self._callbacks.append(callback)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info(f"SatelliteLoader watching {self.path} every {self.poll_interval}s")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.poll_interval + 1)

    def reload_now(self):
        return self._load(initial=False)

    def _watch_loop(self):
        while not self._stop_event.is_set():
            try:
                self._load(initial=False)
            except Exception as e:
                logger.exception(f"Unexpected error: {e}")
            self._stop_event.wait(self.poll_interval)

    def _load(self, initial):
        try:
            stat = self.path.stat()
        except OSError as e:
            logger.error("Cannot stat %s: %s", self.path, e)
            if initial:
                raise
            return False

        if self._mtime is not None and stat.st_mtime == self._mtime:
            return False

        try:
            with self.path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to read/parse %s: %s", self.path, e)
            if initial:
                raise
            return False

        try:
            new_satellites = validate_satellites(raw)
        except SatelliteValidationError as e:
            logger.error("Validation failed for %s: %s", self.path, e)
            if initial:
                raise
            return False

        with self._lock:
            self._satellites = new_satellites
            self._mtime = stat.st_mtime

        if not initial:
            logger.info("Reloaded %s: %d satellite(s)", self.path, len(new_satellites))

        for cb in self._callbacks:
            try:
                cb(new_satellites)
            except Exception:
                logger.exception("Satellite reload callback raised")

        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    loader = SatelliteLoader("config/target_satellites.json")
    loader.on_reload(lambda sats: print(f"[reload] {len(sats)} satellite(s): {[s.name for s in sats]}"))

    print("Initial satellites:")
    for s in loader.get_satellites():
        print(s)

    loader.start()
    print("Watching for changes to target_satellites.json (Ctrl+C to stop)...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        loader.stop()
