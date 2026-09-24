from __future__ import annotations

import csv
import io
import logging

import requests

from config.settings import settings
from core.models import OmmSnapshot
from storage.db import get_connection, insert_omm_snapshot

logger = logging.getLogger(__name__)

SPACE_TRACK_LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
SPACE_TRACK_QUERY_URL = (
    "https://www.space-track.org/basicspacedata/query/class/gp/NORAD_CAT_ID/{norads}"
    "/orderby/NORAD_CAT_ID%20asc/format/csv/emptyresult/show"
)


class OmmUpdateError(Exception):
    pass


def update_omm(satellites) -> list[int]:
    if not satellites:
        return []

    if settings.debug:
        with open("omm_debug.csv") as f:
            response_text = f.read()
    else:
        norads = ",".join(str(sat.id) for sat in satellites)

        with requests.Session() as session:
            login_resp = session.post(
                SPACE_TRACK_LOGIN_URL,
                data={
                    "identity": settings.secrets.space_track_username,
                    "password": settings.secrets.space_track_password,
                },
                timeout=30,
            )
            login_resp.raise_for_status()

            if "Login Failed" in login_resp.text:
                raise OmmUpdateError("Space-Track login failed - check credentials")

            response = session.get(
                SPACE_TRACK_QUERY_URL.format(norads=norads),
                timeout=30,
            )
            response.raise_for_status()
            response_text = response.text

    row_ids = []
    requested_norads = {sat.id for sat in satellites}

    with get_connection() as conn:
        for row in csv.DictReader(io.StringIO(response_text)):
            if int(row["NORAD_CAT_ID"]) not in requested_norads:
                continue

            try:
                snapshot = OmmSnapshot.from_csv_row(row)
            except (KeyError, ValueError) as e:
                logger.warning("Skipping malformed OMM row: %s", e)
                continue

            row_ids.append(insert_omm_snapshot(conn, snapshot))

    logger.info(
        "Updated OMMs for %d satellite(s), %d snapshot(s) stored",
        len(satellites),
        len(row_ids),
    )
    return row_ids
