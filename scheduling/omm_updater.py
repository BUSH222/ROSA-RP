import requests

from config.settings import settings
from core.models import OmmSnapshot
from storage.db import get_connection, insert_omm_snapshot


def update_omm(satellites):
    """Update OMM data and write them in the database"""
    with requests.session() as session:
        login_url = "https://www.space-track.org/ajaxauth/login"
        payload = {
            "identity": settings.secrets.space_track_username,
            "password": settings.secrets.space_track_password,
        }
        session.post(login_url, data=payload)
        target_sat_norads = []
        for sat in satellites:
            target_sat_norads.append(sat.id)
        target_sat_norads = ",".join(str(norad) for norad in target_sat_norads)
        omm_csv = session.get(
            f"https://www.space-track.org/basicspacedata/query/class/gp/NORAD_CAT_ID/{target_sat_norads}/orderby/NORAD_CAT_ID%20asc/format/csv/emptyresult/show"
        )
    with get_connection() as conn:
        for row in omm_csv.text.splitlines()[1:]:
            row_data = row.split(",")
            omm_snapshot = OmmSnapshot.from_csv_row(row_data)
            row_id = insert_omm_snapshot(conn, omm_snapshot)
    return row_id
