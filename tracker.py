import csv
import datetime
import os

import requests
from dotenv import load_dotenv
from skyfield.api import EarthSatellite, load, wgs84

ts = load.timescale()

load_dotenv()
space_track_credentials = {}
space_track_credentials["username"] = os.getenv("SPACE_TRACK_USERNAME")
space_track_credentials["password"] = os.getenv("SPACE_TRACK_PASSWORD")

# variables
LAT, LON, ALT = 55.5269, 37.0888, 172
qth = wgs84.latlon(LAT, LON, elevation_m=ALT)


# Norad_cat_id, [f_center, decimation]
TARGET_SATELLITES = {
    59051: [137.9e6, 8],
    57166: [137.9e6, 8],
    25159: [137.5e6, 2],
}


def get_satellite_name_from_norad_id(id):
    mapping = {
        59051: "METEOR-M 2-4",
        57166: "METEOR-M 2-3",
        25159: "ORBCOMM FM04",
    }
    return mapping.get(id, "Unknown Satellite")


def update_tles(target_sats):
    with requests.session() as session:
        login_url = "https://www.space-track.org/ajaxauth/login"
        payload = {"identity": space_track_credentials["username"], "password": space_track_credentials["password"]}
        session.post(login_url, data=payload)
        target_sat_norads = ",".join(list(map(str, list(target_sats.keys()))))
        omm_csv = session.get(
            f"https://www.space-track.org/basicspacedata/query/class/gp/NORAD_CAT_ID/{target_sat_norads}/orderby/NORAD_CAT_ID%20asc/format/csv/emptyresult/show"
        )

    with open("OMM.csv", "w") as ommfile:
        ommfile.write(omm_csv.text)


def get_schedule(target_sats):
    satellites = []
    with open("OMM.csv") as csvfile:
        ommreader = csv.reader(csvfile)
        header = next(ommreader)
        for row in ommreader:
            element_dict = dict(zip(header, row, strict=True))
            if int(element_dict["NORAD_CAT_ID"]) in target_sats:
                satellites.append(EarthSatellite.from_omm(ts, element_dict))

    now = ts.now()
    t1 = now + datetime.timedelta(hours=25)
    schedule = {}
    for satellite in satellites:
        t, events = satellite.find_events(qth, now, t1, altitude_degrees=0.0)
        intervals = []
        for ti, event in zip(t, events, strict=True):
            if event == 0:
                intervals.append([ti.utc_datetime(), None])
            elif event == 2:
                try:
                    intervals[-1][1] = ti.utc_datetime()
                except IndexError:
                    print(f"Warning: Satellite {satellite.name} is currently above the horizon, omitting")
        schedule[satellite.model.satnum] = intervals

    # cleanup incomplete intervals
    for satnum, intervals in schedule.items():
        schedule[satnum] = [i for i in intervals if i[1] is not None]

    return schedule


if __name__ == "__main__":
    update_tles(TARGET_SATELLITES)
