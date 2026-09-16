from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any

import pandas as pd
import requests

from config.settings import settings

logger = logging.getLogger(__name__)


def safe_get_json(url: str, timeout: float = settings.data_acquisition.helper.default_timeout_s):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Failed to fetch JSON from %s: %s", url, exc)
        return None


def safe_get_text(url: str, timeout: float = settings.data_acquisition.helper.default_timeout_s):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        logger.warning("Failed to fetch text from %s: %s", url, exc)
        return None


def parse_esa_ism_xml(xml_string):
    root = ET.fromstring(xml_string)
    data_type = next(elem for elem in root.iter() if elem.tag.split("}")[-1] == "dataType")

    columns = [elem.text.strip() for elem in data_type if elem.tag.split("}")[-1] == "columns"]

    records = []
    timestamp = None

    for record in data_type:
        if record.tag.split("}")[-1] != "record":
            continue

        row = []
        for child in record:
            tag = child.tag.split("}")[-1]
            if tag == "date":
                if timestamp is None:
                    timestamp = pd.to_datetime(child.text.strip(), utc=True)
            elif tag == "stringValue":
                row.append(child.text.strip())

        records.append(row)

    columns_without_time = [col for col in columns if col != "utc"]
    df = pd.DataFrame(records, columns=columns_without_time)
    df.columns = df.columns.str.replace(" (deg.)", "", regex=False)

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df, timestamp


def filter_local(
    df: pd.DataFrame,
    latitude: float,
    longitude: float,
    lat_pad: float = 2.0,
    lon_pad: float = 3.0,
):
    """Return only the grid rows within several degrees of a site"""
    return df[
        df["latitude"].between(latitude - lat_pad, latitude + lat_pad)
        & df["longitude"].between(longitude - lon_pad, longitude + lon_pad)
    ].reset_index(drop=True)


def dataframe_to_records(df: pd.DataFrame):
    return df.where(df.notna(), None).to_dict(orient="records")


def serialize_timestamp(value: Any):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
