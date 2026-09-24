from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from config.settings import settings
from core.models import OMM_FIELDS, Job, OmmSnapshot, utcnow_iso

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
DEFAULT_DB_PATH = settings.database.sqlite_path


# connection


def connect(db_path=DEFAULT_DB_PATH):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_connection(db_path=DEFAULT_DB_PATH):
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path=DEFAULT_DB_PATH):
    schema_sql = SCHEMA_PATH.read_text()
    with get_connection(db_path) as conn:
        conn.executescript(schema_sql)


# OMMs


def insert_omm_snapshot(conn: sqlite3.Connection, snapshot: OmmSnapshot):
    """Insert an OMM snapshot and return its row id."""
    gp_id = snapshot.raw.get("gp_id")
    if gp_id is not None:
        existing = conn.execute("SELECT id FROM omm_snapshots WHERE gp_id = ?", (gp_id,)).fetchone()
        if existing is not None:
            return existing["id"]

    columns = ["norad_id", "fetched_at", "epoch", *OMM_FIELDS]
    values = [snapshot.norad_id, snapshot.fetched_at, snapshot.epoch]
    values += [snapshot.raw.get(f) for f in OMM_FIELDS]

    placeholders = ", ".join("?" for _ in columns)
    col_list = ", ".join(columns)
    cur = conn.execute(f"INSERT INTO omm_snapshots ({col_list}) VALUES ({placeholders})", values)
    return cur.lastrowid


def get_snapshot(conn: sqlite3.Connection, snapshot_id: int):
    return conn.execute("SELECT * FROM omm_snapshots WHERE id = ?", (snapshot_id,)).fetchone()


def get_latest_snapshot(conn: sqlite3.Connection, norad_id: int, as_of=None):
    if as_of is None:
        return conn.execute(
            "SELECT * FROM omm_snapshots WHERE norad_id = ? ORDER BY epoch DESC LIMIT 1",
            (norad_id,),
        ).fetchone()
    return conn.execute(
        "SELECT * FROM omm_snapshots WHERE norad_id = ? AND epoch <= ? ORDER BY epoch DESC LIMIT 1",
        (norad_id, as_of),
    ).fetchone()


def list_snapshots(conn: sqlite3.Connection, norad_id: int):
    return conn.execute("SELECT * FROM omm_snapshots WHERE norad_id = ? ORDER BY epoch", (norad_id,)).fetchall()


# jobs


def create_job(
    conn: sqlite3.Connection,
    *,
    norad_id: int,
    aos: str,
    los: str,
    f_center: float,
    decimation: int,
    omm_snapshot_id=None,
    observation_dir=None,
    job_id=None,
):
    """Create a pending job for a single pass."""
    loader = settings.loader
    if decimation not in loader.valid_decimation_factors:
        raise ValueError(f"decimation={decimation} is not one of {loader.valid_decimation_factors}")
    if not (loader.min_frequency_hz <= f_center <= loader.max_frequency_hz):
        raise ValueError(f"f_center={f_center} is outside [{loader.min_frequency_hz}, {loader.max_frequency_hz}] Hz")

    now = utcnow_iso()
    job = Job(
        id=job_id or str(uuid.uuid4()),
        norad_id=norad_id,
        aos=aos,
        los=los,
        f_center=f_center,
        decimation=decimation,
        omm_snapshot_id=omm_snapshot_id,
        observation_dir=observation_dir,
        created_at=now,
        updated_at=now,
    )
    conn.execute(
        """
        INSERT INTO jobs (
            id, norad_id, aos, los, f_center, decimation, state,
            omm_snapshot_id, observation_dir, pid, created_at, updated_at, failure_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job.id,
            job.norad_id,
            job.aos,
            job.los,
            job.f_center,
            job.decimation,
            job.state,
            job.omm_snapshot_id,
            job.observation_dir,
            job.pid,
            job.created_at,
            job.updated_at,
            job.failure_reason,
        ),
    )
    return job


def get_job(conn: sqlite3.Connection, job_id: str):
    return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def list_jobs(conn: sqlite3.Connection, state=None):
    if state is None:
        return conn.execute("SELECT * FROM jobs ORDER BY aos").fetchall()
    return conn.execute("SELECT * FROM jobs WHERE state = ? ORDER BY aos", (state,)).fetchall()


def update_job_state(
    conn: sqlite3.Connection,
    job_id: str,
    state: str,
    *,
    pid=None,
    failure_reason=None,
    observation_dir=None,
):
    set_clauses = ["state = ?", "updated_at = ?"]
    values: list = [state, utcnow_iso()]
    if pid is not None:
        set_clauses.append("pid = ?")
        values.append(pid)
    if failure_reason is not None:
        set_clauses.append("failure_reason = ?")
        values.append(failure_reason)
    if observation_dir is not None:
        set_clauses.append("observation_dir = ?")
        values.append(observation_dir)
    values.append(job_id)
    conn.execute(f"UPDATE jobs SET {', '.join(set_clauses)} WHERE id = ?", values)


def set_job_snapshot(conn: sqlite3.Connection, job_id: str, omm_snapshot_id: int):
    """Attach (or change) the OMM snapshot a job should use for propagation."""
    conn.execute(
        "UPDATE jobs SET omm_snapshot_id = ?, updated_at = ? WHERE id = ?",
        (omm_snapshot_id, utcnow_iso(), job_id),
    )


def refresh_pending_job_snapshots(conn: sqlite3.Connection, as_of=None):
    """Re-point every still-pending job at the freshest OMM snapshot available
    for its satellite, using each job's own AOS as the "as of" cutoff (or an
    explicit as_of for all jobs, e.g. right now).

    Call this right after ingesting a new batch of OMMs, so a job scheduled
    hours before its pass ends up using the most recent elements known by
    the time recording actually starts, not whatever was current when the
    job was first created.

    Only touches jobs in state='pending'; a job that's already 'recording'
    (or past) keeps whatever snapshot it locked in.
    Returns the number of jobs whose omm_snapshot_id changed.
    """
    pending = conn.execute("SELECT id, norad_id, aos, omm_snapshot_id FROM jobs WHERE state = 'pending'").fetchall()

    updated = 0
    for job in pending:
        cutoff = as_of or job["aos"]
        latest = get_latest_snapshot(conn, job["norad_id"], as_of=cutoff)
        if latest is not None and latest["id"] != job["omm_snapshot_id"]:
            set_job_snapshot(conn, job["id"], latest["id"])
            updated += 1
    return updated


def delete_job(conn: sqlite3.Connection, job_id: str):
    conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def invalidate_overdue_pending_jobs(
    conn: sqlite3.Connection,
    now: datetime | None = None,
) -> int:
    """Cancel pending jobs whose AOS has already passed."""
    now = now or datetime.now(UTC)

    pending = conn.execute("SELECT id, aos FROM jobs WHERE state = 'pending'").fetchall()

    overdue_ids = []
    for job in pending:
        aos = datetime.fromisoformat(job["aos"].replace("Z", "+00:00"))
        if aos <= now:
            overdue_ids.append(job["id"])

    if not overdue_ids:
        return 0

    updated_at = utcnow_iso()
    conn.executemany(
        """
        UPDATE jobs
        SET state = 'cancelled',
            failure_reason = ?,
            updated_at = ?
        WHERE id = ? AND state = 'pending'
        """,
        [("AOS passed before the job could be scheduled", updated_at, job_id) for job_id in overdue_ids],
    )
    return len(overdue_ids)


if __name__ == "__main__":
    init_db()
    print(f"Initialized {DEFAULT_DB_PATH}")
    if settings.debug:
        with get_connection() as conn:
            latest_snapshot = get_latest_snapshot(conn, 59051)
            print(f"Latest snapshot for 59051: {dict(latest_snapshot)}")
