PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS omm_snapshots (
    id              INTEGER PRIMARY KEY,
    norad_id        INTEGER NOT NULL,
    fetched_at      TEXT NOT NULL,
    epoch           TEXT NOT NULL,

    ccsds_omm_vers      TEXT,
    comment             TEXT,
    creation_date       TEXT,
    originator          TEXT,
    object_name         TEXT,
    object_id           TEXT,
    center_name         TEXT,
    ref_frame           TEXT,
    time_system         TEXT,
    mean_element_theory TEXT,
    mean_motion         REAL,
    eccentricity        REAL,
    inclination         REAL,
    ra_of_asc_node       REAL,
    arg_of_pericenter   REAL,
    mean_anomaly        REAL,
    ephemeris_type      INTEGER,
    classification_type TEXT,
    norad_cat_id        INTEGER,
    element_set_no      INTEGER,
    rev_at_epoch        INTEGER,
    bstar               REAL,
    mean_motion_dot     REAL,
    mean_motion_ddot    REAL,
    semimajor_axis      REAL,
    period              REAL,
    apoapsis            REAL,
    periapsis           REAL,
    object_type         TEXT,
    rcs_size            TEXT,
    country_code        TEXT,
    launch_date         TEXT,
    site                TEXT,
    decay_date          TEXT,
    file                TEXT,
    gp_id               INTEGER,
    tle_line0           TEXT,
    tle_line1           TEXT,
    tle_line2           TEXT
);

CREATE INDEX IF NOT EXISTS idx_omm_norad_epoch
    ON omm_snapshots(norad_id, epoch);

CREATE UNIQUE INDEX IF NOT EXISTS idx_omm_gp_id
    ON omm_snapshots(gp_id) WHERE gp_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT PRIMARY KEY,
    norad_id         INTEGER NOT NULL,
    aos              TEXT NOT NULL,
    los              TEXT NOT NULL,
    f_center         REAL NOT NULL,
    decimation       INTEGER NOT NULL,
    state            TEXT NOT NULL DEFAULT 'pending'
                        CHECK (state IN ('pending', 'recording', 'completed', 'failed', 'cancelled')),
    omm_snapshot_id  INTEGER REFERENCES omm_snapshots(id),
    observation_dir  TEXT,
    pid              INTEGER, -- os pid
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    failure_reason   TEXT,
    UNIQUE(norad_id, aos)
);

CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_norad ON jobs(norad_id, aos);