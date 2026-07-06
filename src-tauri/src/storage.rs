//! Meeting persistence — the append-only session log.
//!
//! Every final transcript line, screen description, clipboard ingest, manual
//! question, and model answer is appended to a local `SQLite` file the moment
//! it exists, so a full interview can be reconstructed (and mined for stats:
//! most frequent questions, what to study) long after the app closed.
//!
//! Producers never touch the database: [`record`] pushes onto an unbounded
//! channel (an in-memory send, nanoseconds, callable from any thread —
//! including async commands) and a dedicated writer thread, sole owner of
//! the connection, does the INSERTs. WAL journaling with `synchronous=NORMAL`
//! makes each insert a page-cache append with fsync deferred to checkpoints:
//! a crash mid-interview loses at most the events still in flight in the
//! channel; everything older is already on disk.
//!
//! One app run = one `sessions` row (the app is opened per interview). Debug
//! and release builds share the file; the session's `build` column separates
//! real interviews from dev runs at query time.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender};
use rusqlite::{params, Connection};

/// One persisted event, already final (never a streaming delta).
pub struct Event {
    /// Source discriminator: `transcript` | `screen` | `clipboard` |
    /// `clipboard_stealth` | `question` | `answer`.
    pub kind: &'static str,
    /// `me` / `interviewer` on transcript events; None elsewhere.
    pub speaker: Option<&'static str>,
    pub content: String,
    /// Seconds since session start — the same clock as the transcript `t_s`.
    pub t_s: u64,
    /// Small JSON bag contextualizing the event (`trace_id`, model, token
    /// usage, trigger…). Queryable in SQL via `json_extract`.
    pub meta: Option<serde_json::Value>,
}

enum Msg {
    Event(Event),
    Shutdown,
}

struct Writer {
    tx: Sender<Msg>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

static WRITER: OnceLock<Writer> = OnceLock::new();

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    build      TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at   TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    kind       TEXT NOT NULL,
    speaker    TEXT,
    content    TEXT NOT NULL,
    t_s        INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    meta       TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_session_kind ON events (session_id, kind);
";

/// `%APPDATA%/gimme-a-chance/sessions.sqlite` — next to the whisper models,
/// one file across debug and release so stats always query one place.
fn db_path() -> PathBuf {
    dirs_next::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("gimme-a-chance")
        .join("sessions.sqlite")
}

/// Open the database, insert this run's session row, and spawn the writer
/// thread. Call once at startup, before any [`record`] site can fire. On
/// failure the app keeps running and [`record`] stays a no-op.
pub fn init() -> Result<()> {
    let path = db_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("create data dir {}", dir.display()))?;
    }
    let conn = open_db(&path)?;
    let session_id = uuid::Uuid::new_v4().to_string();
    start_session(&conn, &session_id)?;
    tracing::info!(path = %path.display(), session_id = %session_id, "session persistence started");

    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = std::thread::Builder::new()
        .name("session-writer".into())
        .spawn(move || writer_loop(&conn, &session_id, &rx))
        .context("spawn session writer thread")?;
    if WRITER
        .set(Writer {
            tx,
            handle: Mutex::new(Some(handle)),
        })
        .is_err()
    {
        anyhow::bail!("storage initialized twice");
    }
    Ok(())
}

/// Queue an event for persistence — non-blocking, safe from any thread, a
/// no-op when [`init`] failed (or after [`shutdown`]).
pub fn record(event: Event) {
    if let Some(w) = WRITER.get() {
        _ = w.tx.send(Msg::Event(event));
    }
}

/// Drain everything still queued, stamp the session's `ended_at`, and close
/// the database. Call once, from the app-exit hook.
pub fn shutdown() {
    let Some(w) = WRITER.get() else { return };
    _ = w.tx.send(Msg::Shutdown);
    let handle = w.handle.lock().ok().and_then(|mut g| g.take());
    if let Some(handle) = handle {
        if handle.join().is_err() {
            tracing::warn!("session writer thread panicked");
        }
    }
}

/// The writer thread: inserts until [`shutdown`] (or every sender dropping)
/// ends the loop, then closes the session. A failed insert is logged and
/// skipped — one bad event must not stop the log.
fn writer_loop(conn: &Connection, session_id: &str, rx: &Receiver<Msg>) {
    while let Ok(Msg::Event(event)) = rx.recv() {
        if let Err(e) = insert_event(conn, session_id, &event) {
            tracing::warn!(error = %e, kind = event.kind, "session event insert failed");
        }
    }
    if let Err(e) = end_session(conn, session_id) {
        tracing::warn!(error = %e, "session close failed");
    }
    // Dropping the last connection checkpoints the WAL back into the file.
}

fn open_db(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path).with_context(|| format!("open {}", path.display()))?;
    // `journal_mode` answers with a result row (the granted mode) — read it.
    let mode: String = conn.query_row("PRAGMA journal_mode = WAL", [], |row| row.get(0))?;
    if !mode.eq_ignore_ascii_case("wal") {
        anyhow::bail!("journal_mode WAL not granted (got {mode})");
    }
    conn.execute_batch("PRAGMA synchronous = NORMAL; PRAGMA foreign_keys = ON;")?;
    conn.execute_batch(SCHEMA)?;
    Ok(conn)
}

fn start_session(conn: &Connection, id: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO sessions (id, name, build, started_at) VALUES (?1, ?2, ?3, ?4)",
        params![
            id,
            chrono::Local::now().format("%Y-%m-%d %H:%M").to_string(),
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            },
            chrono::Utc::now().to_rfc3339(),
        ],
    )?;
    Ok(())
}

fn insert_event(conn: &Connection, session_id: &str, event: &Event) -> Result<()> {
    conn.execute(
        "INSERT INTO events (session_id, kind, speaker, content, t_s, created_at, meta)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            session_id,
            event.kind,
            event.speaker,
            event.content,
            i64::try_from(event.t_s).unwrap_or(i64::MAX),
            chrono::Utc::now().to_rfc3339(),
            event.meta.as_ref().map(ToString::to_string),
        ],
    )?;
    Ok(())
}

fn end_session(conn: &Connection, id: &str) -> Result<()> {
    conn.execute(
        "UPDATE sessions SET ended_at = ?1 WHERE id = ?2",
        params![chrono::Utc::now().to_rfc3339(), id],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pure layer round-trips: schema, session row, both event shapes,
    /// close. The channel/thread plumbing on top is intentionally thin.
    #[test]
    fn round_trip() {
        let dir = std::env::temp_dir().join(format!("gimme-storage-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let conn = open_db(&dir.join("sessions.sqlite")).unwrap();

        start_session(&conn, "s1").unwrap();
        insert_event(
            &conn,
            "s1",
            &Event {
                kind: "transcript",
                speaker: Some("interviewer"),
                content: "tell me about yourself".into(),
                t_s: 42,
                meta: None,
            },
        )
        .unwrap();
        insert_event(
            &conn,
            "s1",
            &Event {
                kind: "answer",
                speaker: None,
                content: "an answer".into(),
                t_s: 43,
                meta: Some(serde_json::json!({ "trigger": "agent" })),
            },
        )
        .unwrap();
        end_session(&conn, "s1").unwrap();

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM events WHERE session_id = 's1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 2);
        let (speaker, t_s): (String, i64) = conn
            .query_row(
                "SELECT speaker, t_s FROM events WHERE kind = 'transcript'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(speaker, "interviewer");
        assert_eq!(t_s, 42);
        let trigger: String = conn
            .query_row(
                "SELECT json_extract(meta, '$.trigger') FROM events WHERE kind = 'answer'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(trigger, "agent");
        let ended: Option<String> = conn
            .query_row("SELECT ended_at FROM sessions WHERE id = 's1'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert!(ended.is_some());

        drop(conn);
        std::fs::remove_dir_all(&dir).ok();
    }
}
