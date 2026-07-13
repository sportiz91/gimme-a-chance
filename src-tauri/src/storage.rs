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
use serde::Serialize;

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

/// This run's session row id — the one the writer thread appends to, and the
/// one the manager panel must never delete.
static SESSION_ID: OnceLock<String> = OnceLock::new();

/// The live session's id, once [`init`] has created its row.
pub fn current_session_id() -> Option<&'static str> {
    SESSION_ID.get().map(String::as_str)
}

// `title`/`description`/`context` are the free-text fields the user fills in
// from the manager panel after the interview; everything else is machine-set.
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    build       TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    title       TEXT,
    description TEXT,
    context     TEXT
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
    _ = SESSION_ID.set(session_id.clone());
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
    // The manager panel opens its own connections next to the writer's; a
    // short busy-wait absorbs the rare write-write overlap with an INSERT.
    conn.pragma_update(None, "busy_timeout", 2000)?;
    conn.execute_batch(SCHEMA)?;
    migrate(&conn)?;
    Ok(conn)
}

/// Columns added after the first release. `CREATE TABLE IF NOT EXISTS` never
/// upgrades an existing file, so databases created before these columns
/// existed get them here — idempotent by checking the live column list.
fn migrate(conn: &Connection) -> Result<()> {
    let mut stmt = conn.prepare("SELECT name FROM pragma_table_info('sessions')")?;
    let existing = stmt
        .query_map([], |r| r.get::<_, String>(0))?
        .collect::<rusqlite::Result<std::collections::HashSet<_>>>()?;
    for col in ["title", "description", "context"] {
        if !existing.contains(col) {
            conn.execute(&format!("ALTER TABLE sessions ADD COLUMN {col} TEXT"), [])?;
        }
    }
    Ok(())
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

// ── Management API — the manager panel's read/write side ───────────────────
//
// The panel opens its own short-lived connections (WAL was chosen exactly so
// readers coexist with the writer thread) and never touches the `record` hot
// path. Everything below is a pure function over a `Connection`, like the
// writer's own helpers above.

/// A connection for the manager panel. Short-lived: each command opens one,
/// works, and drops it.
pub fn open_management() -> Result<Connection> {
    open_db(&db_path())
}

/// One row of the manager's session list — `sessions` plus per-session
/// aggregates.
#[derive(Serialize)]
pub struct SessionSummary {
    pub id: String,
    pub name: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub context: Option<String>,
    pub build: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub event_count: i64,
    /// `t_s` of the session's last event — its duration, and the honest one
    /// even when `ended_at` is missing (crash) or late (app left open).
    pub last_t_s: i64,
    /// This run's session: editable, but not deletable and not injectable.
    pub live: bool,
}

/// One raw timeline event for the manager's viewer / the `.md` export.
#[derive(Serialize)]
pub struct EventRow {
    pub kind: String,
    pub speaker: Option<String>,
    pub content: String,
    pub t_s: i64,
}

const SUMMARY_SELECT: &str = "
    SELECT s.id, s.name, s.title, s.description, s.context, s.build,
           s.started_at, s.ended_at,
           COUNT(e.id), COALESCE(MAX(e.t_s), 0)
    FROM sessions s LEFT JOIN events e ON e.session_id = s.id";

fn map_summary(r: &rusqlite::Row) -> rusqlite::Result<SessionSummary> {
    Ok(SessionSummary {
        id: r.get(0)?,
        name: r.get(1)?,
        title: r.get(2)?,
        description: r.get(3)?,
        context: r.get(4)?,
        build: r.get(5)?,
        started_at: r.get(6)?,
        ended_at: r.get(7)?,
        event_count: r.get(8)?,
        last_t_s: r.get(9)?,
        live: false,
    })
}

/// Every recorded session, newest first.
pub fn list_sessions(conn: &Connection) -> Result<Vec<SessionSummary>> {
    let mut stmt = conn.prepare(&format!(
        "{SUMMARY_SELECT} GROUP BY s.id ORDER BY s.started_at DESC"
    ))?;
    let rows = stmt.query_map([], map_summary)?;
    let current = current_session_id();
    let mut out = Vec::new();
    for row in rows {
        let mut s = row?;
        s.live = current == Some(s.id.as_str());
        out.push(s);
    }
    Ok(out)
}

/// One session's summary row, or an error naming the missing id.
pub fn session_summary(conn: &Connection, id: &str) -> Result<SessionSummary> {
    let mut s = conn
        .query_row(
            &format!("{SUMMARY_SELECT} WHERE s.id = ?1 GROUP BY s.id"),
            [id],
            map_summary,
        )
        .with_context(|| format!("session {id} not found"))?;
    s.live = current_session_id() == Some(s.id.as_str());
    Ok(s)
}

/// A session's full raw timeline, in insertion order.
pub fn session_events(conn: &Connection, session_id: &str) -> Result<Vec<EventRow>> {
    let mut stmt = conn.prepare(
        "SELECT kind, speaker, content, t_s FROM events WHERE session_id = ?1 ORDER BY id",
    )?;
    let rows = stmt.query_map([session_id], |r| {
        Ok(EventRow {
            kind: r.get(0)?,
            speaker: r.get(1)?,
            content: r.get(2)?,
            t_s: r.get(3)?,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// Save the manager panel's free-text fields. Empty fields are stored as
/// NULL so "cleared" and "never set" read the same.
pub fn update_session_meta(
    conn: &Connection,
    id: &str,
    title: &str,
    description: &str,
    context: &str,
) -> Result<()> {
    let none_if_empty = |s: &str| {
        let t = s.trim();
        (!t.is_empty()).then(|| t.to_string())
    };
    let n = conn.execute(
        "UPDATE sessions SET title = ?1, description = ?2, context = ?3 WHERE id = ?4",
        params![
            none_if_empty(title),
            none_if_empty(description),
            none_if_empty(context),
            id
        ],
    )?;
    anyhow::ensure!(n == 1, "session {id} not found");
    Ok(())
}

/// The saved free-text context of a session (None when never filled in).
pub fn session_context(conn: &Connection, id: &str) -> Result<Option<String>> {
    conn.query_row("SELECT context FROM sessions WHERE id = ?1", [id], |r| {
        r.get(0)
    })
    .with_context(|| format!("session {id} not found"))
}

/// Delete a session and its whole timeline. Refuses the live session — the
/// writer thread is still appending to it, and deleting underneath it would
/// leave orphaned events.
pub fn delete_session(conn: &mut Connection, id: &str) -> Result<()> {
    anyhow::ensure!(
        current_session_id() != Some(id),
        "this session is still recording — close the app before deleting it"
    );
    let tx = conn.transaction()?;
    tx.execute("DELETE FROM events WHERE session_id = ?1", [id])?;
    let n = tx.execute("DELETE FROM sessions WHERE id = ?1", [id])?;
    anyhow::ensure!(n == 1, "session {id} not found");
    tx.commit()?;
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

    /// The management layer over a PRE-TITLE database: `migrate` adds the
    /// free-text columns in place, meta round-trips (empty → NULL), the
    /// timeline comes back in order, and delete clears both tables.
    #[test]
    fn management_round_trip() {
        let dir = std::env::temp_dir().join(format!("gimme-mgmt-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sessions.sqlite");
        {
            // The first release's schema, verbatim — what real databases have.
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE sessions (
                     id TEXT PRIMARY KEY, name TEXT NOT NULL, build TEXT NOT NULL,
                     started_at TEXT NOT NULL, ended_at TEXT
                 );
                 CREATE TABLE events (
                     id INTEGER PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                     kind TEXT NOT NULL, speaker TEXT, content TEXT NOT NULL,
                     t_s INTEGER NOT NULL, created_at TEXT NOT NULL, meta TEXT
                 );",
            )
            .unwrap();
        }

        let mut conn = open_db(&path).unwrap(); // migrates in place
        start_session(&conn, "s1").unwrap();
        insert_event(
            &conn,
            "s1",
            &Event {
                kind: "transcript",
                speaker: Some("interviewer"),
                content: "walk me through your resume".into(),
                t_s: 12,
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
                t_s: 30,
                meta: None,
            },
        )
        .unwrap();

        update_session_meta(&conn, "s1", "Privalia r1", "  ", "stack: React 18").unwrap();
        let s = session_summary(&conn, "s1").unwrap();
        assert_eq!(s.title.as_deref(), Some("Privalia r1"));
        assert_eq!(s.description, None); // blank saves as NULL
        assert_eq!(s.event_count, 2);
        assert_eq!(s.last_t_s, 30);
        assert_eq!(
            session_context(&conn, "s1").unwrap().as_deref(),
            Some("stack: React 18")
        );

        let events = session_events(&conn, "s1").unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind, "transcript");
        assert_eq!(events[0].speaker.as_deref(), Some("interviewer"));

        assert_eq!(list_sessions(&conn).unwrap().len(), 1);
        assert!(delete_session(&mut conn, "missing").is_err());
        delete_session(&mut conn, "s1").unwrap();
        assert!(list_sessions(&conn).unwrap().is_empty());
        let orphans: i64 = conn
            .query_row("SELECT COUNT(*) FROM events", [], |r| r.get(0))
            .unwrap();
        assert_eq!(orphans, 0);

        drop(conn);
        std::fs::remove_dir_all(&dir).ok();
    }
}
