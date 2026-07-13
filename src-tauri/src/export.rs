//! Render a recorded session to a self-contained Markdown document — the
//! manager panel's "Export .md". Pure text-building over [`storage`] rows;
//! where the file lands is the command layer's business.

use std::fmt::Write as _;

use crate::storage::{EventRow, SessionSummary};

/// `mm:ss` under an hour, `h:mm:ss` over — interviews cross the hour mark.
fn stamp(t_s: i64) -> String {
    let t = t_s.max(0);
    let (h, m, s) = (t / 3600, (t % 3600) / 60, t % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

/// Human label per event — mirrors the manager timeline's labels.
fn label(event: &EventRow) -> &str {
    match event.kind.as_str() {
        "transcript" => match event.speaker.as_deref() {
            Some("me") => "You",
            _ => "Interviewer",
        },
        "screen" => "Screen",
        "clipboard" | "clipboard_stealth" => "Clipboard",
        "question" => "Question",
        "answer" => "Answer",
        "context" => "Loaded context",
        other => other,
    }
}

/// The whole document: heading + metadata, the user's free-text fields when
/// present, then the full raw timeline. Multi-line content (model answers,
/// pasted code) goes under a blockquote so it can't break the surrounding
/// structure; one-liners stay inline.
pub fn session_markdown(session: &SessionSummary, events: &[EventRow]) -> String {
    let mut md = String::new();
    let title = session.title.as_deref().unwrap_or(&session.name);
    _ = writeln!(md, "# {title}\n");
    _ = writeln!(md, "- **Recorded:** {}", session.name);
    _ = writeln!(md, "- **Duration:** {}", stamp(session.last_t_s));
    _ = writeln!(md, "- **Build:** {}", session.build);
    _ = writeln!(md, "- **Events:** {}", session.event_count);

    if let Some(description) = session.description.as_deref() {
        _ = writeln!(md, "\n## Description\n\n{description}");
    }
    if let Some(context) = session.context.as_deref() {
        _ = writeln!(md, "\n## Context\n\n{context}");
    }

    _ = writeln!(md, "\n## Timeline\n");
    for event in events {
        let head = format!("**[{}] {}:**", stamp(event.t_s), label(event));
        if event.content.contains('\n') {
            _ = writeln!(md, "{head}\n");
            for line in event.content.lines() {
                _ = writeln!(md, "> {line}");
            }
            _ = writeln!(md);
        } else {
            _ = writeln!(md, "{head} {}\n", event.content);
        }
    }
    md
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(kind: &str, speaker: Option<&str>, content: &str, t_s: i64) -> EventRow {
        EventRow {
            kind: kind.into(),
            speaker: speaker.map(Into::into),
            content: content.into(),
            t_s,
        }
    }

    #[test]
    fn renders_meta_fields_and_timeline() {
        let session = SessionSummary {
            id: "s1".into(),
            name: "2026-07-08 14:03".into(),
            title: Some("Privalia round 1".into()),
            description: Some("Code review with Fernando".into()),
            context: Some("They use React 18".into()),
            build: "release".into(),
            started_at: "2026-07-08T12:03:00+00:00".into(),
            ended_at: None,
            event_count: 3,
            last_t_s: 3725,
            live: false,
        };
        let events = vec![
            event(
                "transcript",
                Some("interviewer"),
                "tell me about yourself",
                42,
            ),
            event("screen", None, "A LeetCode problem", 130),
            event("answer", None, "line one\nline two", 3725),
        ];

        let md = session_markdown(&session, &events);
        assert!(md.starts_with("# Privalia round 1\n"));
        assert!(md.contains("- **Duration:** 1:02:05"));
        assert!(md.contains("## Description\n\nCode review with Fernando"));
        assert!(md.contains("## Context\n\nThey use React 18"));
        assert!(md.contains("**[00:42] Interviewer:** tell me about yourself"));
        assert!(md.contains("**[02:10] Screen:** A LeetCode problem"));
        // Multi-line content lands as a blockquote under its header line.
        assert!(md.contains("**[1:02:05] Answer:**\n\n> line one\n> line two"));
    }

    #[test]
    fn untitled_session_falls_back_to_its_name() {
        let session = SessionSummary {
            id: "s2".into(),
            name: "2026-07-01 09:00".into(),
            title: None,
            description: None,
            context: None,
            build: "debug".into(),
            started_at: "2026-07-01T07:00:00+00:00".into(),
            ended_at: None,
            event_count: 0,
            last_t_s: 0,
            live: false,
        };
        let md = session_markdown(&session, &[]);
        assert!(md.starts_with("# 2026-07-01 09:00\n"));
        assert!(!md.contains("## Description"));
        assert!(!md.contains("## Context"));
    }
}
