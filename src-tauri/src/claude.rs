use std::process::Stdio;
use std::time::Instant;

use anyhow::{Context, Result};
use tokio::process::Command;

pub struct ClaudeResult {
    pub answer: String,
    pub spawn_ms: u64,
    pub wait_ms: u64,
}

/// Ask Claude Code CLI a question with context from the interview
#[tracing::instrument(skip(context), fields(question_len = question.len(), context_len = context.len()))]
pub async fn ask(question: &str, context: &str) -> Result<ClaudeResult> {
    let prompt = if context.is_empty() {
        question.to_string()
    } else {
        format!(
            "Interview context (recent transcription):\n\
             ---\n\
             {context}\n\
             ---\n\n\
             The interviewer just asked: \"{question}\"\n\n\
             Give a concise, direct answer I can say out loud. \
             Keep it under 4 sentences unless it's a coding question. \
             If it's a coding question, include the code but explain the approach first in 1-2 sentences."
        )
    };

    // Tauri runs as Windows .exe; claude lives in WSL. Absolute path skips `bash -lc` overhead.
    // stdout/stderr must be piped explicitly — `.spawn()` inherits from parent otherwise,
    // which would print claude's answer to the terminal but leave `output.stdout` empty.
    let mut cmd = Command::new("wsl.exe");
    cmd.arg("--")
        .arg("/home/lasantoneta/.local/bin/claude")
        .arg("--print")
        .arg("--dangerously-skip-permissions")
        .arg(&prompt)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let spawn_start = Instant::now();
    let child = cmd
        .spawn()
        .context("Failed to spawn claude via wsl.exe. Is WSL running and claude installed in WSL?")?;
    let spawn_ms = u64::try_from(spawn_start.elapsed().as_millis()).unwrap_or(u64::MAX);

    let wait_start = Instant::now();
    let output = child
        .wait_with_output()
        .await
        .context("Failed to wait on claude process")?;
    let wait_ms = u64::try_from(wait_start.elapsed().as_millis()).unwrap_or(u64::MAX);

    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();

    tracing::info!(
        spawn_ms,
        wait_ms,
        total_ms = spawn_ms + wait_ms,
        exit_code = output.status.code().unwrap_or(-1),
        stdout_bytes = stdout.len(),
        stderr_bytes = stderr.len(),
        stderr = if stderr.is_empty() { "<empty>".into() } else { stderr.clone() },
        "claude call finished"
    );

    if !output.status.success() {
        anyhow::bail!("Claude CLI failed (exit {:?}): {stderr}", output.status.code());
    }

    Ok(ClaudeResult {
        answer: stdout.trim().to_string(),
        spawn_ms,
        wait_ms,
    })
}
