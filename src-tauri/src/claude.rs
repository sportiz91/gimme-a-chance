use anyhow::{Context, Result};
use tokio::process::Command;

/// Ask Claude Code CLI a question with context from the interview
pub async fn ask(question: &str, context: &str) -> Result<String> {
    let prompt = if context.is_empty() {
        question.to_string()
    } else {
        format!(
            "Interview context (recent transcription):\n\
             ---\n\
             {}\n\
             ---\n\n\
             The interviewer just asked: \"{}\"\n\n\
             Give a concise, direct answer I can say out loud. \
             Keep it under 4 sentences unless it's a coding question. \
             If it's a coding question, include the code but explain the approach first in 1-2 sentences.",
            context, question
        )
    };

    let output = Command::new("claude")
        .arg("--print")
        .arg("--dangerously-skip-permissions")
        .arg(&prompt)
        .output()
        .await
        .context("Failed to run claude CLI. Is it installed?")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Claude CLI failed: {}", stderr);
    }

    let response = String::from_utf8_lossy(&output.stdout).to_string();
    Ok(response.trim().to_string())
}
