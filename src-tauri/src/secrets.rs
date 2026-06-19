//! Secret loading with two complementary layers:
//!   - **At rest**: the OS keyring (Windows Credential Manager), encrypted by the OS.
//!   - **In memory**: `secrecy::SecretString`, which redacts in `Debug`/logs and zeroes
//!     the memory on drop. The raw value is only reachable via `.expose_secret()`.
//!
//! Load order (`load_key`):
//!   1. Keyring — the steady state once seeded.
//!   2. Env var (injected by `dev.ps1` from `~/.claude/.env`) → seed the keyring → use it.
//!      This auto-seeds on the first dev run, so no manual Credential Manager step.
//!   3. None — caller logs a clear startup error (never fails mid-interview).

use secrecy::SecretString;

/// Keyring "service" namespace for all of this app's secrets.
const SERVICE: &str = "gimme-a-chance";

/// Load an API key as a `SecretString`, seeding the keyring from the environment
/// on first use. Returns `None` if the key is in neither the keyring nor the env.
#[must_use]
pub fn load_key(name: &str) -> Option<SecretString> {
    if let Some(secret) = from_keyring(name) {
        tracing::debug!(key = name, "loaded API key from OS keyring");
        return Some(secret);
    }
    if let Ok(value) = std::env::var(name) {
        if !value.is_empty() {
            seed_keyring(name, &value);
            return Some(SecretString::new(value));
        }
    }
    tracing::warn!(key = name, "API key not found in keyring or environment");
    None
}

fn entry(name: &str) -> Option<keyring::Entry> {
    match keyring::Entry::new(SERVICE, name) {
        Ok(e) => Some(e),
        Err(err) => {
            tracing::warn!(key = name, error = %err, "could not open keyring entry");
            None
        }
    }
}

fn from_keyring(name: &str) -> Option<SecretString> {
    let entry = entry(name)?;
    match entry.get_password() {
        Ok(value) if !value.is_empty() => Some(SecretString::new(value)),
        // Empty value, or no entry yet (the normal "not seeded" case) → nothing usable.
        Ok(_) | Err(keyring::Error::NoEntry) => None,
        Err(err) => {
            tracing::warn!(key = name, error = %err, "keyring read failed");
            None
        }
    }
}

fn seed_keyring(name: &str, value: &str) {
    let Some(entry) = entry(name) else { return };
    match entry.set_password(value) {
        Ok(()) => tracing::info!(
            key = name,
            "seeded API key into OS keyring from environment"
        ),
        Err(err) => {
            tracing::warn!(key = name, error = %err, "failed to seed keyring (will use env value this run)");
        }
    }
}
