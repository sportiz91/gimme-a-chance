//! Answering + vision backends over `OpenAI`-compatible HTTP providers.
//!
//! Answering (the "brain") is a fallback chain ordered speed-first so the copilot
//! never goes silent mid-interview:
//!   1. Groq `llama-3.1-8b-instant`   — fastest (~150ms TTFT)
//!   2. Groq `llama-3.3-70b-versatile` — smarter fallback (still Groq)
//!   3. `OpenAI` `gpt-4o-mini`         — survives a full Groq/Cloudflare outage
//!
//! A provider is tried until the FIRST streamed token; a failure before that
//! (HTTP error, connect/first-token timeout) cascades to the next provider with
//! nothing emitted yet. Once tokens start flowing we commit to that provider — a
//! mid-stream failure finalizes the partial answer rather than restarting.
//!
//! The user can pin the brain to a single model (`BrainModel`), and picks a
//! separate vision model (`VisionModel`) that DESCRIBES a screenshot as text
//! (see [`ApiBackend::describe`]). Both share one streaming SSE reader
//! ([`stream_sse_content`]) and one request builder ([`chat_body`]).

use std::borrow::Cow;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use futures_util::StreamExt;
use secrecy::{ExposeSecret, SecretString};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter};

use crate::lang::Language;

/// Browser User-Agent. Groq sits behind Cloudflare, whose bot filter 403s
/// (error 1010) non-browser client signatures like the default reqwest UA.
const BROWSER_UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36";

const OPENAI_URL: &str = "https://api.openai.com/v1/chat/completions";
const GROQ_URL: &str = "https://api.groq.com/openai/v1/chat/completions";

const SYSTEM_PROMPT_EN: &str = "You are a real-time interview copilot. The user reads your \
answer at a glance while speaking, so make it scannable:\n\
- Start with ONE bold sentence (**...**) that directly answers the question — something the \
user can say out loud as-is.\n\
- Then the supporting detail as short bullets, not paragraphs. Under 4 sentences of prose \
total.\n\
- Coding question → give the code first, then 1-2 sentences of approach.\n\
- Conceptual question → include a code snippet ONLY if ~8 lines or fewer scan faster than \
prose would.\n\
- Markdown: always tag fenced code blocks with a language (```python); use inline code for \
identifiers.";
const SYSTEM_PROMPT_ES: &str = "Sos un copiloto de entrevistas en tiempo real. El usuario lee \
tu respuesta de reojo mientras habla, así que hacela escaneable:\n\
- Empezá con UNA oración en negrita (**...**) que responda directo la pregunta — algo que el \
usuario pueda decir en voz alta tal cual.\n\
- Después el detalle en bullets cortos, no párrafos. Máximo 4 oraciones de prosa en total.\n\
- Pregunta de código → primero el código, después 1-2 oraciones de enfoque.\n\
- Pregunta conceptual → incluí un snippet SOLO si ~8 líneas o menos se escanean más rápido \
que la prosa.\n\
- Markdown: siempre etiquetá los bloques de código con lenguaje (```python); usá inline code \
para identificadores.";

fn system_prompt(language: Language) -> &'static str {
    match language {
        Language::English => SYSTEM_PROMPT_EN,
        Language::Spanish => SYSTEM_PROMPT_ES,
    }
}

// ── Caveman style ───────────────────────────────────────────────────────────
//
// Terseness addendum appended to both answer prompts when the UI selects the
// caveman style. Distilled from the caveman-micro benchmark (an 85-token nudge
// beat the original 552-token skill with quality held at 100%): the model
// needs permission to drop filler, not a tutorial. The opening bold line is
// exempt — it must stay speakable out loud. Spanish keeps articles unless
// dropping them is unambiguous: articles carry more disambiguation in Spanish
// than in English.
const CAVEMAN_EN: &str = "\n\
CAVEMAN MODE — the opening bold sentence stays natural and speakable; everything after it is \
telegraphic:\n\
- Cut all filler, keep all technical substance.\n\
- Drop articles (a, an, the), filler words (just, really, basically, actually) and \
pleasantries (sure, certainly, happy to help).\n\
- No hedging. Fragments fine. Short synonyms (fix, not 'implement a solution for').\n\
- Technical terms exact. Code blocks unchanged.\n\
- Bullet pattern: [thing] [action] [reason].";
const CAVEMAN_ES: &str = "\n\
MODO CAVERNÍCOLA — la oración inicial en negrita queda natural y decible; todo lo que sigue, \
telegráfico:\n\
- Cortá todo el relleno, mantené toda la sustancia técnica.\n\
- Sin muletillas (básicamente, realmente, simplemente) ni cortesías (claro, por supuesto, \
encantado de ayudar).\n\
- Sin matizar de más. Fragmentos OK. Sinónimos cortos (arreglá, no 'implementá una solución \
para').\n\
- Mantené los artículos solo cuando quitarlos vuelva ambigua la frase.\n\
- Términos técnicos exactos. Bloques de código intactos.\n\
- Patrón por bullet: [cosa] [acción] [razón].";

fn caveman_suffix(language: Language) -> &'static str {
    match language {
        Language::English => CAVEMAN_EN,
        Language::Spanish => CAVEMAN_ES,
    }
}

/// `base` with the selected style applied: `Normal` borrows the prompt
/// unchanged, `Caveman` appends the terseness addendum.
fn styled_system(
    base: &'static str,
    language: Language,
    style: ResponseStyle,
) -> Cow<'static, str> {
    match style {
        ResponseStyle::Normal => Cow::Borrowed(base),
        ResponseStyle::Caveman => Cow::Owned(format!("{base}{}", caveman_suffix(language))),
    }
}

// The vision model transcribes/describes the screen as text; it must NOT solve
// anything (that's the brain's job) and must answer in the selected language.
// Code fidelity is non-negotiable: a paraphrased or elided statement poisons
// every answer built on it, so the prompt bans ellipsis/summarizing outright.
// Deliberately framed as a neutral TRANSCRIPTIONIST, not an "interview
// copilot": with multiple shots of a problem statement, the copilot framing
// sometimes pattern-matched as exam cheating and drew refusals ("I can't help
// with that"). Transcription itself is innocuous — keep the prompt about that.
// "The user's own screen" is stated (truthfully) because a full-monitor shot
// (tabs, logged-in avatars, taskbar) otherwise reads as surveilling someone
// else's screen — another refusal magnet.
const VISION_SYS_EN: &str = "You are a meticulous screen transcriptionist. The user is sharing \
screenshots of their own screen. Transcribe and describe, faithfully, everything visible: \
problem statements, code, diagrams, error messages, and UI text. Transcribe ALL visible code \
and prose VERBATIM and COMPLETE — never use ellipsis ('...'), never summarize, never skip \
lines. Reproduce code character-for-character as an exact code block. Describe non-text \
elements (diagrams, UI) concisely. Do NOT solve, answer, or comment on anything — only \
transcribe and describe. Write in ENGLISH. Plain text.";
const VISION_SYS_ES: &str = "Sos un transcriptor meticuloso de pantallas. El usuario te \
comparte capturas de su propia pantalla. Transcribí y describí, de forma fiel, todo lo \
visible: enunciados, código, diagramas, mensajes de error y texto de UI. Transcribí TODO el \
código y el texto visibles de forma TEXTUAL y COMPLETA — nunca uses puntos suspensivos \
('...'), nunca resumas, nunca saltees líneas. Reproducí el código carácter por carácter como \
un bloque de código exacto. Los elementos no textuales (diagramas, UI) describilos de forma \
concisa. NO resuelvas, respondas ni comentes nada — solamente transcribí y describí. Escribí \
en ESPAÑOL. Texto plano.";

fn vision_system(language: Language) -> &'static str {
    match language {
        Language::English => VISION_SYS_EN,
        Language::Spanish => VISION_SYS_ES,
    }
}

// ── Agent mode ──────────────────────────────────────────────────────────────
//
// The agent press sends [static system] + [full transcript] + [volatile tail]
// and lets the MODEL infer what help is needed — no question heuristics. Two
// prompt-design rules are borrowed from the harnesses that iterated on this
// the longest (Hermes, OpenClaw): the LAST transcript lines win over anything
// earlier ("latest signal wins"), and decisions already made in the interview
// are binding context, never to be re-litigated.

const AGENT_SYS_EN: &str = "You are a live interview copilot agent. You receive the full \
transcript so far of a job interview happening RIGHT NOW — lines labeled 'Interviewer:' \
(their voice), 'You:' (the candidate, the user you are helping), 'Screen:' (descriptions of \
the user's screen) and 'Clipboard:' (text the user captured) — followed by an INTERVIEW STATE \
summary maintained in the background.\n\
The user pressed the help hotkey. Work out what would help them most at this exact moment \
and provide it. Rules:\n\
- The LAST few transcript lines decide what is needed now; everything earlier is background. \
If the interviewer's latest question or challenge is not fully answered yet, that is the target.\n\
- Honor every decision already made in the interview (INTERVIEW STATE + transcript): build on \
them, never contradict or re-litigate them.\n\
- Stay consistent with everything the user has already claimed about their experience.\n\
- Don't re-cover ground already handled unless the interviewer is revisiting it.\n\
- Format by need: something to SAY out loud → under 4 sentences, natural and speakable. \
Coding → the code plus a 1-2 sentence approach. System design or open-ended discussion → a \
tight bullet skeleton the user can glance at while talking.\n\
- Whatever the need, start with ONE bold line (**...**) — the direct thing to say or do right \
now — then the detail. Always tag fenced code blocks with a language (```python).\n\
- Answer in the language the interview is currently conducted in.";
const AGENT_SYS_ES: &str = "Sos un agente copiloto de entrevistas en vivo. Recibís la \
transcripción completa hasta ahora de una entrevista de trabajo que está pasando AHORA MISMO \
— líneas etiquetadas 'Interviewer:' (la voz del entrevistador), 'You:' (el candidato, el \
usuario al que ayudás), 'Screen:' (descripciones de la pantalla del usuario) y 'Clipboard:' \
(texto que el usuario capturó) — seguida de un resumen INTERVIEW STATE mantenido en segundo \
plano.\n\
El usuario apretó el atajo de ayuda. Deducí qué es lo que más lo ayudaría en este momento \
exacto y dáselo. Reglas:\n\
- Las ÚLTIMAS líneas de la transcripción deciden qué hace falta ahora; todo lo anterior es \
contexto. Si la última pregunta o desafío del entrevistador todavía no está respondido del \
todo, ese es el objetivo.\n\
- Respetá cada decisión ya tomada en la entrevista (INTERVIEW STATE + transcripción): \
construí sobre ellas, nunca las contradigas ni las reabras.\n\
- Mantené consistencia con todo lo que el usuario ya afirmó sobre su experiencia.\n\
- No repitas terreno ya cubierto salvo que el entrevistador esté volviendo sobre él.\n\
- Formato según la necesidad: algo para DECIR en voz alta → menos de 4 oraciones, natural y \
hablable. Código → el código más 1-2 oraciones de enfoque. System design o discusión abierta \
→ un esqueleto de bullets conciso que el usuario pueda mirar de reojo mientras habla.\n\
- Sea cual sea la necesidad, empezá con UNA línea en negrita (**...**) — lo que hay que decir \
o hacer ahora mismo — y después el detalle. Siempre etiquetá los bloques de código con \
lenguaje (```python).\n\
- Respondé en el idioma en el que se está desarrollando la entrevista.";

fn agent_system(language: Language) -> &'static str {
    match language {
        Language::English => AGENT_SYS_EN,
        Language::Spanish => AGENT_SYS_ES,
    }
}

/// The volatile tail: Interview State + the interviewer's in-flight partial
/// (if any) + the press instruction. Rides at the END of the prompt where
/// attention is strongest; framed as REFERENCE ONLY with the live transcript
/// winning conflicts (the Hermes lesson — models otherwise answer questions
/// out of the summary instead of the live moment). The partial lives HERE and
/// never in the transcript: it's provisional text a final will supersede, and
/// the transcript must stay an append-only, byte-stable cache prefix.
fn agent_tail(state_block: &str, live_partial: Option<&str>, language: Language) -> String {
    let state_section = if state_block.is_empty() {
        String::new()
    } else {
        match language {
            Language::English => format!(
                "=== INTERVIEW STATE (background summary — reference only; it may lag the \
                 live transcript by a few minutes, and the transcript WINS on any conflict) ===\n\
                 {state_block}\n\
                 === END INTERVIEW STATE ===\n\n"
            ),
            Language::Spanish => format!(
                "=== INTERVIEW STATE (resumen de fondo — solo referencia; puede correr unos \
                 minutos detrás de la transcripción en vivo, y ante cualquier conflicto GANA \
                 la transcripción) ===\n\
                 {state_block}\n\
                 === FIN INTERVIEW STATE ===\n\n"
            ),
        }
    };
    let partial_section = live_partial.map_or_else(String::new, |p| match language {
        Language::English => format!(
            "=== INTERVIEWER, MID-SENTENCE RIGHT NOW (live partial transcription — may be \
             garbled and incomplete, a final will supersede it; if a question is forming \
             here, THAT question is what the user needs help with) ===\n\
             {p}\n\
             === END MID-SENTENCE ===\n\n"
        ),
        Language::Spanish => format!(
            "=== EL ENTREVISTADOR ESTÁ DICIENDO ESTO AHORA MISMO (transcripción parcial en \
             vivo — puede venir con errores o cortada, un final la va a reemplazar; si acá \
             se está formando una pregunta, ESA pregunta es con lo que el usuario necesita \
             ayuda) ===\n\
             {p}\n\
             === FIN PARCIAL ===\n\n"
        ),
    });
    let request = match language {
        Language::English => {
            "HELP REQUEST: the user pressed the help hotkey NOW. Based on the final \
             transcript lines above (the current moment), give the most useful help right now."
        }
        Language::Spanish => {
            "PEDIDO DE AYUDA: el usuario apretó el atajo de ayuda AHORA. Basándote en las \
             últimas líneas de la transcripción (el momento actual), dale la ayuda más útil \
             ahora mismo."
        }
    };
    format!("{state_section}{partial_section}{request}")
}

// The Interview State maintainer runs on the cheap sibling model in the
// background. Update-mode by design (OpenClaw/Hermes pattern): it PRESERVES
// the previous document and folds the delta in, instead of re-summarizing
// from scratch — repeated re-summarization is what loses the obscure details
// (a decision's rationale, an exact QPS figure) over a long session.
const STATE_MODEL: &str = "gpt-5.4-mini";
const STATE_SYS: &str = "You maintain the INTERVIEW STATE document for a live job interview, \
on behalf of the candidate. Given the previous document and the newest transcript lines \
(labeled Interviewer/You/Screen/Clipboard), output the UPDATED document — the document only, \
no preamble, no code fences.\n\
Use this EXACT format:\n\
## Active Question\n\
[The interviewer's most recent question or challenge that the candidate has not fully \
resolved, VERBATIM in its original language. A question just asked IS active. 'None' only if \
the latest exchange is fully resolved.]\n\
## Decisions Made\n\
[Numbered list. Every decision or agreement reached so far ('we'll go with X', chosen \
approach, agreed constraint) with a brief why. PRESERVE all previous entries and their \
numbering exactly; only append.]\n\
## Current Task Progress\n\
[The live coding/system-design problem, if any: requirements and constraints given, what's \
been covered, what remains.]\n\
## Questions Already Covered\n\
[One line each: questions the interviewer asked that were already answered.]\n\
## Candidate Claims\n\
[What the candidate has stated about themselves: experience, projects, numbers, \
technologies. Keep names and figures verbatim.]\n\
## Interviewer Signals\n\
[Name/role if known, evaluation criteria they stated, topics probed more than once, threads \
left open.]\n\
## Language\n\
[english | spanish | mixed]\n\
Rules: preserve ALL still-relevant information from the previous document (update, don't \
re-summarize); when an Active Question gets resolved, move it into 'Questions Already \
Covered'; keep technology names, numbers and identifiers exactly as said; stay under ~600 \
words total; be concrete. Never output anything but the document.";

// Several shots are described PER IMAGE ("Image N: …"), with scroll-stitching
// as the conditional case. The old wording asserted the shots were "consecutive
// scrolls of the SAME page" — with unrelated shots that false premise both drew
// refusals and made the model merge/drop the later images (measured: ~1650
// chars merged vs ~2000+ per-image on the same two screenshots).
fn vision_instruction(language: Language, shots: usize) -> &'static str {
    match language {
        Language::English => {
            if shots <= 1 {
                "Transcribe and describe this screenshot."
            } else {
                "Transcribe and describe each screenshot SEPARATELY and in order, under the \
                 heading 'Image N:'. If two consecutive screenshots are scrolls of the same \
                 page, continue the transcription without repeating the overlap. Transcribe \
                 ALL visible code COMPLETE — do not cut anything."
            }
        }
        Language::Spanish => {
            if shots <= 1 {
                "Transcribí y describí esta captura de pantalla."
            } else {
                "Transcribí y describí cada captura POR SEPARADO y en orden, con el encabezado \
                 'Imagen N:'. Si dos capturas consecutivas son scrolls de la misma página, \
                 continuá la transcripción sin repetir lo solapado. Todo el código visible: \
                 transcribilo COMPLETO, sin cortar nada."
            }
        }
    }
}

/// Budget to first streamed token (answer path). Exceeding it cascades to the next provider.
const FIRST_TOKEN_TIMEOUT: Duration = Duration::from_secs(4);
/// Max idle between chunks AFTER the first token (a mid-stream stall).
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(10);
/// Vision first-token budget. Reasoning models (gpt-5.x) can think a while before
/// the first visible token, so give more slack than the answer path.
const VISION_FIRST_TOKEN_TIMEOUT: Duration = Duration::from_secs(25);
/// Vision output ceiling. A long `LeetCode` statement + examples + code template
/// runs 1500-2500 tokens; the old 700 silently truncated it, breaking the
/// "transcribe verbatim and complete" contract.
const VISION_MAX_OUT: u32 = 2500;
/// Agent-press first-token budget. gpt-5.5 reasons before the first visible
/// token even at low effort; the ceiling also covers a cold prompt cache over
/// a long transcript.
const AGENT_FIRST_TOKEN_TIMEOUT: Duration = Duration::from_secs(25);
/// Visible-output ceiling for an agent answer (bullets or code).
const AGENT_MAX_OUT: u32 = 1200;
/// Interview State refresh: background call, latency uncritical.
const STATE_TIMEOUT: Duration = Duration::from_secs(45);
const STATE_MAX_OUT: u32 = 900;
/// Reasoning budget for the state refresher, ON TOP of `STATE_MAX_OUT`.
/// gpt-5.4-mini spends reasoning tokens from the same `max_completion_tokens`
/// budget BEFORE emitting any content; with the generic 1024 headroom a large
/// transcript delta ate the whole cap and returned an EMPTY message with
/// `finish_reason` "length" (HTTP 200) — measured 9× on 2026-07-23, killing
/// every refresh from minute 36 on. 4096 matches what the ecosystem converged
/// on for reasoning-mode summarizers.
const STATE_REASONING_HEADROOM: u32 = 4096;

/// Result of one API answer turn.
pub struct ApiOutcome {
    pub answer: String,
    pub ttft_ms: u64,
    pub total_ms: u64,
    /// Which provider actually answered, e.g. `groq/llama-3.1-8b-instant`.
    pub provider: String,
    /// Session-spend accounting. `OpenAI` reports it via `include_usage`,
    /// Groq unprompted under `x_groq`; zeroed if the stream broke early.
    pub usage: TokenUsage,
}

/// Token accounting from the final SSE chunk (`stream_options.include_usage`).
/// `cached` ⊆ `prompt`: the prefix `OpenAI` billed at ~10% — the live check
/// that the append-only transcript discipline is paying off.
#[derive(Clone, Copy, Default)]
pub struct TokenUsage {
    pub prompt: u64,
    pub cached: u64,
    pub completion: u64,
}

impl TokenUsage {
    /// Parse an `OpenAI`-style `usage` object (the chat/completions shape;
    /// Groq's `x_groq.usage` uses the same field names).
    fn from_json(u: &Value) -> Self {
        let field = |ptr: &str| u.pointer(ptr).and_then(Value::as_u64).unwrap_or(0);
        Self {
            prompt: field("/prompt_tokens"),
            cached: field("/prompt_tokens_details/cached_tokens"),
            completion: field("/completion_tokens"),
        }
    }
}

/// Result of one agent press.
pub struct AgentOutcome {
    pub answer: String,
    pub ttft_ms: u64,
    pub total_ms: u64,
    pub model: &'static str,
    pub usage: TokenUsage,
    /// `OpenAI`'s `x-request-id` response header — what their support asks for,
    /// and the disambiguator between retries when reading traces.
    pub request_id: Option<String>,
}

/// Extract `OpenAI`'s `x-request-id` header before the response body is consumed.
fn openai_request_id(resp: &reqwest::Response) -> Option<String> {
    resp.headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(String::from)
}

/// Result of one vision describe turn.
pub struct VisionOutcome {
    pub text: String,
    pub ttft_ms: u64,
    pub total_ms: u64,
    /// Session-spend accounting (includes the image-tile tokens, which only
    /// the server can price).
    pub usage: TokenUsage,
}

/// Streamed content delta emitted to the frontend (same payload for answer & vision;
/// only the event name differs).
#[derive(Clone, serde::Serialize)]
struct StreamDelta {
    trace_id: String,
    text: String,
}

struct Provider {
    /// Human label shown in metrics / logs.
    name: &'static str,
    url: &'static str,
    model: &'static str,
    /// Env/keyring name of the credential this provider uses.
    key_env: &'static str,
}

const OPENAI_GPT4O_MINI: Provider = Provider {
    name: "openai/gpt-4o-mini",
    url: OPENAI_URL,
    model: "gpt-4o-mini",
    key_env: "OPENAI_API_KEY",
};
const OPENAI_GPT55: Provider = Provider {
    name: "openai/gpt-5.5",
    url: OPENAI_URL,
    model: "gpt-5.5",
    key_env: "OPENAI_API_KEY",
};

/// The default brain: speed-first Groq chain, `OpenAI` as the outage backstop.
const CHAIN: &[Provider] = &[
    Provider {
        name: "groq/llama-3.1-8b-instant",
        url: GROQ_URL,
        model: "llama-3.1-8b-instant",
        key_env: "GROQ_API_KEY",
    },
    Provider {
        name: "groq/llama-3.3-70b-versatile",
        url: GROQ_URL,
        model: "llama-3.3-70b-versatile",
        key_env: "GROQ_API_KEY",
    },
    OPENAI_GPT4O_MINI,
];
const BRAIN_4O_MINI: &[Provider] = &[OPENAI_GPT4O_MINI];
const BRAIN_GPT55: &[Provider] = &[OPENAI_GPT55];

/// The model that DESCRIBES the screen (vision-capable). Selected in the UI.
#[derive(Clone, Copy, Debug, Default)]
pub enum VisionModel {
    #[default]
    Gpt4oMini,
    Gpt55,
}

impl VisionModel {
    #[must_use]
    pub fn model_id(self) -> &'static str {
        match self {
            Self::Gpt4oMini => "gpt-4o-mini",
            Self::Gpt55 => "gpt-5.5",
        }
    }

    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::Gpt4oMini => "gpt_4o_mini",
            Self::Gpt55 => "gpt_5_5",
        }
    }

    #[must_use]
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "gpt_4o_mini" | "gpt-4o-mini" => Some(Self::Gpt4oMini),
            "gpt_5_5" | "gpt-5.5" => Some(Self::Gpt55),
            _ => None,
        }
    }
}

/// The model that ANSWERS (later: a wired-in agent). Selected in the UI.
/// `Auto` keeps the historical Groq→OpenAI fallback chain.
#[derive(Clone, Copy, Debug, Default)]
pub enum BrainModel {
    #[default]
    Auto,
    Gpt4oMini,
    Gpt55,
}

impl BrainModel {
    fn providers(self) -> &'static [Provider] {
        match self {
            Self::Auto => CHAIN,
            Self::Gpt4oMini => BRAIN_4O_MINI,
            Self::Gpt55 => BRAIN_GPT55,
        }
    }

    /// The model an AGENT press resolves to: `Auto` (the speed-first
    /// auto-answer chain) upgrades to gpt-5.5; a pinned gpt-4o-mini is
    /// honored. Also the model whose context window the meter shows.
    #[must_use]
    pub fn agent_model_id(self) -> &'static str {
        match self {
            Self::Gpt4oMini => "gpt-4o-mini",
            Self::Auto | Self::Gpt55 => "gpt-5.5",
        }
    }

    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Gpt4oMini => "gpt_4o_mini",
            Self::Gpt55 => "gpt_5_5",
        }
    }

    #[must_use]
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "auto" => Some(Self::Auto),
            "gpt_4o_mini" | "gpt-4o-mini" => Some(Self::Gpt4oMini),
            "gpt_5_5" | "gpt-5.5" => Some(Self::Gpt55),
            _ => None,
        }
    }
}

/// Advertised context window (tokens) of an answering model — the context
/// meter's denominator. Unknown models get the conservative 128K floor.
#[must_use]
pub fn context_window(model: &str) -> u64 {
    match model {
        "gpt-5.5" => 1_050_000,
        // gpt-4o-mini and anything unrecognized.
        _ => 128_000,
    }
}

/// o200k token count of the fixed parts of the NEXT agent prompt (styled
/// system + volatile tail) — the pre-first-press share of the context meter
/// that per-line counts don't cover. Kept here because the prompt builders
/// are private to this module.
#[must_use]
pub fn agent_prompt_base_tokens(
    language: Language,
    style: ResponseStyle,
    state_block: &str,
) -> u64 {
    let system = styled_system(agent_system(language), language, style);
    crate::context_meter::count_tokens(&system)
        + crate::context_meter::count_tokens(&agent_tail(state_block, None, language))
}

/// How answers are WORDED. Selected in the UI. `Normal` keeps the prompts
/// exactly as they always were; `Caveman` appends the terseness addendum to
/// both answer system prompts and asks gpt-5* for low verbosity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResponseStyle {
    #[default]
    Normal,
    Caveman,
}

impl ResponseStyle {
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Caveman => "caveman",
        }
    }

    #[must_use]
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "normal" => Some(Self::Normal),
            "caveman" => Some(Self::Caveman),
            _ => None,
        }
    }
}

pub struct ApiBackend {
    client: reqwest::Client,
    groq: Option<SecretString>,
    openai: Option<SecretString>,
}

impl Default for ApiBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ApiBackend {
    #[must_use]
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .user_agent(BROWSER_UA)
            .build()
            .expect("failed to build reqwest client");
        let groq = crate::secrets::load_key("GROQ_API_KEY");
        let openai = crate::secrets::load_key("OPENAI_API_KEY");
        if groq.is_none() && openai.is_none() {
            tracing::error!(
                "API mode has NO usable keys (GROQ_API_KEY / OPENAI_API_KEY) — \
                 every request will fail. Set them before the interview."
            );
        } else {
            tracing::info!(
                groq_key = groq.is_some(),
                openai_key = openai.is_some(),
                "API backend initialized"
            );
        }
        Self {
            client,
            groq,
            openai,
        }
    }

    fn key_for(&self, key_env: &str) -> Option<&SecretString> {
        match key_env {
            "GROQ_API_KEY" => self.groq.as_ref(),
            "OPENAI_API_KEY" => self.openai.as_ref(),
            _ => None,
        }
    }

    /// Ask the selected brain, cascading on pre-first-token failure. Streams
    /// `answer-delta` events. Errors only if EVERY candidate provider fails.
    #[allow(clippy::too_many_arguments)]
    pub async fn ask(
        &self,
        question: &str,
        context: &str,
        language: Language,
        brain: BrainModel,
        style: ResponseStyle,
        trace_id: &str,
        app: &AppHandle,
    ) -> Result<ApiOutcome> {
        let prompt = build_user(question, context, language);
        let system = styled_system(system_prompt(language), language, style);
        let mut last_err = anyhow!("no providers available (no API keys?)");
        for p in brain.providers() {
            let Some(key) = self.key_for(p.key_env) else {
                tracing::warn!(provider = p.name, "skipping provider — no key");
                continue;
            };
            match self
                .try_provider(p, key, &system, &prompt, style, trace_id, app)
                .await
            {
                Ok(outcome) => {
                    tracing::info!(
                        provider = p.name,
                        ttft_ms = outcome.ttft_ms,
                        total_ms = outcome.total_ms,
                        "API turn answered"
                    );
                    return Ok(outcome);
                }
                Err(e) => {
                    tracing::warn!(provider = p.name, error = %e, "provider failed; cascading to next");
                    last_err = e;
                }
            }
        }
        Err(last_err)
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_provider(
        &self,
        p: &Provider,
        key: &SecretString,
        system: &str,
        prompt: &str,
        style: ResponseStyle,
        trace_id: &str,
        app: &AppHandle,
    ) -> Result<ApiOutcome> {
        let mut body = chat_body(p.model, system, json!(prompt), 500, 0.4);
        // Decoder-level brevity nudge; compounds with the caveman prompt.
        // gpt-5* only — other providers reject the unknown parameter.
        if style == ResponseStyle::Caveman && p.model.starts_with("gpt-5") {
            body["verbosity"] = json!("low");
        }
        let t0 = Instant::now();
        let resp = tokio::time::timeout(
            FIRST_TOKEN_TIMEOUT,
            self.client
                .post(p.url)
                .bearer_auth(key.expose_secret())
                .json(&body)
                .send(),
        )
        .await
        .map_err(|_| anyhow!("connect timed out"))??;

        let (answer, ttft_ms, total_ms, usage) =
            stream_sse_content(resp, "answer-delta", trace_id, app, t0, FIRST_TOKEN_TIMEOUT)
                .await?;
        Ok(ApiOutcome {
            answer,
            ttft_ms,
            total_ms,
            provider: p.name.to_string(),
            usage,
        })
    }

    /// Describe screenshots (base64 JPEGs, no `data:` prefix) as text, streaming
    /// `vision-delta` events. Several shots go in ONE multimodal message — they
    /// are consecutive scrolls of the same page, reconstructed in order. Vision
    /// always goes through `OpenAI` — the only vision-capable provider wired in.
    pub async fn describe(
        &self,
        imgs: &[String],
        model: VisionModel,
        language: Language,
        trace_id: &str,
        app: &AppHandle,
    ) -> Result<VisionOutcome> {
        let key = self
            .openai
            .as_ref()
            .ok_or_else(|| anyhow!("vision needs OPENAI_API_KEY (not set)"))?;
        if imgs.is_empty() {
            bail!("no screenshots to describe");
        }
        let mut parts = vec![json!({
            "type": "text",
            "text": vision_instruction(language, imgs.len()),
        })];
        for img in imgs {
            parts.push(json!({"type": "image_url", "image_url": {
                "url": format!("data:image/jpeg;base64,{img}"),
                "detail": "high",
            }}));
        }
        let content = Value::from(parts);
        let body = chat_body(
            model.model_id(),
            vision_system(language),
            content,
            VISION_MAX_OUT,
            0.2,
        );
        let t0 = Instant::now();
        let resp = tokio::time::timeout(
            VISION_FIRST_TOKEN_TIMEOUT,
            self.client
                .post(OPENAI_URL)
                .bearer_auth(key.expose_secret())
                .json(&body)
                .send(),
        )
        .await
        .map_err(|_| anyhow!("connect timed out"))??;

        let (text, ttft_ms, total_ms, usage) = stream_sse_content(
            resp,
            "vision-delta",
            trace_id,
            app,
            t0,
            VISION_FIRST_TOKEN_TIMEOUT,
        )
        .await?;
        // A refusal streams back as a normal, tiny completion — flag it loudly
        // so log analysis doesn't mistake it for a nearly-empty screen.
        if looks_like_refusal(&text) {
            tracing::warn!(chars = text.len(), text = %text, "vision describe looks like a REFUSAL");
        }
        tracing::info!(
            model = model.model_id(),
            shots = imgs.len(),
            ttft_ms,
            total_ms,
            chars = text.len(),
            "vision describe done"
        );
        Ok(VisionOutcome {
            text,
            ttft_ms,
            total_ms,
            usage,
        })
    }

    /// Agent press: answer from the FULL rolling transcript + Interview State,
    /// streaming `answer-delta` events. `OpenAI`-only — the agent wants a strong
    /// model, so `Auto` (the speed-first auto-answer chain) resolves to gpt-5.5;
    /// a pinned gpt-4o-mini is honored.
    #[allow(clippy::too_many_arguments)]
    pub async fn ask_agent(
        &self,
        transcript: &str,
        state_block: &str,
        live_partial: Option<&str>,
        language: Language,
        brain: BrainModel,
        style: ResponseStyle,
        trace_id: &str,
        app: &AppHandle,
    ) -> Result<AgentOutcome> {
        let key = self
            .openai
            .as_ref()
            .ok_or_else(|| anyhow!("agent mode needs OPENAI_API_KEY (not set)"))?;
        let model = brain.agent_model_id();
        let mut body = agent_body(
            model,
            &styled_system(agent_system(language), language, style),
            transcript,
            &agent_tail(state_block, live_partial, language),
        );
        // Decoder-level brevity nudge; compounds with the caveman prompt.
        if style == ResponseStyle::Caveman && model.starts_with("gpt-5") {
            body["verbosity"] = json!("low");
        }
        let t0 = Instant::now();
        let resp = tokio::time::timeout(
            AGENT_FIRST_TOKEN_TIMEOUT,
            self.client
                .post(OPENAI_URL)
                .bearer_auth(key.expose_secret())
                .json(&body)
                .send(),
        )
        .await
        .map_err(|_| anyhow!("connect timed out"))??;
        let request_id = openai_request_id(&resp);
        let (answer, ttft_ms, total_ms, usage) = stream_sse_content(
            resp,
            "answer-delta",
            trace_id,
            app,
            t0,
            AGENT_FIRST_TOKEN_TIMEOUT,
        )
        .await?;
        Ok(AgentOutcome {
            answer,
            ttft_ms,
            total_ms,
            model,
            usage,
            request_id,
        })
    }

    /// One Interview State update on the cheap sibling model (non-streaming,
    /// background). Returns `(new state document, x-request-id, usage)`.
    pub async fn refresh_interview_state(
        &self,
        prev_state: &str,
        delta: &str,
    ) -> Result<(String, Option<String>, TokenUsage)> {
        let key = self
            .openai
            .as_ref()
            .ok_or_else(|| anyhow!("state refresh needs OPENAI_API_KEY (not set)"))?;
        let prev = if prev_state.is_empty() {
            "(none yet — first update of this interview)"
        } else {
            prev_state
        };
        let user = format!("Previous state document:\n{prev}\n\nNew transcript lines:\n{delta}");
        // Reasoning models pay their thinking from `max_completion_tokens`
        // BEFORE any visible output: too little headroom returns an EMPTY
        // message with finish_reason "length" and HTTP 200. Give the refresher
        // real headroom, and on exactly that signature retry once with double.
        let mut headroom = STATE_REASONING_HEADROOM;
        loop {
            let mut body = chat_body(
                STATE_MODEL,
                STATE_SYS,
                json!(user.clone()),
                STATE_MAX_OUT,
                0.2,
            );
            body["stream"] = json!(false);
            // `stream_options` is only legal on streaming requests.
            if let Some(o) = body.as_object_mut() {
                o.remove("stream_options");
            }
            body["reasoning_effort"] = json!("low");
            body["max_completion_tokens"] = json!(STATE_MAX_OUT + headroom);
            let resp = tokio::time::timeout(
                STATE_TIMEOUT,
                self.client
                    .post(OPENAI_URL)
                    .bearer_auth(key.expose_secret())
                    .json(&body)
                    .send(),
            )
            .await
            .map_err(|_| anyhow!("state refresh timed out"))??;
            let request_id = openai_request_id(&resp);
            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                let snippet: String = body.chars().take(200).collect();
                bail!("HTTP {status}: {snippet}");
            }
            let v: Value = resp.json().await?;
            let text = v
                .pointer("/choices/0/message/content")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            let usage = v
                .get("usage")
                .map(TokenUsage::from_json)
                .unwrap_or_default();
            if !text.is_empty() {
                return Ok((text, request_id, usage));
            }
            let finish_reason = v
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let reasoning_tokens = v
                .pointer("/usage/completion_tokens_details/reasoning_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            if finish_reason == "length" && headroom == STATE_REASONING_HEADROOM {
                tracing::warn!(
                    target: "agent",
                    finish_reason = %finish_reason,
                    reasoning_tokens,
                    headroom,
                    request_id = request_id.as_deref().unwrap_or("-"),
                    "state refresh came back empty (reasoning ate the budget) — retrying with doubled headroom"
                );
                headroom *= 2;
                continue;
            }
            bail!(
                "state refresh returned empty content (finish_reason={finish_reason}, \
                 reasoning_tokens={reasoning_tokens}, headroom={headroom})"
            );
        }
    }

    /// Weigh-in ping (the manager's "⚖️ verify + warm" button): send the agent
    /// prompt EXACTLY as the next press would, non-streaming, output capped to
    /// a single token. Two payoffs for one request: the server's exact
    /// `usage.prompt_tokens` re-anchors the context meter (the number the user
    /// wants right after a 💉 context injection), and the prefill primes
    /// `OpenAI`'s prompt cache so the first real press starts from a warm
    /// prefix — lower TTFT, cached input rate. Costs one uncached read of the
    /// prompt; the cache warm only survives ~5-10 min of inactivity.
    pub async fn warm_agent(
        &self,
        transcript: &str,
        state_block: &str,
        language: Language,
        brain: BrainModel,
        style: ResponseStyle,
    ) -> Result<(TokenUsage, Option<String>)> {
        let key = self
            .openai
            .as_ref()
            .ok_or_else(|| anyhow!("agent mode needs OPENAI_API_KEY (not set)"))?;
        let model = brain.agent_model_id();
        let mut body = agent_body(
            model,
            &styled_system(agent_system(language), language, style),
            transcript,
            &agent_tail(state_block, None, language),
        );
        body["stream"] = json!(false);
        if let Some(o) = body.as_object_mut() {
            o.remove("stream_options");
        }
        // The answer is waste — only the usage receipt matters. 1 visible
        // token is the legal minimum, and the cap also stops gpt-5.5 from
        // spending reasoning tokens before the cutoff.
        if model.starts_with("gpt-5") {
            body["max_completion_tokens"] = json!(1);
        } else {
            body["max_tokens"] = json!(1);
        }
        let resp = tokio::time::timeout(
            // Latency-uncritical background call, but a cold prefill over a
            // long transcript takes a while — same budget as a state refresh.
            STATE_TIMEOUT,
            self.client
                .post(OPENAI_URL)
                .bearer_auth(key.expose_secret())
                .json(&body)
                .send(),
        )
        .await
        .map_err(|_| anyhow!("warm ping timed out"))??;
        let request_id = openai_request_id(&resp);
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let snippet: String = body.chars().take(200).collect();
            bail!("HTTP {status}: {snippet}");
        }
        let v: Value = resp.json().await?;
        let usage = v
            .get("usage")
            .map(TokenUsage::from_json)
            .unwrap_or_default();
        if usage.prompt == 0 {
            bail!("warm ping returned no usage");
        }
        Ok((usage, request_id))
    }
}

/// The agent request body. Three messages: [static system] + [transcript] +
/// [volatile tail]. The ORDER is load-bearing: system+transcript form a
/// stable, append-only prefix that `OpenAI`'s automatic prompt caching bills at
/// ~10% from the second press on (and cuts TTFT); only the tiny tail changes
/// per press. Never insert anything between system and transcript, and never
/// rewrite old transcript lines — either invalidates the whole cache.
fn agent_body(model: &str, system: &str, transcript: &str, tail: &str) -> Value {
    let mut body = json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": transcript},
            {"role": "user", "content": tail},
        ],
        "stream": true,
        // Final pre-[DONE] chunk carries token usage incl. cached_tokens —
        // the debug panel's context/cache meter.
        "stream_options": {"include_usage": true},
    });
    // Stable per-session cache key: routes consecutive presses (and the ⚖️
    // warm ping) to the same cache shard, instead of relying on OpenAI's
    // prefix-hash routing alone — which measured 0/15 hits on 40-59k-token
    // stable prefixes across two real interviews (2026-07-23).
    if let Some(sid) = crate::storage::current_session_id() {
        body["prompt_cache_key"] = json!(format!("gimme-agent-{sid}"));
    }
    if model.starts_with("gpt-5") {
        body["max_completion_tokens"] = json!(AGENT_MAX_OUT + 2048);
        // Latency over depth: the user is live in an interview. "low" keeps
        // gpt-5.5 fast while far above the auto-answer chain's quality.
        body["reasoning_effort"] = json!("low");
    } else {
        body["max_tokens"] = json!(AGENT_MAX_OUT);
        body["temperature"] = json!(0.4);
    }
    body
}

/// Heuristic: a tiny completion whose text apologizes is a refusal, not a
/// nearly-empty screen. Drives the WARN above and the gpt-5.5 retry in
/// `describe_queue`.
#[must_use]
pub fn looks_like_refusal(text: &str) -> bool {
    let lower = text.to_lowercase();
    text.len() < 200
        && ["no puedo", "lo siento", "i can't", "i cannot", "sorry"]
            .iter()
            .any(|m| lower.contains(m))
}

/// Build an `OpenAI`-style chat/completions body. Reasoning models (gpt-5.x) reject
/// `temperature` and cap output via `max_completion_tokens`; classic models use
/// `max_tokens` + `temperature`.
fn chat_body(
    model: &str,
    system: &str,
    user_content: Value,
    max_out: u32,
    temperature: f32,
) -> Value {
    let mut body = json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user"},
        ],
        "stream": true,
    });
    // OpenAI: the final pre-[DONE] chunk carries `usage` for the session
    // spend meter. NOT sent to Groq (the fallback chain must never break
    // over an extra param) — Groq reports usage unprompted under `x_groq`.
    if model.starts_with("gpt-") {
        body["stream_options"] = json!({"include_usage": true});
    }
    // Move the (possibly multimodal) user content in — consuming it.
    body["messages"][1]["content"] = user_content;
    if model.starts_with("gpt-5") {
        // Reasoning + visible tokens share this budget; leave headroom for reasoning.
        body["max_completion_tokens"] = json!(max_out + 1024);
    } else {
        body["max_tokens"] = json!(max_out);
        body["temperature"] = json!(temperature);
    }
    body
}

/// Drain an `OpenAI`-style SSE `chat/completions` stream, emitting each content
/// delta as `event_name` to the frontend. Returns `(full_text, ttft_ms, total_ms,
/// usage)` — usage stays zeroed unless the request asked for
/// `stream_options.include_usage` (the agent path does).
/// Errors before the first token so the caller can cascade; commits after it.
async fn stream_sse_content(
    resp: reqwest::Response,
    event_name: &'static str,
    trace_id: &str,
    app: &AppHandle,
    t0: Instant,
    first_token_timeout: Duration,
) -> Result<(String, u64, u64, TokenUsage)> {
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        let snippet: String = body.chars().take(200).collect();
        bail!("HTTP {status}: {snippet}");
    }

    let mut stream = resp.bytes_stream();
    let mut line_buf = String::new();
    let mut text = String::new();
    let mut ttft_ms = 0_u64;
    let mut got_first = false;
    let mut usage = TokenUsage::default();

    loop {
        let timeout = if got_first {
            STREAM_IDLE_TIMEOUT
        } else {
            first_token_timeout
        };
        let next = tokio::time::timeout(timeout, stream.next()).await;

        let chunk = match next {
            Ok(Some(Ok(bytes))) => bytes,
            Ok(Some(Err(e))) => {
                // Network error: keep what we have if streaming; else cascade.
                if got_first {
                    break;
                }
                return Err(anyhow!("stream error before first token: {e}"));
            }
            Ok(None) => break, // stream ended cleanly
            Err(_) => {
                if got_first {
                    break;
                }
                bail!("no first token within {first_token_timeout:?}");
            }
        };

        line_buf.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(nl) = line_buf.find('\n') {
            let line: String = line_buf.drain(..=nl).collect();
            let line = line.trim();
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                let total_ms = elapsed_ms(t0);
                return Ok((
                    text.trim().to_string(),
                    if ttft_ms == 0 { total_ms } else { ttft_ms },
                    total_ms,
                    usage,
                ));
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            // With include_usage, the final pre-[DONE] chunk carries usage and
            // empty choices; regular chunks carry `"usage": null` — skip those.
            // Groq streams its usage unprompted under `x_groq` instead (no
            // `stream_options` is sent to it — see `chat_body`).
            if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
                usage = TokenUsage::from_json(u);
            } else if let Some(u) = v.pointer("/x_groq/usage").filter(|u| u.is_object()) {
                usage = TokenUsage::from_json(u);
            }
            if let Some(delta) = v
                .pointer("/choices/0/delta/content")
                .and_then(Value::as_str)
            {
                if delta.is_empty() {
                    continue;
                }
                if !got_first {
                    got_first = true;
                    ttft_ms = elapsed_ms(t0);
                }
                text.push_str(delta);
                _ = app.emit(
                    event_name,
                    StreamDelta {
                        trace_id: trace_id.to_string(),
                        text: delta.to_string(),
                    },
                );
            }
        }
    }

    if !got_first {
        bail!("stream produced no content");
    }
    let total_ms = elapsed_ms(t0);
    Ok((text.trim().to_string(), ttft_ms, total_ms, usage))
}

fn build_user(question: &str, context: &str, language: Language) -> String {
    // The language directive lives in the system prompt (sent every turn here), so
    // an empty-context user prompt can stay bare — the model already knows to
    // answer in the chosen language.
    match language {
        Language::English => {
            if context.is_empty() {
                question.to_string()
            } else {
                format!(
                    "Interview context (recent transcription):\n\
                     ---\n\
                     {context}\n\
                     ---\n\n\
                     The interviewer just asked: \"{question}\"\n\n\
                     Give a concise, direct answer I can say out loud."
                )
            }
        }
        Language::Spanish => {
            if context.is_empty() {
                question.to_string()
            } else {
                format!(
                    "Contexto de la entrevista (transcripción reciente):\n\
                     ---\n\
                     {context}\n\
                     ---\n\n\
                     El entrevistador acaba de preguntar: \"{question}\"\n\n\
                     Dame una respuesta concisa y directa para leer en voz alta."
                )
            }
        }
    }
}

fn elapsed_ms(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_millis()).unwrap_or(u64::MAX)
}
