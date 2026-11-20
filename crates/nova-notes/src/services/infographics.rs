//! Static SVG infographics generated from a notebook (background job): the
//! Rust counterpart of `api/routers/infographics.py`,
//! `commands/infographic_commands.py` and `open_notebook/utils/infographic.py`.
//!
//! Styles and layouts are adapted from the `baoyu-infographic` skill in
//! JimLiu/baoyu-skills (MIT licence, (c) Jim Liu), rewritten for vector output.
//!
//! Safety model (ADR-017):
//!
//! - [`sanitize_svg`] parses the reply with `roxmltree` (no DTDs; only the five
//!   XML entities) and rebuilds it from an allow-list: drawing, text, gradient,
//!   pattern and filter elements only. Scripts, `foreignObject`, links,
//!   animation, images, event handlers and every reference that is not a local
//!   `#fragment` are dropped. The result can't run code or load anything.
//! - Effect caps clamp what could make a tab slow: blur and morphology radii,
//!   turbulence octaves, filter chain length and region, how many elements use
//!   a filter, and `<use>` of `<use>` (exponential expansion).
//! - The app shows it as an `<img>` and serves the file with a `sandbox` CSP
//!   ([`crate::services::notes::INFOGRAPHIC_CSP_HEADER`]).
//! - Source text is untrusted: instructions stay in the system message and
//!   sources are passed only inside delimited data blocks (shared with explainers).
//!
//! A note whose attachment is `image/svg+xml` is an infographic: uploads can't
//! be SVG, so only this generator makes them.

use std::collections::HashMap;
use std::sync::LazyLock;
use std::time::Instant;

use regex::{Captures, Regex};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::explainers::notebook_sources_block;
use super::generation_jobs::{
    self, char_prefix, clean_instructions, int_env, GeneratedNote, GenerationJobStarted,
    GenerationJobSummary, JobMessages, MAX_INSTRUCTIONS_CHARS, MAX_TITLE_CHARS,
};
use super::notes::SVG_MIME;
use crate::db::Json;
use crate::domain::{Note, Notebook};
use crate::error::{AppError, AppResult};
use crate::jobs::{JobQueue, RetryPolicy};
use crate::state::AppState;

/// The job's command name.
pub const COMMAND_NAME: &str = "generate_infographic";

pub const SVG_NS: &str = "http://www.w3.org/2000/svg";
pub const XLINK_NS: &str = "http://www.w3.org/1999/xlink";

/// Source text budget (chars, shared between sources). Override with
/// `OPEN_NOTEBOOK_INFOGRAPHIC_CONTEXT_CHARS`.
pub const DEFAULT_CONTEXT_CHARS: usize = 160_000;
/// Output budget. Override with `OPEN_NOTEBOOK_INFOGRAPHIC_MAX_TOKENS`.
pub const DEFAULT_MAX_TOKENS: usize = 32_000;
/// The note's text content (search, embeddings, chat context).
pub const MAX_NOTE_TEXT_CHARS: usize = 20_000;

pub const NO_SOURCES_MESSAGE: &str =
    "Add at least one source to this notebook before generating an infographic";
pub const WORKER_FAILED_MESSAGE: &str = "The background worker could not run the infographic job.";
pub const STALE_JOB_MESSAGE: &str =
    "The infographic job stopped unexpectedly or never started. Check that the \
     worker is running, then start a new infographic.";
const START_FAILED_MESSAGE: &str = "Failed to start the infographic";
const UNEXPECTED_MESSAGE: &str =
    "Infographic generation failed unexpectedly. Check the worker logs.";

const MESSAGES: JobMessages = JobMessages {
    worker_failed: WORKER_FAILED_MESSAGE,
    stale: STALE_JOB_MESSAGE,
};

/// Register the `generate_infographic` job. Not retried automatically (avoids
/// duplicate notes); the user starts a new one from the UI instead.
pub fn register_jobs(jobs: &JobQueue) {
    jobs.register(COMMAND_NAME, RetryPolicy::once(), |ctx, args| async move {
        let input: GenerateInfographicInput = serde_json::from_value(args)
            .map_err(|e| AppError::invalid(format!("Invalid {COMMAND_NAME} input: {e}")))?;
        Ok(generate_infographic_job(&ctx.state, &input).await)
    });
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// A visual style or layout: its id, display name and the prompt guidance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct InfographicOption {
    pub id: &'static str,
    pub name: &'static str,
    pub guidance: &'static str,
}

/// Visual styles that translate to vector art (baoyu's raster-only looks are left out).
pub const STYLES: &[InfographicOption] = &[
    InfographicOption { id: "clean-editorial", name: "Clean editorial", guidance: "Modern magazine infographic. Off-white background (#FAFAF7), near-black text (#1A1A1A), one strong accent (deep teal #0F766E) plus one warm secondary (#E07A2F). Generous whitespace, a strict column grid, thin rules between modules, large confident numerals for key figures. Sans-serif headings (bold, tight), clean sans-serif body." },
    InfographicOption { id: "craft-handmade", name: "Hand-drawn craft", guidance: "Hand-drawn paper-craft feel. Cream paper background (#FFF8F0) with warm pastel blocks like cut construction paper, soft offset drop shadows, slightly irregular shapes (draw with paths that wobble a little, not perfect rectangles), simple cartoon icons built from basic shapes. Rounded casual typography; keywords highlighted with marker-style underlines." },
    InfographicOption { id: "chalkboard", name: "Chalkboard", guidance: "Classroom chalkboard. Background #1C2B1C or #1A1A1A with a faint texture (feTurbulence noise at low opacity). Chalk-white text (#F5F5F5) and coloured-chalk accents: yellow #FFE566, pink #FF9999, blue #66B3FF, green #90EE90. Sketchy imperfect strokes, doodled arrows, underlines, circles and stars. No gradients or gloss." },
    InfographicOption { id: "technical-schematic", name: "Blueprint", guidance: "Engineering blueprint. Deep blue background (#1E3A5F) with a fine grid pattern, white and light-cyan line work, amber highlights (#F59E0B). Precise geometry, consistent stroke weights, dimension lines, callout leaders and small technical annotations. ALL-CAPS labels in a clean sans-serif or monospace." },
    InfographicOption { id: "subway-map", name: "Transit map", guidance: "Metro map. White or light grey background. Thick coloured route lines (red, blue, green, yellow, orange) at 45 and 90 degree angles only, round station markers with white fill, interchange rings where ideas connect, route badges with numbers. Clean sans-serif station labels." },
    InfographicOption { id: "corporate-memphis", name: "Flat vector", guidance: "Flat corporate illustration. Bright, friendly palette (violet #6C5CE7, coral #FF7675, yellow #FDCB6E, teal #00B894) on white. Flat shapes with no outlines, simple geometric people and objects, playful blobs and confetti shapes in the background. Bold rounded sans-serif headings." },
    InfographicOption { id: "cyberpunk-neon", name: "Neon", guidance: "Cyberpunk neon. Near-black background (#0B0B1A) with a faint grid. Glowing magenta (#FF2BD6), cyan (#00F0FF) and acid-yellow (#E6FF00) lines and text, glow via feGaussianBlur merged under the shape. Angular panels with cut corners, HUD-style brackets and scan lines. Monospace or condensed sans-serif labels." },
    InfographicOption { id: "aged-academia", name: "Vintage academic", guidance: "Victorian science plate. Sepia parchment background (#F2E6CF) with a subtle texture, ink-brown line work (#3E2C1C), muted red and olive-green accents. Fine hatching, ornamental rules and corner flourishes, numbered figure captions. Serif typography with small caps." },
    InfographicOption { id: "kawaii", name: "Kawaii", guidance: "Japanese cute. Soft pastel palette (pink #FFC8DD, lavender #CDB4DB, mint #BDE0FE, butter #FFF3B0) on a light background. Rounded bubbly shapes, small smiling faces on icons, sparkles and hearts as decoration, thick soft outlines. Round friendly typography." },
    InfographicOption { id: "pixel-art", name: "Pixel art", guidance: "Retro 8-bit game. Everything built from square pixels on an 8px grid (use rect elements and shape-rendering=crispEdges). Limited bright palette on a dark navy background, chunky pixel icons, game-style panels and health-bar meters for numbers. Monospace labels." },
    InfographicOption { id: "ikea-manual", name: "Instruction manual", guidance: "Flat-pack assembly manual. White background, black uniform-weight line art, no colour except one sparing accent. Numbered step circles, simple stick-figure people, arrows showing motion, minimal words." },
    InfographicOption { id: "bold-graphic", name: "Comic pop", guidance: "Comic-book pop art. Thick black outlines, primary colours (red #E63946, yellow #FFD60A, blue #1D4ED8), halftone dot patterns, starburst callouts and speech-bubble labels. Heavy condensed uppercase headings." },
    InfographicOption { id: "morandi-journal", name: "Journal doodle", guidance: "Bullet-journal page. Warm off-white background, muted Morandi tones (dusty rose #C9A9A6, sage #A3B18A, clay #B08968, slate #6B7B8C). Hand-drawn doodles, washi-tape strips on module corners, dotted dividers, small handwritten-style annotations." },
    InfographicOption { id: "ui-wireframe", name: "Wireframe", guidance: "Greyscale UI wireframe. Light grey background, white cards with 1px grey borders, placeholder-style boxes and crossed image frames, one blue accent for emphasis. Clean system sans-serif, small uppercase section labels." },
];
/// Information structures. `auto` lets the model pick from the material.
pub const LAYOUTS: &[InfographicOption] = &[
    InfographicOption { id: "auto", name: "Best fit", guidance: "Choose the layout below that best fits the material (bento grid when unsure) and follow it." },
    InfographicOption { id: "bento-grid", name: "Bento grid", guidance: "A grid of rectangular modules of mixed sizes. The most important idea gets the largest tile; each tile holds one idea with a heading, an icon or mini-chart and a line or two of text." },
    InfographicOption { id: "linear-progression", name: "Timeline / steps", guidance: "A single path of numbered steps or dated events running top to bottom (portrait) or left to right (landscape), with markers on the line and a short label and note at each." },
    InfographicOption { id: "binary-comparison", name: "A vs B", guidance: "Two halves side by side (or top and bottom) with contrasting colours, matching rows so each point lines up against its counterpart, and a verdict or summary strip at the bottom." },
    InfographicOption { id: "comparison-matrix", name: "Comparison table", guidance: "A table: options as columns, criteria as rows, with icons, ratings or short values in the cells and the strongest cell in each row highlighted." },
    InfographicOption { id: "hierarchical-layers", name: "Pyramid / layers", guidance: "Stacked layers (a pyramid or stack) from foundation to top, each layer labelled with a short description to the side." },
    InfographicOption { id: "tree-branching", name: "Tree", guidance: "A root concept branching into categories and sub-items, with clean orthogonal or curved connectors and aligned levels." },
    InfographicOption { id: "hub-spoke", name: "Hub and spoke", guidance: "A central concept in a prominent hub with 4-8 spokes radiating to nodes, each node with an icon, a label and a one-line description." },
    InfographicOption { id: "funnel", name: "Funnel", guidance: "A funnel narrowing through 3-6 stages, each stage labelled with its name and figure or description, widest at the top." },
    InfographicOption { id: "iceberg", name: "Iceberg", guidance: "An iceberg at a waterline: the visible, obvious points above water, the larger hidden causes or details below." },
    InfographicOption { id: "circular-flow", name: "Cycle", guidance: "4-8 stages arranged in a circle joined by arrows that show the cycle's direction, with the cycle's name in the centre." },
    InfographicOption { id: "venn-diagram", name: "Venn diagram", guidance: "Two or three overlapping translucent circles; what is unique to each in its own region and what is shared in the overlaps." },
    InfographicOption { id: "winding-roadmap", name: "Roadmap", guidance: "A winding road or path with milestone flags along it, from the start to the goal, each milestone with a short label." },
    InfographicOption { id: "dashboard", name: "Dashboard", guidance: "A row of big-number KPI cards at the top, then panels with simple charts (bars, donuts, sparklines) drawn in SVG from figures in the sources only." },
];

pub const DEFAULT_STYLE: &str = "clean-editorial";
pub const DEFAULT_LAYOUT: &str = "auto";

/// Canvas shape.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Aspect {
    #[default]
    Portrait,
    Landscape,
    Square,
}

impl Aspect {
    pub const ALL: [Aspect; 3] = [Aspect::Portrait, Aspect::Landscape, Aspect::Square];

    pub fn as_str(self) -> &'static str {
        match self {
            Aspect::Portrait => "portrait",
            Aspect::Landscape => "landscape",
            Aspect::Square => "square",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|a| a.as_str() == value)
    }

    /// Canvas size in SVG user units.
    pub fn size(self) -> (u32, u32) {
        match self {
            Aspect::Portrait => (1080, 1920),
            Aspect::Landscape => (1920, 1080),
            Aspect::Square => (1440, 1440),
        }
    }
}

pub fn style(id: &str) -> Option<&'static InfographicOption> {
    STYLES.iter().find(|s| s.id == id)
}

pub fn layout(id: &str) -> Option<&'static InfographicOption> {
    LAYOUTS.iter().find(|l| l.id == id)
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

fn default_style() -> String {
    DEFAULT_STYLE.into()
}

fn default_layout() -> String {
    DEFAULT_LAYOUT.into()
}

/// `POST /notebooks/{id}/infographics` body (`InfographicCreateRequest`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfographicCreateRequest {
    /// Visual style id ([`STYLES`]).
    #[serde(default = "default_style")]
    pub style: String,
    /// Layout id ([`LAYOUTS`]), or `auto` to let the model pick.
    #[serde(default = "default_layout")]
    pub layout: String,
    #[serde(default)]
    pub aspect: Aspect,
    /// Optional focus or angle for the reader, at most 2000 characters.
    #[serde(default, deserialize_with = "generation_jobs::instructions_field")]
    pub instructions: Option<String>,
}

impl Default for InfographicCreateRequest {
    fn default() -> Self {
        Self {
            style: default_style(),
            layout: default_layout(),
            aspect: Aspect::default(),
            instructions: None,
        }
    }
}

/// Start generating an infographic; the result appears as a note. Unknown
/// style or layout → 400, missing notebook → 404, no sources → 400.
pub async fn create_infographic(
    state: &AppState,
    notebook_id: &str,
    request: &InfographicCreateRequest,
) -> AppResult<GenerationJobStarted> {
    if style(&request.style).is_none() {
        return Err(AppError::invalid(format!(
            "Unknown infographic style: {}",
            request.style
        )));
    }
    if layout(&request.layout).is_none() {
        return Err(AppError::invalid(format!(
            "Unknown infographic layout: {}",
            request.layout
        )));
    }
    let no_sources = format!("{NO_SOURCES_MESSAGE}.");
    let notebook = generation_jobs::notebook_with_sources(state, notebook_id, &no_sources).await?;
    let args = json!({
        "notebook_id": notebook.id,
        "style": request.style,
        "layout": request.layout,
        "aspect": request.aspect,
        "instructions": clean_instructions(request.instructions.as_deref()),
        "requested_at": generation_jobs::requested_now(),
    });
    generation_jobs::submit_generation(
        state,
        COMMAND_NAME,
        &notebook.id,
        args,
        START_FAILED_MESSAGE,
    )
    .await
}

/// Infographic jobs for this notebook: every queued or running one, plus those
/// requested in the last 24 hours, newest first. A job queued or running for
/// more than 30 minutes is reported as failed. Missing notebook → 404.
pub async fn list_infographic_jobs(
    state: &AppState,
    notebook_id: &str,
) -> AppResult<Vec<GenerationJobSummary>> {
    let notebook = Notebook::get(state, notebook_id).await?;
    generation_jobs::list_generation_jobs(state, COMMAND_NAME, &notebook.id, MESSAGES).await
}

// ---------------------------------------------------------------------------
// The job
// ---------------------------------------------------------------------------

fn default_aspect() -> String {
    Aspect::default().as_str().into()
}

/// `generate_infographic` arguments (`GenerateInfographicInput`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateInfographicInput {
    pub notebook_id: String,
    #[serde(default = "default_style")]
    pub style: String,
    #[serde(default = "default_layout")]
    pub layout: String,
    #[serde(default = "default_aspect")]
    pub aspect: String,
    #[serde(default)]
    pub instructions: Option<String>,
    /// ISO time the API queued the job (the notes pane lists and ages jobs by it).
    #[serde(default)]
    pub requested_at: Option<String>,
}

/// Run one infographic job: the result JSON (`success`, `note_id`, `title`,
/// `error_message`, ...). Failures are reported, not raised.
pub async fn generate_infographic_job(state: &AppState, input: &GenerateInfographicInput) -> Json {
    let started = Instant::now();
    let outcome = run_infographic(state, input).await;
    if let Ok(note) = &outcome {
        tracing::info!("Infographic {} created for {}", note.id, input.notebook_id);
    }
    generation_jobs::generation_output(&input.notebook_id, started, outcome, UNEXPECTED_MESSAGE)
}

/// Draw the infographic and save it as a note in the notebook.
pub async fn run_infographic(
    state: &AppState,
    input: &GenerateInfographicInput,
) -> AppResult<Note> {
    let (Some(style), Some(layout), Some(aspect)) = (
        style(&input.style),
        layout(&input.layout),
        Aspect::parse(&input.aspect),
    ) else {
        return Err(AppError::invalid(format!(
            "Unknown infographic option: {}/{}/{}",
            input.style, input.layout, input.aspect
        )));
    };

    let notebook = Notebook::get(state, &input.notebook_id).await?;
    let block = notebook_sources_block(state, &notebook, context_chars()).await?;

    let (width, height) = aspect.size();
    let instructions = char_prefix(
        input.instructions.as_deref().unwrap_or("").trim(),
        MAX_INSTRUCTIONS_CHARS,
    );
    let system = state.prompts.render(
        "infographic/system",
        json!({
            "width": width,
            "height": height,
            "style_name": style.name,
            "style_guidance": style.guidance,
            "layout_name": layout.name,
            "layout_guidance": layout.guidance,
            "instructions": instructions,
        }),
    )?;
    let human = format!(
        "Notebook title: {}\nNotebook description: {}\n\n<sources>\n{block}\n</sources>\n\nDraw the complete SVG infographic now.",
        notebook.name,
        if notebook.description.is_empty() { "(none)" } else { &notebook.description },
    );

    let reply =
        generation_jobs::generate_reply(state, &system, &human, max_output_tokens()).await?;
    let svg = sanitize_svg(&extract_svg(&reply)?, aspect)?;

    let title = match svg_title(&svg) {
        t if t.is_empty() => format!("Infographic: {}", notebook.name),
        t => t,
    };
    let title = char_prefix(&title, MAX_TITLE_CHARS).to_string();
    let content = match svg_text(&svg) {
        t if t.is_empty() => title.clone(),
        t => t,
    };
    let generated = GeneratedNote {
        file_name: generation_jobs::attachment_file_name(&title, "infographic", ".svg"),
        title,
        content,
        explainer: None,
        ext: ".svg",
        mime: SVG_MIME,
        bytes: svg.into_bytes(),
    };
    generation_jobs::save_generated_note(state, &input.notebook_id, generated).await
}

/// Source text budget (characters).
pub fn context_chars() -> usize {
    int_env(
        "OPEN_NOTEBOOK_INFOGRAPHIC_CONTEXT_CHARS",
        DEFAULT_CONTEXT_CHARS,
    )
}

/// Output token budget.
pub fn max_output_tokens() -> usize {
    int_env("OPEN_NOTEBOOK_INFOGRAPHIC_MAX_TOKENS", DEFAULT_MAX_TOKENS)
}

// ---------------------------------------------------------------------------
// Extracting the SVG from a model reply
// ---------------------------------------------------------------------------

static FENCE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?is)```(?:svg|xml|html)?\s*\n(.*?)```").expect("regex"));
static SVG_START: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)<svg[\s>]").expect("regex"));

/// Pull the `<svg>…</svg>` document out of a model reply. A reply that stops
/// before `</svg>` was cut off by the output limit and is rejected
/// (`IncompleteGeneration`) rather than saved half-drawn.
pub fn extract_svg(text: &str) -> AppResult<String> {
    let mut candidate = text.trim();
    if let Some(inner) = FENCE
        .captures(candidate)
        .and_then(|c| c.get(1))
        .filter(|i| SVG_START.is_match(i.as_str()))
    {
        candidate = inner.as_str();
    }
    let Some(start) = SVG_START.find(candidate) else {
        return Err(AppError::IncompleteGeneration(
            "The model did not return an SVG image. Try again, or pick a stronger default chat model.".into(),
        ));
    };
    let candidate = &candidate[start.start()..];
    let Some(end) = candidate.to_ascii_lowercase().rfind("</svg>") else {
        return Err(AppError::IncompleteGeneration(
            "The infographic was cut off before it finished (the model hit its output limit; hidden reasoning \
             counts towards it). Try again, pick a simpler layout, or raise OPEN_NOTEBOOK_INFOGRAPHIC_MAX_TOKENS."
                .into(),
        ));
    };
    Ok(candidate[..end + "</svg>".len()].to_string())
}

// ---------------------------------------------------------------------------
// Sanitising
// ---------------------------------------------------------------------------

const ALLOWED_ELEMENTS: [&str; 43] = [
    "svg",
    "g",
    "defs",
    "title",
    "desc",
    "symbol",
    "use",
    "style",
    "rect",
    "circle",
    "ellipse",
    "line",
    "polyline",
    "polygon",
    "path",
    "text",
    "tspan",
    "textPath",
    "linearGradient",
    "radialGradient",
    "stop",
    "pattern",
    "clipPath",
    "mask",
    "marker",
    "filter",
    "feBlend",
    "feColorMatrix",
    "feComponentTransfer",
    "feComposite",
    "feDisplacementMap",
    "feDropShadow",
    "feFlood",
    "feFuncA",
    "feFuncB",
    "feFuncG",
    "feFuncR",
    "feGaussianBlur",
    "feMerge",
    "feMergeNode",
    "feMorphology",
    "feOffset",
    "feTurbulence",
];

fn is_allowed(name: &str) -> bool {
    ALLOWED_ELEMENTS.contains(&name)
}

// `url(` followed by anything but an (optionally quoted) local `#fragment`.
static URL_REF: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?i)url\(\s*(["']?#)?"#).expect("regex"));
// CSS the sanitiser refuses outright. Matching the literal text "url(" or
// "@import" is not enough: CSS allows escapes (\75rl( is url(, @\69mport is
// @import) and has URL forms that never contain "url(" (image-set with a
// string, image()). So: no backslash at all, no at-rules except @keyframes,
// and none of the non-url() image functions.
static CSS_AT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)@(keyframes\b)?").expect("regex"));
static CSS_DANGER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\\|expression\s*\(|javascript:|behavior\s*:|image-set|image\s*\(|cross-fade|element\s*\(|(?:^|[^\w-])src\s*\(",
    )
    .expect("regex")
});
static XML_DECL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\A\s*<\?xml[^>]*\?>").expect("regex"));
static DOCTYPE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?is)<!DOCTYPE[^\[>]*(?:\[.*?\])?\s*>").expect("regex"));
static AMP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"&(#\d+;|#x[0-9a-fA-F]+;|[A-Za-z][A-Za-z0-9]*;)?").expect("regex")
});
static NAMED_ENTITY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"&([A-Za-z][A-Za-z0-9]*);").expect("regex"));
static NUMBER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*[0-9.]+").expect("regex"));
static STYLE_FILTER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(?:^|;)\s*filter\s*:[^;]*").expect("regex"));

fn has_remote_url(text: &str) -> bool {
    URL_REF.captures_iter(text).any(|c| c.get(1).is_none())
}

fn css_danger(css: &str) -> bool {
    CSS_AT.captures_iter(css).any(|c| c.get(1).is_none()) || CSS_DANGER.is_match(css)
}

/// Parse model-written SVG safely and rebuild it from an allow-list.
///
/// Not well-formed SVG → `IncompleteGeneration`. Ensures `viewBox`, `width`
/// and `height` are set (an `<img>` and the PNG export need an intrinsic size).
pub fn sanitize_svg(svg: &str, aspect: Aspect) -> AppResult<String> {
    let text = XML_DECL.replace(svg, "");
    let text = DOCTYPE.replace_all(&text, "");
    let text = NAMED_ENTITY.replace_all(&text, |c: &Captures| entity_to_char(&c[1])); // &nbsp; etc. aren't XML
    let text = AMP.replace_all(&text, |c: &Captures| match c.get(1) {
        Some(_) => c[0].to_string(),
        None => "&amp;".to_string(),
    });
    let broken = || {
        AppError::IncompleteGeneration(
            "The model returned a broken SVG image. Try again, or pick a stronger default chat model.".into(),
        )
    };
    let document = roxmltree::Document::parse(&text).map_err(|_| broken())?;
    let root = document.root_element();
    if root.tag_name().name() != "svg" {
        return Err(AppError::IncompleteGeneration(
            "The model did not return an SVG image.".into(),
        ));
    }
    let mut tree = clean(root)?;
    tree.limit_effects()?;
    tree.ensure_size(aspect);
    Ok(tree.serialize())
}

fn entity_to_char(name: &str) -> String {
    if matches!(name, "amp" | "lt" | "gt" | "quot" | "apos") {
        return format!("&{name};");
    }
    match html_unescape_named(name) {
        Some(text) => text
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;"),
        None => format!("&amp;{name};"), // unknown entity: keep it as visible text
    }
}

/// Python's `html.unescape("&name;")`: the HTML5 entity, or the longest
/// legacy entity prefix followed by the rest; `None` when nothing matches.
fn html_unescape_named(name: &str) -> Option<String> {
    let lookup = |key: &str| -> Option<String> {
        let &(first, second) = html5ever::data::NAMED_ENTITIES.get(key)?;
        if first == 0 {
            return None; // a prefix of a longer entity, not an entity itself
        }
        let mut out: String = char::from_u32(first).into_iter().collect();
        if second != 0 {
            out.extend(char::from_u32(second));
        }
        Some(out)
    };
    let full = format!("{name};");
    if let Some(text) = lookup(&full) {
        return Some(text);
    }
    // Names are ASCII here, so byte slicing is safe.
    (2..full.len())
        .rev()
        .find_map(|cut| lookup(&full[..cut]).map(|text| format!("{text}{}", &full[cut..])))
}

/// Keep an attribute (possibly renamed) or drop it.
fn safe_attr(namespace: Option<&str>, name: &str, value: &str) -> Option<(String, String)> {
    if namespace.is_some_and(|ns| ns != SVG_NS && ns != XLINK_NS) {
        return None; // foreign namespaces (sodipodi, inkscape, xml:base, ...)
    }
    let lowered = name.to_lowercase();
    if lowered.starts_with("on") {
        return None;
    }
    if lowered == "href" {
        // Local references only (<use href="#icon">, gradients, textPath).
        if !value.trim().starts_with('#') {
            return None;
        }
        let key = if namespace == Some(XLINK_NS) {
            "xlink:href"
        } else {
            "href"
        };
        return Some((key.to_string(), value.to_string()));
    }
    if has_remote_url(value) || value.to_lowercase().contains("javascript:") {
        return None;
    }
    if lowered == "style" && css_danger(value) {
        return None;
    }
    Some((name.to_string(), value.to_string()))
}

/// One kept element. Children are other elements (by index) and text.
#[derive(Debug, Clone)]
struct Element {
    name: String,
    attrs: Vec<(String, String)>,
    children: Vec<Child>,
}

#[derive(Debug, Clone)]
enum Child {
    Element(usize),
    Text(String),
}

/// The sanitised image: an arena of elements, the root at index 0.
struct SvgTree {
    nodes: Vec<Element>,
}

/// Rebuild the parsed document from the allow-list (iteratively, so deep
/// nesting can't exhaust the stack). More than [`MAX_ELEMENTS`] kept
/// elements → `IncompleteGeneration`.
fn clean(root: roxmltree::Node<'_, '_>) -> AppResult<SvgTree> {
    let mut tree = SvgTree { nodes: Vec::new() };
    // (node, parent index); the root has no parent.
    let mut stack: Vec<(roxmltree::Node<'_, '_>, Option<usize>)> = vec![(root, None)];
    while let Some((node, parent)) = stack.pop() {
        if node.is_text() {
            if let (Some(parent), Some(text)) = (parent, node.text()) {
                if !text.is_empty() {
                    tree.nodes[parent]
                        .children
                        .push(Child::Text(text.to_string()));
                }
            }
            continue;
        }
        if !node.is_element() || !is_allowed(node.tag_name().name()) {
            continue; // comments, processing instructions and dropped elements (with their content)
        }
        if tree.nodes.len() >= MAX_ELEMENTS {
            return Err(too_complex());
        }
        let name = node.tag_name().name().to_string();
        let mut element = Element {
            name,
            attrs: Vec::new(),
            children: Vec::new(),
        };
        for attribute in node.attributes() {
            if let Some((key, value)) =
                safe_attr(attribute.namespace(), attribute.name(), attribute.value())
            {
                set_attr(&mut element.attrs, &key, value);
            }
        }
        let index = tree.nodes.len();
        let is_style = element.name == "style";
        if is_style {
            // Only the text before the first child element, as ElementTree's `.text`.
            let css: String = node
                .children()
                .take_while(|c| !c.is_element())
                .filter_map(|c| c.text())
                .collect();
            if !css.is_empty() && !css_danger(&css) && !has_remote_url(&css) {
                element.children.push(Child::Text(css));
            }
        }
        tree.nodes.push(element);
        if let Some(parent) = parent {
            tree.nodes[parent].children.push(Child::Element(index));
        }
        if !is_style {
            let children: Vec<_> = node.children().collect();
            stack.extend(children.into_iter().rev().map(|child| (child, Some(index))));
        }
    }
    Ok(tree)
}

fn set_attr(attrs: &mut Vec<(String, String)>, key: &str, value: String) {
    match attrs.iter_mut().find(|(k, _)| k == key) {
        Some(slot) => slot.1 = value,
        None => attrs.push((key.to_string(), value)),
    }
}

fn too_complex() -> AppError {
    AppError::IncompleteGeneration(
        "The infographic was too complex to show safely. Try again, or pick a simpler layout."
            .into(),
    )
}

// --- Effect caps --------------------------------------------------------------
// Filters and <use> are allowed for the look, but a few values make a browser
// tab crawl (huge blurs, deep turbulence, long filter chains on hundreds of
// elements, <use> of <use> expanding exponentially). Clamp them so a bad or
// hostile reply can at worst look plainer, never hang the viewer.

pub const MAX_ELEMENTS: usize = 8_000;
/// `stdDeviation`, user units.
pub const MAX_BLUR: f64 = 20.0;
pub const MAX_MORPHOLOGY_RADIUS: f64 = 8.0;
pub const MAX_TURBULENCE_OCTAVES: f64 = 4.0;
/// Per `<filter>`.
pub const MAX_FILTER_PRIMITIVES: usize = 8;
/// Elements carrying a filter attribute or style.
pub const MAX_FILTERED_ELEMENTS: usize = 40;
pub const MAX_USES: usize = 500;
/// Filter region bounds, as fractions of the bounding box.
pub const MAX_FILTER_OFFSET: f64 = 1.0;
pub const MAX_FILTER_EXTENT: f64 = 3.0;

impl SvgTree {
    fn get(&self, index: usize, key: &str) -> Option<&str> {
        self.nodes[index]
            .attrs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    fn set(&mut self, index: usize, key: &str, value: String) {
        set_attr(&mut self.nodes[index].attrs, key, value);
    }

    fn remove(&mut self, index: usize, key: &str) {
        self.nodes[index].attrs.retain(|(k, _)| k != key);
    }

    fn element_children(&self, index: usize) -> Vec<usize> {
        self.nodes[index]
            .children
            .iter()
            .filter_map(|c| match c {
                Child::Element(i) => Some(*i),
                Child::Text(_) => None,
            })
            .collect()
    }

    fn detach(&mut self, parent: usize, child: usize) {
        if let Some(at) = self.nodes[parent]
            .children
            .iter()
            .position(|c| matches!(c, Child::Element(i) if *i == child))
        {
            self.nodes[parent].children.remove(at);
        }
    }

    /// The element and its current descendants, in document order.
    fn iter(&self, index: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack = vec![index];
        while let Some(i) = stack.pop() {
            out.push(i);
            stack.extend(self.element_children(i).into_iter().rev());
        }
        out
    }

    fn limit_effects(&mut self) -> AppResult<()> {
        let elements = self.iter(0);
        if elements.len() > MAX_ELEMENTS {
            return Err(too_complex());
        }
        let mut by_id: HashMap<String, usize> = HashMap::new();
        for &el in &elements {
            if let Some(id) = self.get(el, "id").filter(|id| !id.is_empty()) {
                by_id.insert(id.to_string(), el);
            }
        }

        let mut filtered = 0;
        for &el in &elements {
            match self.nodes[el].name.as_str() {
                "filter" => self.limit_filter(el),
                "feGaussianBlur" | "feDropShadow" => {
                    self.clamp_numbers(el, "stdDeviation", MAX_BLUR, false)
                }
                "feMorphology" => self.clamp_numbers(el, "radius", MAX_MORPHOLOGY_RADIUS, false),
                "feTurbulence" => {
                    self.clamp_numbers(el, "numOctaves", MAX_TURBULENCE_OCTAVES, true)
                }
                _ => {}
            }
            let style = self
                .get(el, "style")
                .filter(|s| !s.is_empty())
                .map(String::from);
            if self.get(el, "filter").is_some()
                || style.as_deref().is_some_and(|s| STYLE_FILTER.is_match(s))
            {
                filtered += 1;
                if filtered > MAX_FILTERED_ELEMENTS {
                    self.remove(el, "filter");
                    if let Some(style) = style {
                        let stripped = STYLE_FILTER.replace_all(&style, "");
                        self.set(
                            el,
                            "style",
                            stripped.trim_start_matches([';', ' ']).to_string(),
                        );
                    }
                }
            }
        }

        let mut uses = 0;
        for &parent in &elements {
            for child in self.element_children(parent) {
                if self.nodes[child].name != "use" {
                    continue;
                }
                uses += 1;
                let href = self
                    .get(child, "href")
                    .filter(|h| !h.is_empty())
                    .or_else(|| self.get(child, "xlink:href"));
                let target = href
                    .and_then(|h| by_id.get(h.trim_start_matches('#')))
                    .copied();
                let nested = target
                    .is_some_and(|t| self.iter(t).iter().any(|&d| self.nodes[d].name == "use"));
                if uses > MAX_USES || nested {
                    self.detach(parent, child);
                }
            }
        }
        Ok(())
    }

    fn limit_filter(&mut self, el: usize) {
        let primitives: Vec<usize> = self
            .element_children(el)
            .into_iter()
            .filter(|&c| self.nodes[c].name.starts_with("fe"))
            .collect();
        for &extra in primitives.iter().skip(MAX_FILTER_PRIMITIVES) {
            self.detach(el, extra);
        }
        if self.get(el, "filterUnits") == Some("userSpaceOnUse") {
            // Absolute regions: fall back to the default bounding-box region.
            for attr in ["filterUnits", "x", "y", "width", "height"] {
                self.remove(el, attr);
            }
            return;
        }
        for (attr, low, high) in [
            ("x", -MAX_FILTER_OFFSET, MAX_FILTER_OFFSET),
            ("y", -MAX_FILTER_OFFSET, MAX_FILTER_OFFSET),
            ("width", 0.0, MAX_FILTER_EXTENT),
            ("height", 0.0, MAX_FILTER_EXTENT),
        ] {
            let raw = self.get(el, attr).unwrap_or("").trim().to_string();
            if raw.is_empty() {
                continue;
            }
            let parsed = match raw.strip_suffix('%') {
                Some(percent) => percent.trim().parse::<f64>().map(|v| v / 100.0),
                None => raw.parse::<f64>(),
            };
            let Ok(value) = parsed else {
                self.remove(el, attr);
                continue;
            };
            let clamped = value.max(low).min(high);
            if clamped != value {
                self.set(el, attr, format!("{}%", format_g(clamped * 100.0)));
            }
        }
    }

    fn clamp_numbers(&mut self, el: usize, attr: &str, limit: f64, integer: bool) {
        let Some(raw) = self.get(el, attr).map(|r| r.replace(',', " ")) else {
            return;
        };
        let mut out = Vec::new();
        for part in raw.split_whitespace() {
            let Ok(value) = part.parse::<f64>() else {
                self.remove(el, attr);
                return;
            };
            let value = value.max(0.0).min(limit);
            out.push(if integer {
                (value.trunc() as i64).to_string()
            } else {
                format_g(value)
            });
        }
        self.set(el, attr, out.join(" "));
    }

    fn ensure_size(&mut self, aspect: Aspect) {
        let (default_w, default_h) = aspect.size();
        let (mut width, mut height) = (i64::from(default_w), i64::from(default_h));
        let view_box = self.get(0, "viewBox").unwrap_or("").replace(',', " ");
        let parts: Vec<&str> = view_box.split_whitespace().collect();
        let mut valid = parts.len() == 4;
        if valid {
            match (parts[2].parse::<f64>(), parts[3].parse::<f64>()) {
                (Ok(w), Ok(h)) if w.is_finite() && h.is_finite() => {
                    width = w.trunc() as i64;
                    height = h.trunc() as i64;
                }
                _ => valid = false,
            }
        }
        if !valid || width <= 0 || height <= 0 {
            width = i64::from(default_w);
            height = i64::from(default_h);
            self.set(0, "viewBox", format!("0 0 {width} {height}"));
        }
        for (attr, size) in [("width", width), ("height", height)] {
            let current = self.get(0, attr).unwrap_or("");
            if !NUMBER.is_match(current) || current.trim().ends_with('%') {
                self.set(0, attr, size.to_string());
            }
        }
    }

    /// Serialise as ElementTree does with the SVG namespace as default:
    /// namespace declarations on the root, `<tag />` for empty elements.
    fn serialize(&self) -> String {
        // Only elements still in the tree count (detached ones are not written).
        let uses_xlink = self.iter(0).into_iter().any(|i| {
            self.nodes[i]
                .attrs
                .iter()
                .any(|(k, _)| k.starts_with("xlink:"))
        });
        enum Step<'a> {
            Open(usize),
            Close(&'a str),
            Text(&'a str),
        }
        let mut out = String::new();
        let mut stack = vec![Step::Open(0)];
        while let Some(step) = stack.pop() {
            match step {
                Step::Text(text) => escape_text(text, &mut out),
                Step::Close(name) => {
                    out.push_str("</");
                    out.push_str(name);
                    out.push('>');
                }
                Step::Open(index) => {
                    let node = &self.nodes[index];
                    out.push('<');
                    out.push_str(&node.name);
                    if index == 0 {
                        out.push_str(&format!(" xmlns=\"{SVG_NS}\""));
                        if uses_xlink {
                            out.push_str(&format!(" xmlns:xlink=\"{XLINK_NS}\""));
                        }
                    }
                    for (key, value) in &node.attrs {
                        out.push(' ');
                        out.push_str(key);
                        out.push_str("=\"");
                        escape_attr(value, &mut out);
                        out.push('"');
                    }
                    if node.children.is_empty() {
                        out.push_str(" />");
                        continue;
                    }
                    out.push('>');
                    stack.push(Step::Close(&node.name));
                    for child in node.children.iter().rev() {
                        stack.push(match child {
                            Child::Element(i) => Step::Open(*i),
                            Child::Text(text) => Step::Text(text),
                        });
                    }
                }
            }
        }
        out
    }
}

fn escape_text(text: &str, out: &mut String) {
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            c => out.push(c),
        }
    }
}

fn escape_attr(text: &str, out: &mut String) {
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\n' => out.push_str("&#10;"),
            '\r' => out.push_str("&#13;"),
            '\t' => out.push_str("&#09;"),
            c => out.push(c),
        }
    }
}

/// Python's `format(value, "g")`: 6 significant digits, trailing zeros
/// dropped, exponent form outside 1e-4 ..< 1e6.
fn format_g(value: f64) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0".into()
        } else {
            "0".into()
        };
    }
    if !value.is_finite() {
        return if value.is_nan() {
            "nan".into()
        } else if value > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    let sci = format!("{value:.5e}");
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let trim = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if (-4..6).contains(&exponent) {
        let decimals = usize::try_from(5 - exponent).unwrap_or(0);
        trim(&format!("{value:.decimals$}"))
    } else {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mantissa), exponent.abs())
    }
}

// --- Reading it back ----------------------------------------------------------

fn words(node: roxmltree::Node<'_, '_>) -> String {
    let joined: Vec<&str> = node
        .descendants()
        .filter(|n| n.is_text())
        .filter_map(|n| n.text())
        .collect();
    joined
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Text of the root `<title>`, if any (`""` otherwise).
pub fn svg_title(svg: &str) -> String {
    let Ok(document) = roxmltree::Document::parse(svg) else {
        return String::new();
    };
    document
        .root_element()
        .children()
        .find(|c| c.is_element() && c.tag_name().name() == "title")
        .map(words)
        .unwrap_or_default()
}

/// The poster's words, one `title` / `desc` / `text` element per line, for
/// the note body (search, embeddings, chat context).
pub fn svg_text(svg: &str) -> String {
    let Ok(document) = roxmltree::Document::parse(svg) else {
        return String::new();
    };
    let mut lines: Vec<String> = Vec::new();
    for node in document.root_element().descendants() {
        if !node.is_element() || !matches!(node.tag_name().name(), "title" | "desc" | "text") {
            continue;
        }
        // Each <tspan> is usually its own visual line: keep a space between them.
        let line = words(node);
        if !line.is_empty() && lines.last() != Some(&line) {
            lines.push(line);
        }
    }
    char_prefix(&lines.join("\n"), MAX_NOTE_TEXT_CHARS).to_string()
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::json;

    use super::*;
    use crate::services::explainers::tests::{notebook_with_source, use_models};
    use crate::testing::TestApp;

    const SVG: &str = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1080 1920\">\
        <title>Tides &amp; the moon</title>\
        <defs><linearGradient id=\"g\"><stop offset=\"0\" stop-color=\"#fff\"/></linearGradient></defs>\
        <rect width=\"1080\" height=\"1920\" fill=\"url(#g)\"/>\
        <text x=\"48\" y=\"120\" font-size=\"64\">Tides<tspan x=\"48\" dy=\"70\">explained</tspan></text>\
        <text x=\"48\" y=\"400\">The moon pulls the sea.</text>\
        </svg>";

    fn sanitize(svg: &str) -> String {
        sanitize_svg(svg, Aspect::Portrait).unwrap()
    }

    fn svg_with(inner: &str, root_attrs: &str) -> String {
        format!("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\" {root_attrs}>{inner}<text>kept</text></svg>")
    }

    fn svg(body: &str) -> String {
        format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" viewBox=\"0 0 100 100\">{body}</svg>"
        )
    }

    #[test]
    fn extract_svg_accepts_bare_fenced_and_chatty_replies() {
        assert_eq!(extract_svg(SVG).unwrap(), SVG);
        assert_eq!(extract_svg(&format!("```svg\n{SVG}\n```")).unwrap(), SVG);
        assert_eq!(
            extract_svg(&format!("Here you go:\n\n{SVG}\n\nEnjoy!")).unwrap(),
            SVG
        );
    }

    #[test]
    fn extract_svg_rejects_missing_or_cut_off_images() {
        for reply in ["", "No image here", "```svg\n<svg><rect/>"] {
            assert!(
                matches!(extract_svg(reply), Err(AppError::IncompleteGeneration(_))),
                "{reply:?}"
            );
        }
    }

    #[test]
    fn sanitize_keeps_drawing_and_local_references() {
        let clean = sanitize(SVG);
        assert!(
            clean
                .starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1080 1920\""),
            "{clean}"
        );
        assert!(
            clean.contains("fill=\"url(#g)\"") && clean.contains("<linearGradient"),
            "{clean}"
        );
        assert!(clean.contains("explained</tspan>"));
        assert!(clean.contains("width=\"1080\"") && clean.contains("height=\"1920\""));
        assert!(
            clean.contains("<stop offset=\"0\" stop-color=\"#fff\" />"),
            "{clean}"
        );
        // Sanitising is stable.
        assert_eq!(sanitize(&clean), clean);
    }

    #[test]
    fn sanitize_strips_everything_that_could_run_or_load() {
        let evil = "<svg xmlns=\"http://www.w3.org/2000/svg\" \
            xmlns:xlink=\"http://www.w3.org/1999/xlink\" viewBox=\"0 0 100 100\" onload=\"alert(1)\">\
            <script>alert(2)</script>\
            <foreignObject><div xmlns=\"http://www.w3.org/1999/xhtml\">x</div></foreignObject>\
            <a href=\"https://evil.example\"><text>link</text></a>\
            <image href=\"https://evil.example/p.png\"/>\
            <use xlink:href=\"https://evil.example/s.svg#x\"/>\
            <use href=\"#ok\"/>\
            <animate attributeName=\"href\" to=\"javascript:alert(3)\"/>\
            <rect fill=\"url(https://evil.example/t)\" style=\"fill:red\"/>\
            <circle style=\"background:url(https://evil.example)\"/>\
            <style>@import url(https://evil.example/x.css);</style>\
            <style>.a{fill:url(#g)}</style>\
            </svg>";
        let clean = sanitize(evil);
        let lowered = clean.to_lowercase();
        for bad in [
            "script",
            "alert",
            "foreignobject",
            "evil.example",
            "<a ",
            "<image",
            "onload",
            "animate",
        ] {
            assert!(!lowered.contains(bad), "{bad}: {clean}");
        }
        assert!(
            clean.contains("href=\"#ok\"") && clean.contains(".a{fill:url(#g)}"),
            "{clean}"
        );
        assert!(clean.contains("style=\"fill:red\""));
        assert!(
            clean.contains("<style />"),
            "dangerous CSS leaves an empty style: {clean}"
        );
    }

    #[test]
    fn sanitize_strips_css_that_loads_by_escape_or_string() {
        let hostile = [
            svg_with("<style>@\\69mport \"https://evil.example/a\";</style>", ""),
            svg_with(
                "<style>svg{background-image:\\75rl(https://evil.example/b)}</style>",
                "",
            ),
            svg_with(
                "",
                "style=\"background-image:\\75rl(https://evil.example/c)\"",
            ),
            svg_with(
                "",
                "style=\"background-image:image-set('https://evil.example/d' 1x)\"",
            ),
            svg_with(
                "",
                "style=\"background-image:-webkit-image-set('https://evil.example/e' 1x)\"",
            ),
            svg_with(
                "",
                "style=\"background-image:image('https://evil.example/f')\"",
            ),
            svg_with(
                "<style>@font-face{font-family:x;src:\\75rl(https://evil.example/g)}</style>",
                "",
            ),
            svg_with(
                "<style>@font-face{font-family:x;src:url(https://evil.example/h)}</style>",
                "",
            ),
            svg_with("<style>@namespace url(https://evil.example/i);</style>", ""),
            svg_with(
                "<style>.x{background:src(\"https://evil.example/j\")}</style>",
                "",
            ),
        ];
        for markup in hostile {
            let clean = sanitize(&markup);
            assert!(!clean.contains("evil.example"), "{clean}");
            assert!(!clean.contains('\\'), "{clean}");
            assert!(clean.contains("<text>kept</text>"), "{clean}");
        }
    }

    #[test]
    fn sanitize_keeps_ordinary_styling() {
        let markup = svg_with(
            "<style>.t{fill:#0f766e;font-family:Inter,'Segoe UI',Helvetica,sans-serif;\
             font-weight:700}.b{stroke:#111;stroke-width:2}\
             @keyframes p{from{opacity:0}to{opacity:1}}</style>\
             <defs><linearGradient id=\"g\"><stop offset=\"0\" stop-color=\"#fff\"/>\
             </linearGradient><filter id=\"f\"><feGaussianBlur stdDeviation=\"2\"/></filter></defs>\
             <rect class=\"b\" fill=\"url(#g)\" filter=\"url(#f)\" width=\"5\" height=\"5\"/>\
             <rect style=\"fill:#e07a2f;opacity:.8;stroke:none\" width=\"5\" height=\"5\"/>",
            "",
        );
        let clean = sanitize(&markup);
        assert!(
            clean.contains(".t{fill:#0f766e;font-family:Inter,'Segoe UI',Helvetica,sans-serif"),
            "{clean}"
        );
        assert!(clean.contains("@keyframes p"));
        assert!(clean.contains("fill=\"url(#g)\"") && clean.contains("filter=\"url(#f)\""));
        assert!(clean.contains("style=\"fill:#e07a2f;opacity:.8;stroke:none\""));
    }

    #[test]
    fn sanitize_repairs_html_entities_and_bare_ampersands() {
        let markup = "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 10 10\">\
            <text>R&D&nbsp;costs &mdash; &bogus; &lt;5% &notit;</text></svg>";
        let clean = sanitize(markup);
        assert_eq!(svg_text(&clean), "R&D costs \u{2014} &bogus; <5% \u{ac}it;");
    }

    #[test]
    fn sanitize_refuses_entities_and_broken_xml() {
        let bomb = "<?xml version=\"1.0\"?><!DOCTYPE svg [<!ENTITY a \"aaaa\">]>\
            <svg xmlns=\"http://www.w3.org/2000/svg\">&a;</svg>";
        let clean = sanitize(bomb); // DOCTYPE dropped, &a; kept as text
        assert!(
            !clean.contains("<!ENTITY") && !clean.contains("aaaa"),
            "{clean}"
        );
        assert!(clean.contains("&amp;a;"));
        assert!(matches!(
            sanitize_svg("<svg><rect></svg>", Aspect::Portrait),
            Err(AppError::IncompleteGeneration(_))
        ));
        assert!(matches!(
            sanitize_svg("<html><body/></html>", Aspect::Portrait),
            Err(AppError::IncompleteGeneration(_))
        ));
    }

    #[test]
    fn sanitize_caps_heavy_filter_values() {
        let clean = sanitize(&svg(&format!(
            "<filter id=\"f\" x=\"-5000%\" y=\"-10%\" width=\"9000%\" height=\"120%\">\
             <feGaussianBlur stdDeviation=\"500 2\"/><feDropShadow stdDeviation=\"1e9\"/>\
             <feMorphology radius=\"300\"/><feTurbulence numOctaves=\"50\"/>{}</filter>\
             <filter id=\"u\" filterUnits=\"userSpaceOnUse\" x=\"0\" y=\"0\" width=\"100000\" height=\"100000\"><feFlood/></filter>",
            "<feOffset/>".repeat(10)
        )));
        assert!(
            clean.contains("stdDeviation=\"20 2\"") && clean.contains("stdDeviation=\"20\""),
            "{clean}"
        );
        assert!(clean.contains("radius=\"8\"") && clean.contains("numOctaves=\"4\""));
        assert!(clean.contains("x=\"-100%\"") && clean.contains("width=\"300%\""));
        assert!(clean.contains("y=\"-10%\"") && clean.contains("height=\"120%\""));
        assert_eq!(clean.matches("<feOffset").count(), 8 - 4);
        assert!(!clean.contains("userSpaceOnUse") && !clean.contains("100000"));
    }

    #[test]
    fn sanitize_caps_how_many_elements_use_a_filter() {
        let rects = format!(
            "{}<rect style=\"fill:red;filter:url(#f)\"/>",
            "<rect filter=\"url(#f)\"/>".repeat(45)
        );
        let clean = sanitize(&svg(&rects));
        assert_eq!(
            clean.matches("filter=\"url(#f)\"").count(),
            MAX_FILTERED_ELEMENTS
        );
        assert!(clean.contains("style=\"fill:red\""), "{clean}");
    }

    #[test]
    fn sanitize_drops_use_of_use_and_too_many_uses() {
        let laughs = "<defs><g id=\"a\"><rect/></g>\
            <g id=\"b\"><use href=\"#a\"/><use href=\"#a\"/></g>\
            <g id=\"c\"><use xlink:href=\"#b\"/><use xlink:href=\"#b\"/></g></defs>\
            <use href=\"#c\"/><use href=\"#a\"/>";
        let clean = sanitize(&svg(laughs));
        assert!(
            !clean.contains("#b") && !clean.contains("href=\"#c\""),
            "{clean}"
        );
        assert_eq!(clean.matches("href=\"#a\"").count(), 3);
        // No xlink:href survived, so no xlink namespace is declared.
        assert!(!clean.contains("xmlns:xlink"), "{clean}");
        let many = sanitize(&svg(&"<use href=\"#x\"/>".repeat(600)));
        assert_eq!(many.matches("<use").count(), MAX_USES);
        let kept_xlink = sanitize(&svg("<g id=\"a\"/><use xlink:href=\"#a\"/>"));
        assert!(
            kept_xlink.contains("xmlns:xlink=\"http://www.w3.org/1999/xlink\"")
                && kept_xlink.contains("xlink:href=\"#a\"")
        );
    }

    #[test]
    fn sanitize_rejects_huge_images() {
        match sanitize_svg(&svg(&"<rect/>".repeat(9_000)), Aspect::Portrait) {
            Err(AppError::IncompleteGeneration(message)) => {
                assert!(message.contains("too complex"))
            }
            other => panic!("{other:?}"),
        }
        // Deep nesting is handled without recursion.
        let deep = format!("{}{}", "<g>".repeat(5_000), "</g>".repeat(5_000));
        assert!(sanitize_svg(&svg(&deep), Aspect::Portrait).is_ok());
    }

    #[test]
    fn sanitize_sets_a_size_from_the_aspect_when_missing() {
        let clean = sanitize_svg(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"100%\"><rect/></svg>",
            Aspect::Landscape,
        )
        .unwrap();
        assert!(clean.contains("viewBox=\"0 0 1920 1080\""), "{clean}");
        assert!(
            clean.contains("width=\"1920\"") && clean.contains("height=\"1080\""),
            "{clean}"
        );
        let sized = sanitize_svg(
            "<svg viewBox=\"0,0,300,200\" width=\"30em\"><rect/></svg>",
            Aspect::Square,
        )
        .unwrap();
        assert!(
            sized.contains("width=\"30em\"") && sized.contains("height=\"200\""),
            "{sized}"
        );
    }

    #[test]
    fn title_and_text() {
        let clean = sanitize(SVG);
        assert_eq!(svg_title(&clean), "Tides & the moon");
        assert_eq!(
            svg_text(&clean).lines().collect::<Vec<_>>(),
            [
                "Tides & the moon",
                "Tides explained",
                "The moon pulls the sea."
            ]
        );
    }

    #[test]
    fn format_g_matches_python() {
        for (value, expected) in [
            (20.0, "20"),
            (2.5, "2.5"),
            (-100.0, "-100"),
            (300.0, "300"),
            (0.0, "0"),
            (1e9, "1e+09"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (123456.0, "123456"),
            (1234567.0, "1.23457e+06"),
        ] {
            assert_eq!(format_g(value), expected, "{value}");
        }
    }

    #[test]
    fn options_match_the_frontend() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../open-notebook/frontend/src/lib/infographic-options.ts");
        let Ok(source) = std::fs::read_to_string(path) else {
            return; // reference checkout not present
        };
        let re = Regex::new(r"id: '([a-z-]+)'").unwrap();
        let mut found: Vec<String> = re
            .captures_iter(&source)
            .map(|c| c[1].to_string())
            .collect();
        found.sort();
        found.dedup();
        let mut expected: Vec<String> = STYLES
            .iter()
            .chain(LAYOUTS)
            .map(|o| o.id.to_string())
            .collect();
        expected.extend(Aspect::ALL.iter().map(|a| a.as_str().to_string()));
        expected.sort();
        expected.dedup();
        assert_eq!(found, expected);
        assert!(style(DEFAULT_STYLE).is_some() && layout(DEFAULT_LAYOUT).is_some());
    }

    // --- the job and the API --------------------------------------------------

    #[tokio::test]
    async fn infographic_job_saves_a_sanitised_svg_note() {
        let app = TestApp::new().await;
        let (notebook_id, source) = notebook_with_source(&app).await;
        let reply = format!(
            "```svg\n{}\n```",
            SVG.replace("<rect", "<script>x()</script><rect")
        );
        let model = use_models(&app, &[&reply]);

        let (status, body) = app
            .post(
                &format!("/api/notebooks/{notebook_id}/infographics"),
                json!({ "style": "chalkboard", "layout": "hub-spoke", "aspect": "square", "instructions": " for kids " }),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["status"], "queued");
        let job_id = body["job_id"].as_str().unwrap().to_string();
        assert_eq!(app.wait_for_job(&job_id).await, "completed");
        let record = app.state.db.select(&job_id).await.unwrap().unwrap();
        let args = &record["args"];
        assert_eq!(
            (args["style"].as_str(), args["layout"].as_str()),
            (Some("chalkboard"), Some("hub-spoke"))
        );
        assert_eq!(
            (args["aspect"].as_str(), args["instructions"].as_str()),
            (Some("square"), Some("for kids"))
        );
        assert_eq!(record["result"]["success"], true, "{}", record["result"]);
        let note_id = record["result"]["note_id"].as_str().unwrap().to_string();

        let system = model.calls.lock().unwrap()[0][0].content.clone();
        assert!(
            system.contains("STYLE: Chalkboard") && system.contains("LAYOUT: Hub and spoke"),
            "{system}"
        );
        assert!(system.contains("viewBox=\"0 0 1440 1440\"") && system.contains("for kids"));
        let human = model.calls.lock().unwrap()[0][1].content.clone();
        assert!(human.contains(&format!(
            "<source n=\"1\" id=\"{}\" title=\"Moon\">",
            source.id
        )));
        assert!(human.ends_with("Draw the complete SVG infographic now."));
        assert_eq!(
            model.options.lock().unwrap()[0].max_tokens,
            Some(DEFAULT_MAX_TOKENS as u32)
        );

        let note = Note::get(&app.state, &note_id).await.unwrap();
        assert_eq!(note.note_type.as_deref(), Some("ai"));
        assert_eq!(note.explainer, None);
        assert_eq!(note.title.as_deref(), Some("Tides & the moon"));
        assert_eq!(note.attachment_mime.as_deref(), Some(SVG_MIME));
        assert_eq!(note.attachment_name.as_deref(), Some("Tides-the-moon.svg"));
        assert!(note
            .content
            .as_deref()
            .unwrap()
            .contains("The moon pulls the sea."));
        let path = crate::domain::note::note_file_path(
            &app.state.config,
            note.attachment_file.as_deref().unwrap(),
        )
        .unwrap();
        let stored = std::fs::read_to_string(path).unwrap();
        assert!(
            stored.starts_with("<svg") && !stored.contains("script"),
            "{stored}"
        );

        let (_, viewed) = app.get(&format!("/api/notes/{note_id}")).await;
        assert_eq!(
            (
                viewed["infographic"].as_bool(),
                viewed["explainer"].as_bool()
            ),
            (Some(true), Some(false)),
            "{viewed}"
        );
        let (status, served) = app
            .text(
                axum::http::Method::GET,
                &format!("/api/notes/{note_id}/file"),
                None,
            )
            .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(served, stored);

        let (status, jobs) = app
            .get(&format!("/api/notebooks/{notebook_id}/infographics/jobs"))
            .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(jobs.as_array().unwrap().len(), 1);
        assert_eq!(jobs[0]["note_id"], note_id.as_str());
        assert_eq!(jobs[0]["title"], "Tides & the moon");
        // Explainer jobs are a separate list.
        let (_, explainer_jobs) = app
            .get(&format!("/api/notebooks/{notebook_id}/explainers/jobs"))
            .await;
        assert_eq!(explainer_jobs, json!([]));
    }

    #[tokio::test]
    async fn job_reports_failures_instead_of_raising() {
        let app = TestApp::new().await;
        let (notebook_id, _) = notebook_with_source(&app).await;
        use_models(&app, &["sorry, no"]);
        let input = GenerateInfographicInput {
            notebook_id: notebook_id.clone(),
            style: DEFAULT_STYLE.into(),
            layout: DEFAULT_LAYOUT.into(),
            aspect: "portrait".into(),
            instructions: None,
            requested_at: None,
        };
        let out = generate_infographic_job(&app.state, &input).await;
        assert_eq!(out["success"], false);
        assert!(
            out["error_message"]
                .as_str()
                .unwrap()
                .contains("did not return an SVG"),
            "{out}"
        );

        let unknown = GenerateInfographicInput {
            style: "nope".into(),
            ..input
        };
        let out = generate_infographic_job(&app.state, &unknown).await;
        assert_eq!(out["success"], false);
        assert!(
            out["error_message"]
                .as_str()
                .unwrap()
                .contains("Unknown infographic option"),
            "{out}"
        );
        assert!(Notebook::get(&app.state, &notebook_id)
            .await
            .unwrap()
            .notes(&app.state, false)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn create_infographic_validates_options_and_sources() {
        let app = TestApp::new().await;
        let (_, nb) = app.post("/api/notebooks", json!({ "name": "Empty" })).await;
        let url = format!("/api/notebooks/{}/infographics", nb["id"].as_str().unwrap());
        let (status, body) = app.post(&url, json!({})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["detail"], format!("{NO_SOURCES_MESSAGE}."));
        let (status, body) = app.post(&url, json!({ "style": "nope" })).await;
        assert_eq!(
            (status, body["detail"].as_str()),
            (
                StatusCode::BAD_REQUEST,
                Some("Unknown infographic style: nope")
            )
        );
        let (status, body) = app.post(&url, json!({ "layout": "nope" })).await;
        assert_eq!(
            (status, body["detail"].as_str()),
            (
                StatusCode::BAD_REQUEST,
                Some("Unknown infographic layout: nope")
            )
        );
        assert_eq!(
            app.post(&url, json!({ "aspect": "tall" })).await.0,
            StatusCode::UNPROCESSABLE_ENTITY
        );
        assert_eq!(
            app.post(&url, json!({ "style": null })).await.0,
            StatusCode::UNPROCESSABLE_ENTITY
        );
        // Options are checked before the notebook.
        let (status, _) = app
            .post(
                "/api/notebooks/notebook:missing/infographics",
                json!({ "style": "nope" }),
            )
            .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _) = app
            .post("/api/notebooks/notebook:missing/infographics", json!({}))
            .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn infographic_jobs_age_out_stale_ones() {
        let app = TestApp::new().await;
        let (notebook_id, _) = notebook_with_source(&app).await;
        app.state
            .db
            .query(
                "CREATE command:stuck CONTENT $content",
                crate::db::Params::new().set(
                    "content",
                    json!({
                        "app": "open_notebook", "name": COMMAND_NAME, "status": "new",
                        "args": { "notebook_id": notebook_id, "requested_at": "2020-01-01T00:00:00+00:00" },
                    }),
                ),
            )
            .await
            .unwrap();
        // Queued for years: reported as failed (stale), which then ages out.
        let (status, jobs) = app
            .get(&format!("/api/notebooks/{notebook_id}/infographics/jobs"))
            .await;
        assert_eq!((status, jobs), (StatusCode::OK, json!([])));
        let (status, _) = app
            .get("/api/notebooks/notebook:missing/infographics/jobs")
            .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
}
