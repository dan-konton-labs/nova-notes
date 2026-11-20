//! Prompt templates.
//!
//! The Jinja files under `prompts/` are the Python build's templates, embedded
//! in the binary and rendered with minijinja. `Prompter::render("ask/entry", data)`
//! corresponds to Python's `Prompter(prompt_template="ask/entry").render(data)`.

use std::sync::Arc;

use include_dir::{include_dir, Dir};
use minijinja::{Environment, Value};
use serde::Serialize;

use crate::error::{AppError, AppResult};

static PROMPTS: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/prompts");

#[derive(Clone)]
pub struct Prompter {
    env: Arc<Environment<'static>>,
}

fn collect(dir: &'static Dir<'static>, env: &mut Environment<'static>) -> AppResult<()> {
    for file in dir.files() {
        let path = file.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jinja") {
            continue;
        }
        let name = path.with_extension("").to_string_lossy().replace('\\', "/");
        let source = file
            .contents_utf8()
            .ok_or_else(|| AppError::internal(format!("Prompt template {name} is not UTF-8")))?;
        env.add_template_owned(name, source.to_string())?;
    }
    for sub in dir.dirs() {
        collect(sub, env)?;
    }
    Ok(())
}

impl Prompter {
    pub fn new() -> AppResult<Self> {
        let mut env = Environment::new();
        env.set_keep_trailing_newline(false);
        collect(&PROMPTS, &mut env)?;
        Ok(Self { env: Arc::new(env) })
    }

    /// Names of every embedded template (`ask/entry`, `chat/system`, ...).
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.env.templates().map(|(name, _)| name.to_string()).collect();
        names.sort();
        names
    }

    fn context(data: impl Serialize) -> Value {
        // ai-prompter injects the current time into every render.
        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let base = Value::from_serialize(serde_json::json!({ "current_time": now }));
        // Later contexts win, so caller data can override `current_time`.
        minijinja::context! { ..Value::from_serialize(data), ..base }
    }

    /// Render an embedded template by name (forward slashes, no extension).
    pub fn render(&self, template: &str, data: impl Serialize) -> AppResult<String> {
        let tmpl = self
            .env
            .get_template(template)
            .map_err(|_| AppError::internal(format!("Unknown prompt template: {template}")))?;
        Ok(tmpl.render(Self::context(data))?)
    }

    /// Render ad-hoc template text (user-written transformation prompts).
    pub fn render_str(&self, source: &str, data: impl Serialize) -> AppResult<String> {
        Ok(self.env.render_str(source, Self::context(data))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn every_embedded_template_parses_and_renders() {
        let prompter = Prompter::new().unwrap();
        let names = prompter.names();
        for expected in [
            "ask/entry",
            "ask/query_process",
            "ask/final_answer",
            "chat/system",
            "source_chat/system",
            "transformation/execute",
            "pattern/generic",
            "explainer/system",
            "infographic/system",
        ] {
            assert!(names.iter().any(|n| n == expected), "missing template {expected}");
        }
        for name in names {
            prompter.render(&name, json!({})).unwrap_or_else(|e| panic!("{name} failed to render: {e}"));
        }
    }

    #[test]
    fn data_and_current_time_are_available() {
        let prompter = Prompter::new().unwrap();
        let out = prompter.render_str("{{ name }} at {{ current_time }}", json!({ "name": "Nova" })).unwrap();
        assert!(out.starts_with("Nova at 20"), "{out}");
        let out = prompter.render_str("{{ current_time }}", json!({ "current_time": "fixed" })).unwrap();
        assert_eq!(out, "fixed");
        assert!(prompter.render("nope/missing", json!({})).is_err());
    }
}
