//! Small helpers shared by the page areas: responses for htmx and plain
//! forms, form-field parsing, ids and paths.

use std::collections::HashSet;

use axum::extract::Multipart;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Redirect, Response};
use maud::Markup;

use super::{human_date, is_htmx};
use crate::error::AppError;

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

/// Add an `HX-Trigger` header (for example `{"toast": "..."}` or
/// `{"notes-changed": true}`) to a response.
pub fn with_trigger(mut response: Response, events: serde_json::Value) -> Response {
    if let Ok(value) = HeaderValue::from_str(&events.to_string()) {
        response.headers_mut().insert("HX-Trigger", value);
    }
    response
}

/// Ask the page for a toast (`app.js` listens for the `toast` event).
pub fn with_toast(response: Response, message: &str) -> Response {
    with_trigger(response, serde_json::json!({ "toast": message }))
}

/// htmx gets the fragment; a plain form post is redirected to `target`.
pub fn fragment_or_redirect(headers: &HeaderMap, fragment: Markup, target: &str) -> Response {
    if is_htmx(headers) {
        fragment.into_response()
    } else {
        Redirect::to(target).into_response()
    }
}

/// Go to `target` after a form: htmx follows `HX-Redirect`, a plain form
/// post gets a 303.
pub fn redirect(headers: &HeaderMap, target: &str) -> Response {
    if is_htmx(headers) {
        let mut response = StatusCode::OK.into_response();
        if let Ok(value) = HeaderValue::from_str(target) {
            response.headers_mut().insert("HX-Redirect", value);
        }
        response
    } else {
        Redirect::to(target).into_response()
    }
}

// ---------------------------------------------------------------------------
// Paths and ids
// ---------------------------------------------------------------------------

/// A same-site relative path (`/...`), or `None` for anything that could
/// leave the site (`//host`, `/\host`, absolute URLs).
///
/// Browsers drop tab, CR and LF from URLs and read `\` as `/`, so
/// `/<TAB>/evil.example` would become `//evil.example`: any control character
/// or backslash is refused outright. What is left must resolve, against a
/// placeholder origin, to that same origin.
pub fn safe_path(path: Option<&str>) -> Option<String> {
    let path = path?;
    if !path.starts_with('/') || path.starts_with("//") || path.chars().any(|c| c.is_control() || c == '\\') {
        return None;
    }
    let base = url::Url::parse("http://nova-notes.invalid/").ok()?;
    let joined = base.join(path).ok()?;
    (joined.origin() == base.origin()).then(|| path.to_string())
}

/// [`safe_path`], falling back to `fallback`.
pub fn safe_path_or(path: Option<&str>, fallback: &str) -> String {
    safe_path(path).unwrap_or_else(|| fallback.to_string())
}

/// Percent-encode one path segment; record ids keep their `table:` prefix.
pub fn seg(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~' | b':') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

/// A string safe to use as an element id (`credential:abc` → `credential-abc`).
pub fn dom_id(id: &str) -> String {
    id.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect()
}

/// An RFC 3339 timestamp from a service response as a short human date.
pub fn human_date_str(value: &str) -> String {
    human_date(chrono::DateTime::parse_from_rfc3339(value).ok().map(|d| d.with_timezone(&chrono::Utc)))
}

// ---------------------------------------------------------------------------
// Forms
// ---------------------------------------------------------------------------

/// A submitted form as its raw `(name, value)` pairs. Unlike `axum::Form`,
/// repeated names (checkbox groups) and "was this field on the form at all"
/// both survive. Where a name repeats, single-value reads take the last one,
/// so a hidden `false` before a checkbox works.
#[derive(Debug, Clone, Default)]
pub struct Fields(pub Vec<(String, String)>);

impl Fields {
    /// A URL-encoded body or query string.
    pub fn parse(raw: &[u8]) -> Self {
        Self(url::form_urlencoded::parse(raw).into_owned().collect())
    }

    /// A query string, which may be absent.
    pub fn query(raw: Option<&str>) -> Self {
        Self::parse(raw.unwrap_or_default().as_bytes())
    }

    pub fn has(&self, key: &str) -> bool {
        self.0.iter().any(|(k, _)| k == key)
    }

    /// The last value of `key`, as submitted.
    pub fn raw(&self, key: &str) -> Option<&str> {
        self.0.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    /// The last value of `key` as submitted, or `""`.
    pub fn string(&self, key: &str) -> String {
        self.raw(key).unwrap_or_default().to_string()
    }

    /// The last value of `key`, trimmed; `None` when absent or blank.
    pub fn text(&self, key: &str) -> Option<String> {
        self.raw(key).map(str::trim).filter(|v| !v.is_empty()).map(String::from)
    }

    /// [`Fields::text`], or a 400 with `message` when it is missing.
    pub fn required(&self, key: &str, message: &str) -> Result<String, AppError> {
        self.text(key).ok_or_else(|| AppError::invalid(message))
    }

    /// Every non-blank value of `key`, trimmed, in order, without duplicates.
    pub fn all(&self, key: &str) -> Vec<String> {
        let mut seen = HashSet::new();
        self.0
            .iter()
            .filter(|(k, v)| k == key && !v.trim().is_empty())
            .map(|(_, v)| v.trim().to_string())
            .filter(|v| seen.insert(v.clone()))
            .collect()
    }

    /// For a field the form showed: `Some(None)` when left blank (clear it),
    /// `Some(Some(v))` when filled, `None` when the form did not have it.
    pub fn present(&self, key: &str) -> Option<Option<String>> {
        self.has(key).then(|| self.text(key))
    }

    /// A checkbox rendered after a hidden `false` input: the last value wins.
    /// `None` when neither was on the form.
    pub fn flag(&self, key: &str) -> Option<bool> {
        self.raw(key).map(|v| matches!(v, "true" | "on" | "1" | "yes"))
    }

    /// A plain checkbox: ticked when present with any value but
    /// `false` / `off` / `0`.
    pub fn checked(&self, key: &str) -> bool {
        self.0.iter().any(|(k, v)| k == key && !matches!(v.as_str(), "false" | "off" | "0"))
    }
}

/// One uploaded file from a multipart form.
#[derive(Debug, Clone)]
pub struct UploadedFile {
    /// The form field it came from.
    pub field: String,
    pub filename: String,
    pub data: Vec<u8>,
}

/// The text fields and files of a multipart form. Empty file parts (a file
/// input with nothing chosen) are dropped. Reading errors keep the status
/// axum gives them (for example 413 when the body is too large).
pub async fn read_multipart(mut multipart: Multipart) -> Result<(Fields, Vec<UploadedFile>), AppError> {
    let (mut fields, mut files) = (Vec::new(), Vec::new());
    while let Some(part) = multipart.next_field().await.map_err(multipart_error)? {
        let field = part.name().unwrap_or_default().to_string();
        match part.file_name().map(str::to_string) {
            Some(filename) => {
                let data = part.bytes().await.map_err(multipart_error)?.to_vec();
                if !(filename.is_empty() && data.is_empty()) {
                    files.push(UploadedFile { field, filename, data });
                }
            }
            None => fields.push((field, part.text().await.map_err(multipart_error)?)),
        }
    }
    Ok((Fields(fields), files))
}

/// A multipart reading error, with the status axum chose for it.
pub fn multipart_error(e: axum::extract::multipart::MultipartError) -> AppError {
    AppError::http(e.status(), e.body_text())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_paths_stay_on_site() {
        assert_eq!(safe_path(Some("/notebooks/x")).as_deref(), Some("/notebooks/x"));
        assert_eq!(safe_path(Some("//evil.example")), None);
        assert_eq!(safe_path(Some("/\\evil.example")), None);
        assert_eq!(safe_path(Some("https://evil.example")), None);
        for bad in ["/\t/evil.example", "/\\evil", "/a\\b", "/\r\nX: y", "/\n", "/\u{7f}/x", "/\u{85}x", ""] {
            assert_eq!(safe_path(Some(bad)), None, "{bad:?}");
        }
        assert_eq!(safe_path(Some("/notebooks/x?y=1#z")).as_deref(), Some("/notebooks/x?y=1#z"));
        assert_eq!(safe_path_or(None, "/sources"), "/sources");
    }

    #[test]
    fn ids_and_segments() {
        assert_eq!(seg("notebook:a b/c"), "notebook:a%20b%2Fc");
        assert_eq!(dom_id("credential:abc"), "credential-abc");
    }

    #[test]
    fn fields_keep_repeats_and_presence() {
        let f = Fields::parse(b"q=+hi+&tag=a&tag=b&tag=a&tag=&embed=false&embed=true&blank=&off=false");
        assert_eq!(f.string("q"), " hi ");
        assert_eq!(f.text("q").as_deref(), Some("hi"));
        assert_eq!(f.all("tag"), ["a", "b"]);
        assert_eq!(f.flag("embed"), Some(true));
        assert_eq!(f.flag("missing"), None);
        assert_eq!(f.present("blank"), Some(None));
        assert_eq!(f.present("missing"), None);
        assert!(f.checked("embed") && !f.checked("off") && !f.checked("missing"));
        assert!(f.required("missing", "Missing").is_err());
    }
}
