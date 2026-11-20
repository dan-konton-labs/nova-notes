//! Language code to language name, for the prompt's language instruction.
//!
//! An episode profile stores a BCP 47 code (`pt-BR`); the prompts name the
//! language in English ("Portuguese"). Same rule as podcast-creator: only the
//! primary subtag is used, looked up as ISO 639-1 and then ISO 639-3.

use isolang::Language;

use crate::error::{PodcastError, Result};

/// Resolve a BCP 47 or ISO 639 code to its English language name.
pub fn resolve_language_name(language_code: &str) -> Result<String> {
    if language_code.trim().is_empty() {
        return Err(PodcastError::InvalidInput("Language code cannot be empty".into()));
    }

    let code = language_code.trim().split('-').next().unwrap_or_default().to_lowercase();
    let language = Language::from_639_1(&code).or_else(|| Language::from_639_3(&code)).ok_or_else(|| {
        PodcastError::InvalidInput(format!(
            "Invalid language code: '{language_code}'. \
             Use ISO 639-1 (e.g., 'pt') or BCP 47 (e.g., 'pt-BR') format."
        ))
    })?;

    let name = language.to_name();
    Ok(name.split(';').next().unwrap_or(name).trim().to_string())
}

/// [`resolve_language_name`] for an optional profile value: `None` and the
/// empty string both mean "no language instruction".
pub fn resolve_optional_language(language_code: Option<&str>) -> Result<Option<String>> {
    match language_code {
        Some(code) if !code.is_empty() => resolve_language_name(code).map(Some),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_two_letter_and_regional_codes() {
        assert_eq!(resolve_language_name("pt").unwrap(), "Portuguese");
        assert_eq!(resolve_language_name("pt-BR").unwrap(), "Portuguese");
        assert_eq!(resolve_language_name("en-US").unwrap(), "English");
        assert_eq!(resolve_language_name(" ES ").unwrap(), "Spanish");
        assert_eq!(resolve_language_name("he").unwrap(), "Hebrew");
        assert_eq!(resolve_language_name("zh-Hans-CN").unwrap(), "Chinese");
    }

    #[test]
    fn resolves_three_letter_codes() {
        assert_eq!(resolve_language_name("deu").unwrap(), "German");
    }

    #[test]
    fn rejects_unknown_and_empty_codes() {
        let unknown = resolve_language_name("english").unwrap_err().to_string();
        assert!(unknown.starts_with("Invalid language code: 'english'."), "{unknown}");
        assert_eq!(
            resolve_language_name("  ").unwrap_err().to_string(),
            "Language code cannot be empty"
        );
    }

    #[test]
    fn unset_language_means_no_instruction() {
        assert_eq!(resolve_optional_language(None).unwrap(), None);
        assert_eq!(resolve_optional_language(Some("")).unwrap(), None);
        assert_eq!(resolve_optional_language(Some("fr")).unwrap().as_deref(), Some("French"));
    }
}
