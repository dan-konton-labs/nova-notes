//! Application settings, runtime capabilities and the language list.
//!
//! The Rust counterpart of `api/routers/settings.py`, `capabilities.py`,
//! `languages.py` and `open_notebook/utils/runtime_capabilities.py`.

use serde::{Deserialize, Serialize};

use crate::config::env_nonempty;
use crate::domain::ContentSettings;
use crate::error::{AppError, AppResult};
use crate::jobs::JobQueue;
use crate::state::AppState;

/// Settings have no background jobs.
pub fn register_jobs(_jobs: &JobQueue) {}

// ---------------------------------------------------------------------------
// Content settings
// ---------------------------------------------------------------------------

/// Engines a document can be processed with.
pub const DOCUMENT_ENGINES: [&str; 3] = ["auto", "docling", "simple"];
/// Engines a URL can be processed with.
pub const URL_ENGINES: [&str; 5] = ["auto", "firecrawl", "jina", "crawl4ai", "simple"];
/// When a new source is embedded for vector search.
pub const EMBEDDING_OPTIONS: [&str; 3] = ["ask", "always", "never"];
/// Whether uploaded files are deleted once processed.
pub const AUTO_DELETE_OPTIONS: [&str; 2] = ["yes", "no"];

/// Body of `PUT /settings`. Absent and `null` fields are left alone.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SettingsUpdate {
    #[serde(default)]
    pub default_content_processing_engine_doc: Option<String>,
    #[serde(default)]
    pub default_content_processing_engine_url: Option<String>,
    #[serde(default)]
    pub default_embedding_option: Option<String>,
    #[serde(default)]
    pub auto_delete_files: Option<String>,
    #[serde(default)]
    pub docling_ocr: Option<bool>,
    #[serde(default)]
    pub docling_formulas: Option<bool>,
    #[serde(default)]
    pub docling_vision: Option<bool>,
    #[serde(default)]
    pub youtube_preferred_languages: Option<Vec<String>>,
}

fn choice(field: &str, value: &Option<String>, allowed: &[&str]) -> AppResult<Option<String>> {
    match value {
        Some(value) if !allowed.contains(&value.as_str()) => {
            Err(AppError::invalid(format!("Invalid {field}: '{value}'. Allowed values: {}", allowed.join(", "))))
        }
        other => Ok(other.clone()),
    }
}

/// The content settings (`SettingsResponse`), with defaults for anything unset.
pub async fn get_settings(state: &AppState) -> AppResult<ContentSettings> {
    ContentSettings::load(state).await
}

/// Change the given settings and return all of them as stored.
pub async fn update_settings(state: &AppState, update: &SettingsUpdate) -> AppResult<ContentSettings> {
    // Validate everything before anything is written.
    let engine_doc = choice("default_content_processing_engine_doc", &update.default_content_processing_engine_doc, &DOCUMENT_ENGINES)?;
    let engine_url = choice("default_content_processing_engine_url", &update.default_content_processing_engine_url, &URL_ENGINES)?;
    let embedding = choice("default_embedding_option", &update.default_embedding_option, &EMBEDDING_OPTIONS)?;
    let auto_delete = choice("auto_delete_files", &update.auto_delete_files, &AUTO_DELETE_OPTIONS)?;

    let mut settings = ContentSettings::load(state).await?;
    if engine_doc.is_some() {
        settings.default_content_processing_engine_doc = engine_doc;
    }
    if engine_url.is_some() {
        settings.default_content_processing_engine_url = engine_url;
    }
    if embedding.is_some() {
        settings.default_embedding_option = embedding;
    }
    if auto_delete.is_some() {
        settings.auto_delete_files = auto_delete;
    }
    if update.docling_ocr.is_some() {
        settings.docling_ocr = update.docling_ocr;
    }
    if update.docling_formulas.is_some() {
        settings.docling_formulas = update.docling_formulas;
    }
    if update.docling_vision.is_some() {
        settings.docling_vision = update.docling_vision;
    }
    if update.youtube_preferred_languages.is_some() {
        settings.youtube_preferred_languages = update.youtube_preferred_languages.clone();
    }
    settings.save(state).await
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

/// Which opt-in extraction runtimes can actually be used (`CapabilitiesResponse`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    /// The Docling document engine, its OCR toggle and image sources work.
    pub docling_available: bool,
    /// Crawl4AI can run: locally, or offloaded to a remote server.
    pub crawl4ai_available: bool,
    /// A remote Crawl4AI server is configured (`CRAWL4AI_API_URL`).
    pub crawl4ai_remote_configured: bool,
}

impl Capabilities {
    /// This build has no Docling and no local Crawl4AI / Chromium, so the only
    /// opt-in runtime it can use is a remote Crawl4AI server.
    pub fn with_crawl4ai_api_url(url: Option<&str>) -> Self {
        let remote = url.is_some_and(|url| !url.trim().is_empty());
        Self { docling_available: false, crawl4ai_available: remote, crawl4ai_remote_configured: remote }
    }
}

/// The runtimes available to this process.
pub fn capabilities() -> Capabilities {
    Capabilities::with_crawl4ai_api_url(env_nonempty("CRAWL4AI_API_URL").as_deref())
}

/// The environment variable that would make `engine` usable, or `None` when
/// it is usable (or needs no opt-in runtime at all). The engine choice lives
/// in the database and so outlives the environment that made it available:
/// check this before handing extraction an engine.
pub fn engine_runtime_missing(engine: Option<&str>) -> Option<&'static str> {
    engine_runtime_missing_with(engine, capabilities())
}

/// [`engine_runtime_missing`] against a given set of capabilities.
pub fn engine_runtime_missing_with(engine: Option<&str>, capabilities: Capabilities) -> Option<&'static str> {
    match engine?.trim().to_lowercase().as_str() {
        "crawl4ai" if !capabilities.crawl4ai_available => Some("OPEN_NOTEBOOK_ENABLE_CRAWL4AI"),
        "docling" if !capabilities.docling_available => Some("OPEN_NOTEBOOK_ENABLE_DOCLING"),
        // auto / simple / firecrawl / jina need no runtime; unknown engines
        // are the extractor's problem.
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Languages
// ---------------------------------------------------------------------------

/// A language choice: a BCP 47 locale code and its English name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Language {
    pub code: &'static str,
    pub name: &'static str,
}

/// The languages offered for podcasts and transcripts, sorted by name.
pub fn languages() -> Vec<Language> {
    LANGUAGES.iter().map(|&(code, name)| Language { code, name }).collect()
}

/// What the Python build's `GET /languages` returns (pycountry 26.2 with
/// Babel 2.18 / CLDR): every ISO 639-1 language as its most likely locale,
/// or as a bare code where CLDR knows no territory for it, plus the regional
/// variants that matter for speech (accent, vocabulary, spelling). Kept in
/// the order the endpoint serves: by name.
const LANGUAGES: [(&str, &str); 199] = [
    ("ab-GE", "Abkhazian (Georgia)"),
    ("aa-ET", "Afar (Ethiopia)"),
    ("af-ZA", "Afrikaans (South Africa)"),
    ("ak-GH", "Akan (Ghana)"),
    ("sq-AL", "Albanian (Albania)"),
    ("am-ET", "Amharic (Ethiopia)"),
    ("ar-EG", "Arabic (Egypt)"),
    ("ar-SA", "Arabic (Saudi Arabia)"),
    ("an-ES", "Aragonese (Spain)"),
    ("hy-AM", "Armenian (Armenia)"),
    ("as-IN", "Assamese (India)"),
    ("av", "Avaric"),
    ("ae", "Avestan"),
    ("ay", "Aymara"),
    ("az-AZ", "Azerbaijani (Latin, Azerbaijan)"),
    ("bm-ML", "Bambara (Mali)"),
    ("bn-BD", "Bangla (Bangladesh)"),
    ("ba-RU", "Bashkir (Russia)"),
    ("eu-ES", "Basque (Spain)"),
    ("be-BY", "Belarusian (Belarus)"),
    ("bi", "Bislama"),
    ("bs-BA", "Bosnian (Latin, Bosnia & Herzegovina)"),
    ("br-FR", "Breton (France)"),
    ("bg-BG", "Bulgarian (Bulgaria)"),
    ("my-MM", "Burmese (Myanmar (Burma))"),
    ("ca-ES", "Catalan (Spain)"),
    ("ch", "Chamorro"),
    ("ce-RU", "Chechen (Russia)"),
    ("zh-CN", "Chinese (Simplified, China)"),
    ("zh-HK", "Chinese (Traditional, Hong Kong SAR China)"),
    ("zh-TW", "Chinese (Traditional, Taiwan)"),
    ("cu-RU", "Church Slavic (Russia)"),
    ("cv-RU", "Chuvash (Russia)"),
    ("kw-GB", "Cornish (United Kingdom)"),
    ("co-FR", "Corsican (France)"),
    ("cr", "Cree"),
    ("hr-HR", "Croatian (Croatia)"),
    ("cs-CZ", "Czech (Czechia)"),
    ("da-DK", "Danish (Denmark)"),
    ("dv-MV", "Divehi (Maldives)"),
    ("nl-BE", "Dutch (Belgium)"),
    ("nl-NL", "Dutch (Netherlands)"),
    ("dz-BT", "Dzongkha (Bhutan)"),
    ("en-AU", "English (Australia)"),
    ("en-IN", "English (India)"),
    ("en-GB", "English (United Kingdom)"),
    ("en-US", "English (United States)"),
    ("eo-001", "Esperanto (world)"),
    ("et-EE", "Estonian (Estonia)"),
    ("ee-GH", "Ewe (Ghana)"),
    ("fo-FO", "Faroese (Faroe Islands)"),
    ("fj", "Fijian"),
    ("fil-PH", "Filipino (Philippines)"),
    ("fi-FI", "Finnish (Finland)"),
    ("fr-CA", "French (Canada)"),
    ("fr-FR", "French (France)"),
    ("fr-CH", "French (Switzerland)"),
    ("ff-SN", "Fula (Latin, Senegal)"),
    ("gl-ES", "Galician (Spain)"),
    ("lg-UG", "Ganda (Uganda)"),
    ("ka-GE", "Georgian (Georgia)"),
    ("de-AT", "German (Austria)"),
    ("de-DE", "German (Germany)"),
    ("de-CH", "German (Switzerland)"),
    ("el-GR", "Greek (Greece)"),
    ("gn-PY", "Guarani (Paraguay)"),
    ("gu-IN", "Gujarati (India)"),
    ("ht-HT", "Haitian Creole (Haiti)"),
    ("ha-NG", "Hausa (Nigeria)"),
    ("he-IL", "Hebrew (Israel)"),
    ("hz", "Herero"),
    ("hi-IN", "Hindi (India)"),
    ("ho", "Hiri Motu"),
    ("hu-HU", "Hungarian (Hungary)"),
    ("is-IS", "Icelandic (Iceland)"),
    ("io-001", "Ido (world)"),
    ("ig-NG", "Igbo (Nigeria)"),
    ("id-ID", "Indonesian (Indonesia)"),
    ("ia-001", "Interlingua (world)"),
    ("ie-EE", "Interlingue (Estonia)"),
    ("iu-CA", "Inuktitut (Canada)"),
    ("ik", "Inupiaq"),
    ("ga-IE", "Irish (Ireland)"),
    ("it-IT", "Italian (Italy)"),
    ("ja-JP", "Japanese (Japan)"),
    ("jv-ID", "Javanese (Indonesia)"),
    ("kl-GL", "Kalaallisut (Greenland)"),
    ("kn-IN", "Kannada (India)"),
    ("kr", "Kanuri"),
    ("ks-IN", "Kashmiri (Arabic, India)"),
    ("kk-KZ", "Kazakh (Cyrillic, Kazakhstan)"),
    ("km-KH", "Khmer (Cambodia)"),
    ("ki-KE", "Kikuyu (Kenya)"),
    ("rw-RW", "Kinyarwanda (Rwanda)"),
    ("kv", "Komi"),
    ("kg", "Kongo"),
    ("ko-KR", "Korean (South Korea)"),
    ("kj", "Kuanyama"),
    ("ku-TR", "Kurdish (Türkiye)"),
    ("ky-KG", "Kyrgyz (Kyrgyzstan)"),
    ("lo-LA", "Lao (Laos)"),
    ("la-VA", "Latin (Vatican City)"),
    ("lv-LV", "Latvian (Latvia)"),
    ("li", "Limburgan"),
    ("ln-CD", "Lingala (Congo - Kinshasa)"),
    ("lt-LT", "Lithuanian (Lithuania)"),
    ("lu-CD", "Luba-Katanga (Congo - Kinshasa)"),
    ("lb-LU", "Luxembourgish (Luxembourg)"),
    ("mk-MK", "Macedonian (North Macedonia)"),
    ("mg-MG", "Malagasy (Madagascar)"),
    ("ms-MY", "Malay (Malaysia)"),
    ("ml-IN", "Malayalam (India)"),
    ("mt-MT", "Maltese (Malta)"),
    ("gv-IM", "Manx (Isle of Man)"),
    ("mr-IN", "Marathi (India)"),
    ("mh", "Marshallese"),
    ("mn-MN", "Mongolian (Mongolia)"),
    ("mi-NZ", "Māori (New Zealand)"),
    ("na", "Nauru"),
    ("nv-US", "Navajo (United States)"),
    ("ng", "Ndonga"),
    ("ne-NP", "Nepali (Nepal)"),
    ("nd-ZW", "North Ndebele (Zimbabwe)"),
    ("se-NO", "Northern Sami (Norway)"),
    ("no", "Norwegian"),
    ("nb-NO", "Norwegian Bokmål (Norway)"),
    ("nn-NO", "Norwegian Nynorsk (Norway)"),
    ("ny-MW", "Nyanja (Malawi)"),
    ("oc-FR", "Occitan (France)"),
    ("or-IN", "Odia (India)"),
    ("oj", "Ojibwa"),
    ("om-ET", "Oromo (Ethiopia)"),
    ("os-GE", "Ossetic (Georgia)"),
    ("pi", "Pali"),
    ("ps-AF", "Pashto (Afghanistan)"),
    ("fa-IR", "Persian (Iran)"),
    ("pl-PL", "Polish (Poland)"),
    ("pt-BR", "Portuguese (Brazil)"),
    ("pt-PT", "Portuguese (Portugal)"),
    ("pa-IN", "Punjabi (Gurmukhi, India)"),
    ("qu-PE", "Quechua (Peru)"),
    ("ro-RO", "Romanian (Romania)"),
    ("rm-CH", "Romansh (Switzerland)"),
    ("rn-BI", "Rundi (Burundi)"),
    ("ru-RU", "Russian (Russia)"),
    ("sm", "Samoan"),
    ("sg-CF", "Sango (Central African Republic)"),
    ("sa-IN", "Sanskrit (India)"),
    ("sc-IT", "Sardinian (Italy)"),
    ("gd-GB", "Scottish Gaelic (United Kingdom)"),
    ("sr-RS", "Serbian (Cyrillic, Serbia)"),
    ("sh", "Serbo-Croatian"),
    ("sn-ZW", "Shona (Zimbabwe)"),
    ("ii-CN", "Sichuan Yi (China)"),
    ("sd-PK", "Sindhi (Arabic, Pakistan)"),
    ("si-LK", "Sinhala (Sri Lanka)"),
    ("sk-SK", "Slovak (Slovakia)"),
    ("sl-SI", "Slovenian (Slovenia)"),
    ("so-SO", "Somali (Somalia)"),
    ("nr-ZA", "South Ndebele (South Africa)"),
    ("st-ZA", "Southern Sotho (South Africa)"),
    ("es-AR", "Spanish (Argentina)"),
    ("es-CO", "Spanish (Colombia)"),
    ("es-MX", "Spanish (Mexico)"),
    ("es-ES", "Spanish (Spain)"),
    ("su-ID", "Sundanese (Latin, Indonesia)"),
    ("sw-TZ", "Swahili (Tanzania)"),
    ("ss-ZA", "Swati (South Africa)"),
    ("sv-SE", "Swedish (Sweden)"),
    ("ty", "Tahitian"),
    ("tg-TJ", "Tajik (Tajikistan)"),
    ("ta-IN", "Tamil (India)"),
    ("tt-RU", "Tatar (Russia)"),
    ("te-IN", "Telugu (India)"),
    ("th-TH", "Thai (Thailand)"),
    ("bo-CN", "Tibetan (China)"),
    ("ti-ET", "Tigrinya (Ethiopia)"),
    ("to-TO", "Tongan (Tonga)"),
    ("ts-ZA", "Tsonga (South Africa)"),
    ("tn-ZA", "Tswana (South Africa)"),
    ("tr-TR", "Turkish (Türkiye)"),
    ("tk-TM", "Turkmen (Turkmenistan)"),
    ("tw", "Twi"),
    ("uk-UA", "Ukrainian (Ukraine)"),
    ("ur-PK", "Urdu (Pakistan)"),
    ("ug-CN", "Uyghur (China)"),
    ("uz-UZ", "Uzbek (Latin, Uzbekistan)"),
    ("ve-ZA", "Venda (South Africa)"),
    ("vi-VN", "Vietnamese (Vietnam)"),
    ("vo-001", "Volapük (world)"),
    ("wa-BE", "Walloon (Belgium)"),
    ("cy-GB", "Welsh (United Kingdom)"),
    ("fy-NL", "Western Frisian (Netherlands)"),
    ("wo-SN", "Wolof (Senegal)"),
    ("xh-ZA", "Xhosa (South Africa)"),
    ("yi-UA", "Yiddish (Ukraine)"),
    ("yo-NG", "Yoruba (Nigeria)"),
    ("za-CN", "Zhuang (China)"),
    ("zu-ZA", "Zulu (South Africa)"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_remote_crawl4ai_server_is_ever_available() {
        assert_eq!(
            Capabilities::with_crawl4ai_api_url(None),
            Capabilities { docling_available: false, crawl4ai_available: false, crawl4ai_remote_configured: false }
        );
        assert_eq!(Capabilities::with_crawl4ai_api_url(Some("  ")), Capabilities::with_crawl4ai_api_url(None));
        assert_eq!(
            Capabilities::with_crawl4ai_api_url(Some("http://crawl4ai:11235")),
            Capabilities { docling_available: false, crawl4ai_available: true, crawl4ai_remote_configured: true }
        );
    }

    #[test]
    fn engines_report_the_runtime_they_miss() {
        let none = Capabilities::with_crawl4ai_api_url(None);
        let remote = Capabilities::with_crawl4ai_api_url(Some("http://crawl4ai:11235"));
        let everything = Capabilities { docling_available: true, crawl4ai_available: true, crawl4ai_remote_configured: false };

        for engine in ["auto", "simple", "firecrawl", "jina", "some-future-engine", ""] {
            assert_eq!(engine_runtime_missing_with(Some(engine), none), None, "{engine}");
        }
        assert_eq!(engine_runtime_missing_with(None, none), None);

        assert_eq!(engine_runtime_missing_with(Some("crawl4ai"), none), Some("OPEN_NOTEBOOK_ENABLE_CRAWL4AI"));
        assert_eq!(engine_runtime_missing_with(Some("  Crawl4AI  "), none), Some("OPEN_NOTEBOOK_ENABLE_CRAWL4AI"));
        assert_eq!(engine_runtime_missing_with(Some("crawl4ai"), remote), None, "a remote server counts");
        assert_eq!(engine_runtime_missing_with(Some("docling"), remote), Some("OPEN_NOTEBOOK_ENABLE_DOCLING"));
        assert_eq!(engine_runtime_missing_with(Some("docling"), everything), None);
    }

    #[test]
    fn languages_are_unique_and_sorted_by_name() {
        let all = languages();
        assert_eq!(all.len(), 199);
        assert!(all.windows(2).all(|pair| pair[0].name <= pair[1].name), "sorted by name");
        let codes: std::collections::HashSet<_> = all.iter().map(|l| l.code).collect();
        assert_eq!(codes.len(), all.len(), "codes are unique");
        for (code, name) in [
            ("en-US", "English (United States)"),
            ("en-GB", "English (United Kingdom)"),
            ("pt-BR", "Portuguese (Brazil)"),
            ("pt-PT", "Portuguese (Portugal)"),
            ("zh-TW", "Chinese (Traditional, Taiwan)"),
            ("no", "Norwegian"),
        ] {
            assert!(all.contains(&Language { code, name }), "{code} {name}");
        }
    }
}
