//! Web UI for Discover: a notebook built from a web search.
//!
//! The notebooks page shows two buttons ("Discover notebook" and "Discover +
//! podcast") that open one dialog in the matching mode. Its form posts to
//! `POST /notebooks/discover`, which calls [`discovery::discover_notebook`]
//! and sends the browser to the new notebook, whose sources pane polls while
//! the background job adds and processes the sources. A rejected form comes
//! back with the message in place (htmx swaps it; see `static/js/discover.js`).

use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::get;
use axum::Router;
use maud::{html, Markup};

use super::{is_htmx, page, page_head, redirect, seg, Fields, PageContext, WebResult};
use crate::error::AppError;
use crate::services::discovery::{self, DiscoverNotebookRequest, DEFAULT_NUM_SOURCES};
use crate::services::podcasts::{self, EpisodeProfileView, SpeakerProfileView};
use crate::state::AppState;

const DIALOG_ID: &str = "disc-dialog";
const MAX_FORM_SOURCES: i64 = 50;

pub fn router() -> Router<AppState> {
    Router::new().route("/notebooks/discover", get(|| async { Redirect::to("/notebooks") }).post(submit))
}

/// Notebook only, or notebook plus podcast.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Mode {
    #[default]
    Notebook,
    Podcast,
}

/// The form as submitted, kept as text so a rejected form shows it again.
#[derive(Debug, Clone)]
struct FormValues {
    mode: Mode,
    prompt: String,
    num_sources: String,
    episode_profile: String,
    speaker_profile: String,
}

impl Default for FormValues {
    fn default() -> Self {
        Self {
            mode: Mode::Notebook,
            prompt: String::new(),
            num_sources: DEFAULT_NUM_SOURCES.to_string(),
            episode_profile: String::new(),
            speaker_profile: String::new(),
        }
    }
}

impl FormValues {
    fn from_fields(fields: &Fields) -> Self {
        Self {
            mode: if fields.raw("mode") == Some("podcast") { Mode::Podcast } else { Mode::Notebook },
            prompt: fields.string("prompt"),
            num_sources: fields.string("num_sources"),
            episode_profile: fields.text("episode_profile").unwrap_or_default(),
            speaker_profile: fields.text("speaker_profile").unwrap_or_default(),
        }
    }

    /// The dialog's own checks (as in the old frontend), then the request.
    fn request(&self) -> Result<DiscoverNotebookRequest, AppError> {
        let prompt = self.prompt.trim();
        if prompt.chars().count() < 3 {
            return Err(AppError::unprocessable("Describe what to research in at least 3 characters."));
        }
        if self.prompt.chars().count() > 2000 {
            return Err(AppError::unprocessable("Keep the description under 2000 characters."));
        }
        let num_sources = match self.num_sources.trim() {
            "" => DEFAULT_NUM_SOURCES,
            raw => match raw.parse::<i64>() {
                Ok(n) if (1..=MAX_FORM_SOURCES).contains(&n) => n,
                _ => {
                    return Err(AppError::unprocessable(format!(
                        "Number of sources must be a whole number from 1 to {MAX_FORM_SOURCES}."
                    )))
                }
            },
        };
        let podcast = self.mode == Mode::Podcast;
        if podcast && self.episode_profile.is_empty() {
            return Err(AppError::unprocessable("Choose an episode profile for the podcast."));
        }
        Ok(DiscoverNotebookRequest {
            prompt: prompt.to_string(),
            create_podcast: podcast,
            episode_profile: podcast.then(|| self.episode_profile.clone()),
            speaker_profile: (podcast && !self.speaker_profile.is_empty()).then(|| self.speaker_profile.clone()),
            num_sources,
        })
    }
}

struct Profiles {
    episodes: Vec<EpisodeProfileView>,
    speakers: Vec<SpeakerProfileView>,
}

async fn load_profiles(state: &AppState) -> WebResult<Profiles> {
    Ok(Profiles {
        episodes: podcasts::list_episode_profiles(state).await?,
        speakers: podcasts::list_speaker_profiles(state).await?,
    })
}

/// The Discover buttons and dialog for the notebooks page header.
pub async fn discover_controls(state: &AppState) -> WebResult<Markup> {
    let profiles = load_profiles(state).await?;
    Ok(html! {
        button.btn.disc-open type="button" data-open=(format!("#{DIALOG_ID}")) data-disc-mode="notebook" { "Discover notebook" }
        button.btn.disc-open type="button" data-open=(format!("#{DIALOG_ID}")) data-disc-mode="podcast" { "Discover + podcast" }
        dialog.disc-dialog id=(DIALOG_ID) aria-labelledby="disc-title" {
            div.dialog-head {
                h3 #disc-title { "Discover a notebook" }
                button.btn.btn-ghost.btn-sm type="button" data-close aria-label="Close" { "Close" }
            }
            (discover_form(&profiles, &FormValues::default(), None))
        }
    })
}

fn discover_form(profiles: &Profiles, values: &FormValues, error: Option<&str>) -> Markup {
    let podcast = values.mode == Mode::Podcast;
    html! {
        form.disc-form #disc-form method="post" action="/notebooks/discover"
            hx-post="/notebooks/discover" hx-target="this" hx-swap="outerHTML"
            hx-disabled-elt="find button[type='submit']" {
            div.dialog-body {
                p.muted {
                    "Describe a topic. Nova Notes searches the web, picks a balanced set of sources "
                    "and adds them to a new notebook."
                }
                @if let Some(error) = error {
                    p.disc-error role="alert" { (error) }
                }
                fieldset.disc-mode {
                    legend { "Create" }
                    label.disc-choice {
                        input type="radio" name="mode" value="notebook" checked[!podcast];
                        " A notebook"
                    }
                    label.disc-choice {
                        input type="radio" name="mode" value="podcast" checked[podcast];
                        " A notebook and a podcast"
                    }
                }
                div.field {
                    label for="disc-prompt" { "What should it research?" }
                    textarea #disc-prompt name="prompt" required minlength="3" maxlength="2000"
                        placeholder="e.g. How do heat pumps work, and are they worth it in cold climates?" { (values.prompt) }
                }
                div.field.disc-count {
                    label for="disc-num" { "Number of sources" }
                    input #disc-num type="number" name="num_sources" min="1" max=(MAX_FORM_SOURCES) step="1" value=(values.num_sources);
                    p.hint { "Between 3 and 15 sources are added. Default: 10." }
                }
                div.disc-podcast {
                    div.field {
                        label for="disc-episode" { "Episode profile" }
                        select #disc-episode name="episode_profile" {
                            option value="" { "Choose an episode profile" }
                            @for profile in &profiles.episodes {
                                option value=(profile.name) selected[values.episode_profile == profile.name] { (profile.name) }
                            }
                        }
                        @if profiles.episodes.is_empty() {
                            p.hint { "No episode profiles yet. Create one under " a href="/podcasts" { "Podcasts" } "." }
                        }
                    }
                    div.field {
                        label for="disc-speaker" { "Speaker profile (optional)" }
                        select #disc-speaker name="speaker_profile" {
                            option value="" { "The episode profile's speakers" }
                            @for profile in &profiles.speakers {
                                option value=(profile.name) selected[values.speaker_profile == profile.name] { (profile.name) }
                            }
                        }
                    }
                    p.hint { "The podcast starts once at least three sources are processed." }
                }
            }
            div.dialog-foot {
                button.btn type="button" data-close { "Cancel" }
                button.btn.btn-primary type="submit" {
                    span.disc-idle { "Discover" }
                    span.disc-busy { "Starting…" }
                }
            }
        }
    }
}

/// `POST /notebooks/discover`: start a discovery and go to the new notebook.
/// A rejected request (bad input, unknown profile) shows the form again with
/// the reason.
async fn submit(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> WebResult<Response> {
    let values = FormValues::from_fields(&Fields::parse(&body));
    let outcome = match values.request() {
        Ok(request) => discovery::discover_notebook(&state, &request).await,
        Err(e) => Err(e),
    };
    match outcome {
        Ok(started) => Ok(redirect(&headers, &format!("/notebooks/{}", seg(&started.notebook_id)))),
        Err(e) if e.status().is_client_error() => {
            let profiles = load_profiles(&state).await?;
            let form = discover_form(&profiles, &values, Some(&e.detail()));
            if is_htmx(&headers) {
                return Ok((e.status(), form).into_response());
            }
            let ctx = PageContext::new(&state, "/notebooks");
            let body = html! {
                (page_head("Discover a notebook", None, html! { a.btn href="/notebooks" { "Back to notebooks" } }))
                div.card.disc-page { (form) }
            };
            Ok((e.status(), page(&ctx, "Discover a notebook", body)).into_response())
        }
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{header, HeaderMap, Method, Request, StatusCode};
    use http_body_util::BodyExt;

    use crate::api::episode_profiles::tests::create_episode_profile;
    use crate::api::speaker_profiles::tests::{create_speaker_profile, model};
    use crate::services::discovery::tests::{mock_models, mock_search, pages};
    use crate::testing::TestApp;

    async fn form(app: &TestApp, body: &str, htmx: bool) -> (StatusCode, HeaderMap, String) {
        let mut builder = Request::builder()
            .method(Method::POST)
            .uri("/notebooks/discover")
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded");
        if htmx {
            builder = builder.header("hx-request", "true");
        }
        let response = app.send(builder.body(Body::from(body.to_string())).unwrap()).await;
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, headers, String::from_utf8_lossy(&bytes).into_owned())
    }

    #[tokio::test]
    async fn controls_render_buttons_dialog_and_profiles() {
        let app = TestApp::new().await;
        let llm = model(&app, "llm1", "language").await;
        let tts = model(&app, "tts1", "text_to_speech").await;
        let speaker = create_speaker_profile(&app, "disc_speakers", Some(&tts)).await;
        create_episode_profile(&app, "disc_show", speaker["id"].as_str().unwrap(), &llm).await;

        let markup = super::discover_controls(&app.state).await.unwrap().into_string();
        assert!(markup.contains("Discover notebook") && markup.contains("Discover + podcast"), "{markup}");
        assert_eq!(markup.matches("<dialog").count(), 1);
        assert!(markup.contains(r##"data-open="#disc-dialog""##));
        assert!(markup.contains(r#"<option value="disc_show">disc_show</option>"#), "{markup}");
        assert!(markup.contains(r#"<option value="disc_speakers">disc_speakers</option>"#), "{markup}");
        assert!(markup.contains(r#"name="num_sources" min="1" max="50" step="1" value="10""#), "{markup}");

        // The notebooks page carries it in its header.
        let (status, page) = app.text(Method::GET, "/notebooks", None).await;
        assert_eq!(status, StatusCode::OK);
        assert!(page.contains(r#"id="disc-dialog""#));
    }

    #[tokio::test]
    async fn a_successful_post_goes_to_the_new_notebook() {
        let app = TestApp::new().await;
        mock_models(&app, &["Heat Pumps", r#"["heat pumps"]"#, r#"{"selected": [1, 2, 3]}"#]);
        let _server = mock_search(&app, |uri| pages(uri, 3)).await;

        let (status, headers, body) = form(&app, "mode=notebook&prompt=heat+pumps&num_sources=3", true).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let target = headers["hx-redirect"].to_str().unwrap().to_string();
        assert!(target.starts_with("/notebooks/notebook:"), "{target}");
        let notebook_id = target.trim_start_matches("/notebooks/");
        let jobs = app.state.jobs.list(Some("discover_notebook"), None, 10).await.unwrap();
        assert_eq!(app.wait_for_job(&jobs[0].id).await, "completed");
        let (_, notebook) = app.get(&format!("/api/notebooks/{notebook_id}")).await;
        assert_eq!((notebook["name"].as_str(), notebook["source_count"].as_i64()), (Some("Heat Pumps"), Some(3)));

        // A plain form post is redirected with a 303.
        let (status, headers, _) = form(&app, "prompt=more+heat+pumps", false).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert!(headers[header::LOCATION].to_str().unwrap().starts_with("/notebooks/notebook:"));
    }

    #[tokio::test]
    async fn rejected_posts_show_the_form_with_the_reason() {
        let app = TestApp::new().await;
        mock_models(&app, &["Title"]);

        let (status, _, body) = form(&app, "mode=notebook&prompt=ab&num_sources=10", true).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(body.contains(r#"id="disc-form""#) && body.contains("at least 3 characters"), "{body}");
        assert!(!body.contains("<html"), "htmx gets the form alone: {body}");

        let (status, _, body) = form(&app, "prompt=heat+pumps&num_sources=99", true).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(body.contains("from 1 to 50") && body.contains("heat pumps"), "{body}");

        let (status, _, body) = form(&app, "mode=podcast&prompt=heat+pumps&episode_profile=", true).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(body.contains("Choose an episode profile"), "{body}");
        assert!(body.contains(r#"value="podcast" checked"#), "the mode is kept: {body}");

        let (status, _, body) = form(&app, "mode=podcast&prompt=heat+pumps&episode_profile=nope", true).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body.contains("Episode profile 'nope' not found"), "{body}");

        // Without htmx the form comes back as a page.
        let (status, _, body) = form(&app, "prompt=ab", false).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(body.contains("<html") && body.contains("at least 3 characters"), "{body}");
        assert!(app.state.jobs.list(Some("discover_notebook"), None, 10).await.unwrap().is_empty());
    }
}
