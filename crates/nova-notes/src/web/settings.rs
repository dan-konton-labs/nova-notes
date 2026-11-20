//! Web UI for the "Manage" pages: AI setup (`/settings/models`), the
//! content-processing settings (`/settings`) and the advanced tools
//! (`/advanced`: system info and the embedding rebuild).
//!
//! Every page works as plain HTML forms; with htmx the forms open in one
//! shared dialog and answer with a fragment instead of a redirect. API keys
//! are never rendered: credentials only expose `has_api_key`.

use std::collections::{BTreeMap, HashSet};
use std::time::Duration;

use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{Form, Router};
use maud::{html, Markup};
use serde::Deserialize;

use super::{dom_id, empty_state, is_htmx, page, page_head, status_badge, with_toast, Fields, PageContext, WebResult};
use crate::domain::{ContentSettings, DefaultModels};
use crate::error::AppError;
use crate::jobs::status as job_status;
use crate::services::credentials::{
    self, CredentialResponse, CredentialStatus, CredentialUpdate, NewCredential, ProviderInfoResponse, RegisterModel,
};
use crate::services::embedding::{self, RebuildMode, RebuildRequest, RebuildStatus};
use crate::services::models::{self, DefaultModelsUpdate, ModelResponse, NewModel};
use crate::services::settings::{self as app_settings, Capabilities, SettingsUpdate};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/settings", get(settings_page).post(settings_save))
        .route("/settings/models", get(models_page))
        .route("/settings/models/credentials", post(credential_create))
        .route("/settings/models/credentials/new", get(credential_new))
        .route("/settings/models/credentials/{id}", post(credential_update))
        .route("/settings/models/credentials/{id}/edit", get(credential_edit))
        .route("/settings/models/credentials/{id}/delete", get(credential_delete_confirm).post(credential_delete))
        .route("/settings/models/credentials/{id}/test", post(credential_test))
        .route("/settings/models/credentials/{id}/discover", get(credential_discover))
        .route("/settings/models/credentials/{id}/register", post(credential_register))
        .route("/settings/models/migrate-env", post(migrate_env))
        .route("/settings/models/defaults", post(defaults_save))
        .route("/settings/models/auto-assign", post(auto_assign))
        .route("/settings/models/models", post(model_create))
        .route("/settings/models/models/{id}/test", post(model_test))
        .route("/settings/models/models/{id}/delete", post(model_delete))
        .route("/advanced", get(advanced_page))
        .route("/advanced/rebuild", post(rebuild_start))
        .route("/advanced/rebuild/estimate", post(rebuild_estimate))
        .route("/advanced/rebuild/{id}/status", get(rebuild_status))
}

const MODELS_PATH: &str = "/settings/models";
const BODY_TARGET: &str = "#set-models-body";
const DIALOG_TARGET: &str = "#set-dialog-content";

// ---------------------------------------------------------------------------
// Small shared pieces
// ---------------------------------------------------------------------------

/// A form's htmx attributes, or none when it is rendered on its own page.
struct Hx<'a> {
    post: Option<&'a str>,
    target: &'a str,
}

fn type_label(model_type: &str) -> &str {
    match model_type {
        "language" => "Language",
        "embedding" => "Embedding",
        "text_to_speech" => "TTS",
        "speech_to_text" => "STT",
        other => other,
    }
}

/// A full page holding one card, for the no-JavaScript versions of the dialogs.
fn card_page(state: &AppState, path: &str, title: &str, body: Markup) -> Markup {
    let ctx = PageContext::new(state, path);
    page(
        &ctx,
        title,
        html! {
            div.crumbs { a href=(MODELS_PATH) { "Models" } " / " (title) }
            div.card.stack.set-narrow { h2 { (title) } (body) }
        },
    )
}

/// The dialog contents for htmx, or a full page otherwise.
fn dialog_or_page(state: &AppState, headers: &HeaderMap, path: &str, title: &str, body: Markup) -> Response {
    if is_htmx(headers) {
        html! {
            div.dialog-head {
                h3 { (title) }
                button.btn.btn-ghost.btn-sm type="button" data-close aria-label="Close" { "Close" }
            }
            div.dialog-body { (body) }
        }
        .into_response()
    } else {
        card_page(state, path, title, body).into_response()
    }
}

/// The shared dialog the "open in a dialog" links load into.
fn shared_dialog() -> Markup {
    html! {
        dialog #set-dialog.wide aria-label="Settings dialog" {
            div #set-dialog-content {
                div.dialog-body { span.spinner {} }
            }
        }
    }
}

/// A link that opens `href` in the shared dialog with htmx, or navigates to
/// it as a page without JavaScript.
fn dialog_link(href: &str, label: &str, class: &str, disabled: bool) -> Markup {
    html! {
        @if disabled {
            button class=(class) type="button" disabled { (label) }
        } @else {
            a class=(class) href=(href) hx-get=(href) hx-target=(DIALOG_TARGET) data-open="#set-dialog" data-set-dialog { (label) }
        }
    }
}

// ---------------------------------------------------------------------------
// /settings/models: data
// ---------------------------------------------------------------------------

/// Default-model slots as the old page grouped them.
struct Slot {
    key: &'static str,
    label: &'static str,
    description: &'static str,
    model_type: &'static str,
    required: bool,
    falls_back_to_chat: bool,
    short: &'static str,
}

const PRIMARY_SLOTS: [Slot; 4] = [
    Slot { key: "default_chat_model", label: "Chat Model", description: "Used for chat conversations", model_type: "language", required: true, falls_back_to_chat: false, short: "Chat" },
    Slot { key: "default_embedding_model", label: "Embedding Model", description: "Used for semantic search and vector embeddings", model_type: "embedding", required: true, falls_back_to_chat: false, short: "Embedding" },
    Slot { key: "default_text_to_speech_model", label: "Text-to-Speech Model", description: "Used for podcast generation", model_type: "text_to_speech", required: false, falls_back_to_chat: false, short: "TTS" },
    Slot { key: "default_speech_to_text_model", label: "Speech-to-Text Model", description: "Used for audio transcription", model_type: "speech_to_text", required: false, falls_back_to_chat: false, short: "STT" },
];

const ADVANCED_SLOTS: [Slot; 3] = [
    Slot { key: "default_transformation_model", label: "Transformation Model", description: "Used for summaries, insights, and transformations", model_type: "language", required: false, falls_back_to_chat: true, short: "Transform" },
    Slot { key: "default_tools_model", label: "Tools Model", description: "Used for function calling - OpenAI or Anthropic recommended", model_type: "language", required: false, falls_back_to_chat: true, short: "Tools" },
    Slot { key: "large_context_model", label: "Large Context Model", description: "Used for processing large documents - Gemini recommended", model_type: "language", required: false, falls_back_to_chat: true, short: "Large Ctx" },
];

fn all_slots() -> impl Iterator<Item = &'static Slot> {
    PRIMARY_SLOTS.iter().chain(ADVANCED_SLOTS.iter())
}

struct ModelsData {
    providers: Vec<ProviderInfoResponse>,
    credentials: Vec<CredentialResponse>,
    models: Vec<ModelResponse>,
    defaults: DefaultModels,
    status: CredentialStatus,
}

impl ModelsData {
    async fn load(state: &AppState) -> Result<Self, AppError> {
        Ok(Self {
            providers: credentials::list_providers(),
            credentials: credentials::list_credentials(state, None).await?,
            models: models::list_models(state, None).await?,
            defaults: models::get_default_models(state).await?,
            status: credentials::provider_status(state).await,
        })
    }

    fn encryption_ready(&self) -> bool {
        self.status.encryption_configured
    }

    /// Providers configured only through environment variables.
    fn providers_to_migrate(&self) -> Vec<&ProviderInfoResponse> {
        self.providers
            .iter()
            .filter(|p| p.env_configured && self.status.source.get(&p.name).map(String::as_str) == Some("environment"))
            .collect()
    }

    fn credentials_of(&self, provider: &str) -> Vec<&CredentialResponse> {
        self.credentials.iter().filter(|c| c.provider == provider).collect()
    }

    fn models_of_type(&self, model_type: &str) -> Vec<&ModelResponse> {
        let mut list: Vec<&ModelResponse> = self.models.iter().filter(|m| m.model_type == model_type).collect();
        list.sort_by_key(|m| m.name.to_lowercase());
        list
    }

    /// Model id → the short name of the default slot it fills.
    fn default_slots(&self) -> BTreeMap<String, &'static str> {
        all_slots()
            .filter_map(|slot| self.defaults.slot(slot.key).map(|id| (id.to_string(), slot.short)))
            .collect()
    }

    fn missing_required(&self) -> Vec<&'static str> {
        all_slots()
            .filter(|slot| slot.required)
            .filter(|slot| match self.defaults.slot(slot.key) {
                Some(id) => !self.models.iter().any(|m| m.id == id && m.model_type == slot.model_type),
                None => true,
            })
            .map(|slot| slot.label)
            .collect()
    }

    fn display_name<'a>(&'a self, provider: &'a str) -> &'a str {
        self.providers.iter().find(|p| p.name == provider).map(|p| p.display_name.as_str()).unwrap_or(provider)
    }
}

fn provider_info(provider: &str) -> Result<ProviderInfoResponse, AppError> {
    credentials::list_providers()
        .into_iter()
        .find(|p| p.name == provider)
        .ok_or_else(|| AppError::invalid(credentials::supported_provider_message()))
}

// ---------------------------------------------------------------------------
// /settings/models: rendering
// ---------------------------------------------------------------------------

async fn models_page(State(state): State<AppState>) -> WebResult<Markup> {
    let data = ModelsData::load(&state).await?;
    let ctx = PageContext::new(&state, MODELS_PATH);
    Ok(page(
        &ctx,
        "Models",
        html! {
            (page_head("Models", Some("Connect your AI providers and securely store their credentials."), html! {}))
            (models_body(&data, None))
            (shared_dialog())
        },
    ))
}

/// Everything below the heading; mutating forms swap it with htmx.
fn models_body(data: &ModelsData, notice: Option<Markup>) -> Markup {
    let encryption_ready = data.encryption_ready();
    let to_migrate = data.providers_to_migrate();
    let mut providers: Vec<&ProviderInfoResponse> = data.providers.iter().collect();
    // Configured providers first; the registry owns the base order.
    providers.sort_by_key(|p| data.credentials_of(&p.name).is_empty());
    html! {
        div #set-models-body.stack {
            @if let Some(notice) = notice { (notice) }
            @if !encryption_ready {
                div.flash.bad role="alert" {
                    strong { "Encryption key not configured" }
                    p.small { "Set the " code { "OPEN_NOTEBOOK_ENCRYPTION_KEY" } " environment variable to any secret string to enable storing API keys in the database." }
                }
            } @else if !to_migrate.is_empty() {
                div.flash.warn.row-between {
                    div {
                        strong { "Environment Variables Detected" }
                        p.small {
                            (to_migrate.len()) " API key(s) are configured via environment variables and can be migrated to the database for easier management: "
                            (to_migrate.iter().map(|p| p.display_name.as_str()).collect::<Vec<_>>().join(", "))
                        }
                    }
                    form.inline method="post" action="/settings/models/migrate-env" hx-post="/settings/models/migrate-env" hx-target=(BODY_TARGET) hx-swap="outerHTML" {
                        button.btn.btn-sm type="submit" { "Migrate to Database" }
                    }
                }
            }
            (defaults_card(data))
            div.stack #set-providers {
                @for provider in providers {
                    (provider_card(data, provider, encryption_ready))
                }
            }
            (all_models_card(data))
            p.small.set-help {
                a href="https://github.com/lfnovo/open-notebook/blob/main/docs/5-CONFIGURATION/ai-providers.md" target="_blank" rel="noopener noreferrer" {
                    "Learn how to connect AI providers →"
                }
            }
        }
    }
}

fn defaults_card(data: &ModelsData) -> Markup {
    let missing = data.missing_required();
    let chat_name = data
        .defaults
        .slot("default_chat_model")
        .and_then(|id| data.models.iter().find(|m| m.id == id))
        .map(|m| m.name.as_str());
    html! {
        section.card.stack #set-defaults {
            div {
                h2 { "Default Model Assignments" }
                p.muted.small { "Configure which models to use for different purposes across Nova-Notes" }
            }
            @if !missing.is_empty() {
                div.flash.warn.row-between {
                    span { "Missing required models: " (missing.join(", ")) ". Nova-Notes may not function properly without these." }
                    form.inline method="post" action="/settings/models/auto-assign" hx-post="/settings/models/auto-assign" hx-target=(BODY_TARGET) hx-swap="outerHTML" {
                        button.btn.btn-sm type="submit" { "Auto-assign Defaults" }
                    }
                }
            }
            form.stack method="post" action="/settings/models/defaults"
                hx-post="/settings/models/defaults" hx-trigger="change" hx-target=(BODY_TARGET) hx-swap="outerHTML" {
                div.set-slots.set-slots-4 {
                    @for slot in &PRIMARY_SLOTS { (slot_select(data, slot, chat_name, false)) }
                }
                div.set-advanced {
                    p.faint.small { "Advanced" }
                    div.set-slots.set-slots-3 {
                        @for slot in &ADVANCED_SLOTS { (slot_select(data, slot, chat_name, true)) }
                    }
                }
                noscript { div.row { button.btn.btn-primary type="submit" { "Save defaults" } } }
            }
        }
    }
}

fn slot_select(data: &ModelsData, slot: &Slot, chat_name: Option<&str>, show_description: bool) -> Markup {
    let available = data.models_of_type(slot.model_type);
    let current = data.defaults.slot(slot.key).filter(|id| available.iter().any(|m| m.id == *id));
    let id = format!("set-slot-{}", slot.key);
    let invalid = slot.required && current.is_none() && !available.is_empty();
    let hint = if slot.required || current.is_some() {
        None
    } else if slot.falls_back_to_chat {
        chat_name.map(|name| format!("Using chat model ({name})"))
    } else if slot.model_type == "text_to_speech" {
        Some("Not set — audio generation unavailable until configured".to_string())
    } else if slot.model_type == "speech_to_text" {
        Some("Not set — transcription unavailable until configured".to_string())
    } else {
        None
    };
    html! {
        div.field {
            label for=(id) {
                (slot.label)
                @if slot.required { span.set-required { "*" } }
            }
            select #(id) name=(slot.key) .set-invalid[invalid]
                data-set-embedding[slot.key == "default_embedding_model"]
                data-current=(current.unwrap_or("")) {
                @if slot.required {
                    @if current.is_none() {
                        option value="" selected { (if invalid { "Required - Select a model" } else { "Select a model" }) }
                    }
                } @else {
                    option value="" selected[current.is_none()] {
                        (if slot.falls_back_to_chat { "Use fallback (chat default)" } else { "None" })
                    }
                }
                @for model in &available {
                    option value=(model.id) selected[current == Some(model.id.as_str())] { (model.name) " · " (model.provider) }
                }
            }
            @if let Some(hint) = hint { p.hint.set-italic { (hint) } }
            @if show_description { p.hint { (slot.description) } }
        }
    }
}

fn provider_card(data: &ModelsData, provider: &ProviderInfoResponse, encryption_ready: bool) -> Markup {
    let creds = data.credentials_of(&provider.name);
    let configured = !creds.is_empty();
    let modalities: Vec<&str> = if provider.modalities.is_empty() {
        vec!["language"]
    } else {
        provider.modalities.iter().map(String::as_str).collect()
    };
    let active: HashSet<&str> = data
        .models
        .iter()
        .filter(|m| creds.iter().any(|c| Some(&c.id) == m.credential.as_ref()))
        .map(|m| m.model_type.as_str())
        .collect();
    let from_env = data.status.source.get(&provider.name).map(String::as_str) == Some("environment");
    let add_href = format!("/settings/models/credentials/new?provider={}", provider.name);
    html! {
        section.card.stack-sm.set-provider .set-configured[configured] id=(format!("set-provider-{}", provider.name)) {
            div.row-between {
                div.row {
                    h3.set-provider-name .faint[!configured] { (provider.display_name) }
                    @for modality in &modalities {
                        span.badge .info[active.contains(modality)] { (type_label(modality)) }
                    }
                }
                div.row {
                    @if from_env { span.badge.warn title="Configured through environment variables" { "From environment" } }
                    @if configured {
                        span.badge.ok { "Configured" }
                    } @else {
                        span.badge { "Not configured" }
                    }
                }
            }
            @for credential in &creds {
                (credential_item(data, credential))
            }
            (dialog_link(&add_href, "+ Add Configuration", "btn btn-sm set-add", !encryption_ready))
        }
    }
}

fn credential_item(data: &ModelsData, credential: &CredentialResponse) -> Markup {
    let linked: Vec<&ModelResponse> =
        data.models.iter().filter(|m| m.credential.as_deref() == Some(credential.id.as_str())).collect();
    let active: HashSet<&str> = linked.iter().map(|m| m.model_type.as_str()).collect();
    let slots = data.default_slots();
    let broken = credential.decryption_error.is_some();
    let base = format!("/settings/models/credentials/{}", credential.id);
    let test_target = format!("set-test-{}", dom_id(&credential.id));
    html! {
        div.item.set-cred {
            div.stack-sm.grow {
                div.row-between {
                    div.row {
                        span.item-title { (credential.name) }
                        @for modality in &credential.modalities {
                            span.badge .info[active.contains(modality.as_str())] { (type_label(modality)) }
                        }
                        @if credential.has_api_key { span.badge title="An API key is stored (encrypted)" { "Key" } }
                    }
                    div.row {
                        span id=(test_target) {}
                        form.inline method="post" action=(format!("{base}/test")) hx-post=(format!("{base}/test")) hx-target=(format!("#{test_target}")) {
                            button.btn.btn-ghost.btn-sm type="submit" disabled[broken] title="Test Connection" {
                                "Test" span.spinner.htmx-indicator {}
                            }
                        }
                        (dialog_link(&format!("{base}/discover"), "Models", "btn btn-ghost btn-sm", broken))
                        (dialog_link(&format!("{base}/edit"), "Edit", "btn btn-ghost btn-sm", broken))
                        (dialog_link(&format!("{base}/delete"), "Delete", "btn btn-ghost btn-sm set-danger", false))
                    }
                }
                @if broken {
                    div.flash.warn {
                        strong { "Decryption Error" }
                        p.small { "This credential's API key could not be decrypted. The encryption key may have changed. Delete this credential and re-create it with the correct key." }
                    }
                }
                @for model_type in crate::domain::model::MODEL_TYPES {
                    @let of_type: Vec<&&ModelResponse> = linked.iter().filter(|m| m.model_type == model_type).collect();
                    @if !of_type.is_empty() {
                        div.set-model-row {
                            span.badge.info { (type_label(model_type)) }
                            div.row.set-chips {
                                @for model in of_type { (model_chip(model, slots.get(&model.id).copied())) }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A model name with its default slot and test / delete buttons.
fn model_chip(model: &ModelResponse, slot: Option<&str>) -> Markup {
    let base = format!("/settings/models/models/{}", model.id);
    html! {
        span.set-chip .set-chip-default[slot.is_some()] {
            code { (model.name) }
            @if let Some(slot) = slot { span.small { "(" (slot) ")" } }
            form.inline method="post" action=(format!("{base}/test")) hx-post=(format!("{base}/test")) hx-target=(DIALOG_TARGET) {
                button.set-chip-btn type="submit" title="Test Model" data-open="#set-dialog" data-set-dialog { "test" }
            }
            form.inline method="post" action=(format!("{base}/delete")) hx-post=(format!("{base}/delete")) hx-target=(BODY_TARGET) hx-swap="outerHTML"
                hx-confirm=(format!("Delete the model {}?", model.name)) {
                button.set-chip-btn.set-danger type="submit" title="Delete Model" aria-label=(format!("Delete {}", model.name)) { "×" }
            }
        }
    }
}

fn all_models_card(data: &ModelsData) -> Markup {
    let slots = data.default_slots();
    let credential_name = |id: &Option<String>| {
        id.as_ref()
            .and_then(|id| data.credentials.iter().find(|c| &c.id == id))
            .map(|c| c.name.clone())
            .unwrap_or_else(|| "Provider default".to_string())
    };
    let mut list: Vec<&ModelResponse> = data.models.iter().collect();
    list.sort_by_key(|m| (m.provider.clone(), m.model_type.clone(), m.name.to_lowercase()));
    html! {
        section.card.stack #set-models {
            div {
                h2 { "All Models" }
                p.muted.small { "Every configured model, including those that use a provider's environment variables." }
            }
            @if list.is_empty() {
                (empty_state("No models", "Add a configuration above and use its Models button to discover models, or add one by name below."))
            } @else {
                div.table-wrap {
                    table {
                        thead { tr { th { "Model" } th { "Provider" } th { "Type" } th { "Configuration" } th { "" } } }
                        tbody {
                            @for model in &list {
                                tr {
                                    td {
                                        code { (model.name) }
                                        @if let Some(slot) = slots.get(&model.id) { " " span.badge.ok { (slot) } }
                                    }
                                    td { (data.display_name(&model.provider)) }
                                    td { (type_label(&model.model_type)) }
                                    td.muted { (credential_name(&model.credential)) }
                                    td.set-actions {
                                        @let base = format!("/settings/models/models/{}", model.id);
                                        form.inline method="post" action=(format!("{base}/test")) hx-post=(format!("{base}/test")) hx-target=(DIALOG_TARGET) {
                                            button.btn.btn-ghost.btn-sm type="submit" data-open="#set-dialog" data-set-dialog { "Test" }
                                        }
                                        form.inline method="post" action=(format!("{base}/delete")) hx-post=(format!("{base}/delete")) hx-target=(BODY_TARGET) hx-swap="outerHTML"
                                            hx-confirm=(format!("Delete the model {}?", model.name)) {
                                            button.btn.btn-ghost.btn-sm.set-danger type="submit" { "Delete" }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            details.set-add-model {
                summary { "Add a model by name" }
                form.set-form-grid method="post" action="/settings/models/models" hx-post="/settings/models/models" hx-target=(BODY_TARGET) hx-swap="outerHTML" {
                    div.field {
                        label for="set-new-model-provider" { "Provider" }
                        select #set-new-model-provider name="provider" required {
                            @for provider in &data.providers { option value=(provider.name) { (provider.display_name) } }
                        }
                    }
                    div.field {
                        label for="set-new-model-name" { "Model name" }
                        input #set-new-model-name type="text" name="name" required placeholder="gpt-4o-mini";
                    }
                    div.field {
                        label for="set-new-model-type" { "Model Type" }
                        select #set-new-model-type name="type" {
                            @for model_type in crate::domain::model::MODEL_TYPES { option value=(model_type) { (type_label(model_type)) } }
                        }
                    }
                    div.field {
                        label for="set-new-model-credential" { "Configuration" }
                        select #set-new-model-credential name="credential" {
                            option value="" { "Provider default" }
                            @for credential in &data.credentials {
                                option value=(credential.id) { (data.display_name(&credential.provider)) " · " (credential.name) }
                            }
                        }
                    }
                    div.row { button.btn.btn-primary type="submit" { "Add model" } }
                }
            }
        }
    }
}

/// Re-render the models body for htmx, or go back to the page.
async fn models_done(state: &AppState, headers: &HeaderMap, toast: Option<String>, notice: Option<Markup>) -> WebResult<Response> {
    if !is_htmx(headers) {
        return Ok(Redirect::to(MODELS_PATH).into_response());
    }
    let data = ModelsData::load(state).await?;
    let response = models_body(&data, notice).into_response();
    Ok(match toast {
        Some(message) => with_toast(response, &message),
        None => response,
    })
}

// ---------------------------------------------------------------------------
// Credentials
// ---------------------------------------------------------------------------

const OMLX_DEFAULT_BASE_URL: &str = "http://localhost:11435/v1";

fn credential_form(provider: &ProviderInfoResponse, credential: Option<&CredentialResponse>, hx: Hx<'_>) -> Markup {
    let name = provider.name.as_str();
    let is_vertex = name == "vertex";
    let is_ollama = name == "ollama";
    let is_omlx = name == "omlx";
    let is_azure = name == "azure";
    let is_compatible = name == "openai_compatible";
    let requires_key = !matches!(name, "vertex" | "ollama" | "omlx" | "openai_compatible");
    let editing = credential.is_some();
    let action = match credential {
        Some(c) => format!("/settings/models/credentials/{}", c.id),
        None => "/settings/models/credentials".to_string(),
    };
    let value = |pick: fn(&CredentialResponse) -> Option<&String>| credential.and_then(pick).cloned().unwrap_or_default();
    let modalities: Vec<&str> = if provider.modalities.is_empty() {
        vec!["language"]
    } else {
        provider.modalities.iter().map(String::as_str).collect()
    };
    let chosen: Vec<String> = credential.map(|c| c.modalities.clone()).unwrap_or_else(|| modalities.iter().map(|m| m.to_string()).collect());
    let base_url_default = if editing { value(|c| c.base_url.as_ref()) } else if is_omlx { OMLX_DEFAULT_BASE_URL.to_string() } else { String::new() };
    html! {
        form.stack method="post" action=(action)
            hx-post=[hx.post] hx-target=[hx.post.map(|_| hx.target)] hx-swap=[hx.post.map(|_| "outerHTML")] data-close-on-success[hx.post.is_some()] {
            input type="hidden" name="provider" value=(name);
            div.field {
                label for="set-cred-name" { "Configuration Name" }
                input #set-cred-name type="text" name="name" value=(credential.map(|c| c.name.as_str()).unwrap_or(""))
                    placeholder=(format!("{} Production", provider.display_name)) required[editing];
                p.hint { "A descriptive name for this configuration (e.g., 'Production', 'Development')" }
            }
            @if is_vertex {
                div.field {
                    label for="set-cred-project" { "GCP Project ID" }
                    input #set-cred-project type="text" name="project" value=(value(|c| c.project.as_ref())) placeholder="my-gcp-project" required;
                }
                div.field {
                    label for="set-cred-location" { "Region" }
                    input #set-cred-location type="text" name="location" value=(value(|c| c.location.as_ref())) placeholder="us-central1" required;
                }
                div.field {
                    label for="set-cred-path" { "Service Account JSON Path " span.faint { "(optional)" } }
                    input #set-cred-path type="text" name="credentials_path" value=(value(|c| c.credentials_path.as_ref())) placeholder="/path/to/service-account.json";
                }
            } @else {
                div.field {
                    label for="set-cred-key" {
                        "API Key"
                        @if !requires_key || editing { " " span.faint { "(optional)" } }
                    }
                    div.set-key {
                        input #set-cred-key type="password" name="api_key" autocomplete="off" spellcheck="false"
                            placeholder=(if editing { "••••••••••••" } else { "sk-..." }) required[requires_key && !editing];
                        button.btn.btn-ghost.btn-sm type="button" data-set-reveal="#set-cred-key" { "Show" }
                    }
                    @if editing {
                        p.hint {
                            @if credential.is_some_and(|c| c.has_api_key) { "A key is stored. " }
                            "Leave blank to keep the existing API key"
                        }
                    }
                    @if let Some(docs) = &provider.docs_url {
                        a.small href=(docs) target="_blank" rel="noopener noreferrer" { "Get API Key →" }
                    }
                }
                div.field {
                    label for="set-cred-url" { (if is_azure { "Endpoint (Base URL)" } else { "Base URL" }) }
                    input #set-cred-url type="url" name="base_url" value=(base_url_default)
                        placeholder=(if is_ollama { "http://localhost:11434" } else if is_omlx { OMLX_DEFAULT_BASE_URL } else if is_azure { "https://my-resource.openai.azure.com" } else { "https://api.example.com/v1" });
                    p.hint {
                        @if is_compatible {
                            "Include the API version path required by your server. For LM Studio in Docker, use http://host.docker.internal:1234/v1."
                        } @else if is_azure {
                            "Your Azure OpenAI resource endpoint."
                        } @else {
                            "Only change this if you need to override the provider's default API endpoint."
                        }
                    }
                }
                @if is_azure {
                    div.field {
                        label for="set-cred-version" { "API Version" }
                        input #set-cred-version type="text" name="api_version" value=(value(|c| c.api_version.as_ref())) placeholder="2024-10-21";
                    }
                }
            }
            @if is_ollama {
                div.field {
                    label for="set-cred-ctx" { "Context Window (num_ctx) " span.faint { "(optional)" } }
                    input #set-cred-ctx type="number" min="1" name="num_ctx" placeholder="8192"
                        value=(credential.and_then(|c| c.num_ctx).map(|n| n.to_string()).unwrap_or_default());
                    p.hint { "Maximum context window for Ollama models. Leave empty to use the default (8192). Increase only if your hardware can handle a larger window." }
                }
            }
            @if modalities.len() > 1 {
                fieldset.set-fieldset {
                    legend { "Model types" }
                    input type="hidden" name="modalities_shown" value="1";
                    div.row {
                        @for modality in &modalities {
                            label.check {
                                input type="checkbox" name="modalities" value=(modality) checked[chosen.iter().any(|c| c == modality)];
                                (type_label(modality))
                            }
                        }
                    }
                }
            } @else {
                @for modality in &modalities { input type="hidden" name="modalities" value=(modality); }
            }
            div.row.set-foot {
                @if hx.post.is_some() {
                    button.btn type="button" data-close { "Cancel" }
                } @else {
                    a.btn href=(MODELS_PATH) { "Cancel" }
                }
                button.btn.btn-primary type="submit" { (if editing { "Save" } else { "Add Configuration" }) }
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct NewCredentialQuery {
    provider: Option<String>,
}

async fn credential_new(State(state): State<AppState>, headers: HeaderMap, Query(query): Query<NewCredentialQuery>) -> WebResult<Response> {
    let provider = provider_info(query.provider.as_deref().unwrap_or(""))?;
    let hx = Hx { post: is_htmx(&headers).then_some("/settings/models/credentials"), target: BODY_TARGET };
    let title = format!("Add Configuration · {}", provider.display_name);
    Ok(dialog_or_page(&state, &headers, MODELS_PATH, &title, credential_form(&provider, None, hx)))
}

async fn credential_edit(State(state): State<AppState>, headers: HeaderMap, Path(id): Path<String>) -> WebResult<Response> {
    let credential = credentials::get_credential(&state, &id).await?;
    let provider = provider_info(&credential.provider)?;
    let action = format!("/settings/models/credentials/{}", credential.id);
    let hx = Hx { post: is_htmx(&headers).then_some(action.as_str()), target: BODY_TARGET };
    let title = format!("Edit Configuration · {}", provider.display_name);
    let form = credential_form(&provider, Some(&credential), hx);
    Ok(dialog_or_page(&state, &headers, MODELS_PATH, &title, form))
}

fn parse_num_ctx(fields: &Fields) -> Result<Option<Option<i64>>, AppError> {
    match fields.present("num_ctx") {
        None => Ok(None),
        Some(None) => Ok(Some(None)),
        Some(Some(raw)) => raw
            .parse::<i64>()
            .ok()
            .filter(|n| *n > 0)
            .map(|n| Some(Some(n)))
            .ok_or_else(|| AppError::invalid("The context window must be a positive whole number")),
    }
}

fn chosen_modalities(fields: &Fields) -> Result<Vec<String>, AppError> {
    let modalities = fields.all("modalities");
    if modalities.is_empty() {
        return Err(AppError::invalid("Choose at least one model type"));
    }
    Ok(modalities)
}

async fn credential_create(State(state): State<AppState>, headers: HeaderMap, Form(pairs): Form<Vec<(String, String)>>) -> WebResult<Response> {
    let fields = Fields(pairs);
    let provider = provider_info(fields.raw("provider").unwrap_or(""))?;
    let input = NewCredential {
        name: fields.text("name").unwrap_or_else(|| format!("{} Config", provider.display_name)),
        provider: provider.name.clone(),
        modalities: chosen_modalities(&fields)?,
        api_key: fields.text("api_key"),
        base_url: fields.text("base_url"),
        api_version: fields.text("api_version"),
        project: fields.text("project"),
        location: fields.text("location"),
        credentials_path: fields.text("credentials_path"),
        num_ctx: parse_num_ctx(&fields)?.flatten(),
        ..NewCredential::default()
    };
    let created = credentials::create_credential(&state, &input).await?;
    models_done(&state, &headers, Some(format!("Configuration '{}' saved", created.name)), None).await
}

async fn credential_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Form(pairs): Form<Vec<(String, String)>>,
) -> WebResult<Response> {
    let fields = Fields(pairs);
    let update = CredentialUpdate {
        name: fields.text("name"),
        modalities: if fields.has("modalities_shown") { Some(chosen_modalities(&fields)?) } else { None },
        // A blank key field keeps the stored key.
        api_key: fields.text("api_key"),
        base_url: fields.present("base_url"),
        api_version: fields.present("api_version"),
        project: fields.present("project"),
        location: fields.present("location"),
        credentials_path: fields.present("credentials_path"),
        num_ctx: parse_num_ctx(&fields)?,
        ..CredentialUpdate::default()
    };
    let updated = credentials::update_credential(&state, &id, &update).await?;
    models_done(&state, &headers, Some(format!("Configuration '{}' updated", updated.name)), None).await
}

async fn credential_delete_confirm(State(state): State<AppState>, headers: HeaderMap, Path(id): Path<String>) -> WebResult<Response> {
    let credential = credentials::get_credential(&state, &id).await?;
    let others: Vec<CredentialResponse> = credentials::list_credentials(&state, Some(&credential.provider))
        .await?
        .into_iter()
        .filter(|c| c.id != credential.id)
        .collect();
    let action = format!("/settings/models/credentials/{}/delete", credential.id);
    let htmx = is_htmx(&headers);
    let body = html! {
        form.stack method="post" action=(action)
            hx-post=[htmx.then_some(action.as_str())] hx-target=[htmx.then_some(BODY_TARGET)] hx-swap=[htmx.then_some("outerHTML")] data-close-on-success[htmx] {
            p { "Are you sure you want to delete '" (credential.name) "'? This cannot be undone." }
            @if credential.model_count > 0 {
                div.flash.warn.stack-sm {
                    p { "This credential has " (credential.model_count) " linked model(s)." }
                    @if !others.is_empty() {
                        div.field {
                            label for="set-migrate-to" { "Migrate models to:" }
                            select #set-migrate-to name="migrate_to" {
                                option value="" { "Select credential" }
                                @for other in &others { option value=(other.id) { (other.name) } }
                            }
                        }
                    }
                }
            }
            div.row.set-foot {
                @if htmx { button.btn type="button" data-close { "Cancel" } } @else { a.btn href=(MODELS_PATH) { "Cancel" } }
                @if credential.model_count > 0 && !others.is_empty() {
                    button.btn type="submit" name="action" value="migrate" { "Migrate & Delete" }
                }
                button.btn.btn-danger type="submit" name="action" value="delete" {
                    (if credential.model_count > 0 { "Delete with Models" } else { "Delete" })
                }
            }
        }
    };
    Ok(dialog_or_page(&state, &headers, MODELS_PATH, "Delete Configuration", body))
}

async fn credential_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Form(pairs): Form<Vec<(String, String)>>,
) -> WebResult<Response> {
    let fields = Fields(pairs);
    let migrate_to = if fields.raw("action") == Some("migrate") {
        Some(fields.text("migrate_to").ok_or_else(|| AppError::invalid("Choose the configuration to move the models to"))?)
    } else {
        None
    };
    let result = credentials::delete_credential(&state, &id, migrate_to.as_deref()).await?;
    let message = match (migrate_to.is_some(), result.deleted_models) {
        (true, _) => "Configuration deleted; its models were moved".to_string(),
        (false, 0) => "Configuration deleted successfully".to_string(),
        (false, n) => format!("Configuration deleted with {n} model(s)"),
    };
    models_done(&state, &headers, Some(message), None).await
}

async fn credential_test(State(state): State<AppState>, headers: HeaderMap, Path(id): Path<String>) -> WebResult<Response> {
    let result = credentials::test_credential(&state, &id).await;
    let fragment = html! {
        @if result.success {
            span.badge.ok title=(result.message) { "Connected" }
        } @else {
            span.badge.bad title=(result.message) { "Failed" }
            " " span.small.set-test-message { (result.message) }
        }
    };
    if is_htmx(&headers) {
        return Ok(fragment.into_response());
    }
    let body = html! {
        p { (if result.success { "Connection successful" } else { "Connection test failed" }) }
        p.muted { (result.message) }
        div.row { a.btn href=(MODELS_PATH) { "Back to models" } }
    };
    Ok(card_page(&state, MODELS_PATH, "Test Connection", body).into_response())
}

async fn credential_discover(State(state): State<AppState>, headers: HeaderMap, Path(id): Path<String>) -> WebResult<Response> {
    let credential = credentials::get_credential(&state, &id).await?;
    let provider = provider_info(&credential.provider)?;
    let found = credentials::discover_credential_models(&state, &credential.id).await?;
    let mut seen = HashSet::new();
    let discovered: Vec<_> = found.discovered.into_iter().filter(|m| seen.insert(m.name.clone())).collect();
    let existing: HashSet<String> = models::list_models(&state, None)
        .await?
        .into_iter()
        .filter(|m| m.credential.as_deref() == Some(credential.id.as_str()))
        .map(|m| m.name.to_lowercase())
        .collect();
    let types: Vec<String> = if provider.modalities.is_empty() { credential.modalities.clone() } else { provider.modalities.clone() };
    let first_type = credential.modalities.first().cloned().unwrap_or_else(|| "language".into());
    let action = format!("/settings/models/credentials/{}/register", credential.id);
    let htmx = is_htmx(&headers);
    let body = html! {
        form.stack method="post" action=(action) data-set-discover
            hx-post=[htmx.then_some(action.as_str())] hx-target=[htmx.then_some(BODY_TARGET)] hx-swap=[htmx.then_some("outerHTML")] data-close-on-success[htmx] {
            p.muted { (credential.name) }
            div.field {
                label for="set-discover-type" { "Model Type" }
                select #set-discover-type name="model_type" {
                    @for model_type in &types { option value=(model_type) selected[*model_type == first_type] { (type_label(model_type)) } }
                }
                p.hint { "Select the type for the models you want to add. If you need different types, add them in separate batches." }
            }
            input type="search" placeholder="Search models..." data-set-filter aria-label="Search models";
            @if discovered.is_empty() {
                p.muted.small { "No models found from this provider" }
            } @else {
                div.row-between {
                    button.btn.btn-sm type="button" data-set-toggle-all { "Select all" }
                    span.small.faint { (discovered.len()) " found" }
                }
                div.set-discover-list {
                    @for model in &discovered {
                        label.check.set-discover-item data-name=(model.name.to_lowercase()) {
                            input type="checkbox" name="model" value=(model.name);
                            span.truncate { (model.name) }
                            @if let Some(description) = model.description.as_ref().filter(|d| **d != model.name) {
                                span.small.faint.truncate { "(" (description) ")" }
                            }
                            @if existing.contains(&model.name.to_lowercase()) { span.badge { "added" } }
                        }
                    }
                }
            }
            div.field {
                label for="set-discover-custom" { "Or add a model by name" }
                input #set-discover-custom type="text" name="custom" placeholder="Type a model name...";
            }
            div.row.set-foot {
                @if htmx { button.btn type="button" data-close { "Cancel" } } @else { a.btn href=(MODELS_PATH) { "Cancel" } }
                button.btn.btn-primary type="submit" { "Add selected" }
            }
        }
    };
    let title = format!("Discover Models - {}", provider.display_name);
    Ok(dialog_or_page(&state, &headers, MODELS_PATH, &title, body))
}

async fn credential_register(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Form(pairs): Form<Vec<(String, String)>>,
) -> WebResult<Response> {
    let fields = Fields(pairs);
    let credential = credentials::get_credential(&state, &id).await?;
    let model_type = fields.text("model_type").unwrap_or_else(|| "language".into());
    if !crate::domain::model::is_valid_model_type(&model_type) {
        return Err(AppError::invalid(format!("Unknown model type: {model_type}")).into());
    }
    let mut names = fields.all("model");
    if let Some(custom) = fields.text("custom") {
        if !names.iter().any(|n| n.eq_ignore_ascii_case(&custom)) {
            names.push(custom);
        }
    }
    if names.is_empty() {
        return Err(AppError::invalid("Select at least one model to add").into());
    }
    let requested: Vec<RegisterModel> = names
        .into_iter()
        .map(|name| RegisterModel { name, provider: credential.provider.clone(), model_type: model_type.clone() })
        .collect();
    let result = credentials::register_credential_models(&state, &credential.id, &requested).await?;
    let message = match result.existing {
        0 => format!("Added {} model(s)", result.created),
        n => format!("Added {} model(s); {n} already registered", result.created),
    };
    models_done(&state, &headers, Some(message), None).await
}

async fn migrate_env(State(state): State<AppState>, headers: HeaderMap) -> WebResult<Response> {
    let result = credentials::migrate_from_env(&state).await?;
    let message = if !result.errors.is_empty() {
        format!("{} key(s) failed to migrate", result.errors.len())
    } else if result.migrated.is_empty() {
        "All keys are already in the database".to_string()
    } else {
        format!("{} API key(s) migrated successfully", result.migrated.len())
    };
    let notice = (!result.errors.is_empty()).then(|| {
        html! {
            div.flash.bad role="alert" {
                strong { "Some keys could not be migrated" }
                ul.small { @for error in &result.errors { li { (error) } } }
            }
        }
    });
    models_done(&state, &headers, Some(message), notice).await
}

// ---------------------------------------------------------------------------
// Models and defaults
// ---------------------------------------------------------------------------

async fn defaults_save(State(state): State<AppState>, headers: HeaderMap, Form(pairs): Form<Vec<(String, String)>>) -> WebResult<Response> {
    let fields = Fields(pairs);
    let before = models::get_default_models(&state).await?;
    let mut update = DefaultModelsUpdate::default();
    for slot in all_slots() {
        if !fields.has(slot.key) {
            continue;
        }
        let value = fields.text(slot.key);
        // A required slot can only be reassigned; an empty choice leaves it.
        if slot.required && value.is_none() {
            continue;
        }
        let field = match slot.key {
            "default_chat_model" => &mut update.default_chat_model,
            "default_embedding_model" => &mut update.default_embedding_model,
            "default_text_to_speech_model" => &mut update.default_text_to_speech_model,
            "default_speech_to_text_model" => &mut update.default_speech_to_text_model,
            "default_transformation_model" => &mut update.default_transformation_model,
            "default_tools_model" => &mut update.default_tools_model,
            "large_context_model" => &mut update.large_context_model,
            _ => continue,
        };
        *field = Some(value);
    }
    let after = models::update_default_models(&state, &update).await?;
    let old_embedding = before.slot("default_embedding_model");
    let embedding_changed = old_embedding.is_some() && old_embedding != after.slot("default_embedding_model");
    let notice = embedding_changed.then(|| {
        html! {
            div.flash.warn.row-between role="status" {
                div {
                    strong { "Important: Rebuild Required" }
                    p.small { "Changing your embedding model requires rebuilding all existing embeddings to maintain consistency. Without rebuilding, your searches may return incorrect or incomplete results." }
                }
                a.btn.btn-sm href="/advanced" { "Go to Rebuild" }
            }
        }
    });
    models_done(&state, &headers, Some("Default models saved".into()), notice).await
}

async fn auto_assign(State(state): State<AppState>, headers: HeaderMap) -> WebResult<Response> {
    let result = models::auto_assign_defaults(&state).await?;
    let message = if !result.assigned.is_empty() {
        format!("{} default models automatically assigned", result.assigned.len())
    } else if !result.missing.is_empty() {
        "No models available to assign. Please sync models first.".to_string()
    } else {
        "All default models are already configured".to_string()
    };
    models_done(&state, &headers, Some(message), None).await
}

async fn model_create(State(state): State<AppState>, headers: HeaderMap, Form(pairs): Form<Vec<(String, String)>>) -> WebResult<Response> {
    let fields = Fields(pairs);
    let input = NewModel {
        name: fields.text("name").ok_or_else(|| AppError::invalid("Enter the model name"))?,
        provider: provider_info(fields.raw("provider").unwrap_or(""))?.name,
        model_type: fields.text("type").unwrap_or_default(),
        credential: fields.text("credential"),
    };
    let created = models::create_model(&state, &input).await?;
    models_done(&state, &headers, Some(format!("Model '{}' added", created.name)), None).await
}

async fn model_delete(State(state): State<AppState>, headers: HeaderMap, Path(id): Path<String>) -> WebResult<Response> {
    models::delete_model(&state, &id).await?;
    models_done(&state, &headers, Some("Model deleted successfully".into()), None).await
}

async fn model_test(State(state): State<AppState>, headers: HeaderMap, Path(id): Path<String>) -> WebResult<Response> {
    let model = models::list_models(&state, None)
        .await?
        .into_iter()
        .find(|m| m.id == id || m.id == format!("model:{id}"))
        .ok_or_else(|| AppError::not_found("Model not found"))?;
    let result = models::test_model(&state, &model.id).await?;
    let title = if result.success { "Model Test Passed" } else { "Model Test Failed" };
    let body = html! {
        div.stack-sm {
            p.row {
                span.badge .ok[result.success] .bad[!result.success] { (if result.success { "Passed" } else { "Failed" }) }
                code { (model.name) }
            }
            p { (result.message) }
            @if let Some(details) = &result.details { pre.pre-wrap.set-details { (details) } }
            div.row.set-foot {
                @if is_htmx(&headers) { button.btn type="button" data-close { "Done" } } @else { a.btn href=(MODELS_PATH) { "Done" } }
            }
        }
    };
    Ok(dialog_or_page(&state, &headers, MODELS_PATH, title, body))
}

// ---------------------------------------------------------------------------
// /settings
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct SavedQuery {
    saved: Option<String>,
}

async fn settings_page(State(state): State<AppState>, Query(query): Query<SavedQuery>) -> WebResult<Markup> {
    let current = app_settings::get_settings(&state).await?;
    let ctx = PageContext::new(&state, "/settings");
    Ok(page(
        &ctx,
        "Settings",
        html! {
            (page_head("Settings", None, html! {}))
            (settings_form(&current, app_settings::capabilities(), query.saved.is_some()))
        },
    ))
}

fn option(value: &str, label: &str, current: Option<&str>, disabled: bool) -> Markup {
    html! { option value=(value) selected[current == Some(value)] disabled[disabled] { (label) } }
}

fn help(text: &str) -> Markup {
    html! { details.set-help-toggle { summary { "Help me choose" } p.small.muted { (text) } } }
}

fn docling_toggle(key: &str, label: &str, hint: &str, checked: bool, available: bool) -> Markup {
    let id = format!("set-{key}");
    html! {
        div.stack-sm {
            label.check for=(id) {
                // Only an enabled toggle is submitted; a disabled one leaves the stored value alone.
                @if available { input type="hidden" name=(key) value="false"; }
                input #(id) type="checkbox" name=(key) value="true" checked[checked] disabled[!available];
                (label)
            }
            p.hint { (hint) }
        }
    }
}

fn settings_form(current: &ContentSettings, capabilities: Capabilities, saved: bool) -> Markup {
    let doc = current.default_content_processing_engine_doc.as_deref();
    let url = current.default_content_processing_engine_url.as_deref();
    let embedding = current.default_embedding_option.as_deref();
    let auto_delete = current.auto_delete_files.as_deref();
    let docling = capabilities.docling_available;
    let crawl4ai = capabilities.crawl4ai_available;
    html! {
        form #set-settings.stack.set-settings method="post" action="/settings" hx-post="/settings" hx-target="this" hx-swap="outerHTML" {
            @if saved { div.flash.ok role="status" { "Settings saved" } }
            section.card.stack {
                div { h2 { "Content Processing" } p.muted.small { "Configure how documents and URLs are processed" } }
                div.field {
                    label for="set-doc-engine" { "Document Processing Engine" }
                    select #set-doc-engine name="default_content_processing_engine_doc" {
                        (option("auto", "Auto (Recommended)", doc, false))
                        (option("docling", "Docling", doc, !docling))
                        (option("simple", "Simple", doc, false))
                    }
                    @if !docling { p.hint { "Docling is optional. Enable it with OPEN_NOTEBOOK_ENABLE_DOCLING=true — it installs on first startup (a large download)." } }
                    (help("· Docling is a little slower but more accurate, specially if the documents contain tables and images. · Simple will extract any content from the document without formatting it. · Auto (recommended) will try to process through docling and default to simple."))
                }
                (docling_toggle("docling_ocr", "Enable OCR", "Extract text from scanned PDFs and images when using the Docling engine. Slower to process.", current.docling_ocr.unwrap_or(true), docling))
                (docling_toggle("docling_formulas", "Extract formulas", "Extract mathematical formulas as structured markup when using the Docling engine. Adds processing time.", current.docling_formulas.unwrap_or(false), docling))
                (docling_toggle("docling_vision", "Describe images and charts", "Use a vision model to describe images and extract chart data when using the Docling engine. Significantly slower and may call a vision model.", current.docling_vision.unwrap_or(false), docling))
                div.field {
                    label for="set-url-engine" { "URL Processing Engine" }
                    select #set-url-engine name="default_content_processing_engine_url" {
                        (option("auto", "Auto (Recommended)", url, false))
                        (option("firecrawl", "Firecrawl", url, false))
                        (option("jina", "Jina", url, false))
                        (option("crawl4ai", "Crawl4AI", url, !crawl4ai))
                        (option("simple", "Simple", url, false))
                    }
                    @if !crawl4ai { p.hint { "Local Crawl4AI is optional. Enable it with OPEN_NOTEBOOK_ENABLE_CRAWL4AI=true (installs on first startup), or set CRAWL4AI_API_URL to use a remote server." } }
                    (help("· Firecrawl is a paid service (with a free tier), and very powerful. · Jina is a good option as well and also has a free tier. · Crawl4AI renders JavaScript pages locally (or via a Crawl4AI server when CRAWL4AI_API_URL is set) with no API key. · Simple will use basic HTTP extraction and will miss content on javascript-based websites. · Auto (recommended) will try Firecrawl, then Jina, then Crawl4AI, finally falling back to simple."))
                }
            }
            section.card.stack {
                div { h2 { "Embedding and Search" } p.muted.small { "Configure search and embedding options" } }
                div.field {
                    label for="set-embedding" { "Default Embedding Option" }
                    select #set-embedding name="default_embedding_option" {
                        (option("ask", "Ask", embedding, false))
                        (option("always", "Always", embedding, false))
                        (option("never", "Never", embedding, false))
                    }
                    (help("Embedding the content will make it easier to find by you and by your AI agents. If you are running a local embedding model (Ollama, for example), you shouldn't worry about cost and just embed everything."))
                }
            }
            section.card.stack {
                div { h2 { "File Management" } p.muted.small { "Configure file handling and storage options" } }
                div.field {
                    label for="set-auto-delete" { "Auto Delete Files" }
                    select #set-auto-delete name="auto_delete_files" {
                        (option("yes", "Yes", auto_delete, false))
                        (option("no", "No", auto_delete, false))
                    }
                    (help("Once your files are uploaded and processed, they are not required anymore. Most users should allow Nova-Notes to delete uploaded files from the upload folder automatically."))
                }
            }
            div.row.set-end { button.btn.btn-primary type="submit" { "Save" } }
        }
    }
}

async fn settings_save(State(state): State<AppState>, headers: HeaderMap, Form(pairs): Form<Vec<(String, String)>>) -> WebResult<Response> {
    let fields = Fields(pairs);
    let update = SettingsUpdate {
        default_content_processing_engine_doc: fields.text("default_content_processing_engine_doc"),
        default_content_processing_engine_url: fields.text("default_content_processing_engine_url"),
        default_embedding_option: fields.text("default_embedding_option"),
        auto_delete_files: fields.text("auto_delete_files"),
        docling_ocr: fields.flag("docling_ocr"),
        docling_formulas: fields.flag("docling_formulas"),
        docling_vision: fields.flag("docling_vision"),
        youtube_preferred_languages: None,
    };
    let saved = app_settings::update_settings(&state, &update).await?;
    if is_htmx(&headers) {
        let response = settings_form(&saved, app_settings::capabilities(), true).into_response();
        return Ok(with_toast(response, "Settings saved"));
    }
    Ok(Redirect::to("/settings?saved=1").into_response())
}

// ---------------------------------------------------------------------------
// /advanced
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct AdvancedQuery {
    command_id: Option<String>,
}

async fn advanced_page(State(state): State<AppState>, Query(query): Query<AdvancedQuery>) -> WebResult<Markup> {
    let db_online = matches!(tokio::time::timeout(Duration::from_secs(2), state.db.ping()).await, Ok(Ok(())));
    let default_request = RebuildRequest { mode: RebuildMode::Existing, include_sources: true, include_notes: true, include_insights: true };
    let estimate = embedding::estimate_rebuild(&state, &default_request).await?;
    let status = match query.command_id.as_deref().filter(|id| !id.is_empty()) {
        Some(id) => Some(embedding::rebuild_status(&state, id).await?),
        None => None,
    };
    let ctx = PageContext::new(&state, "/advanced");
    Ok(page(
        &ctx,
        "Advanced",
        html! {
            (page_head("Advanced Tools", Some("Advanced tools and utilities for power users"), html! {}))
            div.stack.set-narrow-wide {
                section.card.stack-sm {
                    h2 { "System Info" }
                    div.row-between { span { "Current Version" } span.badge { code { (crate::API_COMPAT_VERSION) } } }
                    div.row-between { span { "Build" } span.badge { code { "nova-notes-rust " (crate::VERSION) } } }
                    div.row-between {
                        span { "Database" }
                        @if db_online { span.badge.ok { "Online" } } @else { span.badge.bad { "Offline" } }
                    }
                    p.hint { "This build does not check GitHub for newer releases." }
                }
                section.card.stack {
                    div {
                        h2 { "Rebuild Embeddings" }
                        p.muted.small { "Rebuild vector search index for all sources" }
                    }
                    div #set-rebuild {
                        @match &status {
                            Some(status) => (rebuild_status_view(status)),
                            None => (rebuild_form(estimate)),
                        }
                    }
                    (rebuild_help())
                }
            }
        },
    ))
}

fn rebuild_form(estimate: i64) -> Markup {
    html! {
        form.stack method="post" action="/advanced/rebuild" hx-post="/advanced/rebuild" hx-target="#set-rebuild" {
            div.field {
                label for="set-rebuild-mode" { "Rebuild Mode" }
                select #set-rebuild-mode name="mode" hx-post="/advanced/rebuild/estimate" hx-trigger="change" hx-target="#set-estimate" hx-include="closest form" {
                    option value="existing" selected { "Existing" }
                    option value="all" { "All" }
                }
                p.hint {
                    strong { "Existing: " } "Re-embed only items that already have embeddings (faster, for model switching)"
                    br;
                    strong { "All: " } "Re-embed existing items + create embeddings for items without any (slower, comprehensive)"
                }
            }
            fieldset.set-fieldset hx-post="/advanced/rebuild/estimate" hx-trigger="change" hx-target="#set-estimate" hx-include="closest form" {
                legend { "Include in Rebuild" }
                div.row {
                    @for (key, label) in [("include_sources", "Sources"), ("include_notes", "Notes"), ("include_insights", "Insights")] {
                        label.check {
                            input type="hidden" name=(key) value="false";
                            input type="checkbox" name=(key) value="true" checked;
                            (label)
                        }
                    }
                }
            }
            div #set-estimate { (estimate_line(Ok(estimate))) }
            button.btn.btn-primary type="submit" { "Start Rebuild" span.spinner.htmx-indicator {} }
        }
    }
}

fn estimate_line(estimate: Result<i64, &str>) -> Markup {
    html! {
        @match estimate {
            Ok(count) => p.small.muted { "About " strong { (count) } " item(s) will be re-embedded." },
            Err(message) => div.flash.bad role="alert" { (message) },
        }
    }
}

fn rebuild_request(fields: &Fields) -> Result<RebuildRequest, AppError> {
    let mode = match fields.raw("mode").unwrap_or("existing") {
        "existing" => RebuildMode::Existing,
        "all" => RebuildMode::All,
        other => return Err(AppError::invalid(format!("Unknown rebuild mode: {other}"))),
    };
    Ok(RebuildRequest {
        mode,
        include_sources: fields.flag("include_sources").unwrap_or(false),
        include_notes: fields.flag("include_notes").unwrap_or(false),
        include_insights: fields.flag("include_insights").unwrap_or(false),
    })
}

const SELECT_ONE: &str = "Please select at least one item type to rebuild";

fn selects_something(request: &RebuildRequest) -> bool {
    request.include_sources || request.include_notes || request.include_insights
}

async fn rebuild_estimate(State(state): State<AppState>, Form(pairs): Form<Vec<(String, String)>>) -> WebResult<Markup> {
    let request = rebuild_request(&Fields(pairs))?;
    if !selects_something(&request) {
        return Ok(estimate_line(Err(SELECT_ONE)));
    }
    Ok(estimate_line(Ok(embedding::estimate_rebuild(&state, &request).await?)))
}

async fn rebuild_start(State(state): State<AppState>, headers: HeaderMap, Form(pairs): Form<Vec<(String, String)>>) -> WebResult<Response> {
    let request = rebuild_request(&Fields(pairs))?;
    if !selects_something(&request) {
        return Err(AppError::invalid(SELECT_ONE).into());
    }
    let started = embedding::start_rebuild(&state, &request).await?;
    if !is_htmx(&headers) {
        let target = format!("/advanced?command_id={}", crate::api::auth::encode_query_value(&started.command_id));
        return Ok(Redirect::to(&target).into_response());
    }
    let status = embedding::rebuild_status(&state, &started.command_id).await?;
    Ok(with_toast(rebuild_status_view(&status).into_response(), &started.message))
}

async fn rebuild_status(State(state): State<AppState>, Path(id): Path<String>) -> WebResult<Markup> {
    Ok(rebuild_status_view(&embedding::rebuild_status(&state, &id).await?))
}

fn seconds_between(start: Option<&str>, end: Option<&str>) -> Option<f64> {
    let parse = |value: Option<&str>| value.and_then(|v| chrono::DateTime::parse_from_rfc3339(v).ok());
    let (start, end) = (parse(start)?, parse(end)?);
    Some((end - start).num_milliseconds() as f64 / 1000.0)
}

fn rebuild_status_view(status: &RebuildStatus) -> Markup {
    let finished = job_status::is_terminal(&status.status);
    let label = match status.status.as_str() {
        job_status::NEW => "Queued",
        job_status::RUNNING => "Submitting jobs...",
        job_status::COMPLETED => "Jobs Submitted!",
        job_status::FAILED => "Failed",
        job_status::CANCELED => "Canceled",
        other => other,
    };
    let poll_url = format!("/advanced/rebuild/{}/status", status.command_id);
    let parsed = |value: &Option<String>| {
        value
            .as_deref()
            .and_then(|v| chrono::DateTime::parse_from_rfc3339(v).ok())
            .map(|d| super::human_date(Some(d.with_timezone(&chrono::Utc))))
    };
    let duration = if finished { seconds_between(status.started_at.as_deref(), status.completed_at.as_deref()) } else { None };
    html! {
        div.stack.set-rebuild-status
            hx-get=[(!finished).then_some(poll_url.as_str())] hx-trigger=[(!finished).then_some("every 3s")] hx-swap=[(!finished).then_some("outerHTML")] {
            div.row-between {
                div.row {
                    (status_badge(Some(&status.status)))
                    strong { (label) }
                    @if !finished { span.spinner {} }
                }
                @if finished {
                    a.btn.btn-sm href="/advanced" { "Start New Rebuild" }
                } @else {
                    a.btn.btn-ghost.btn-sm href=(format!("/advanced?command_id={}", crate::api::auth::encode_query_value(&status.command_id))) { "Refresh" }
                }
            }
            @if !finished { p.small.muted { "You can leave this page as this will run in the background" } }
            @if let Some(progress) = &status.progress {
                div.stack-sm {
                    div.row-between.small {
                        span { "Progress" }
                        strong { (progress.processed) "/" (progress.total) " jobs submitted (" (format!("{:.1}", progress.percentage)) "%)" }
                    }
                    progress max="100" value=(format!("{:.1}", progress.percentage)) {}
                }
            }
            @if let Some(stats) = &status.stats {
                @if stats.failed > 0 { p.small.set-warn { (stats.failed) " jobs failed to submit" } }
                div.set-stats {
                    div { span.small.muted { "Sources" } strong { (stats.sources) } }
                    div { span.small.muted { "Notes" } strong { (stats.notes) } }
                    div { span.small.muted { "Insights" } strong { (stats.insights) } }
                    div { span.small.muted { "Time" } strong { (duration.map(|s| format!("{s:.1}s")).unwrap_or_else(|| "—".into())) } }
                }
            }
            @if let Some(error) = &status.error_message { div.flash.bad role="alert" { (error) } }
            @if let Some(started) = parsed(&status.started_at) {
                p.small.faint {
                    "Started " (started)
                    @if finished { @if let Some(done) = parsed(&status.completed_at) { " · finished " (done) } }
                }
            }
        }
    }
}

fn rebuild_help() -> Markup {
    html! {
        div.stack-sm.set-faq {
            details { summary { "When should I rebuild embeddings?" } p.small.muted { "You should rebuild when switching models, upgrading versions, fixing corruption, or after bulk imports." } }
            details { summary { "How long does rebuilding take?" } p.small.muted { "Processing time depends on item count, model speed, and API rate limits. Local models are usually very fast." } }
            details { summary { "Is it safe to rebuild while using the app?" } p.small.muted { "Yes, rebuilding is safe! It doesn't delete content, only replaces embeddings, and handles errors gracefully." } }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{header, Method, Request, StatusCode};
    use http_body_util::BodyExt;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::domain::{Credential, DefaultModels, Model};
    use crate::testing::TestApp;

    /// Send a urlencoded form, optionally as htmx; returns status, headers and body.
    async fn form(app: &TestApp, uri: &str, body: &str, htmx: bool) -> (StatusCode, axum::http::HeaderMap, String) {
        let mut builder = Request::builder()
            .method(Method::POST)
            .uri(uri)
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

    async fn get_htmx(app: &TestApp, uri: &str) -> (StatusCode, String) {
        let request = Request::builder().uri(uri).header("hx-request", "true").body(Body::empty()).unwrap();
        let response = app.send(request).await;
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    async fn page(app: &TestApp, uri: &str) -> String {
        let (status, body) = app.text(Method::GET, uri, None).await;
        assert_eq!(status, StatusCode::OK, "{uri}: {body}");
        body
    }

    async fn credential(app: &TestApp, body: serde_json::Value) -> String {
        let (status, created) = app.post("/api/credentials", body).await;
        assert_eq!(status, StatusCode::CREATED, "{created}");
        created["id"].as_str().unwrap().to_string()
    }

    async fn model(app: &TestApp, name: &str, provider: &str, model_type: &str, credential: Option<&str>) -> String {
        let mut model = Model {
            name: name.into(),
            provider: provider.into(),
            model_type: model_type.into(),
            credential: credential.map(String::from),
            ..Model::default()
        };
        model.save(&app.state).await.unwrap();
        model.id
    }

    fn location(headers: &axum::http::HeaderMap) -> &str {
        headers[header::LOCATION].to_str().unwrap()
    }

    #[test]
    fn fields_keep_repeats_presence_and_the_last_value() {
        let fields = super::Fields(vec![
            ("a".into(), "false".into()),
            ("a".into(), "true".into()),
            ("m".into(), "x".into()),
            ("m".into(), " x ".into()),
            ("m".into(), "y".into()),
            ("blank".into(), "  ".into()),
        ]);
        assert_eq!(fields.flag("a"), Some(true));
        assert_eq!(fields.all("m"), ["x", "y"]);
        assert_eq!(fields.present("blank"), Some(None));
        assert_eq!(fields.present("absent"), None);
        assert_eq!(super::dom_id("credential:a.b"), "credential-a-b");
    }

    #[tokio::test]
    async fn models_page_shows_providers_and_empty_states() {
        let app = TestApp::new().await;
        let body = page(&app, "/settings/models").await;
        assert!(body.contains("Default Model Assignments"));
        assert!(body.contains("OpenAI") && body.contains("Ollama"));
        assert!(body.contains("Not configured"));
        assert!(body.contains("Missing required models: Chat Model, Embedding Model"));
        assert!(body.contains("No models"));
        assert!(body.contains("/settings/models/credentials/new?provider=openai"));
        assert!(!body.contains("Encryption key not configured"));
    }

    #[tokio::test]
    async fn models_page_warns_without_an_encryption_key() {
        let mut app = TestApp::new().await;
        let mut config = (*app.state.config).clone();
        config.encryption_key = None;
        app.state.config = std::sync::Arc::new(config);
        app.router = crate::api::build_router(app.state.clone());
        let body = page(&app, "/settings/models").await;
        assert!(body.contains("Encryption key not configured"));
        assert!(body.contains("OPEN_NOTEBOOK_ENCRYPTION_KEY"));
    }

    #[tokio::test]
    async fn credential_forms_render_per_provider() {
        let app = TestApp::new().await;
        let body = page(&app, "/settings/models/credentials/new?provider=vertex").await;
        assert!(body.contains("GCP Project ID") && !body.contains("name=\"api_key\""));
        let (status, fragment) = get_htmx(&app, "/settings/models/credentials/new?provider=ollama").await;
        assert_eq!(status, StatusCode::OK);
        assert!(fragment.contains("num_ctx") && fragment.contains("hx-post=\"/settings/models/credentials\""));
        assert!(!fragment.contains("<html"), "htmx gets a fragment");
        let (status, _) = app.text(Method::GET, "/settings/models/credentials/new?provider=nope", None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn create_edit_and_delete_a_credential_without_showing_the_key() {
        let app = TestApp::new().await;
        let (status, headers, _) = form(
            &app,
            "/settings/models/credentials",
            "provider=openai&name=Main&api_key=sk-super-secret&base_url=&modalities=language&modalities=embedding",
            false,
        )
        .await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(location(&headers), "/settings/models");
        let stored = Credential::get_by_provider(&app.state, "openai").await.unwrap();
        assert_eq!(stored.len(), 1);
        let id = stored[0].id.clone();
        assert_eq!(stored[0].api_key.as_ref().unwrap().expose(), "sk-super-secret");
        assert_eq!(stored[0].modalities, ["language", "embedding"]);

        let body = page(&app, "/settings/models").await;
        assert!(body.contains("Main") && body.contains("Configured"));
        assert!(!body.contains("sk-super-secret"));
        let edit = page(&app, &format!("/settings/models/credentials/{id}/edit")).await;
        assert!(edit.contains("Leave blank to keep the existing API key"));
        assert!(!edit.contains("sk-super-secret"));

        // A blank key keeps the stored one; htmx gets the refreshed body and a toast.
        let (status, headers, fragment) = form(
            &app,
            &format!("/settings/models/credentials/{id}"),
            "provider=openai&name=Renamed&api_key=&base_url=https%3A%2F%2Fapi.openai.com%2Fv1&modalities_shown=1&modalities=language",
            true,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{fragment}");
        assert!(fragment.contains("id=\"set-models-body\"") && fragment.contains("Renamed"));
        assert!(headers["hx-trigger"].to_str().unwrap().contains("updated"));
        let updated = Credential::get(&app.state, &id).await.unwrap();
        assert_eq!(updated.name, "Renamed");
        assert_eq!(updated.api_key.unwrap().expose(), "sk-super-secret");
        assert_eq!(updated.base_url.as_deref(), Some("https://api.openai.com/v1"));
        assert_eq!(updated.modalities, ["language"]);

        // Clearing the base URL.
        form(&app, &format!("/settings/models/credentials/{id}"), "provider=openai&name=Renamed&base_url=", false).await;
        assert_eq!(Credential::get(&app.state, &id).await.unwrap().base_url, None);

        // Delete takes the models with it.
        let model_id = model(&app, "gpt-4o", "openai", "language", Some(&id)).await;
        let confirm = page(&app, &format!("/settings/models/credentials/{id}/delete")).await;
        assert!(confirm.contains("1 linked model(s)") && confirm.contains("Delete with Models"));
        let (status, _, _) = form(&app, &format!("/settings/models/credentials/{id}/delete"), "action=delete", false).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert!(Credential::get_by_provider(&app.state, "openai").await.unwrap().is_empty());
        assert!(Model::get(&app.state, &model_id).await.is_err());
    }

    #[tokio::test]
    async fn deleting_can_migrate_models_to_another_credential() {
        let app = TestApp::new().await;
        let first = credential(&app, json!({ "name": "One", "provider": "openai", "api_key": "sk-1" })).await;
        let second = credential(&app, json!({ "name": "Two", "provider": "openai", "api_key": "sk-2" })).await;
        let model_id = model(&app, "gpt-4o", "openai", "language", Some(&first)).await;
        let confirm = page(&app, &format!("/settings/models/credentials/{first}/delete")).await;
        assert!(confirm.contains("Migrate &amp; Delete") && confirm.contains(&second));

        let (status, _, body) = form(&app, &format!("/settings/models/credentials/{first}/delete"), "action=migrate&migrate_to=", false).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

        let body = format!("action=migrate&migrate_to={}", second.replace(':', "%3A"));
        let (status, _, _) = form(&app, &format!("/settings/models/credentials/{first}/delete"), &body, true).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(Model::get(&app.state, &model_id).await.unwrap().credential.as_deref(), Some(second.as_str()));
        let (status, _) = app.text(Method::GET, &format!("/settings/models/credentials/{first}/delete"), None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn missing_credentials_and_models_are_404s() {
        let app = TestApp::new().await;
        for uri in ["/settings/models/credentials/credential:nope/edit", "/settings/models/credentials/credential:nope/discover"] {
            let (status, body) = app.text(Method::GET, uri, None).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{uri}");
            assert!(body.contains("data-error-detail"));
        }
        let (status, _, _) = form(&app, "/settings/models/models/model:nope/delete", "", true).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _, _) = form(&app, "/settings/models/models/model:nope/test", "", true).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _, body) = form(&app, "/settings/models/credentials/credential:nope/test", "", true).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("Credential not found"));
    }

    #[tokio::test]
    async fn discover_and_register_models() {
        let app = TestApp::new().await;
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": [{ "id": "qwen3" }, { "id": "nomic-embed" }] })))
            .mount(&server)
            .await;
        let id = credential(&app, json!({ "name": "Local", "provider": "openai_compatible", "base_url": format!("{}/v1", server.uri()) })).await;

        let (status, fragment) = get_htmx(&app, &format!("/settings/models/credentials/{id}/discover")).await;
        assert_eq!(status, StatusCode::OK, "{fragment}");
        assert!(fragment.contains("value=\"qwen3\"") && fragment.contains("value=\"nomic-embed\""));

        let (status, headers, body) = form(
            &app,
            &format!("/settings/models/credentials/{id}/register"),
            "model_type=language&model=qwen3&custom=my-custom",
            true,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(headers["hx-trigger"].to_str().unwrap().contains("Added 2 model(s)"));
        assert!(body.contains("qwen3") && body.contains("my-custom"));
        let linked = Model::get_by_credential(&app.state, &id).await.unwrap();
        assert_eq!(linked.len(), 2);
        assert!(linked.iter().all(|m| m.model_type == "language" && m.provider == "openai_compatible"));

        let (status, _, _) = form(&app, &format!("/settings/models/credentials/{id}/register"), "model_type=language", true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn defaults_save_and_auto_assign() {
        let app = TestApp::new().await;
        let chat = model(&app, "gpt-4o", "openai", "language", None).await;
        let other = model(&app, "gpt-4o-mini", "openai", "language", None).await;
        let embed = model(&app, "text-embedding-3-small", "openai", "embedding", None).await;
        let embed2 = model(&app, "text-embedding-3-large", "openai", "embedding", None).await;

        let (status, headers, body) = form(&app, "/settings/models/auto-assign", "", true).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(headers["hx-trigger"].to_str().unwrap().contains("automatically assigned"));
        let defaults = DefaultModels::load(&app.state).await.unwrap();
        assert_eq!(defaults.default_chat_model.as_deref(), Some(chat.as_str()));
        let assigned_embedding = defaults.default_embedding_model.clone().unwrap();
        let new_embedding = if assigned_embedding == embed { embed2.clone() } else { embed.clone() };
        assert!(!body.contains("Missing required models"));

        // Assign an optional slot, keep required ones when left blank, change the embedding model.
        let encode = |s: &str| s.replace(':', "%3A");
        let body = format!(
            "default_chat_model=&default_embedding_model={}&default_transformation_model={}&default_tools_model=&large_context_model=",
            encode(&new_embedding),
            encode(&other)
        );
        let (status, _, fragment) = form(&app, "/settings/models/defaults", &body, true).await;
        assert_eq!(status, StatusCode::OK, "{fragment}");
        assert!(fragment.contains("Rebuild Required"));
        let defaults = DefaultModels::load(&app.state).await.unwrap();
        assert_eq!(defaults.default_chat_model.as_deref(), Some(chat.as_str()));
        assert_eq!(defaults.default_embedding_model.as_deref(), Some(new_embedding.as_str()));
        assert_eq!(defaults.default_transformation_model.as_deref(), Some(other.as_str()));

        // Clearing an optional slot (plain form → redirect).
        let (status, headers, _) = form(&app, "/settings/models/defaults", "default_transformation_model=", false).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(location(&headers), "/settings/models");
        assert_eq!(DefaultModels::load(&app.state).await.unwrap().default_transformation_model, None);

        let page_body = page(&app, "/settings/models").await;
        assert!(page_body.contains("(Chat)") || page_body.contains(">Chat<"), "default slot badges are shown");
    }

    #[tokio::test]
    async fn add_test_and_delete_a_model() {
        let app = TestApp::new().await;
        // The model test makes a real call: point it at a local mock that refuses it.
        let server = MockServer::start().await;
        let cred = credential(&app, json!({ "name": "Local", "provider": "openai_compatible", "base_url": format!("{}/v1", server.uri()) })).await;
        let body = format!("provider=openai_compatible&name=llama3&type=language&credential={}", cred.replace(':', "%3A"));
        let (status, _, body) = form(&app, "/settings/models/models", &body, true).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("llama3"));
        let created = Model::get_by_type(&app.state, "language").await.unwrap();
        assert_eq!(created.len(), 1);
        assert_eq!(created[0].credential.as_deref(), Some(cred.as_str()));
        let id = created[0].id.clone();

        let (status, _, body) = form(&app, "/settings/models/models", "provider=openai_compatible&name=llama3&type=language", true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "duplicates are refused: {body}");
        let (status, _, _) = form(&app, "/settings/models/models", "provider=ollama&name=x&type=vision", true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        // A failed call is reported in the dialog, never as an error page.
        let (status, _, body) = form(&app, &format!("/settings/models/models/{id}/test"), "", true).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("Model Test Failed") && body.contains("llama3"), "{body}");

        let (status, headers, _) = form(&app, &format!("/settings/models/models/{id}/delete"), "", false).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(location(&headers), "/settings/models");
        assert!(Model::get_by_type(&app.state, "language").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn migrate_from_env_reports_back() {
        let app = TestApp::new().await;
        let (status, headers, _) = form(&app, "/settings/models/migrate-env", "", true).await;
        assert_eq!(status, StatusCode::OK);
        assert!(headers.contains_key("hx-trigger"));
    }

    #[tokio::test]
    async fn settings_page_shows_and_saves_settings() {
        let app = TestApp::new().await;
        let body = page(&app, "/settings").await;
        assert!(body.contains("Content Processing") && body.contains("Document Processing Engine"));
        assert!(body.contains("OPEN_NOTEBOOK_ENABLE_DOCLING"), "docling is not available in this build");

        let (status, headers, _) = form(
            &app,
            "/settings",
            "default_content_processing_engine_doc=simple&default_content_processing_engine_url=jina&default_embedding_option=always&auto_delete_files=no",
            false,
        )
        .await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(location(&headers), "/settings?saved=1");
        let (_, saved) = app.get("/api/settings").await;
        assert_eq!(saved["default_content_processing_engine_doc"], "simple");
        assert_eq!(saved["default_content_processing_engine_url"], "jina");
        assert_eq!(saved["default_embedding_option"], "always");
        assert_eq!(saved["auto_delete_files"], "no");
        assert_eq!(saved["docling_ocr"], true, "a disabled toggle is left alone");
        assert!(page(&app, "/settings?saved=1").await.contains("Settings saved"));

        let (status, _, fragment) = form(&app, "/settings", "default_embedding_option=never", true).await;
        assert_eq!(status, StatusCode::OK);
        assert!(fragment.contains("id=\"set-settings\"") && fragment.contains("Settings saved"));
        let (status, _, _) = form(&app, "/settings", "default_embedding_option=sometimes", true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn advanced_page_estimates_starts_and_polls_a_rebuild() {
        let app = TestApp::new().await;
        let body = page(&app, "/advanced").await;
        assert!(body.contains("System Info") && body.contains("Rebuild Embeddings"));
        assert!(body.contains(crate::API_COMPAT_VERSION) && body.contains("Online"));
        assert!(body.contains("About <strong>0</strong> item(s)"));

        let (status, _, fragment) = form(&app, "/advanced/rebuild/estimate", "mode=all&include_sources=false&include_notes=false&include_insights=false", true).await;
        assert_eq!(status, StatusCode::OK);
        assert!(fragment.contains("select at least one"));
        let (status, _, _) = form(&app, "/advanced/rebuild", "mode=all&include_sources=false", true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let (status, headers, _) = form(&app, "/advanced/rebuild", "mode=all&include_sources=false&include_sources=true", false).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        let target = location(&headers).to_string();
        assert!(target.starts_with("/advanced?command_id=command"), "{target}");
        let command_id = target.split("command_id=").nth(1).unwrap().replace("%3A", ":");
        app.wait_for_job(&command_id).await;
        let status_page = page(&app, &target).await;
        assert!(status_page.contains("Jobs Submitted!") && status_page.contains("Start New Rebuild"), "{status_page}");

        let (status, fragment) = get_htmx(&app, &format!("/advanced/rebuild/{command_id}/status")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(!fragment.contains("every 3s"), "a finished rebuild stops polling");

        let (status, headers, fragment) = form(&app, "/advanced/rebuild", "mode=existing&include_notes=true", true).await;
        assert_eq!(status, StatusCode::OK, "{fragment}");
        assert!(headers.contains_key("hx-trigger"));
        assert!(fragment.contains("set-rebuild-status"));

        let (status, _) = app.text(Method::GET, "/advanced/rebuild/command:nope/status", None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
}
