//! Field-level encryption for API keys stored in the database.
//!
//! The Rust counterpart of `open_notebook/utils/encryption.py`, byte-compatible
//! with it so a database written by the Python build opens here (and back):
//!
//! - New values are Fernet tokens (AES-128-CBC + HMAC-SHA256) under a key
//!   derived from the passphrase with PBKDF2-HMAC-SHA256 (fixed salt, 600 000
//!   iterations), stored with the `pbkdf2v1:` marker in front.
//! - Unmarked values are legacy: Fernet under an unsalted SHA-256 of the
//!   passphrase (decrypt only), or plaintext that predates encryption.
//!
//! The passphrase is `Config::encryption_key` (`OPEN_NOTEBOOK_ENCRYPTION_KEY`
//! or its `_FILE` variant). PBKDF2 is deliberately slow, so the derived keys
//! are cached per passphrase for the life of the process ([`Cipher`]).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use aes::cipher::block_padding::Pkcs7;
use aes::cipher::{BlockDecryptMut, BlockEncryptMut, KeyIvInit};
use base64::engine::general_purpose::URL_SAFE;
use base64::Engine;
use hmac::{Hmac, Mac};
use rand::RngCore;
use sha2::{Digest, Sha256};

use crate::config::Config;
use crate::db::Json;
use crate::error::{AppError, AppResult};

/// Version marker in front of every value written by the PBKDF2 derivation.
/// Unmarked values are legacy (bare SHA-256 derivation) or plaintext.
pub const PBKDF2_MARKER: &str = "pbkdf2v1:";

/// PBKDF2 cost parameter (OWASP guidance for HMAC-SHA256). Changing it needs
/// a new marker version.
pub const PBKDF2_ITERATIONS: u32 = 600_000;

/// Fixed application salt, shared by every deployment by design (ADR-009):
/// domain separation and brute-force slowdown, not a per-install secret.
pub const PBKDF2_SALT: &[u8] = b"open-notebook-credential-v1";

/// What the domain layer shows in place of a key it could not decrypt. Never
/// a secret, and never to be encrypted as one.
pub const UNDECRYPTABLE_PLACEHOLDER: &str = "UNDECRYPTABLE";

const MISSING_KEY: &str = "OPEN_NOTEBOOK_ENCRYPTION_KEY is not set. Set this environment variable to any secret string \
     to enable encrypted storage of API keys in the database.";
const MARKED_FAILED: &str = "Decryption failed: versioned value could not be decrypted. \
     Check OPEN_NOTEBOOK_ENCRYPTION_KEY configuration.";
const WRONG_KEY: &str = "Decryption failed: data appears to be encrypted but key is incorrect. \
     Check OPEN_NOTEBOOK_ENCRYPTION_KEY configuration.";
const UNABLE: &str = "Decryption failed: unable to decrypt value.";
const KEY_NOT_CONFIGURED: &str =
    "Encryption key not configured. Set OPEN_NOTEBOOK_ENCRYPTION_KEY to enable storing API keys.";

// ---------------------------------------------------------------------------
// SecretString
// ---------------------------------------------------------------------------

/// A decrypted secret held in memory. `Debug` never prints it and it has no
/// `Display` or `Serialize`: reading the value takes an explicit
/// [`expose`](Self::expose).
#[derive(Clone, PartialEq, Eq)]
pub struct SecretString(String);

impl SecretString {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// The secret itself. Never log or return the result.
    pub fn expose(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Debug for SecretString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SecretString(**********)")
    }
}

impl From<String> for SecretString {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for SecretString {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

// ---------------------------------------------------------------------------
// Key derivation
// ---------------------------------------------------------------------------

fn pbkdf2_key(passphrase: &str, iterations: u32) -> [u8; 32] {
    let mut key = [0u8; 32];
    pbkdf2::pbkdf2_hmac::<Sha256>(passphrase.as_bytes(), PBKDF2_SALT, iterations, &mut key);
    key
}

fn legacy_key(passphrase: &str) -> [u8; 32] {
    Sha256::digest(passphrase.as_bytes()).into()
}

/// The Fernet key for a passphrase: URL-safe base64 of PBKDF2-HMAC-SHA256
/// with the fixed salt and [`PBKDF2_ITERATIONS`]. Slow on purpose and not
/// cached; use [`Cipher`] to encrypt or decrypt.
pub fn derive_fernet_key(passphrase: &str) -> String {
    URL_SAFE.encode(pbkdf2_key(passphrase, PBKDF2_ITERATIONS))
}

/// The Fernet key older versions used (unsalted SHA-256 of the passphrase).
/// Only ever used to read values written before the PBKDF2 scheme.
pub fn derive_legacy_fernet_key(passphrase: &str) -> String {
    URL_SAFE.encode(legacy_key(passphrase))
}

// ---------------------------------------------------------------------------
// Fernet
// ---------------------------------------------------------------------------

type Aes128CbcEnc = cbc::Encryptor<aes::Aes128>;
type Aes128CbcDec = cbc::Decryptor<aes::Aes128>;

const FERNET_VERSION: u8 = 0x80;
/// version (1) + timestamp (8) + IV (16).
const FERNET_HEADER: usize = 25;
const FERNET_MAC: usize = 32;
const AES_BLOCK: usize = 16;

/// The token is not valid under this key (Python's `InvalidToken`).
struct InvalidToken;

/// A Fernet key: 16 bytes for signing followed by 16 for encryption.
struct FernetKey {
    signing: [u8; 16],
    encryption: [u8; 16],
}

impl FernetKey {
    fn new(key: [u8; 32]) -> Self {
        let mut signing = [0u8; 16];
        let mut encryption = [0u8; 16];
        signing.copy_from_slice(&key[..16]);
        encryption.copy_from_slice(&key[16..]);
        Self { signing, encryption }
    }

    fn mac(&self) -> Hmac<Sha256> {
        // HMAC accepts a key of any length, so this cannot fail.
        Hmac::<Sha256>::new_from_slice(&self.signing).expect("HMAC accepts any key length")
    }

    fn encrypt(&self, plaintext: &[u8]) -> String {
        let mut iv = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut iv);
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        self.encrypt_at(plaintext, now, iv)
    }

    fn encrypt_at(&self, plaintext: &[u8], timestamp: u64, iv: [u8; 16]) -> String {
        let ciphertext =
            Aes128CbcEnc::new(&self.encryption.into(), &iv.into()).encrypt_padded_vec_mut::<Pkcs7>(plaintext);
        let mut token = Vec::with_capacity(FERNET_HEADER + ciphertext.len() + FERNET_MAC);
        token.push(FERNET_VERSION);
        token.extend_from_slice(&timestamp.to_be_bytes());
        token.extend_from_slice(&iv);
        token.extend_from_slice(&ciphertext);
        let mut mac = self.mac();
        mac.update(&token);
        token.extend_from_slice(&mac.finalize().into_bytes());
        URL_SAFE.encode(token)
    }

    /// Verify and decrypt a token. No expiry check: stored values never expire.
    fn decrypt(&self, token: &[u8]) -> Result<Vec<u8>, InvalidToken> {
        let data = python_urlsafe_b64decode(token).ok_or(InvalidToken)?;
        if data.first() != Some(&FERNET_VERSION) || data.len() < FERNET_HEADER + FERNET_MAC {
            return Err(InvalidToken);
        }
        let (signed, signature) = data.split_at(data.len() - FERNET_MAC);
        let mut mac = self.mac();
        mac.update(signed);
        mac.verify_slice(signature).map_err(|_| InvalidToken)?;

        let iv: [u8; 16] = signed[9..FERNET_HEADER].try_into().map_err(|_| InvalidToken)?;
        let ciphertext = &signed[FERNET_HEADER..];
        if ciphertext.is_empty() || ciphertext.len() % AES_BLOCK != 0 {
            return Err(InvalidToken);
        }
        Aes128CbcDec::new(&self.encryption.into(), &iv.into())
            .decrypt_padded_vec_mut::<Pkcs7>(ciphertext)
            .map_err(|_| InvalidToken)
    }
}

/// Decode base64 the way Python's `base64.urlsafe_b64decode` does, which is
/// what `cryptography`'s Fernet and `looks_like_fernet_token` both rely on.
///
/// It is lenient: both alphabets are accepted, characters outside them
/// (whitespace, punctuation, non-ASCII bytes) are skipped, and anything after
/// complete padding is ignored. Only a dangling partial group is an error.
/// Matching it exactly keeps the token / plaintext classification of legacy
/// values identical in both builds.
fn python_urlsafe_b64decode(input: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() / 4 * 3 + 3);
    let mut quad_pos = 0u8;
    let mut pads = 0u8;
    let mut left = 0u8;
    for &ch in input {
        if ch == b'=' {
            if quad_pos >= 2 {
                pads += 1;
                if quad_pos + pads >= 4 {
                    return Some(out);
                }
            }
            continue;
        }
        let sextet = match ch {
            b'A'..=b'Z' => ch - b'A',
            b'a'..=b'z' => ch - b'a' + 26,
            b'0'..=b'9' => ch - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => continue,
        };
        pads = 0;
        match quad_pos {
            0 => left = sextet,
            1 => {
                out.push((left << 2) | (sextet >> 4));
                left = sextet & 0x0f;
            }
            2 => {
                out.push((left << 4) | (sextet >> 2));
                left = sextet & 0x03;
            }
            _ => out.push((left << 6) | sextet),
        }
        quad_pos = (quad_pos + 1) % 4;
    }
    (quad_pos == 0).then_some(out)
}

/// Whether a string has the shape of a Fernet token: at least 100 characters
/// that decode to version (1) + timestamp (8) + IV (16) + a whole number of
/// AES blocks + HMAC (32). Used to tell "encrypted under another key" apart
/// from legacy plaintext.
pub fn looks_like_fernet_token(s: &str) -> bool {
    // Python refuses to decode a non-ASCII `str`, so such a value is never a token.
    if s.len() < 100 || !s.is_ascii() {
        return false;
    }
    match python_urlsafe_b64decode(s.as_bytes()) {
        Some(decoded) if decoded.len() >= 73 => (decoded.len() - 57) % AES_BLOCK == 0,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Cipher
// ---------------------------------------------------------------------------

/// Encrypts and decrypts stored values for one passphrase.
///
/// Both keys are derived on first use and kept: the PBKDF2 key costs a few
/// hundred milliseconds (seconds in an unoptimised build), the legacy key is
/// only needed to read unmarked values.
pub struct Cipher {
    passphrase: String,
    iterations: u32,
    pbkdf2: OnceLock<FernetKey>,
    legacy: OnceLock<FernetKey>,
}

impl std::fmt::Debug for Cipher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Cipher(**********)")
    }
}

impl Cipher {
    fn with_iterations(passphrase: &str, iterations: u32) -> Self {
        Self { passphrase: passphrase.to_string(), iterations, pbkdf2: OnceLock::new(), legacy: OnceLock::new() }
    }

    /// The cipher for a passphrase. One instance per passphrase is shared by
    /// the whole process, so each key is derived at most once.
    pub fn for_passphrase(passphrase: &str) -> Arc<Self> {
        static CIPHERS: OnceLock<Mutex<HashMap<String, Arc<Cipher>>>> = OnceLock::new();
        let mut ciphers = CIPHERS.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
        ciphers
            .entry(passphrase.to_string())
            .or_insert_with(|| Arc::new(Self::with_iterations(passphrase, PBKDF2_ITERATIONS)))
            .clone()
    }

    /// The cipher for the configured encryption key. No key configured is
    /// `AppError::Configuration`.
    pub fn from_config(config: &Config) -> AppResult<Arc<Self>> {
        match config.encryption_key.as_deref() {
            Some(passphrase) if !passphrase.is_empty() => Ok(Self::for_passphrase(passphrase)),
            _ => Err(AppError::config(MISSING_KEY)),
        }
    }

    fn pbkdf2_key(&self) -> &FernetKey {
        self.pbkdf2.get_or_init(|| FernetKey::new(pbkdf2_key(&self.passphrase, self.iterations)))
    }

    fn legacy_key(&self) -> &FernetKey {
        self.legacy.get_or_init(|| FernetKey::new(legacy_key(&self.passphrase)))
    }

    /// Encrypt a value into its stored, marked form.
    ///
    /// Input that already is valid marked ciphertext comes back unchanged, so
    /// a read / save round trip can never double-wrap. Anything else,
    /// marker-like plaintext included, is encrypted normally.
    pub fn encrypt(&self, value: &str) -> String {
        if value.starts_with(PBKDF2_MARKER) && self.decrypt(value).is_ok() {
            return value.to_string();
        }
        format!("{PBKDF2_MARKER}{}", self.pbkdf2_key().encrypt(value.as_bytes()))
    }

    /// Decrypt a stored value.
    ///
    /// Marked values decrypt under PBKDF2 only, and any failure is an error:
    /// the input is never handed back as if it were a usable key. Unmarked
    /// values try the legacy key; if that fails they are returned as they are
    /// (legacy plaintext) unless they look like a token, which means the key
    /// is wrong. Failures are `AppError::Configuration`; see
    /// [`is_decryption_error`].
    pub fn decrypt(&self, value: &str) -> AppResult<String> {
        let (key, token, failure) = match value.strip_prefix(PBKDF2_MARKER) {
            Some(token) => (self.pbkdf2_key(), token, Some(MARKED_FAILED)),
            None => (self.legacy_key(), value, looks_like_fernet_token(value).then_some(WRONG_KEY)),
        };
        match key.decrypt(token.as_bytes()) {
            Ok(plaintext) => String::from_utf8(plaintext).map_err(|_| AppError::config(UNABLE)),
            Err(InvalidToken) => match failure {
                Some(message) => Err(AppError::config(message)),
                None => Ok(value.to_string()),
            },
        }
    }
}

/// Encrypt a value with the configured key. See [`Cipher::encrypt`].
pub fn encrypt_value(config: &Config, value: &str) -> AppResult<String> {
    Ok(Cipher::from_config(config)?.encrypt(value))
}

/// Decrypt a stored value with the configured key. See [`Cipher::decrypt`].
pub fn decrypt_value(config: &Config, value: &str) -> AppResult<String> {
    Cipher::from_config(config)?.decrypt(value)
}

/// Whether an encryption key is configured at all.
pub fn is_configured(config: &Config) -> bool {
    config.encryption_key.as_deref().is_some_and(|key| !key.is_empty())
}

/// The gate the credential endpoints run before any write: 400 with the
/// Python service's message when no encryption key is configured.
pub fn require_encryption_key(config: &Config) -> AppResult<()> {
    if is_configured(config) {
        Ok(())
    } else {
        Err(AppError::invalid(KEY_NOT_CONFIGURED))
    }
}

/// Whether an error is a failed decryption (wrong key, corrupt value), as
/// opposed to a missing record or a missing key. The credential endpoints
/// treat an undecryptable record differently from a missing one.
pub fn is_decryption_error(error: &AppError) -> bool {
    matches!(error, AppError::Configuration(message) if [MARKED_FAILED, WRONG_KEY, UNABLE].contains(&message.as_str()))
}

// ---------------------------------------------------------------------------
// Re-encryption (the one-shot upgrade to the PBKDF2 scheme)
// ---------------------------------------------------------------------------

/// The fate of one stored `api_key` value in the encryption-scheme pass.
/// Reasons are short codes that carry no key material.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reencrypt {
    /// It decrypted under the legacy scheme; this is its new marked form.
    Migrated(String),
    /// Nothing to do: `empty` or `already-migrated`.
    Skipped(&'static str),
    /// Left untouched: `unexpected-type`, `decrypt-placeholder`,
    /// `undecryptable-marked` or `undecryptable-legacy`.
    Error(&'static str),
}

/// Decide what the encryption-scheme pass does with one raw stored value.
pub fn reencrypt_stored_value(cipher: &Cipher, stored: &Json) -> Reencrypt {
    let stored = match stored {
        Json::Null => return Reencrypt::Skipped("empty"),
        Json::String(s) if s.is_empty() => return Reencrypt::Skipped("empty"),
        Json::String(s) => s,
        _ => return Reencrypt::Error("unexpected-type"),
    };
    if stored == UNDECRYPTABLE_PLACEHOLDER {
        return Reencrypt::Error("decrypt-placeholder");
    }
    if stored.starts_with(PBKDF2_MARKER) {
        return match cipher.decrypt(stored) {
            Ok(_) => Reencrypt::Skipped("already-migrated"),
            Err(_) => Reencrypt::Error("undecryptable-marked"),
        };
    }
    match cipher.decrypt(stored) {
        Ok(secret) => Reencrypt::Migrated(cipher.encrypt(&secret)),
        Err(_) => Reencrypt::Error("undecryptable-legacy"),
    }
}

/// Reference values produced by the Python build
/// (`open_notebook/utils/encryption.py` with `cryptography` 50.0.2, checked on
/// Python 3.9 and 3.11). Shared by the tests of the records that store keys.
#[cfg(test)]
pub(crate) mod reference {
    /// The passphrase of `Config::for_tests`, so one PBKDF2 derivation serves
    /// every test in the binary.
    pub const PASSPHRASE: &str = "test-encryption-key";
    /// `derive_fernet_key(PASSPHRASE)`.
    pub const DERIVED_KEY: &str = "JGDDvFwjvqAKdpqP_0Em1N0ySEqHg0bhfVbJIz4hU1Y=";
    /// `_derive_legacy_fernet_key(PASSPHRASE)`.
    pub const LEGACY_KEY: &str = "TU_pBcXXXWPceeZh0Zt76EgWHOl78yIlUaMoRPQZeME=";

    pub const MARKED_PLAINTEXT: &str = "sk-python-reference-key-1234567890";
    /// `encrypt_value(MARKED_PLAINTEXT)`.
    pub const MARKED: &str = "pbkdf2v1:gAAAAABq9h_1MeAHDHQr-2nPw7JX_ZpVzyujeoJvSFEHBRMrMTkyoRRwu9NdGo-LZN0L4TOam8KJXXxGFFRrY6BhyPkHjMC44M3eoxmtiwHDpW6xcw6yjtwZy2IkDz2donNMUQfwWjdp";

    pub const UNICODE_PLAINTEXT: &str =
        "clé-secrète-日本語-🔑 with spaces and a longer body to span several AES blocks";
    /// `encrypt_value(UNICODE_PLAINTEXT)`.
    pub const MARKED_UNICODE: &str = "pbkdf2v1:gAAAAABq9h_1Le3d48fdyWjvN3fXl50BoMicQtojnb9cyZ4YBfKki4ozJUApSdv7Wj6HQX23RKpYdEL9VY_ZBhakvavUEkdmEUbD7j_HrUT8p31YiB-2gjM5jF-G0XIXzIcvIl0I5b3KM-gFbMJ-lFhgKdz4iynZa6U3yjxtSSBJkJ6iAU1vW-QNSazfZhmFPFPwbBQTkod1";
    /// `encrypt_value("")`.
    pub const MARKED_EMPTY: &str = "pbkdf2v1:gAAAAABq9h_12hMb1m3qy51r3ffOp4M9ZbhRajTQzMdurmVdrh9b5Gf9SnW5Uxj5bzNTrcoBmcDlMlbfq07K2ImzmwXuAMqPSA==";

    pub const LEGACY_PLAINTEXT: &str = "sk-python-legacy-key-0987654321";
    /// A Fernet token under the pre-PBKDF2 key (bare SHA-256 of PASSPHRASE).
    pub const LEGACY: &str = "gAAAAABq9h_1HzqdY5hjPbWv3LHmqDg13_7WgZjaYl_3FY4Wy1QdaWgkXbmDoVor1A6x_P-rNiWBeSvQbH-rvVmcjAAx_5pyeEM0uDCA28zT7cUu_GJ89so=";

    /// `encrypt_value("sk-foreign-marked")` under the passphrase `another-passphrase`.
    pub const MARKED_OTHER_KEY: &str = "pbkdf2v1:gAAAAABq9h_1ZpHJtgB1GtnX3Xlv4c6DTcnvnhIwXjJIIZYWzdBbXkFpEyLXW9kyYlQ79QJx98s4HV6IRagWoiCD1gv2eYxsskVkPGgu2K8uHuBOAPKeFK4=";
    /// A legacy token for `sk-foreign-legacy` under `another-passphrase`.
    pub const LEGACY_OTHER_KEY: &str = "gAAAAABq9h_1irIX8YEPrNl1cIuAnZ4yq4oO2D0_-UMAADmaebFWKy6yKMLAT79FoBy3dOD8Z00A6lQYiu95HkPp0ylQVd-d2p6XFX0Qxg6Q_1cdYjkDUBQ=";
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::json;

    use super::reference::*;
    use super::*;

    /// A cipher with a cheap KDF, like the Python suite's `fast_kdf` fixture.
    /// Its legacy key is the real one (the legacy derivation has no cost
    /// parameter); its PBKDF2 key is not the production key.
    fn fast(passphrase: &str) -> Cipher {
        Cipher::with_iterations(passphrase, 1_000)
    }

    /// The production cipher for the reference passphrase (cached, so the
    /// 600 000-iteration derivation runs once per test binary).
    fn production() -> Arc<Cipher> {
        Cipher::for_passphrase(PASSPHRASE)
    }

    fn config(key: Option<&str>) -> Config {
        let mut config = Config::for_tests(Path::new("/tmp/nn-crypto-tests"));
        config.encryption_key = key.map(String::from);
        config
    }

    fn message(result: AppResult<String>) -> String {
        match result {
            Err(AppError::Configuration(message)) => message,
            other => panic!("expected a configuration error, got {other:?}"),
        }
    }

    #[test]
    fn production_parameters() {
        assert_eq!(PBKDF2_ITERATIONS, 600_000);
        assert_eq!(PBKDF2_SALT, b"open-notebook-credential-v1");
        assert_eq!(PBKDF2_MARKER, "pbkdf2v1:");
    }

    #[test]
    fn kdf_is_pbkdf2_and_matches_python() {
        // hashlib.pbkdf2_hmac("sha256", b"test-passphrase", PBKDF2_SALT, 1000)
        assert_eq!(
            URL_SAFE.encode(pbkdf2_key("test-passphrase", 1_000)),
            "0f6-CODlcPqeU0Zx_3gU5PFAMKDY72e9_3TFavVu9TU="
        );
        // hashlib.sha256(b"test-passphrase")
        assert_eq!(derive_legacy_fernet_key("test-passphrase"), "dXTwG569OyXjZA-IQnJg9gWHS6dv7-2AJCHYup4jjJM=");
        assert_eq!(derive_legacy_fernet_key(PASSPHRASE), LEGACY_KEY);

        // The production derivation (600 000 iterations), through the cache.
        let cipher = production();
        let key = cipher.pbkdf2_key();
        let derived = URL_SAFE.encode([key.signing, key.encryption].concat());
        assert_eq!(derived, DERIVED_KEY);
        assert_ne!(derived, LEGACY_KEY);
    }

    #[test]
    fn fernet_spec_vectors() {
        // https://github.com/fernet/spec: generate.json and verify.json.
        let key: [u8; 32] =
            URL_SAFE.decode("cw_0x689RpI-jtRR7oE8h_eQsKImvJapLeSbXpwF4e4=").unwrap().try_into().unwrap();
        let key = FernetKey::new(key);
        let token = "gAAAAAAdwJ6wAAECAwQFBgcICQoLDA0ODy021cpGVWKZ_eEwCGM4BLLF_5CV9dOPmrhuVUPgJobwOz7JcbmrR64jVmpU4IwqDA==";
        let iv: [u8; 16] = std::array::from_fn(|i| i as u8);
        assert_eq!(key.encrypt_at(b"hello", 499_162_800, iv), token);
        assert_eq!(key.decrypt(token.as_bytes()).ok().unwrap(), b"hello");
        // One flipped bit anywhere fails the signature.
        let tampered = token.replacen("AdwJ6w", "AdwJ6x", 1);
        assert!(key.decrypt(tampered.as_bytes()).is_err());
    }

    #[test]
    fn python_tokens_decrypt() {
        let cipher = production();
        assert_eq!(cipher.decrypt(MARKED).unwrap(), MARKED_PLAINTEXT);
        assert_eq!(cipher.decrypt(MARKED_UNICODE).unwrap(), UNICODE_PLAINTEXT);
        assert_eq!(cipher.decrypt(MARKED_EMPTY).unwrap(), "");
        assert_eq!(cipher.decrypt(LEGACY).unwrap(), LEGACY_PLAINTEXT);
        // The same through the configured key.
        let config = config(Some(PASSPHRASE));
        assert_eq!(decrypt_value(&config, MARKED).unwrap(), MARKED_PLAINTEXT);
        assert_eq!(decrypt_value(&config, LEGACY).unwrap(), LEGACY_PLAINTEXT);
    }

    #[test]
    fn round_trip_new_format() {
        let cipher = production();
        let stored = cipher.encrypt("sk-live-secret");
        assert!(stored.starts_with(PBKDF2_MARKER));
        assert!(!stored.contains("sk-live-secret"));
        assert_eq!(cipher.decrypt(&stored).unwrap(), "sk-live-secret");
        // A fresh IV every time.
        assert_ne!(cipher.encrypt("sk-live-secret"), stored);

        let config = config(Some(PASSPHRASE));
        let stored = encrypt_value(&config, "sk-domain-shape").unwrap();
        assert_eq!(decrypt_value(&config, &stored).unwrap(), "sk-domain-shape");
        assert_eq!(cipher.decrypt(&stored).unwrap(), "sk-domain-shape");

        for plaintext in ["", "x", "exactly-16-bytes", "ünïcödé 🔑", &"long ".repeat(200)] {
            assert_eq!(cipher.decrypt(&cipher.encrypt(plaintext)).unwrap(), plaintext);
        }
    }

    #[test]
    fn legacy_fernet_fallback() {
        let cipher = fast("test-passphrase");
        let legacy = cipher.legacy_key().encrypt(b"sk-legacy");
        assert!(!legacy.starts_with(PBKDF2_MARKER));
        assert_eq!(cipher.decrypt(&legacy).unwrap(), "sk-legacy");
    }

    #[test]
    fn legacy_plaintext_passes_through() {
        let cipher = fast("test-passphrase");
        assert_eq!(cipher.decrypt("sk-plain-legacy").unwrap(), "sk-plain-legacy");
        assert_eq!(cipher.decrypt("").unwrap(), "");
        assert_eq!(cipher.decrypt(UNDECRYPTABLE_PLACEHOLDER).unwrap(), UNDECRYPTABLE_PLACEHOLDER);
    }

    #[test]
    fn marked_value_under_the_wrong_key_is_an_error() {
        let stored = fast("test-passphrase").encrypt("sk-live-secret");
        let error = fast("wrong-passphrase").decrypt(&stored);
        assert_eq!(message(error), MARKED_FAILED);

        // A value the Python build wrote under another passphrase.
        let error = production().decrypt(MARKED_OTHER_KEY).unwrap_err();
        assert!(is_decryption_error(&error));
        assert_eq!(message(Err(error)), MARKED_FAILED);
        assert_eq!(message(production().decrypt(LEGACY_OTHER_KEY)), WRONG_KEY);
        // The marker selects the scheme: a legacy token behind it does not decrypt.
        assert_eq!(message(production().decrypt(&format!("{PBKDF2_MARKER}{LEGACY}"))), MARKED_FAILED);
    }

    #[test]
    fn corrupt_marked_values_are_errors() {
        let cipher = production();
        let truncated = &MARKED[..MARKED.len() - 4];
        let double = format!("{PBKDF2_MARKER}{MARKED}");
        for corrupt in ["pbkdf2v1:", "pbkdf2v1:not-valid-base64!!!", "pbkdf2v1:aGk=", truncated, double.as_str()] {
            assert_eq!(message(cipher.decrypt(corrupt)), MARKED_FAILED, "{corrupt}");
        }
        // Python's decoder skips stray whitespace, so this still decrypts there and here.
        assert_eq!(cipher.decrypt(&format!("{MARKED}\n")).unwrap(), MARKED_PLAINTEXT);
    }

    #[test]
    fn marker_like_plaintext_is_encrypted() {
        let cipher = fast("test-passphrase");
        let plaintext = format!("{PBKDF2_MARKER}looks-like-a-key");
        let stored = cipher.encrypt(&plaintext);
        assert!(stored.starts_with(PBKDF2_MARKER));
        assert_ne!(stored, plaintext);
        assert_eq!(cipher.decrypt(&stored).unwrap(), plaintext);

        // Corrupt marked input is encrypted too, never stored as it came.
        let corrupt = format!("{PBKDF2_MARKER}aGk=");
        let stored = cipher.encrypt(&corrupt);
        assert_ne!(stored, corrupt);
        assert_eq!(cipher.decrypt(&stored).unwrap(), corrupt);
    }

    #[test]
    fn no_double_encryption() {
        let cipher = fast("test-passphrase");
        let stored = cipher.encrypt("sk-live-secret");
        assert_eq!(cipher.encrypt(&stored), stored);
        assert_eq!(cipher.decrypt(&cipher.encrypt(&stored)).unwrap(), "sk-live-secret");

        // A value the Python build encrypted is left alone as well.
        assert_eq!(production().encrypt(MARKED), MARKED);
        assert_eq!(encrypt_value(&config(Some(PASSPHRASE)), MARKED).unwrap(), MARKED);
        // Marked under another key is not valid ciphertext here, so it is wrapped.
        let wrapped = production().encrypt(MARKED_OTHER_KEY);
        assert_ne!(wrapped, MARKED_OTHER_KEY);
        assert_eq!(production().decrypt(&wrapped).unwrap(), MARKED_OTHER_KEY);
    }

    #[test]
    fn missing_key_is_a_configuration_error() {
        for config in [config(None), config(Some(""))] {
            assert!(!is_configured(&config));
            for result in [encrypt_value(&config, "sk-x"), decrypt_value(&config, "sk-x"), decrypt_value(&config, MARKED)]
            {
                let error = result.unwrap_err();
                assert!(!is_decryption_error(&error));
                assert_eq!(error.status(), axum::http::StatusCode::UNPROCESSABLE_ENTITY);
                assert!(error.to_string().starts_with("OPEN_NOTEBOOK_ENCRYPTION_KEY is not set. "), "{error}");
            }
            let gate = require_encryption_key(&config).unwrap_err();
            assert_eq!(gate.status(), axum::http::StatusCode::BAD_REQUEST);
            assert_eq!(gate.to_string(), KEY_NOT_CONFIGURED);
        }
        let configured = config(Some(PASSPHRASE));
        assert!(is_configured(&configured));
        assert!(require_encryption_key(&configured).is_ok());
    }

    /// The outcome of `decrypt_value` on an unmarked value.
    #[derive(Debug, PartialEq)]
    enum Unmarked {
        /// Returned unchanged (legacy plaintext).
        Plaintext,
        /// Decrypted under the legacy key.
        Decrypts(&'static str),
        /// Looks encrypted, key does not fit.
        WrongKey,
    }

    /// Every row was produced by the Python implementation: its
    /// `looks_like_fernet_token` answer and what `decrypt_value` did.
    #[test]
    fn unmarked_values_classify_like_python() {
        use Unmarked::*;
        let a = |n: usize| "A".repeat(n);
        let cases: Vec<(String, bool, Unmarked)> = vec![
            ("sk-plain-legacy".into(), false, Plaintext),
            (String::new(), false, Plaintext),
            ("a".repeat(99), false, Plaintext),
            ("a".repeat(100), false, Plaintext),
            ("a".repeat(101), false, Plaintext),
            (a(120), false, Plaintext),
            (a(124), false, Plaintext),
            (a(140), true, WrongKey),
            (format!("{}=", a(119)), true, WrongKey),
            (format!("sk-proj-{}x", "Ab1_-".repeat(31)), false, Plaintext),
            (format!("sk-ant-api03-{}", "Zz9-_".repeat(19)), false, Plaintext),
            (format!("{}!", a(139)), false, Plaintext),
            (format!("{}!!!!", a(140)), true, WrongKey),
            (format!("{} {}", a(70), a(70)), true, WrongKey),
            (format!("{}\n{}", a(70), a(70)), true, WrongKey),
            (format!("{}==", a(138)), false, Plaintext),
            (format!("{}=", a(138)), false, Plaintext),
            (a(141), false, Plaintext),
            (a(142), false, Plaintext),
            (a(143), false, Plaintext),
            ("+/".repeat(70), true, WrongKey),
            ("-_".repeat(70), true, WrongKey),
            ("é".repeat(140), false, Plaintext),
            (format!("{}é", a(139)), false, Plaintext),
            (format!("gAAAAA{}", "B".repeat(134)), true, WrongKey),
            (LEGACY[..LEGACY.len() - 1].to_string(), false, Plaintext),
            (format!("{LEGACY}\n"), true, Decrypts(LEGACY_PLAINTEXT)),
            (format!(" {LEGACY}"), true, Decrypts(LEGACY_PLAINTEXT)),
            (format!("{}\n{}", &LEGACY[..50], &LEGACY[50..]), true, Decrypts(LEGACY_PLAINTEXT)),
            (LEGACY.replace('-', "+").replace('_', "/"), true, Decrypts(LEGACY_PLAINTEXT)),
            (LEGACY.to_string(), true, Decrypts(LEGACY_PLAINTEXT)),
            (LEGACY_OTHER_KEY.to_string(), true, WrongKey),
            (MARKED[PBKDF2_MARKER.len()..].to_string(), true, WrongKey),
            (UNDECRYPTABLE_PLACEHOLDER.to_string(), false, Plaintext),
        ];
        // Unmarked values only ever need the legacy key, which has no cost parameter.
        let cipher = fast(PASSPHRASE);
        for (input, looks, expected) in cases {
            assert_eq!(looks_like_fernet_token(&input), looks, "looks_like_fernet_token({input:?})");
            let outcome = match cipher.decrypt(&input) {
                Ok(value) if value == input => Plaintext,
                Ok(value) if value == LEGACY_PLAINTEXT => Decrypts(LEGACY_PLAINTEXT),
                Ok(value) => panic!("{input:?} decrypted to something unexpected: {value:?}"),
                Err(error) => {
                    assert_eq!(error.to_string(), WRONG_KEY, "{input:?}");
                    WrongKey
                }
            };
            assert_eq!(outcome, expected, "decrypt({input:?})");
        }
    }

    #[test]
    fn reencryption_decides_per_value() {
        let cipher = fast("test-passphrase");
        let legacy = cipher.legacy_key().encrypt(b"sk-legacy");
        let marked = cipher.encrypt("sk-new");

        match reencrypt_stored_value(&cipher, &json!(legacy)) {
            Reencrypt::Migrated(stored) => {
                assert!(stored.starts_with(PBKDF2_MARKER));
                assert_eq!(cipher.decrypt(&stored).unwrap(), "sk-legacy");
                // A second pass leaves it alone.
                assert_eq!(reencrypt_stored_value(&cipher, &json!(stored)), Reencrypt::Skipped("already-migrated"));
            }
            other => panic!("expected a migration, got {other:?}"),
        }
        // Legacy plaintext is encrypted as well.
        match reencrypt_stored_value(&cipher, &json!("sk-plain-legacy")) {
            Reencrypt::Migrated(stored) => assert_eq!(cipher.decrypt(&stored).unwrap(), "sk-plain-legacy"),
            other => panic!("expected a migration, got {other:?}"),
        }
        assert_eq!(reencrypt_stored_value(&cipher, &json!(marked)), Reencrypt::Skipped("already-migrated"));
        assert_eq!(reencrypt_stored_value(&cipher, &Json::Null), Reencrypt::Skipped("empty"));
        assert_eq!(reencrypt_stored_value(&cipher, &json!("")), Reencrypt::Skipped("empty"));
        assert_eq!(reencrypt_stored_value(&cipher, &json!(12345)), Reencrypt::Error("unexpected-type"));
        assert_eq!(reencrypt_stored_value(&cipher, &json!(["sk-list"])), Reencrypt::Error("unexpected-type"));
        assert_eq!(reencrypt_stored_value(&cipher, &json!("UNDECRYPTABLE")), Reencrypt::Error("decrypt-placeholder"));
        assert_eq!(
            reencrypt_stored_value(&cipher, &json!("pbkdf2v1:aGk=")),
            Reencrypt::Error("undecryptable-marked")
        );
        let foreign = fast("another-passphrase").legacy_key().encrypt(b"sk-foreign");
        assert_eq!(reencrypt_stored_value(&cipher, &json!(foreign)), Reencrypt::Error("undecryptable-legacy"));
    }

    #[test]
    fn secrets_and_ciphers_do_not_print() {
        let secret = SecretString::new("sk-very-secret");
        assert_eq!(secret.expose(), "sk-very-secret");
        assert!(!format!("{secret:?}").contains("sk-very-secret"));
        assert!(!format!("{:?}", Some(secret)).contains("very"));
        assert!(!format!("{:?}", fast("my-passphrase")).contains("my-passphrase"));
    }
}
