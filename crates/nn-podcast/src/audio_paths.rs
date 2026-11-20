//! Single choke point for podcast episode audio file paths.
//!
//! `PodcastEpisode::audio_file` stores a path RELATIVE to the podcasts folder
//! (e.g. `episodes/<uuid>/audio/<uuid>.mp3`). Two helpers enforce that
//! contract at the only two places paths cross the database boundary:
//!
//! - [`to_relative_audio_path`], write side (after generation): converts the
//!   generated file path to the relative storage form and refuses to produce
//!   a value outside the podcasts root, so the database never holds an
//!   absolute or escaping path.
//! - [`resolve_contained_audio_path`], read side (stream, list, get, delete,
//!   retry, publish, feed): joins the stored value with the podcasts folder,
//!   resolves symlinks and `..`, and verifies containment. Any absolute,
//!   `scheme://` or escaping value is treated as invalid and gives `None`.
//!
//! The original read the podcasts folder from its configuration module; here
//! every function takes it as an argument.

use std::path::{Component, Path, PathBuf};

use crate::error::{PodcastError, Result};

/// Symlinks followed for one path before giving up (a loop).
const MAX_SYMLINKS: usize = 40;

/// Resolve `path` like Python's `os.path.realpath`: make it absolute, fold
/// `.` and `..`, and follow symlinks, without requiring the path (or its
/// tail) to exist.
pub fn realpath(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")).join(path)
    };
    let mut links_left = MAX_SYMLINKS;
    resolve_components(PathBuf::new(), &absolute, &mut links_left)
}

fn resolve_components(mut resolved: PathBuf, rest: &Path, links_left: &mut usize) -> PathBuf {
    for component in rest.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => resolved.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Normal(name) => {
                let candidate = resolved.join(name);
                let target = match std::fs::symlink_metadata(&candidate) {
                    Ok(metadata) if metadata.file_type().is_symlink() && *links_left > 0 => {
                        std::fs::read_link(&candidate).ok()
                    }
                    _ => None,
                };
                match target {
                    Some(target) => {
                        *links_left -= 1;
                        // A relative target is resolved against the directory
                        // holding the link; an absolute one starts over.
                        let base = if target.is_absolute() { PathBuf::new() } else { resolved };
                        resolved = resolve_components(base, &target, links_left);
                    }
                    None => resolved = candidate,
                }
            }
        }
    }
    resolved
}

/// Real (symlink-resolved, absolute) path of the podcasts output root.
pub fn podcasts_root(podcasts_folder: &Path) -> PathBuf {
    realpath(podcasts_folder)
}

/// Where a new episode's files go: `(<uuid>, <podcasts_folder>/episodes/<uuid>)`.
///
/// The directory name is a fresh UUID, never the episode name, so it is safe
/// whatever the user typed. It is built under the same root that
/// [`to_relative_audio_path`] validates against.
pub fn build_episode_output_dir(podcasts_folder: &Path) -> (String, PathBuf) {
    let episode_dir_name = uuid::Uuid::new_v4().to_string();
    let output_dir = podcasts_folder.join("episodes").join(&episode_dir_name);
    (episode_dir_name, output_dir)
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = (bytes[i] == b'%').then(|| bytes.get(i + 1..i + 3)).flatten();
        match hex.and_then(|hex| std::str::from_utf8(hex).ok()).and_then(|hex| u8::from_str_radix(hex, 16).ok()) {
            Some(byte) => {
                out.push(byte);
                i += 3;
            }
            None => {
                out.push(bytes[i]);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The decoded path component of a `file://` URI.
fn file_uri_path(uri: &str) -> String {
    let rest = &uri["file://".len()..];
    // Authority (usually empty) ends at the first slash; query and fragment
    // are not part of the path.
    let path = rest.find('/').map_or("", |slash| &rest[slash..]);
    let path = path.split(['?', '#']).next().unwrap_or_default();
    percent_decode(path)
}

/// Convert a generated audio file path to the database storage form.
///
/// Accepts an absolute or working-directory-relative path, including the
/// legacy `file://` URI form, and returns it relative to `podcasts_folder` as
/// a `/`-separated string.
///
/// Fails with [`PodcastError::InvalidInput`] when the path resolves outside
/// the podcasts root (or is the root itself): the database must never hold an
/// absolute or escaping value.
pub fn to_relative_audio_path(podcasts_folder: &Path, audio_path: impl AsRef<Path>) -> Result<String> {
    let audio_path = audio_path.as_ref();
    let shown = audio_path.to_string_lossy();
    let raw = if shown.starts_with("file://") { PathBuf::from(file_uri_path(&shown)) } else { audio_path.to_path_buf() };

    let resolved = realpath(&raw);
    let root = podcasts_root(podcasts_folder);
    match resolved.strip_prefix(&root) {
        Ok(relative) if !relative.as_os_str().is_empty() => Ok(relative
            .components()
            .map(|component| component.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/")),
        _ => Err(PodcastError::InvalidInput(format!(
            "Generated audio file path is outside the podcasts folder: {shown}"
        ))),
    }
}

/// Resolve a stored `audio_file` value to a real filesystem path.
///
/// Returns `None` for anything that must not be followed:
/// - empty or missing values;
/// - absolute paths and `scheme://` URIs (legacy rows);
/// - relative paths that escape the root (`..` or symlink traversal);
/// - the root itself.
///
/// The file does not have to exist: callers check that separately.
pub fn resolve_contained_audio_path(podcasts_folder: &Path, audio_file: Option<&str>) -> Option<PathBuf> {
    let audio_file = audio_file.filter(|value| !value.is_empty())?;
    if audio_file.contains("://") {
        return None;
    }
    let candidate = Path::new(audio_file);
    if candidate.is_absolute() || audio_file.starts_with('/') || audio_file.starts_with('\\') {
        return None;
    }
    let root = podcasts_root(podcasts_folder);
    let resolved = realpath(&root.join(candidate));
    (resolved != root && resolved.starts_with(&root)).then_some(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    // --- write side --------------------------------------------------------

    #[test]
    fn absolute_path_under_root_becomes_relative() {
        let dir = tmp();
        let audio = dir.path().join("episodes/uuid-1/audio/uuid-1.mp3");
        assert_eq!(
            to_relative_audio_path(dir.path(), &audio).unwrap(),
            "episodes/uuid-1/audio/uuid-1.mp3"
        );
        assert_eq!(
            to_relative_audio_path(dir.path(), audio.to_str().unwrap()).unwrap(),
            "episodes/uuid-1/audio/uuid-1.mp3"
        );
    }

    #[test]
    fn file_uri_under_root_becomes_relative() {
        let dir = tmp();
        let uri = format!("file://{}/episodes/uuid-3/a.mp3", dir.path().display());
        assert_eq!(to_relative_audio_path(dir.path(), &uri).unwrap(), "episodes/uuid-3/a.mp3");

        let encoded = format!("file://{}/episodes/my%20show/a.mp3?x=1#t", dir.path().display());
        assert_eq!(to_relative_audio_path(dir.path(), &encoded).unwrap(), "episodes/my show/a.mp3");
    }

    #[test]
    fn relative_root_and_relative_path_are_normalised_against_the_working_directory() {
        // The original's case: the data folder configured as "./data".
        let cwd = std::env::current_dir().unwrap();
        let rel = to_relative_audio_path(
            Path::new("./nn-podcast-test-data/podcasts"),
            "nn-podcast-test-data/podcasts/episodes/uuid-4/a.mp3",
        )
        .unwrap();
        assert_eq!(rel, "episodes/uuid-4/a.mp3");
        assert!(podcasts_root(Path::new("./nn-podcast-test-data/podcasts")).starts_with(realpath(&cwd)));
    }

    #[test]
    fn path_outside_root_is_refused() {
        let dir = tmp();
        let root = dir.path().join("podcasts");
        std::fs::create_dir(&root).unwrap();
        let error = to_relative_audio_path(&root, dir.path().join("elsewhere.mp3")).unwrap_err();
        assert!(matches!(error, PodcastError::InvalidInput(_)));
        assert!(error.to_string().contains("outside the podcasts folder"));
    }

    #[test]
    fn traversal_out_of_root_is_refused() {
        let dir = tmp();
        let root = dir.path().join("podcasts");
        std::fs::create_dir(&root).unwrap();
        let error = to_relative_audio_path(&root, root.join("..").join("escape.mp3")).unwrap_err();
        assert!(error.to_string().contains("outside the podcasts folder"));
    }

    #[test]
    fn root_itself_is_refused() {
        let dir = tmp();
        let error = to_relative_audio_path(dir.path(), dir.path()).unwrap_err();
        assert!(error.to_string().contains("outside the podcasts folder"));
    }

    #[test]
    fn sibling_with_matching_prefix_is_refused_on_write() {
        let dir = tmp();
        let root = dir.path().join("podcasts");
        std::fs::create_dir(&root).unwrap();
        assert!(to_relative_audio_path(&root, dir.path().join("podcasts_evil/a.mp3")).is_err());
    }

    #[test]
    fn roundtrips_with_read_helper() {
        let dir = tmp();
        let audio = dir.path().join("episodes/uuid-5/a.mp3");
        std::fs::create_dir_all(audio.parent().unwrap()).unwrap();
        std::fs::write(&audio, b"audio").unwrap();

        let stored = to_relative_audio_path(dir.path(), &audio).unwrap();
        assert_eq!(
            resolve_contained_audio_path(dir.path(), Some(&stored)),
            Some(std::fs::canonicalize(&audio).unwrap())
        );
    }

    #[test]
    fn podcasts_root_is_absolute() {
        assert!(podcasts_root(Path::new("data/podcasts")).is_absolute());
    }

    #[test]
    fn episode_output_dir_is_a_uuid_under_episodes() {
        let (name, output_dir) = build_episode_output_dir(Path::new("/data/podcasts"));
        assert_eq!(uuid::Uuid::parse_str(&name).unwrap().to_string(), name);
        assert_eq!(output_dir, Path::new("/data/podcasts/episodes").join(&name));
        assert!(name.chars().all(|c| "0123456789abcdef-".contains(c)));

        let (other, _) = build_episode_output_dir(Path::new("/data/podcasts"));
        assert_ne!(name, other);

        let dir = tmp();
        let (_, creatable) = build_episode_output_dir(dir.path());
        std::fs::create_dir_all(&creatable).unwrap();
        assert!(creatable.is_dir());
    }

    // --- read side ---------------------------------------------------------

    #[test]
    fn relative_path_inside_root_resolves() {
        let dir = tmp();
        let episode_dir = dir.path().join("episodes/some-uuid");
        std::fs::create_dir_all(&episode_dir).unwrap();
        std::fs::write(episode_dir.join("final.mp3"), b"fake audio").unwrap();

        assert_eq!(
            resolve_contained_audio_path(dir.path(), Some("episodes/some-uuid/final.mp3")),
            Some(std::fs::canonicalize(episode_dir.join("final.mp3")).unwrap())
        );
    }

    #[test]
    fn a_file_that_does_not_exist_still_resolves() {
        let dir = tmp();
        let resolved = resolve_contained_audio_path(dir.path(), Some("episodes/abc/out.mp3")).unwrap();
        assert!(resolved.ends_with("episodes/abc/out.mp3"));
        assert!(resolved.starts_with(std::fs::canonicalize(dir.path()).unwrap()));
    }

    #[test]
    fn dotdot_escape_is_rejected() {
        let dir = tmp();
        let root = dir.path().join("podcasts");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(dir.path().join("outside.mp3"), b"etc passwd style file").unwrap();

        assert_eq!(resolve_contained_audio_path(&root, Some("../outside.mp3")), None);
        assert_eq!(resolve_contained_audio_path(&root, Some("episodes/../../outside.mp3")), None);
        assert_eq!(resolve_contained_audio_path(&root, Some("../../etc/passwd")), None);
    }

    #[test]
    fn dotdot_that_stays_inside_is_allowed() {
        let dir = tmp();
        let resolved = resolve_contained_audio_path(dir.path(), Some("episodes/x/../y/a.mp3")).unwrap();
        assert!(resolved.ends_with("episodes/y/a.mp3"));
    }

    #[test]
    fn absolute_path_is_rejected_even_when_inside_root() {
        let dir = tmp();
        let inside = dir.path().join("episodes/u/a.mp3");
        std::fs::create_dir_all(inside.parent().unwrap()).unwrap();
        std::fs::write(&inside, b"audio").unwrap();

        assert_eq!(resolve_contained_audio_path(dir.path(), inside.to_str()), None);
        assert_eq!(resolve_contained_audio_path(dir.path(), Some("/etc/passwd")), None);
    }

    #[test]
    fn file_uri_is_rejected() {
        let dir = tmp();
        assert_eq!(
            resolve_contained_audio_path(dir.path(), Some("file:///data/podcasts/episodes/x/a.mp3")),
            None
        );
        assert_eq!(resolve_contained_audio_path(dir.path(), Some("https://example.com/a.mp3")), None);
    }

    #[test]
    fn sibling_directory_with_matching_prefix_is_rejected() {
        let dir = tmp();
        let root = dir.path().join("podcasts");
        std::fs::create_dir(&root).unwrap();
        let sibling = dir.path().join("podcasts_evil");
        std::fs::create_dir(&sibling).unwrap();
        std::fs::write(sibling.join("secret.mp3"), b"not yours").unwrap();

        assert_eq!(resolve_contained_audio_path(&root, Some("../podcasts_evil/secret.mp3")), None);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escape_is_rejected() {
        let dir = tmp();
        let root = dir.path().join("podcasts");
        std::fs::create_dir(&root).unwrap();
        let outside = dir.path().join("outside.mp3");
        std::fs::write(&outside, b"secret").unwrap();
        std::os::unix::fs::symlink(&outside, root.join("link.mp3")).unwrap();
        std::os::unix::fs::symlink(dir.path(), root.join("up")).unwrap();
        std::os::unix::fs::symlink("../outside.mp3", root.join("relative-link.mp3")).unwrap();

        assert_eq!(resolve_contained_audio_path(&root, Some("link.mp3")), None);
        assert_eq!(resolve_contained_audio_path(&root, Some("up/outside.mp3")), None);
        assert_eq!(resolve_contained_audio_path(&root, Some("relative-link.mp3")), None);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_that_stays_inside_resolves_to_its_target() {
        let dir = tmp();
        let root = dir.path().join("podcasts");
        std::fs::create_dir_all(root.join("episodes/a")).unwrap();
        std::fs::write(root.join("episodes/a/real.mp3"), b"audio").unwrap();
        std::os::unix::fs::symlink("episodes/a/real.mp3", root.join("alias.mp3")).unwrap();
        std::os::unix::fs::symlink(&root, root.join("loop")).unwrap();

        assert_eq!(
            resolve_contained_audio_path(&root, Some("alias.mp3")),
            Some(std::fs::canonicalize(root.join("episodes/a/real.mp3")).unwrap())
        );
        // A link back to the root is the root, which is not a file.
        assert_eq!(resolve_contained_audio_path(&root, Some("loop")), None);
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_root_is_resolved_once_for_both_sides() {
        let dir = tmp();
        let real = dir.path().join("real-podcasts");
        std::fs::create_dir_all(real.join("episodes/u")).unwrap();
        std::fs::write(real.join("episodes/u/a.mp3"), b"audio").unwrap();
        let linked = dir.path().join("podcasts");
        std::os::unix::fs::symlink(&real, &linked).unwrap();

        let stored = to_relative_audio_path(&linked, linked.join("episodes/u/a.mp3")).unwrap();
        assert_eq!(stored, "episodes/u/a.mp3");
        assert_eq!(
            resolve_contained_audio_path(&linked, Some(&stored)),
            Some(std::fs::canonicalize(real.join("episodes/u/a.mp3")).unwrap())
        );
    }

    #[test]
    fn empty_and_none_are_rejected() {
        let dir = tmp();
        assert_eq!(resolve_contained_audio_path(dir.path(), None), None);
        assert_eq!(resolve_contained_audio_path(dir.path(), Some("")), None);
    }

    #[test]
    fn root_itself_is_rejected() {
        let dir = tmp();
        assert_eq!(resolve_contained_audio_path(dir.path(), Some(".")), None);
        assert_eq!(resolve_contained_audio_path(dir.path(), Some("episodes/..")), None);
    }

    #[test]
    fn realpath_matches_canonicalize_for_existing_paths() {
        let dir = tmp();
        let nested = dir.path().join("a/b");
        std::fs::create_dir_all(&nested).unwrap();
        assert_eq!(realpath(&nested), std::fs::canonicalize(&nested).unwrap());
        assert_eq!(realpath(&nested.join("../b/./missing/x")), std::fs::canonicalize(&nested).unwrap().join("missing/x"));
    }
}
