//! Asset source abstraction for `.misa` loading.
//!
//! The core parser ([`crate::native::parse`]) needs to read mesh files and
//! other auxiliary data referenced from the `.misa` TOML, but it should
//! NOT depend on `std::fs` directly — that would prevent the parser from
//! being used in embedded targets, WASM sandboxes, or test fixtures.
//!
//! This module defines [`AssetSource`] — a trait that abstracts byte
//! retrieval — and ships several built-in implementations:
//!
//! - [`FileSystemSource`] — reads from disk relative to a root directory.
//! - [`InMemorySource`] — serves bytes from an owned `BTreeMap`. Useful
//!   for tests and for shipping fixtures bundled with a binary.
//! - [`StaticBundleSource`] — serves bytes from a `&'static` slice table.
//!   Pairs well with `build.rs` + `include_bytes!` for firmware images.
//! - [`NullSource`] — refuses every read. Use it when the caller only
//!   wants the structural data and meshes are irrelevant.
//!
//! # Path conventions
//!
//! Paths passed to [`AssetSource::read`] are **logical paths relative to
//! the `.misa` file** (e.g. `"meshes/trunk.stl"`). Implementations MUST
//! enforce sandbox semantics:
//!
//! - Reject absolute paths (`/`, `\`, drive letters).
//! - Reject `..` segments.
//! - When mapping to a real filesystem, verify the resolved path stays
//!   under the configured root.
//!
//! These rules are enforced inside [`FileSystemSource`]; custom
//! implementations are responsible for their own sandbox.

use std::collections::BTreeMap;
use std::path::PathBuf;

/// Errors produced by an [`AssetSource`].
#[derive(Debug, Clone)]
pub enum AssetError {
    /// No asset exists at the requested path.
    NotFound,
    /// The path violates sandbox rules (absolute, contains `..`, escapes root).
    AccessDenied,
    /// Underlying I/O error stringified — keeps the trait `std::io`-free.
    Io(String),
}

impl std::fmt::Display for AssetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssetError::NotFound => f.write_str("asset not found"),
            AssetError::AccessDenied => f.write_str("asset access denied"),
            AssetError::Io(msg) => write!(f, "asset I/O error: {msg}"),
        }
    }
}

impl std::error::Error for AssetError {}

/// Abstract byte source for assets referenced from a `.misa` file.
///
/// Implementations must enforce sandbox semantics — see the module-level
/// docs for the full rule list.
pub trait AssetSource {
    /// Read the bytes at `path` (logical, relative to the `.misa` file).
    fn read(&self, path: &str) -> Result<Vec<u8>, AssetError>;

    /// Cheap existence check. Default implementation calls `read` and
    /// discards the result; override for sources that can answer faster
    /// (e.g. an `lstat` instead of a full read).
    fn exists(&self, path: &str) -> bool {
        self.read(path).is_ok()
    }
}

// ─── Path validation helper (shared by built-in sources) ──────────────────

/// Reject paths that violate the sandbox conventions documented on
/// [`AssetSource`]. Implementations of [`AssetSource`] should call this
/// before doing any I/O.
pub fn validate_logical_path(path: &str) -> Result<(), AssetError> {
    if path.is_empty() {
        return Err(AssetError::AccessDenied);
    }
    // Absolute paths in any flavour.
    if path.starts_with('/') || path.starts_with('\\') {
        return Err(AssetError::AccessDenied);
    }
    // Windows drive-letter prefix (`C:`).
    if path.len() >= 2 {
        let bytes = path.as_bytes();
        if bytes[1] == b':' && bytes[0].is_ascii_alphabetic() {
            return Err(AssetError::AccessDenied);
        }
    }
    // Any `..` segment.
    for seg in path.split(['/', '\\']) {
        if seg == ".." {
            return Err(AssetError::AccessDenied);
        }
    }
    Ok(())
}

// ─── FileSystemSource ─────────────────────────────────────────────────────

/// Reads assets from disk under a fixed root directory.
///
/// `read("meshes/trunk.stl")` resolves to `<root>/meshes/trunk.stl`. Path
/// validation rejects absolute paths and `..` segments before touching
/// the filesystem; the resolved path is also re-checked against `root`
/// to defend against symlink shenanigans.
#[derive(Debug, Clone)]
pub struct FileSystemSource {
    root: PathBuf,
}

impl FileSystemSource {
    /// Build a source rooted at `root`. The directory does not have to
    /// exist at construction time — failure surfaces on the first `read`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &std::path::Path {
        &self.root
    }
}

impl AssetSource for FileSystemSource {
    fn read(&self, path: &str) -> Result<Vec<u8>, AssetError> {
        validate_logical_path(path)?;
        let candidate = self.root.join(path);

        // Defence in depth: after canonicalising, ensure we're still under
        // root. Skipped when canonicalize fails (e.g. file missing) — the
        // subsequent read will produce NotFound which is the right answer.
        if let (Ok(canon_root), Ok(canon)) = (self.root.canonicalize(), candidate.canonicalize()) {
            if !canon.starts_with(&canon_root) {
                return Err(AssetError::AccessDenied);
            }
        }

        match std::fs::read(&candidate) {
            Ok(bytes) => Ok(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(AssetError::NotFound),
            Err(e) => Err(AssetError::Io(e.to_string())),
        }
    }

    fn exists(&self, path: &str) -> bool {
        if validate_logical_path(path).is_err() {
            return false;
        }
        self.root.join(path).exists()
    }
}

// ─── InMemorySource ───────────────────────────────────────────────────────

/// Serves assets from an owned in-memory map. Convenient for tests and
/// for callers that have already loaded everything (e.g. after fetching
/// over a network).
#[derive(Debug, Clone, Default)]
pub struct InMemorySource {
    files: BTreeMap<String, Vec<u8>>,
}

impl InMemorySource {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) bytes at `path`.
    pub fn insert(&mut self, path: impl Into<String>, bytes: impl Into<Vec<u8>>) {
        self.files.insert(path.into(), bytes.into());
    }
}

impl AssetSource for InMemorySource {
    fn read(&self, path: &str) -> Result<Vec<u8>, AssetError> {
        validate_logical_path(path)?;
        self.files
            .get(path)
            .cloned()
            .ok_or(AssetError::NotFound)
    }

    fn exists(&self, path: &str) -> bool {
        validate_logical_path(path).is_ok() && self.files.contains_key(path)
    }
}

// ─── StaticBundleSource ───────────────────────────────────────────────────

/// Serves assets from a `&'static` table — pair with `build.rs` +
/// `include_bytes!` to ship a `.misa` and its meshes inside a binary.
///
/// ```ignore
/// const ASSETS: &[(&str, &[u8])] = &[
///     ("namiashi.misa",    include_bytes!("../assets/namiashi.misa")),
///     ("meshes/trunk.stl", include_bytes!("../assets/meshes/trunk.stl")),
/// ];
/// let source = StaticBundleSource::new(ASSETS);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct StaticBundleSource<'a> {
    files: &'a [(&'a str, &'a [u8])],
}

impl<'a> StaticBundleSource<'a> {
    pub const fn new(files: &'a [(&'a str, &'a [u8])]) -> Self {
        Self { files }
    }
}

impl AssetSource for StaticBundleSource<'_> {
    fn read(&self, path: &str) -> Result<Vec<u8>, AssetError> {
        validate_logical_path(path)?;
        for (name, bytes) in self.files {
            if *name == path {
                return Ok(bytes.to_vec());
            }
        }
        Err(AssetError::NotFound)
    }

    fn exists(&self, path: &str) -> bool {
        if validate_logical_path(path).is_err() {
            return false;
        }
        self.files.iter().any(|(name, _)| *name == path)
    }
}

// ─── NullSource ───────────────────────────────────────────────────────────

/// Refuses every asset read. Use it to parse a `.misa` file's structure
/// without resolving any mesh references — the parser will accept this
/// and simply skip mesh loading, deferring it to a later
/// `load_meshes(&assets)` call with a real source.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullSource;

impl AssetSource for NullSource {
    fn read(&self, _path: &str) -> Result<Vec<u8>, AssetError> {
        Err(AssetError::NotFound)
    }

    fn exists(&self, _path: &str) -> bool {
        false
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_absolute_paths() {
        assert!(matches!(
            validate_logical_path("/etc/passwd"),
            Err(AssetError::AccessDenied)
        ));
        assert!(matches!(
            validate_logical_path("\\Windows"),
            Err(AssetError::AccessDenied)
        ));
        assert!(matches!(
            validate_logical_path("C:/Windows"),
            Err(AssetError::AccessDenied)
        ));
    }

    #[test]
    fn validate_rejects_dot_dot() {
        assert!(matches!(
            validate_logical_path("../etc/passwd"),
            Err(AssetError::AccessDenied)
        ));
        assert!(matches!(
            validate_logical_path("meshes/../../secret"),
            Err(AssetError::AccessDenied)
        ));
    }

    #[test]
    fn validate_accepts_normal_paths() {
        assert!(validate_logical_path("meshes/trunk.stl").is_ok());
        assert!(validate_logical_path("a.txt").is_ok());
        assert!(validate_logical_path("deep/nested/path/file.bin").is_ok());
    }

    #[test]
    fn validate_rejects_empty() {
        assert!(matches!(
            validate_logical_path(""),
            Err(AssetError::AccessDenied)
        ));
    }

    #[test]
    fn in_memory_source_round_trip() {
        let mut src = InMemorySource::new();
        src.insert("meshes/trunk.stl", b"hello".to_vec());
        assert!(src.exists("meshes/trunk.stl"));
        assert_eq!(src.read("meshes/trunk.stl").unwrap(), b"hello");
        assert!(matches!(
            src.read("missing"),
            Err(AssetError::NotFound)
        ));
    }

    #[test]
    fn in_memory_source_blocks_traversal() {
        let mut src = InMemorySource::new();
        src.insert("../etc/passwd", b"x".to_vec());
        // The insertion is allowed (raw map), but `read` rejects the path.
        assert!(matches!(
            src.read("../etc/passwd"),
            Err(AssetError::AccessDenied)
        ));
    }

    #[test]
    fn static_bundle_source() {
        const ASSETS: &[(&str, &[u8])] = &[("a.txt", b"alpha"), ("b/c.bin", b"beta")];
        let src = StaticBundleSource::new(ASSETS);
        assert_eq!(src.read("a.txt").unwrap(), b"alpha");
        assert_eq!(src.read("b/c.bin").unwrap(), b"beta");
        assert!(matches!(src.read("missing"), Err(AssetError::NotFound)));
        assert!(src.exists("a.txt"));
        assert!(!src.exists("missing"));
    }

    #[test]
    fn null_source_refuses_all() {
        let src = NullSource;
        assert!(matches!(src.read("any"), Err(AssetError::NotFound)));
        assert!(!src.exists("any"));
    }
}
