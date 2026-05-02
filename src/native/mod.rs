//! `.misa` — misarta's native master format.
//!
//! `.misa` is a TOML-encoded, articara/misarta-native description of a robot
//! that supersedes the URDF + `.misarta.toml` sidecar split: a single file
//! holds the full kinematic tree, geometry, materials, mimic / loop-closure /
//! collision-pair / sensor / actuator definitions, plus editor metadata
//! (poses, sequences, gaits, home pose).
//!
//! See `doc/refactor_20260502.md` for the design rationale; the on-disk
//! schema lives in [`schema`].
//!
//! # Layered API
//!
//! The module is split into three layers so the file-system dependency
//! can be swapped out for embedded / WASM / test scenarios:
//!
//! - **Layer 1 — [`source`]**: the [`AssetSource`] trait plus four
//!   built-in implementations ([`FileSystemSource`], [`InMemorySource`],
//!   [`StaticBundleSource`], [`NullSource`]).
//! - **Layer 2 — [`parse`]**: [`parse_str`] takes a TOML string + an
//!   `AssetSource` and produces a [`ParseOutput`]. No `std::fs` access.
//! - **Layer 3 — convenience**: [`load`] / [`save`] wrap layer 2 with
//!   `std::fs` for the common "I have a path on disk" case.
//!
//! # Quick reference
//!
//! ```ignore
//! // Common case: read a .misa from disk.
//! let out = misarta::native::load("robots/namiashi/namiashi.misa")?;
//! if !out.report.is_empty() {
//!     show_dialog(&out.report);
//! }
//!
//! // Embedded case: parse from a memory buffer with bundled meshes.
//! const ASSETS: &[(&str, &[u8])] = &[ /* ... */ ];
//! let source = StaticBundleSource::new(ASSETS);
//! let text = std::str::from_utf8(source.read("robot.misa")?)?;
//! let out = misarta::native::parse_str(text, &source)?;
//! ```

pub mod report;
pub mod schema;
pub mod source;

pub use report::{is_valid_identifier, sanitize_identifier, LoadReport,
                 MaterialCollision, NameSanitization};
pub use schema::{
    Actuator, ActuatorJointRef, ActuatorMode, CollisionPair, ColorSpec, Gait, GaitTypeConfig,
    Geom, Home, Inertial, Joint, JointDynamics, JointKind, JointLimit, Link, LoopClosure,
    Material, MisaFile, Mimic, Origin, Pose, RobotMeta, Sensor, SensorKind, Sequence,
    SequenceStep, Visual, Collision, CURRENT_VERSION, SCHEMA_TAG,
};
pub use source::{
    validate_logical_path, AssetError, AssetSource, FileSystemSource, InMemorySource, NullSource,
    StaticBundleSource,
};

use std::path::Path;

// ─── Errors ────────────────────────────────────────────────────────────────

/// Top-level error type for `.misa` load / parse / save.
#[derive(Debug, Clone)]
pub enum NativeError {
    /// I/O or asset access failed.
    Io(String),
    /// TOML failed to parse, or the document didn't match the schema.
    Toml(String),
    /// `schema = "..."` header is missing, malformed, or names a version
    /// this build can't read.
    UnsupportedSchema(String),
    /// Structural validation failed (e.g. joint references unknown link,
    /// duplicate name, root link not in `link` list).
    Validation(String),
    /// An [`AssetSource`] reported failure for a required asset.
    Asset(AssetError),
}

impl std::fmt::Display for NativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NativeError::Io(m) => write!(f, "I/O error: {m}"),
            NativeError::Toml(m) => write!(f, "TOML error: {m}"),
            NativeError::UnsupportedSchema(m) => write!(f, "unsupported .misa schema: {m}"),
            NativeError::Validation(m) => write!(f, "validation error: {m}"),
            NativeError::Asset(e) => write!(f, "asset error: {e}"),
        }
    }
}

impl std::error::Error for NativeError {}

impl From<AssetError> for NativeError {
    fn from(e: AssetError) -> Self {
        NativeError::Asset(e)
    }
}

// ─── ParseOutput ───────────────────────────────────────────────────────────

/// The fully-decoded contents of a `.misa` file.
///
/// At the layer-2 boundary we return the raw schema struct rather than
/// converting straight to [`crate::model::Model`]; the conversion lives
/// in a separate step so callers that only want the structural data
/// (e.g. a model linter, a documentation generator) don't pay the cost
/// of building the dynamics model.
///
/// Use [`build_model`] when you want a runtime [`crate::model::Model`].
#[derive(Debug, Clone)]
pub struct ParseOutput {
    /// The parsed document, post-sanitisation.
    pub file: schema::MisaFile,
    /// Diagnostics — sanitised names, missing meshes, warnings.
    pub report: LoadReport,
}

// ─── Layer 2: parse_str ────────────────────────────────────────────────────

/// Parse a `.misa` TOML string and resolve any required assets via
/// `assets`.
///
/// Pass [`NullSource`] when meshes are not needed; the parser will
/// record any references as `report.missing_meshes` but won't fail.
///
/// **Status**: stub. The actual TOML → schema decoding, identifier
/// sanitisation, and validation passes will be filled in by follow-up
/// PRs (see ToDo list in `doc/refactor_20260502.md`).
pub fn parse_str(
    _text: &str,
    _assets: &dyn AssetSource,
) -> Result<ParseOutput, NativeError> {
    Err(NativeError::Validation(
        "parse_str: not yet implemented (schema and traits are in place; \
         follow-up will wire decoding + validation)"
            .into(),
    ))
}

/// Serialise a [`schema::MisaFile`] to a TOML string.
///
/// **Status**: stub. Will validate the file before serialising and emit
/// sections in canonical order to keep diffs stable.
pub fn write_str(_file: &schema::MisaFile) -> Result<String, NativeError> {
    Err(NativeError::Validation(
        "write_str: not yet implemented (schema in place; follow-up will \
         wire validation + canonical-order emission)"
            .into(),
    ))
}

// ─── Layer 3: load / save (std::fs convenience) ────────────────────────────

/// Read a `.misa` file from disk.
///
/// Wraps [`parse_str`] with a [`FileSystemSource`] rooted at the file's
/// parent directory, so mesh references like `"meshes/trunk.stl"` resolve
/// relative to the `.misa` location.
pub fn load(path: impl AsRef<Path>) -> Result<ParseOutput, NativeError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| {
        NativeError::Io(format!("read {}: {e}", path.display()))
    })?;
    let root = path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let assets = FileSystemSource::new(root);
    parse_str(&text, &assets)
}

/// Write a `.misa` file to disk. The caller is responsible for ensuring
/// `path.parent()` exists.
pub fn save(path: impl AsRef<Path>, file: &schema::MisaFile) -> Result<(), NativeError> {
    let path = path.as_ref();
    let text = write_str(file)?;
    std::fs::write(path, text).map_err(|e| {
        NativeError::Io(format!("write {}: {e}", path.display()))
    })
}

// ─── Build runtime Model from parsed file (stub) ──────────────────────────

/// Convert a parsed [`schema::MisaFile`] into a runtime
/// [`crate::model::Model`] plus [`crate::geometry::GeometryModel`].
///
/// **Status**: stub. The conversion mirrors the existing
/// `crate::urdf::load_urdf_string` / `crate::sdf` paths but consumes the
/// `.misa` schema directly. Will be wired up alongside `parse_str`.
pub fn build_model(
    _file: &schema::MisaFile,
) -> Result<
    (
        crate::model::Model<f64>,
        crate::geometry::GeometryModel,
    ),
    NativeError,
> {
    Err(NativeError::Validation(
        "build_model: not yet implemented (depends on parse_str)".into(),
    ))
}
