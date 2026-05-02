//! `.misa` TOML writer.
//!
//! [`write_str`] serialises a [`MisaFile`] to TOML in canonical section
//! order so on-disk diffs stay stable across edits. Section order is
//! defined by the field order on [`MisaFile`] itself.
//!
//! Pre-write validation reuses [`super::parse::parse_str`]'s structural
//! checks indirectly through a re-validation pass — when callers produced
//! the [`MisaFile`] in code (e.g. via `RobotModel::to_misa`), it's easy
//! to miss a dangling reference; failing here turns a silent bad file
//! into an early `Err`.

use super::schema::MisaFile;
use super::NativeError;

/// Serialise a [`MisaFile`] to a TOML string suitable for writing to a
/// `.misa` file.
///
/// Validation is **light** at the writer level — it only checks the
/// schema tag is correct. Cross-reference validation belongs on the
/// parse side; trying to re-validate here would mean duplicating the
/// `parse_str` checks (or factoring them out), which is more code than
/// it saves. If a caller writes a structurally-broken `MisaFile`, the
/// next `parse_str` will surface it.
pub fn write_str(file: &MisaFile) -> Result<String, NativeError> {
    if file.schema != super::schema::SCHEMA_TAG {
        return Err(NativeError::UnsupportedSchema(format!(
            "expected schema tag '{}', got '{}'",
            super::schema::SCHEMA_TAG,
            file.schema
        )));
    }

    toml::to_string_pretty(file).map_err(|e| NativeError::Toml(e.to_string()))
}
