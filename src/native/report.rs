//! Diagnostics returned by [`crate::native::parse_str`] / `load`.
//!
//! Loading a `.misa` file is intentionally lenient — the parser fixes
//! up identifier names that violate the schema's character-set rules
//! and renames materials that collide on import — but each fix-up is
//! recorded here so the host can surface them to the user (the editor
//! shows them in a confirmation dialog, headless callers can log them).
//!
//! Construct an empty report with [`LoadReport::default`]; check whether
//! anything was reported with [`LoadReport::is_empty`].

use serde::{Deserialize, Serialize};

/// Aggregated load-time diagnostics.
///
/// All fields are append-only collections. Order within each vector is
/// the order in which the parser encountered the issue, so the dialog
/// can replay events linearly.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoadReport {
    /// Names that were sanitised because they failed the
    /// `^[A-Za-z_][A-Za-z0-9_]*$` rule. Each entry records what
    /// changed and why.
    pub sanitized_names: Vec<NameSanitization>,

    /// Materials whose names collided with already-loaded entries on
    /// import. The conflicting copy is renamed; the original keeps
    /// its name.
    pub material_collisions: Vec<MaterialCollision>,

    /// Mesh paths referenced from `.misa` that the [`AssetSource`]
    /// could not resolve. Non-fatal — the parser still returns a model;
    /// affected visuals are loaded without their mesh data.
    ///
    /// [`AssetSource`]: crate::native::AssetSource
    pub missing_meshes: Vec<String>,

    /// Free-form warnings (e.g. "deprecated field used", "ignored
    /// unknown sensor type"). Always non-fatal; fatal problems return
    /// `Err` from the parser instead of populating the report.
    pub warnings: Vec<String>,
}

impl LoadReport {
    /// True when the report has nothing to surface — the host can use
    /// this to decide whether to bother showing a dialog.
    pub fn is_empty(&self) -> bool {
        self.sanitized_names.is_empty()
            && self.material_collisions.is_empty()
            && self.missing_meshes.is_empty()
            && self.warnings.is_empty()
    }

    /// Total number of items across all categories. Useful for showing a
    /// compact "12 issues" badge before the user opens the full dialog.
    pub fn total(&self) -> usize {
        self.sanitized_names.len()
            + self.material_collisions.len()
            + self.missing_meshes.len()
            + self.warnings.len()
    }

    pub fn push_sanitization(
        &mut self,
        category: impl Into<String>,
        original: impl Into<String>,
        sanitized: impl Into<String>,
        reason: impl Into<String>,
        occurrence_index: usize,
    ) {
        self.sanitized_names.push(NameSanitization {
            category: category.into(),
            original: original.into(),
            sanitized: sanitized.into(),
            reason: reason.into(),
            occurrence_index,
        });
    }

    pub fn push_material_collision(
        &mut self,
        original: impl Into<String>,
        renamed_to: impl Into<String>,
    ) {
        self.material_collisions.push(MaterialCollision {
            original: original.into(),
            renamed_to: renamed_to.into(),
        });
    }

    pub fn push_missing_mesh(&mut self, path: impl Into<String>) {
        self.missing_meshes.push(path.into());
    }

    pub fn push_warning(&mut self, message: impl Into<String>) {
        self.warnings.push(message.into());
    }
}

/// One identifier that was rewritten during load.
///
/// Categories use snake_case singular nouns matching the table they came
/// from — `"link"`, `"joint"`, `"material"`, `"sensor"`, `"pose"`,
/// `"sequence"`, `"actuator"`, `"mimic"`, `"loop_closure"`,
/// `"collision_pair"`, `"gait"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NameSanitization {
    pub category: String,
    pub original: String,
    pub sanitized: String,
    /// Human-readable explanation, e.g. `"contained '-'"`,
    /// `"started with digit"`, `"contained whitespace"`.
    pub reason: String,
    /// 0-based position within `category` — disambiguates duplicate
    /// originals so the dialog can show "the 3rd link named 'foo'".
    pub occurrence_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialCollision {
    pub original: String,
    pub renamed_to: String,
}

// ─── Identifier sanitisation ───────────────────────────────────────────────

/// Sanitisation policy for entity names.
///
/// Rules applied in order:
/// 1. Replace `-` with `_`.
/// 2. Replace any whitespace run with `_`.
/// 3. Strip any remaining char outside `[A-Za-z0-9_]`.
/// 4. If the result is empty, return `"_"`.
/// 5. If the first char is a digit, prepend `_`.
///
/// Returns `(sanitized, reason)` where `reason` is `None` if the input
/// was already valid (no change needed).
pub fn sanitize_identifier(input: &str) -> (String, Option<&'static str>) {
    if input.is_empty() {
        return ("_".into(), Some("empty identifier"));
    }

    // Fast-path: already valid.
    if is_valid_identifier(input) {
        return (input.to_string(), None);
    }

    // Build sanitised form, remembering the most user-relevant reason.
    let mut reason: Option<&'static str> = None;
    let mut out = String::with_capacity(input.len());

    for ch in input.chars() {
        match ch {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '_' => out.push(ch),
            '-' => {
                out.push('_');
                reason.get_or_insert("contained '-'");
            }
            c if c.is_whitespace() => {
                if !out.ends_with('_') {
                    out.push('_');
                }
                reason.get_or_insert("contained whitespace");
            }
            _ => {
                reason.get_or_insert("contained non-identifier characters");
            }
        }
    }

    if out.is_empty() {
        return ("_".into(), Some("only non-identifier characters"));
    }

    if out.starts_with(|c: char| c.is_ascii_digit()) {
        out.insert(0, '_');
        reason.get_or_insert("started with digit");
    }

    (out, reason.or(Some("contained invalid characters")))
}

/// True when `s` matches `^[A-Za-z_][A-Za-z0-9_]*$`.
pub fn is_valid_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_report_is_empty() {
        let r = LoadReport::default();
        assert!(r.is_empty());
        assert_eq!(r.total(), 0);
    }

    #[test]
    fn report_aggregates_all_categories() {
        let mut r = LoadReport::default();
        r.push_sanitization("link", "trunk-1", "trunk_1", "contained '-'", 0);
        r.push_material_collision("red", "red_2");
        r.push_missing_mesh("meshes/missing.stl");
        r.push_warning("deprecated field 'foo'");
        assert!(!r.is_empty());
        assert_eq!(r.total(), 4);
    }

    #[test]
    fn valid_identifier_recognised() {
        assert!(is_valid_identifier("trunk"));
        assert!(is_valid_identifier("FL_calf_joint"));
        assert!(is_valid_identifier("_private"));
        assert!(is_valid_identifier("a1"));
        assert!(!is_valid_identifier(""));
        assert!(!is_valid_identifier("1leading_digit"));
        assert!(!is_valid_identifier("with-dash"));
        assert!(!is_valid_identifier("with space"));
        assert!(!is_valid_identifier("dot.name"));
    }

    #[test]
    fn sanitize_passes_valid_unchanged() {
        let (s, reason) = sanitize_identifier("FL_hip_joint");
        assert_eq!(s, "FL_hip_joint");
        assert!(reason.is_none());
    }

    #[test]
    fn sanitize_replaces_dash() {
        let (s, reason) = sanitize_identifier("front-left-leg");
        assert_eq!(s, "front_left_leg");
        assert_eq!(reason, Some("contained '-'"));
    }

    #[test]
    fn sanitize_replaces_whitespace() {
        let (s, reason) = sanitize_identifier("link  with   spaces");
        assert_eq!(s, "link_with_spaces");
        assert_eq!(reason, Some("contained whitespace"));
    }

    #[test]
    fn sanitize_prefixes_leading_digit() {
        let (s, reason) = sanitize_identifier("3wheel");
        assert_eq!(s, "_3wheel");
        assert_eq!(reason, Some("started with digit"));
    }

    #[test]
    fn sanitize_strips_invalid_chars() {
        let (s, reason) = sanitize_identifier("link.foo!");
        assert_eq!(s, "linkfoo");
        assert_eq!(reason, Some("contained non-identifier characters"));
    }

    #[test]
    fn sanitize_handles_empty() {
        let (s, reason) = sanitize_identifier("");
        assert_eq!(s, "_");
        assert_eq!(reason, Some("empty identifier"));
    }

    #[test]
    fn sanitize_handles_only_invalid_chars() {
        let (s, reason) = sanitize_identifier("...");
        assert_eq!(s, "_");
        assert_eq!(reason, Some("only non-identifier characters"));
    }
}
