//! `.misa` TOML parser.
//!
//! [`parse_str`] takes a TOML string + [`AssetSource`], decodes it via
//! `serde`, runs identifier sanitisation and structural validation, and
//! returns a [`ParseOutput`] suitable for either inspection or
//! conversion to a runtime [`crate::model::Model`] via
//! [`super::build::build_model`].
//!
//! Sanitisation is **non-fatal** — names that fail the identifier rule
//! get rewritten and recorded in [`LoadReport::sanitized_names`].
//! Structural problems (unknown joint parent / child, duplicate names,
//! missing root) are fatal and surface as [`NativeError::Validation`].

use std::collections::{BTreeMap, HashMap, HashSet};

use super::report::{sanitize_identifier, LoadReport};
use super::schema::{
    Actuator, ActuatorJointRef, CollisionPair, Geom, Joint, Link, LoopClosure, Material, MisaFile,
    Mimic, Pose, Sensor, Sequence, SCHEMA_TAG,
};
use super::source::AssetSource;
use super::{NativeError, ParseOutput};

/// Parse a `.misa` TOML string and resolve mesh references via `assets`.
///
/// See module-level docs for behaviour. Returns [`NativeError::Toml`] for
/// syntactic problems, [`NativeError::UnsupportedSchema`] for an unknown
/// schema tag, and [`NativeError::Validation`] for structural errors that
/// would prevent building a runtime model.
pub fn parse_str(text: &str, assets: &dyn AssetSource) -> Result<ParseOutput, NativeError> {
    // ── 1. TOML decode ──────────────────────────────────────────────────
    let mut file: MisaFile =
        toml::from_str(text).map_err(|e| NativeError::Toml(e.to_string()))?;

    // ── 2. Schema tag ───────────────────────────────────────────────────
    check_schema_tag(&file.schema)?;

    // ── 3. Sanitise identifiers + record actions ────────────────────────
    let mut report = LoadReport::default();
    sanitise_all_identifiers(&mut file, &mut report);

    // ── 4. Structural validation ────────────────────────────────────────
    validate_structure(&file)?;

    // ── 5. Mesh existence (non-fatal) ───────────────────────────────────
    check_mesh_assets(&file, assets, &mut report);

    Ok(ParseOutput { file, report })
}

// ─── Schema check ──────────────────────────────────────────────────────────

fn check_schema_tag(tag: &str) -> Result<(), NativeError> {
    let (vendor, major) = MisaFile::parse_schema(tag).ok_or_else(|| {
        NativeError::UnsupportedSchema(format!(
            "schema tag '{tag}' is not in 'vendor/major' format (expected '{SCHEMA_TAG}')"
        ))
    })?;
    if vendor != "misarta" {
        return Err(NativeError::UnsupportedSchema(format!(
            "schema vendor '{vendor}' is not recognised (expected 'misarta')"
        )));
    }
    if major != super::schema::CURRENT_VERSION {
        return Err(NativeError::UnsupportedSchema(format!(
            "schema version {major} is newer than this build supports ({})",
            super::schema::CURRENT_VERSION
        )));
    }
    Ok(())
}

// ─── Identifier sanitisation ──────────────────────────────────────────────

fn sanitise_all_identifiers(file: &mut MisaFile, report: &mut LoadReport) {
    // Each category sanitises in declaration order. We also build a
    // rename map from old → new for each category so cross-references
    // (joint.parent, actuator.joints[*].name, mimic.source / .joint, etc.)
    // can be patched in a second pass.

    let link_renames = sanitise_named_collection(
        "link",
        file.link.iter_mut().map(|l| &mut l.name),
        report,
    );
    let joint_renames = sanitise_named_collection(
        "joint",
        file.joint.iter_mut().map(|j| &mut j.name),
        report,
    );
    let material_renames = sanitise_named_collection(
        "material",
        file.material.iter_mut().map(|m| &mut m.name),
        report,
    );
    let sensor_renames = sanitise_named_collection(
        "sensor",
        file.sensor.iter_mut().map(|s| &mut s.name),
        report,
    );
    let actuator_renames = sanitise_named_collection(
        "actuator",
        file.actuator.iter_mut().map(|a| &mut a.name),
        report,
    );
    let pose_renames = sanitise_named_collection(
        "pose",
        file.pose.iter_mut().map(|p| &mut p.name),
        report,
    );
    let sequence_renames = sanitise_named_collection(
        "sequence",
        file.sequence.iter_mut().map(|s| &mut s.name),
        report,
    );
    let loop_closure_renames = sanitise_named_collection(
        "loop_closure",
        file.loop_closure.iter_mut().map(|lc| &mut lc.name),
        report,
    );
    let gait_renames = sanitise_named_collection(
        "gait",
        file.gait.iter_mut().map(|g| &mut g.name),
        report,
    );

    // Suppress unused warnings for renames we don't currently propagate
    // (sequence/loop_closure/gait names are only referenced internally
    // and by the user, never by other tables).
    let _ = (sequence_renames, loop_closure_renames, gait_renames);

    // ── Patch cross-references ──────────────────────────────────────────
    // robot.root → link
    apply_rename(&mut file.robot.root, &link_renames);

    // joint.parent / .child → link
    for j in &mut file.joint {
        apply_rename(&mut j.parent, &link_renames);
        apply_rename(&mut j.child, &link_renames);
    }

    // visual.material → material
    for l in &mut file.link {
        for v in &mut l.visual {
            if let Some(name) = &mut v.material {
                apply_rename(name, &material_renames);
            }
        }
    }

    // sensor.link → link, sensor.kind.{joint,partner} → link/joint
    for s in &mut file.sensor {
        apply_rename(&mut s.link, &link_renames);
        match &mut s.kind {
            super::schema::SensorKind::ForceTorque { joint } => {
                if let Some(j) = joint {
                    apply_rename(j, &joint_renames);
                }
            }
            super::schema::SensorKind::Contact { partner } => {
                if let Some(p) = partner {
                    apply_rename(p, &link_renames);
                }
            }
            _ => {}
        }
    }

    // actuator.joints[*].name → joint
    for a in &mut file.actuator {
        for jr in &mut a.joints {
            apply_rename(&mut jr.name, &joint_renames);
        }
    }

    // mimic.joint / .source → joint
    for m in &mut file.mimic {
        apply_rename(&mut m.joint, &joint_renames);
        apply_rename(&mut m.source, &joint_renames);
    }

    // loop_closure.link_a / .link_b → link
    for lc in &mut file.loop_closure {
        apply_rename(&mut lc.link_a, &link_renames);
        apply_rename(&mut lc.link_b, &link_renames);
    }

    // collision_pair.link_a / .link_b → link
    for cp in &mut file.collision_pair {
        apply_rename(&mut cp.link_a, &link_renames);
        apply_rename(&mut cp.link_b, &link_renames);
    }

    // pose.angles keys (joint names) — BTreeMap rebuild
    if !joint_renames.is_empty() {
        for p in &mut file.pose {
            patch_angle_map(&mut p.angles, &joint_renames);
        }
        patch_angle_map(&mut file.home.joint_positions, &joint_renames);
    }

    // sequence.steps[].pose_name → pose
    if !pose_renames.is_empty() {
        for s in &mut file.sequence {
            for step in &mut s.steps {
                apply_rename(&mut step.pose_name, &pose_renames);
            }
        }
    }

    let _ = actuator_renames; // actuator names aren't referenced elsewhere
    let _ = sensor_renames; // sensor names aren't referenced elsewhere
}

/// Sanitise every name in an iterator. Returns a map of original → new
/// (omits entries that didn't change).
fn sanitise_named_collection<'a, I: IntoIterator<Item = &'a mut String>>(
    category: &str,
    names: I,
    report: &mut LoadReport,
) -> HashMap<String, String> {
    let mut renames = HashMap::new();
    for (idx, name) in names.into_iter().enumerate() {
        let (sanitised, reason) = sanitize_identifier(name);
        if let Some(reason) = reason {
            renames.insert(name.clone(), sanitised.clone());
            report.push_sanitization(category, name.clone(), sanitised.clone(), reason, idx);
            *name = sanitised;
        }
    }
    renames
}

fn apply_rename(s: &mut String, renames: &HashMap<String, String>) {
    if let Some(new) = renames.get(s) {
        *s = new.clone();
    }
}

fn patch_angle_map(map: &mut BTreeMap<String, f64>, renames: &HashMap<String, String>) {
    if renames.is_empty() {
        return;
    }
    let to_rename: Vec<(String, String)> = map
        .keys()
        .filter_map(|k| renames.get(k).map(|new| (k.clone(), new.clone())))
        .collect();
    for (old, new) in to_rename {
        if let Some(v) = map.remove(&old) {
            map.insert(new, v);
        }
    }
}

// ─── Structural validation ────────────────────────────────────────────────

fn validate_structure(file: &MisaFile) -> Result<(), NativeError> {
    // No empty robot name (keeps downstream logging useful)
    if file.robot.name.is_empty() {
        return Err(NativeError::Validation(
            "[robot] name is empty".into(),
        ));
    }

    // Duplicate link / joint names
    check_duplicates("link", file.link.iter().map(|l| l.name.as_str()))?;
    check_duplicates("joint", file.joint.iter().map(|j| j.name.as_str()))?;
    check_duplicates("material", file.material.iter().map(|m| m.name.as_str()))?;
    check_duplicates("actuator", file.actuator.iter().map(|a| a.name.as_str()))?;
    check_duplicates("sensor", file.sensor.iter().map(|s| s.name.as_str()))?;
    check_duplicates("pose", file.pose.iter().map(|p| p.name.as_str()))?;

    // Build link / joint name sets for cross-reference checks
    let link_names: HashSet<&str> = file.link.iter().map(|l| l.name.as_str()).collect();
    let joint_names: HashSet<&str> = file.joint.iter().map(|j| j.name.as_str()).collect();
    let material_names: HashSet<&str> =
        file.material.iter().map(|m| m.name.as_str()).collect();
    let pose_names: HashSet<&str> = file.pose.iter().map(|p| p.name.as_str()).collect();

    // Root link must exist
    if !link_names.contains(file.robot.root.as_str()) {
        return Err(NativeError::Validation(format!(
            "robot.root = '{}' does not name any [[link]]",
            file.robot.root,
        )));
    }

    // Joint parent / child must reference known links
    for j in &file.joint {
        if !link_names.contains(j.parent.as_str()) {
            return Err(NativeError::Validation(format!(
                "joint '{}': parent link '{}' is not declared",
                j.name, j.parent,
            )));
        }
        if !link_names.contains(j.child.as_str()) {
            return Err(NativeError::Validation(format!(
                "joint '{}': child link '{}' is not declared",
                j.name, j.child,
            )));
        }
        if j.parent == j.child {
            return Err(NativeError::Validation(format!(
                "joint '{}': parent and child are the same link",
                j.name,
            )));
        }
    }

    // No two joints may share the same child (tree topology). Loop closures
    // are expressed separately via [[loop_closure]], not by reusing joints.
    let mut child_owner: HashMap<&str, &str> = HashMap::new();
    for j in &file.joint {
        if let Some(prev) = child_owner.insert(j.child.as_str(), j.name.as_str()) {
            return Err(NativeError::Validation(format!(
                "links '{}' is the child of two joints ('{}' and '{}'); \
                 use [[loop_closure]] for closed kinematic chains",
                j.child, prev, j.name,
            )));
        }
    }

    // Visual.material references must resolve, and color/material are
    // mutually exclusive.
    for l in &file.link {
        for v in &l.visual {
            if v.color.is_some() && v.material.is_some() {
                return Err(NativeError::Validation(format!(
                    "link '{}': visual has both `color` and `material` set; choose one",
                    l.name,
                )));
            }
            if let Some(name) = &v.material {
                if !material_names.contains(name.as_str()) {
                    return Err(NativeError::Validation(format!(
                        "link '{}': visual references unknown material '{}'",
                        l.name, name,
                    )));
                }
            }
        }
    }

    // Sensor.link must exist; sensor kind references where applicable.
    for s in &file.sensor {
        if !link_names.contains(s.link.as_str()) {
            return Err(NativeError::Validation(format!(
                "sensor '{}': link '{}' is not declared",
                s.name, s.link,
            )));
        }
        if let super::schema::SensorKind::ForceTorque { joint: Some(j) } = &s.kind {
            if !joint_names.contains(j.as_str()) {
                return Err(NativeError::Validation(format!(
                    "sensor '{}': force_torque joint '{}' is not declared",
                    s.name, j,
                )));
            }
        }
        if let super::schema::SensorKind::Contact { partner: Some(p) } = &s.kind {
            if !link_names.contains(p.as_str()) {
                return Err(NativeError::Validation(format!(
                    "sensor '{}': contact partner '{}' is not declared",
                    s.name, p,
                )));
            }
        }
    }

    // Actuator references must resolve to declared joints, and `joints`
    // must be non-empty.
    for a in &file.actuator {
        if a.joints.is_empty() {
            return Err(NativeError::Validation(format!(
                "actuator '{}': must drive at least one joint",
                a.name,
            )));
        }
        for jr in &a.joints {
            if !joint_names.contains(jr.name.as_str()) {
                return Err(NativeError::Validation(format!(
                    "actuator '{}': drives unknown joint '{}'",
                    a.name, jr.name,
                )));
            }
        }
    }

    // Mimic source / target must be declared joints and not equal.
    for m in &file.mimic {
        if !joint_names.contains(m.joint.as_str()) {
            return Err(NativeError::Validation(format!(
                "mimic: target joint '{}' is not declared",
                m.joint,
            )));
        }
        if !joint_names.contains(m.source.as_str()) {
            return Err(NativeError::Validation(format!(
                "mimic: source joint '{}' is not declared",
                m.source,
            )));
        }
        if m.joint == m.source {
            return Err(NativeError::Validation(format!(
                "mimic: joint and source are the same ('{}')",
                m.joint,
            )));
        }
    }

    // Loop closure links must exist.
    for lc in &file.loop_closure {
        if !link_names.contains(lc.link_a.as_str()) {
            return Err(NativeError::Validation(format!(
                "loop_closure '{}': link_a '{}' is not declared",
                lc.name, lc.link_a,
            )));
        }
        if !link_names.contains(lc.link_b.as_str()) {
            return Err(NativeError::Validation(format!(
                "loop_closure '{}': link_b '{}' is not declared",
                lc.name, lc.link_b,
            )));
        }
    }

    // Collision pair links must exist.
    for cp in &file.collision_pair {
        if !link_names.contains(cp.link_a.as_str()) {
            return Err(NativeError::Validation(format!(
                "collision_pair: link_a '{}' is not declared",
                cp.link_a,
            )));
        }
        if !link_names.contains(cp.link_b.as_str()) {
            return Err(NativeError::Validation(format!(
                "collision_pair: link_b '{}' is not declared",
                cp.link_b,
            )));
        }
    }

    // Sequence step pose references must resolve.
    for s in &file.sequence {
        for step in &s.steps {
            if !pose_names.contains(step.pose_name.as_str()) {
                return Err(NativeError::Validation(format!(
                    "sequence '{}': step references unknown pose '{}'",
                    s.name, step.pose_name,
                )));
            }
        }
    }

    Ok(())
}

fn check_duplicates<'a>(
    category: &str,
    names: impl Iterator<Item = &'a str>,
) -> Result<(), NativeError> {
    let mut seen: HashSet<&str> = HashSet::new();
    for name in names {
        if !seen.insert(name) {
            return Err(NativeError::Validation(format!(
                "duplicate {category} name: '{name}'",
            )));
        }
    }
    Ok(())
}

// ─── Mesh asset existence check ────────────────────────────────────────────

fn check_mesh_assets(file: &MisaFile, assets: &dyn AssetSource, report: &mut LoadReport) {
    let mut seen: HashSet<&str> = HashSet::new();
    for l in &file.link {
        for v in &l.visual {
            collect_missing_mesh(&v.geom, assets, &mut seen, report);
        }
        for c in &l.collision {
            collect_missing_mesh(&c.geom, assets, &mut seen, report);
        }
    }
}

fn collect_missing_mesh<'a>(
    geom: &'a Geom,
    assets: &dyn AssetSource,
    seen: &mut HashSet<&'a str>,
    report: &mut LoadReport,
) {
    if let Geom::Mesh { file: path, .. } = geom {
        if seen.insert(path.as_str()) && !assets.exists(path) {
            report.push_missing_mesh(path);
        }
    }
}

// ─── Suppress unused warnings for forward-compat re-exports ───────────────
#[allow(dead_code)]
fn _shape_check_compile_time() {
    // Keep these symbols referenced so compiler surfaces type errors here
    // if their schemas drift. Cheaper than writing dummy tests for each.
    let _ = std::any::type_name::<Actuator>();
    let _ = std::any::type_name::<ActuatorJointRef>();
    let _ = std::any::type_name::<CollisionPair>();
    let _ = std::any::type_name::<Joint>();
    let _ = std::any::type_name::<Link>();
    let _ = std::any::type_name::<LoopClosure>();
    let _ = std::any::type_name::<Material>();
    let _ = std::any::type_name::<Mimic>();
    let _ = std::any::type_name::<Pose>();
    let _ = std::any::type_name::<Sensor>();
    let _ = std::any::type_name::<Sequence>();
}
