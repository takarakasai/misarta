//! USD ASCII (.usda) ⇄ [`MisaFile`] conversion.
//!
//! The exporter generates a `.usda` conforming to UsdPhysics (suitable
//! for NVIDIA Isaac Sim / Omniverse); the importer parses files that
//! follow those conventions (and tolerates typical Isaac-flavoured
//! files). Ported from articara's `src/usd.rs` / `src/usd_import.rs` (A5).
//!
//! # Host boundary
//!
//! Two pieces of data live outside the schema and cross the boundary
//! explicitly:
//!
//! - **Posed world transforms** — the exporter writes each link at its
//!   world rest pose. By default that pose is chained from the joint
//!   origins at q = 0; a host with a posed model supplies
//!   [`UsdExportRefs::link_world_tf`].
//! - **Mesh geometry** — USD embeds mesh data inline (`points` /
//!   `normals`), but [`mn::Geom::Mesh`] only carries a file reference.
//!   On export the host supplies triangle soups via
//!   [`UsdExportRefs::mesh_vertices`]; on import, inline meshes are
//!   returned in [`UsdImport::inline_meshes`] with a sentinel
//!   (`__usd_inline__<n>`) in the schema's `file` field.
//!
//! USD has no native mimic / sensor / actuator concepts in plain
//! UsdPhysics — those `MisaFile` sections are dropped on export.

use std::collections::HashMap;

use misarta::mesh::MeshData;
use misarta::native as mn;
use misarta::native::MisaFile;
use nalgebra as na;

use crate::util::{fmt, origin_iso, resolve_visual_rgba};

/// Which geometry list a geom occurrence lives in, and at what index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeomSlot {
    Visual(usize),
    Collision(usize),
}

/// Prefix of the sentinel written into [`mn::Geom::Mesh`]'s `file` for
/// meshes that arrived inline (no backing file). Pair each schema entry
/// with [`UsdImport::inline_meshes`] to get the actual geometry.
pub const INLINE_MESH_PREFIX: &str = "__usd_inline__";

// ═══════════════════════════════ Import ════════════════════════════════

/// One inline mesh parsed from a USD `Mesh` prim.
#[derive(Debug, Clone)]
pub struct InlineMesh {
    /// Link the geom belongs to.
    pub link: String,
    pub slot: GeomSlot,
    pub mesh: MeshData,
}

/// Result of a successful USD import.
#[derive(Debug, Clone)]
pub struct UsdImport {
    pub file: MisaFile,
    /// Inline mesh payloads for every `Geom::Mesh` whose `file` starts
    /// with [`INLINE_MESH_PREFIX`].
    pub inline_meshes: Vec<InlineMesh>,
    pub warnings: Vec<String>,
}

/// Import a `.usda` file into a [`MisaFile`].
pub fn import(path: &std::path::Path) -> Result<UsdImport, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("Read USDA: {e}"))?;
    import_str(&text)
}

/// Import from USDA text.
pub fn import_str(text: &str) -> Result<UsdImport, String> {
    let top_prims = parse_usda(text);
    if top_prims.is_empty() {
        return Err("No prims found in USDA file".into());
    }

    // Find the World prim (or use the first top-level prim).
    let world = top_prims
        .iter()
        .find(|p| p.name == "World")
        .or_else(|| top_prims.first())
        .ok_or("No World prim found")?;

    // Robot root: first Xform child with ArticulationRoot / link children,
    // falling back to any Xform that isn't the PhysicsScene.
    let robot_prim = world
        .children
        .iter()
        .find(|p| {
            p.prim_type == "Xform"
                && (p.api_schemas.iter().any(|s| s.contains("ArticulationRoot"))
                    || p.children.iter().any(is_link_prim))
        })
        .or_else(|| {
            world
                .children
                .iter()
                .find(|p| p.prim_type == "Xform" && p.name != "PhysicsScene")
        })
        .ok_or("No robot prim found under World")?;

    let mut file = MisaFile::new(robot_prim.name.clone(), "");
    let mut inline_meshes: Vec<InlineMesh> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // ── Materials ───────────────────────────────────────────────────────
    let mut material_colors: HashMap<String, [f32; 4]> = HashMap::new();
    if let Some(mat_scope) = find_child(robot_prim, "Materials") {
        for mat_prim in &mat_scope.children {
            if mat_prim.prim_type == "Material" {
                material_colors.insert(mat_prim.name.clone(), parse_material_color(mat_prim));
            }
        }
    }

    // ── Links ───────────────────────────────────────────────────────────
    for child in &robot_prim.children {
        if !is_link_prim(child) {
            continue;
        }
        let link_name = child.name.clone();

        let mass = child
            .props
            .get("physics:mass")
            .map(|s| parse_f64(s))
            .unwrap_or(1.0);
        let diag = child
            .props
            .get("physics:diagonalInertia")
            .map(|s| parse_f3(s))
            .unwrap_or((0.001, 0.001, 0.001));
        let com = child
            .props
            .get("physics:centerOfMass")
            .map(|s| parse_f3(s))
            .unwrap_or((0.0, 0.0, 0.0));
        let inertial = mn::Inertial {
            mass,
            ixx: diag.0 as f64,
            iyy: diag.1 as f64,
            izz: diag.2 as f64,
            ixy: 0.0,
            ixz: 0.0,
            iyz: 0.0,
            origin: mn::Origin {
                xyz: [com.0 as f64, com.1 as f64, com.2 as f64],
                rpy: None,
                quat: None,
            },
        };

        let mut visual: Vec<mn::Visual> = Vec::new();
        if let Some(vis_scope) = find_child(child, "visuals") {
            for (gi, vis_prim) in vis_scope.children.iter().enumerate() {
                let (geom, origin) = parse_geom_prim(
                    vis_prim,
                    &link_name,
                    GeomSlot::Visual(gi),
                    &mut inline_meshes,
                );
                let color = vis_prim
                    .props
                    .get("material:binding")
                    .and_then(|path| extract_material_path(path))
                    .and_then(|full_path| {
                        let mat_name = full_path.rsplit('/').next()?;
                        material_colors.get(mat_name).copied()
                    })
                    .unwrap_or([0.7, 0.7, 0.7, 1.0]);
                visual.push(mn::Visual {
                    origin,
                    geom,
                    color: Some(mn::ColorSpec::Rgba(color)),
                    material: None,
                });
            }
        }

        let mut collision: Vec<mn::Collision> = Vec::new();
        if let Some(col_scope) = find_child(child, "collisions") {
            for (gi, col_prim) in col_scope.children.iter().enumerate() {
                let (geom, origin) = parse_geom_prim(
                    col_prim,
                    &link_name,
                    GeomSlot::Collision(gi),
                    &mut inline_meshes,
                );
                collision.push(mn::Collision {
                    origin,
                    geom,
                    physics: None,
                });
            }
        }

        // `physics:filteredPairs` on the link → disabled collision pairs.
        // Single-line rel values keep their surrounding brackets.
        if let Some(fp) = child.props.get("physics:filteredPairs") {
            for target in fp.trim().trim_matches(['[', ']']).split(',') {
                let partner = extract_rel_name(target);
                if !partner.is_empty() {
                    file.collision_pair.push(mn::CollisionPair {
                        link_a: link_name.clone(),
                        link_b: partner,
                        enabled: false,
                    });
                }
            }
        }

        file.link.push(mn::Link {
            name: link_name,
            description: String::new(),
            inertial,
            visual,
            collision,
            collision_enabled: true,
        });
    }

    if file.link.is_empty() {
        return Err("No links found in USDA file".into());
    }

    // ── Joints ──────────────────────────────────────────────────────────
    let mut child_links: std::collections::HashSet<String> = std::collections::HashSet::new();
    for child in &robot_prim.children {
        if !is_joint_prim(child) {
            continue;
        }
        let joint_name = child.name.clone();
        let parent = child
            .props
            .get("physics:body0")
            .map(|s| extract_rel_name(s))
            .unwrap_or_default();
        let child_link = child
            .props
            .get("physics:body1")
            .map(|s| extract_rel_name(s))
            .unwrap_or_default();
        if parent.is_empty() || child_link.is_empty() {
            warnings.push(format!(
                "joint '{joint_name}': missing body0/body1 — skipped"
            ));
            continue;
        }

        let lower_raw = child
            .props
            .get("physics:lowerLimit")
            .map(|s| parse_f64(s))
            .unwrap_or(0.0);
        let upper_raw = child
            .props
            .get("physics:upperLimit")
            .map(|s| parse_f64(s))
            .unwrap_or(0.0);

        let (kind, lower, upper) = if child.prim_type.contains("Revolute") {
            // Distinguish revolute from continuous by the ±360° export
            // convention for unlimited joints.
            if lower_raw <= -360.0 && upper_raw >= 360.0 {
                (
                    mn::JointKind::Continuous,
                    -std::f64::consts::TAU,
                    std::f64::consts::TAU,
                )
            } else {
                // Revolute limits are stored in degrees.
                (
                    mn::JointKind::Revolute,
                    lower_raw.to_radians(),
                    upper_raw.to_radians(),
                )
            }
        } else if child.prim_type.contains("Prismatic") {
            (mn::JointKind::Prismatic, lower_raw, upper_raw)
        } else {
            (mn::JointKind::Fixed, 0.0, 0.0)
        };

        // Reconstruct the joint origin from the USD local transforms.
        // On export: localRot0 = origin.rotation · extra_rot and
        // localRot1 = extra_rot, so origin.rotation = rot0 · rot1⁻¹.
        let local_pos0 = child
            .props
            .get("physics:localPos0")
            .map(|s| parse_f3(s))
            .unwrap_or((0.0, 0.0, 0.0));
        let local_rot0 = child
            .props
            .get("physics:localRot0")
            .map(|s| parse_quat(s))
            .unwrap_or_else(na::UnitQuaternion::identity);
        let local_rot1 = child
            .props
            .get("physics:localRot1")
            .map(|s| parse_quat(s))
            .unwrap_or_else(na::UnitQuaternion::identity);
        let origin_rot = local_rot0 * local_rot1.inverse();
        let q = origin_rot.quaternion();
        let origin = mn::Origin {
            xyz: [
                local_pos0.0 as f64,
                local_pos0.1 as f64,
                local_pos0.2 as f64,
            ],
            rpy: None,
            quat: if (q.w - 1.0).abs() < 1e-12 && q.i == 0.0 && q.j == 0.0 && q.k == 0.0 {
                None
            } else {
                Some([q.i, q.j, q.k, q.w])
            },
        };

        // Axis: extra_rot (= localRot1) maps the source axis onto the USD
        // principal axis, so source_axis = rot1⁻¹ · usd_axis.
        let usd_principal = match child
            .props
            .get("physics:axis")
            .map(|s| s.trim().trim_matches('"').to_string())
            .unwrap_or_else(|| "Z".to_string())
            .as_str()
        {
            "X" => na::Vector3::x(),
            "Y" => na::Vector3::y(),
            _ => na::Vector3::z(),
        };
        let axis = if kind == mn::JointKind::Fixed {
            [0.0, 0.0, 1.0]
        } else {
            let a = local_rot1.inverse() * usd_principal;
            [a.x, a.y, a.z]
        };

        child_links.insert(child_link.clone());
        file.joint.push(mn::Joint {
            name: joint_name,
            kind,
            parent,
            child: child_link,
            axis,
            origin,
            limit: mn::JointLimit {
                lower,
                upper,
                effort: 0.0,
                velocity: 0.0,
            },
            dynamics: mn::JointDynamics::default(),
        });
    }

    file.robot.root = file
        .link
        .iter()
        .find(|l| !child_links.contains(&l.name))
        .map(|l| l.name.clone())
        .unwrap_or_else(|| file.link[0].name.clone());

    Ok(UsdImport {
        file,
        inline_meshes,
        warnings,
    })
}

// ─── Prim tree parser ───────────────────────────────────────────────────

/// A parsed USD prim with its type, name, properties, and children.
#[derive(Debug, Clone)]
struct UsdPrim {
    prim_type: String,
    name: String,
    api_schemas: Vec<String>,
    props: HashMap<String, String>,
    children: Vec<UsdPrim>,
}

/// Parse a USDA text into a tree of `UsdPrim`s.
fn parse_usda(text: &str) -> Vec<UsdPrim> {
    let lines: Vec<&str> = text.lines().collect();
    let mut pos = 0;
    // Skip header
    while pos < lines.len() {
        if lines[pos].trim().starts_with("def ") {
            break;
        }
        pos += 1;
    }
    let mut prims = Vec::new();
    while pos < lines.len() {
        if let Some((prim, next)) = parse_prim(&lines, pos) {
            prims.push(prim);
            pos = next;
        } else {
            pos += 1;
        }
    }
    prims
}

/// Parse a single prim starting at `start`. Returns the parsed prim and
/// the line index after the closing brace.
fn parse_prim(lines: &[&str], start: usize) -> Option<(UsdPrim, usize)> {
    let trimmed = lines[start].trim();
    if !trimmed.starts_with("def ") {
        return None;
    }
    let rest = &trimmed[4..];
    let (prim_type, rest) = split_first_word(rest);
    let name = extract_quoted(rest).unwrap_or_default();

    let mut api_schemas = Vec::new();
    let mut pos = start + 1;

    // Metadata block ( ... ) before {.
    let has_metadata = rest.contains('(') && !rest.contains('{');
    if has_metadata {
        while pos < lines.len() {
            let t = lines[pos].trim();
            if t.contains("apiSchemas") {
                let combined = collect_bracket_content(lines, &mut pos, '[', ']');
                for part in combined.split(',') {
                    let s = part.trim().trim_matches('"').trim().to_string();
                    if !s.is_empty() {
                        api_schemas.push(s);
                    }
                }
                continue;
            }
            if t.starts_with(')') || t.ends_with(')') {
                pos += 1;
                break;
            }
            pos += 1;
        }
    }

    // Find opening brace.
    while pos < lines.len() {
        let t = lines[pos].trim();
        if t.starts_with('{') || t == "{" {
            pos += 1;
            break;
        }
        if lines.get(start).is_some_and(|l| l.contains('{')) && pos == start + 1 {
            break;
        }
        pos += 1;
    }

    let mut props = HashMap::new();
    let mut children = Vec::new();

    // Parse contents until the matching }.
    let mut depth = 1u32;
    while pos < lines.len() && depth > 0 {
        let t = lines[pos].trim();

        if t == "}" || t == "}," {
            depth -= 1;
            if depth == 0 {
                pos += 1;
                break;
            }
            pos += 1;
            continue;
        }

        if t.starts_with("def ") {
            if let Some((child, next)) = parse_prim(lines, pos) {
                children.push(child);
                pos = next;
                continue;
            }
        }

        if t == "{" {
            depth += 1;
            pos += 1;
            continue;
        }

        if let Some((key, value)) = parse_property_line(lines, &mut pos) {
            props.insert(key, value);
        } else {
            pos += 1;
        }
    }

    Some((
        UsdPrim {
            prim_type,
            name,
            api_schemas,
            props,
            children,
        },
        pos,
    ))
}

fn split_first_word(s: &str) -> (String, &str) {
    let s = s.trim();
    if let Some(idx) = s.find(|c: char| c.is_whitespace()) {
        (s[..idx].to_string(), &s[idx..])
    } else {
        (s.to_string(), "")
    }
}

fn extract_quoted(s: &str) -> Option<String> {
    let start = s.find('"')? + 1;
    let end = s[start..].find('"')? + start;
    Some(s[start..end].to_string())
}

/// Parse a property line `[qualifiers] type key = value`, reading ahead
/// for multi-line `[ ... ]` arrays.
fn parse_property_line(lines: &[&str], pos: &mut usize) -> Option<(String, String)> {
    let line = lines[*pos].trim();
    if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
        return None;
    }
    if !line.contains('=') {
        return None;
    }

    let eq_idx = line.find('=')?;
    let lhs = line[..eq_idx].trim();
    let rhs_raw = line[eq_idx + 1..].trim();

    let key = extract_prop_key(lhs);

    if rhs_raw.starts_with('[') && !rhs_raw.contains(']') {
        let value = collect_bracket_content(lines, pos, '[', ']');
        return Some((key, value));
    }

    *pos += 1;
    Some((key, rhs_raw.to_string()))
}

/// `"float physics:mass"` → `"physics:mass"`.
fn extract_prop_key(lhs: &str) -> String {
    lhs.split_whitespace().last().unwrap_or(lhs).to_string()
}

/// Collect content within matched brackets, possibly across lines.
fn collect_bracket_content(lines: &[&str], pos: &mut usize, open: char, close: char) -> String {
    let mut result = String::new();
    let mut depth = 0i32;
    while *pos < lines.len() {
        let line = lines[*pos].trim();
        for c in line.chars() {
            if c == open {
                depth += 1;
            } else if c == close {
                depth -= 1;
            }
        }
        result.push_str(line);
        result.push(' ');
        *pos += 1;
        if depth <= 0 {
            break;
        }
    }
    let trimmed = result.trim();
    if let (Some(s), Some(e)) = (trimmed.find(open), trimmed.rfind(close)) {
        trimmed[s + 1..e].to_string()
    } else {
        trimmed.to_string()
    }
}

// ─── Value parsers ──────────────────────────────────────────────────────

fn parse_float(s: &str) -> f32 {
    s.trim().parse::<f32>().unwrap_or(0.0)
}

fn parse_f64(s: &str) -> f64 {
    s.trim().parse::<f64>().unwrap_or(0.0)
}

/// Parse `(x, y, z)` into 3 f32 values.
fn parse_f3(s: &str) -> (f32, f32, f32) {
    let s = s.trim().trim_start_matches('(').trim_end_matches(')');
    let parts: Vec<f32> = s
        .split(',')
        .map(|p| p.trim().parse().unwrap_or(0.0))
        .collect();
    (
        parts.first().copied().unwrap_or(0.0),
        parts.get(1).copied().unwrap_or(0.0),
        parts.get(2).copied().unwrap_or(0.0),
    )
}

/// Parse a quaternion `(w, x, y, z)`.
fn parse_quat(s: &str) -> na::UnitQuaternion<f64> {
    let s = s.trim().trim_start_matches('(').trim_end_matches(')');
    let parts: Vec<f64> = s
        .split(',')
        .map(|p| p.trim().parse().unwrap_or(0.0))
        .collect();
    let w = parts.first().copied().unwrap_or(1.0);
    let x = parts.get(1).copied().unwrap_or(0.0);
    let y = parts.get(2).copied().unwrap_or(0.0);
    let z = parts.get(3).copied().unwrap_or(0.0);
    na::UnitQuaternion::from_quaternion(na::Quaternion::new(w, x, y, z))
}

/// Parse a `point3f[]` / `normal3f[]` array into tuples.
fn parse_point_array(s: &str) -> Vec<(f32, f32, f32)> {
    let mut result = Vec::new();
    let mut i = 0;
    let bytes = s.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'(' {
            if let Some(end) = s[i..].find(')') {
                result.push(parse_f3(&s[i..i + end + 1]));
                i += end + 1;
            } else {
                break;
            }
        } else {
            i += 1;
        }
    }
    result
}

/// `</World/robot/link_name>` → `link_name`.
fn extract_rel_name(s: &str) -> String {
    let s = s.trim().trim_start_matches('<').trim_end_matches('>');
    s.rsplit('/').next().unwrap_or(s).to_string()
}

fn extract_material_path(s: &str) -> Option<String> {
    let s = s.trim().trim_start_matches('<').trim_end_matches('>');
    if s.is_empty() { None } else { Some(s.to_string()) }
}

// ─── Prim → schema conversion ───────────────────────────────────────────

fn parse_material_color(prim: &UsdPrim) -> [f32; 4] {
    let mut color = [0.7f32, 0.7, 0.7, 1.0];
    for child in &prim.children {
        if let Some(diff) = child.props.get("inputs:diffuseColor") {
            let (r, g, b) = parse_f3(diff);
            color[0] = r;
            color[1] = g;
            color[2] = b;
        }
        if let Some(opacity) = child.props.get("inputs:opacity") {
            color[3] = parse_float(opacity);
        }
    }
    // Flat layout: properties directly on the Material prim.
    if let Some(diff) = prim.props.get("inputs:diffuseColor") {
        let (r, g, b) = parse_f3(diff);
        color[0] = r;
        color[1] = g;
        color[2] = b;
    }
    if let Some(opacity) = prim.props.get("inputs:opacity") {
        color[3] = parse_float(opacity);
    }
    color
}

/// Convert a geometry prim (Cube/Cylinder/Sphere/Capsule/Mesh) to a
/// schema geom + origin. Inline mesh payloads go to `inline_meshes`.
fn parse_geom_prim(
    prim: &UsdPrim,
    link: &str,
    slot: GeomSlot,
    inline_meshes: &mut Vec<InlineMesh>,
) -> (mn::Geom, mn::Origin) {
    let origin = parse_xform_origin(prim);

    let geom = match prim.prim_type.as_str() {
        "Cube" => {
            // The scale xformOp encodes half-extents; `size` is always 2.
            let (sx, sy, sz) = prim
                .props
                .get("xformOp:scale")
                .map(|s| parse_f3(s))
                .unwrap_or((0.05, 0.05, 0.05));
            mn::Geom::Box {
                size: [
                    sx.abs() as f64 * 2.0,
                    sy.abs() as f64 * 2.0,
                    sz.abs() as f64 * 2.0,
                ],
            }
        }
        "Cylinder" => {
            let radius = prim
                .props
                .get("radius")
                .map(|s| parse_f64(s))
                .unwrap_or(0.02);
            let height = prim
                .props
                .get("height")
                .map(|s| parse_f64(s))
                .unwrap_or(0.2);
            mn::Geom::Cylinder {
                radius,
                length: height,
            }
        }
        "Sphere" => mn::Geom::Sphere {
            radius: prim
                .props
                .get("radius")
                .map(|s| parse_f64(s))
                .unwrap_or(0.05),
        },
        "Capsule" => {
            // Export writes total height = length + 2·radius.
            let radius = prim
                .props
                .get("radius")
                .map(|s| parse_f64(s))
                .unwrap_or(0.02);
            let height = prim
                .props
                .get("height")
                .map(|s| parse_f64(s))
                .unwrap_or(0.2);
            mn::Geom::Capsule {
                radius,
                length: (height - 2.0 * radius).max(0.0),
            }
        }
        "Mesh" => {
            let points = prim
                .props
                .get("points")
                .map(|s| parse_point_array(s))
                .unwrap_or_default();
            let normals = prim
                .props
                .get("normals")
                .map(|s| parse_point_array(s))
                .unwrap_or_default();
            let mut vertices = Vec::with_capacity(points.len() * 6);
            for (i, (px, py, pz)) in points.iter().enumerate() {
                let (nx, ny, nz) = normals.get(i).copied().unwrap_or((0.0, 0.0, 1.0));
                vertices.extend_from_slice(&[*px, *py, *pz, nx, ny, nz]);
            }
            let sentinel = format!("{INLINE_MESH_PREFIX}{}", inline_meshes.len());
            inline_meshes.push(InlineMesh {
                link: link.to_string(),
                slot,
                mesh: MeshData::from_flat_vertices_f32(&vertices),
            });
            mn::Geom::Mesh {
                file: sentinel,
                scale: [1.0, 1.0, 1.0],
            }
        }
        _ => mn::Geom::Box {
            size: [0.02, 0.02, 0.02],
        },
    };

    (geom, origin)
}

/// Parse `xformOp:translate` + `xformOp:orient` into an [`mn::Origin`].
fn parse_xform_origin(prim: &UsdPrim) -> mn::Origin {
    let xyz = prim
        .props
        .get("xformOp:translate")
        .map(|s| {
            let (x, y, z) = parse_f3(s);
            [x as f64, y as f64, z as f64]
        })
        .unwrap_or([0.0; 3]);
    let quat = prim.props.get("xformOp:orient").map(|s| {
        let q = parse_quat(s);
        let q = q.quaternion();
        [q.i, q.j, q.k, q.w]
    });
    mn::Origin {
        xyz,
        rpy: None,
        quat,
    }
}

fn find_child<'a>(prim: &'a UsdPrim, name: &str) -> Option<&'a UsdPrim> {
    prim.children.iter().find(|c| c.name == name)
}

/// A link prim has PhysicsRigidBodyAPI or visual/collision scopes.
fn is_link_prim(prim: &UsdPrim) -> bool {
    if prim.prim_type != "Xform" {
        return false;
    }
    prim.api_schemas
        .iter()
        .any(|s| s.contains("PhysicsRigidBodyAPI"))
        || prim
            .children
            .iter()
            .any(|c| c.name == "visuals" || c.name == "collisions")
}

fn is_joint_prim(prim: &UsdPrim) -> bool {
    prim.prim_type.contains("PhysicsRevoluteJoint")
        || prim.prim_type.contains("PhysicsPrismaticJoint")
        || prim.prim_type.contains("PhysicsFixedJoint")
        || prim.prim_type.contains("Joint")
}

// ═══════════════════════════════ Export ════════════════════════════════

/// Host-supplied data the emitter needs but the schema doesn't carry.
/// Both members are optional; the defaults are the q = 0 rest pose and
/// empty mesh prims.
#[derive(Default)]
pub struct UsdExportRefs<'a> {
    /// Posed world transform per link (e.g. an editor's current FK).
    /// Links it returns `None` for fall back to the q = 0 rest chain.
    pub link_world_tf: Option<&'a dyn Fn(&str) -> Option<na::Isometry3<f64>>>,
    /// Flat `[x, y, z, nx, ny, nz]` triangle soup for a mesh geom
    /// occurrence. `None` emits the Mesh prim without geometry data.
    pub mesh_vertices: Option<&'a dyn Fn(&str, GeomSlot) -> Option<Vec<f32>>>,
}

/// Export a [`MisaFile`] as USD ASCII (.usda) text.
///
/// Mimics, sensors and actuators have no plain-UsdPhysics equivalent and
/// are not emitted — hosts should warn the user when dropping them.
pub fn export(file: &MisaFile, refs: &UsdExportRefs) -> String {
    let mut s = String::with_capacity(16 * 1024);
    let robot_name = sanitize_name(&file.robot.name);
    let robot_path = format!("/World/{robot_name}");

    // ---- Header ----
    s.push_str("#usda 1.0\n");
    s.push_str("(\n");
    s.push_str("    defaultPrim = \"World\"\n");
    s.push_str(&format!(
        "    doc = \"Generated by misarta-formats — {}\"\n",
        file.robot.name
    ));
    s.push_str("    metersPerUnit = 1.0\n");
    s.push_str("    upAxis = \"Z\"\n");
    s.push_str(")\n\n");

    // ---- Rest-pose world transforms (q = 0 chain, host override) ----
    let rest = rest_transforms(file);
    let world_tf = |link: &str| -> na::Isometry3<f64> {
        if let Some(f) = refs.link_world_tf {
            if let Some(tf) = f(link) {
                return tf;
            }
        }
        rest.get(link)
            .copied()
            .unwrap_or_else(na::Isometry3::identity)
    };

    // ---- Unique materials over resolved visual colours ----
    let named_materials: HashMap<&str, [f32; 4]> = file
        .material
        .iter()
        .map(|m| (m.name.as_str(), crate::util::color_spec_to_rgba(&m.color)))
        .collect();
    let mut materials: Vec<[f32; 4]> = Vec::new();
    let mut material_map: HashMap<u64, usize> = HashMap::new();
    for link in &file.link {
        for vis in &link.visual {
            let color = resolve_visual_rgba(vis, &named_materials);
            let key = color_key(&color);
            if !material_map.contains_key(&key) {
                material_map.insert(key, materials.len());
                materials.push(color);
            }
        }
    }

    // ---- World scope ----
    s.push_str("def Xform \"World\"\n{\n");
    s.push_str("    def PhysicsScene \"PhysicsScene\"\n    {\n");
    s.push_str("        vector3f physics:gravityDirection = (0, 0, -1)\n");
    s.push_str("        float physics:gravityMagnitude = 9.81\n");
    s.push_str("    }\n\n");

    // ---- Robot root prim ----
    s.push_str(&format!("    def Xform \"{robot_name}\" (\n"));
    s.push_str("        prepend apiSchemas = [\"PhysicsArticulationRootAPI\"]\n");
    s.push_str("    )\n    {\n");

    // Disabled collision pairs → physics:filteredPairs, listed on the
    // pair's first link (the rel is symmetric in USD physics semantics).
    let link_names: std::collections::HashSet<&str> =
        file.link.iter().map(|l| l.name.as_str()).collect();
    let mut filter_map: HashMap<&str, Vec<&str>> = HashMap::new();
    for cp in &file.collision_pair {
        if cp.enabled
            || !link_names.contains(cp.link_a.as_str())
            || !link_names.contains(cp.link_b.as_str())
        {
            continue;
        }
        filter_map
            .entry(cp.link_a.as_str())
            .or_default()
            .push(cp.link_b.as_str());
    }

    for (li, link) in file.link.iter().enumerate() {
        let tf = world_tf(&link.name);
        let partners = filter_map
            .get(link.name.as_str())
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        write_link(
            &mut s,
            file,
            li,
            &tf,
            &robot_path,
            &named_materials,
            &material_map,
            "        ",
            partners,
            refs,
        );
    }

    for joint in &file.joint {
        write_joint(&mut s, joint, &robot_path, "        ");
    }

    if !materials.is_empty() {
        s.push_str("        def Scope \"Materials\"\n        {\n");
        for (i, color) in materials.iter().enumerate() {
            write_material(&mut s, i, color, &robot_path, "            ");
        }
        s.push_str("        }\n");
    }

    s.push_str("    }\n"); // close robot
    s.push_str("}\n"); // close World

    s
}

/// World transforms of every link at q = 0: chained joint origins from
/// the root.
fn rest_transforms(file: &MisaFile) -> HashMap<String, na::Isometry3<f64>> {
    let mut children: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, j) in file.joint.iter().enumerate() {
        children.entry(j.parent.as_str()).or_default().push(i);
    }
    let mut out: HashMap<String, na::Isometry3<f64>> = HashMap::new();
    let mut stack: Vec<(String, na::Isometry3<f64>)> =
        vec![(file.robot.root.clone(), na::Isometry3::identity())];
    while let Some((link, tf)) = stack.pop() {
        if out.contains_key(&link) {
            continue; // guard against cycles
        }
        out.insert(link.clone(), tf);
        if let Some(js) = children.get(link.as_str()) {
            for &ji in js {
                let j = &file.joint[ji];
                stack.push((j.child.clone(), tf * origin_iso(&j.origin)));
            }
        }
    }
    out
}

// ─── Emit helpers ───────────────────────────────────────────────────────

/// Sanitise a name for use as a USD prim-path component
/// (`[a-zA-Z_][a-zA-Z0-9_]*`).
fn sanitize_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for (i, c) in name.chars().enumerate() {
        if c.is_ascii_alphanumeric() || c == '_' {
            if i == 0 && c.is_ascii_digit() {
                out.push('_');
            }
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push_str("prim");
    }
    out
}

/// Hash a colour for material de-duplication.
fn color_key(color: &[f32; 4]) -> u64 {
    let r = (color[0] * 10000.0) as u64;
    let g = (color[1] * 10000.0) as u64;
    let b = (color[2] * 10000.0) as u64;
    let a = (color[3] * 10000.0) as u64;
    r | (g << 16) | (b << 32) | (a << 48)
}

fn fmt_3(x: f64, y: f64, z: f64) -> String {
    format!("({}, {}, {})", fmt(x), fmt(y), fmt(z))
}

fn fmt_quat(q: &na::UnitQuaternion<f64>) -> String {
    let q = q.quaternion();
    format!("({}, {}, {}, {})", fmt(q.w), fmt(q.i), fmt(q.j), fmt(q.k))
}

/// Pick the USD principal axis for a joint axis; the returned extra
/// rotation aligns that principal axis with the source direction.
fn determine_usd_axis(axis: &[f64; 3]) -> (&'static str, na::UnitQuaternion<f64>) {
    let v = na::Vector3::new(axis[0], axis[1], axis[2]);
    let a = if v.norm() > 1e-6 {
        v.normalize()
    } else {
        na::Vector3::z()
    };
    let eps = 1e-3;

    if (a.x.abs() - 1.0).abs() < eps && a.y.abs() < eps && a.z.abs() < eps {
        if a.x > 0.0 {
            ("X", na::UnitQuaternion::identity())
        } else {
            (
                "X",
                na::UnitQuaternion::from_axis_angle(&na::Vector3::z_axis(), std::f64::consts::PI),
            )
        }
    } else if a.x.abs() < eps && (a.y.abs() - 1.0).abs() < eps && a.z.abs() < eps {
        if a.y > 0.0 {
            ("Y", na::UnitQuaternion::identity())
        } else {
            (
                "Y",
                na::UnitQuaternion::from_axis_angle(&na::Vector3::z_axis(), std::f64::consts::PI),
            )
        }
    } else if a.x.abs() < eps && a.y.abs() < eps && (a.z.abs() - 1.0).abs() < eps {
        if a.z > 0.0 {
            ("Z", na::UnitQuaternion::identity())
        } else {
            (
                "Z",
                na::UnitQuaternion::from_axis_angle(&na::Vector3::x_axis(), std::f64::consts::PI),
            )
        }
    } else {
        // Arbitrary axis — align Z with it.
        let rot = na::UnitQuaternion::rotation_between(&na::Vector3::z(), &a)
            .unwrap_or_else(na::UnitQuaternion::identity);
        ("Z", rot)
    }
}

/// `xformOp:translate` + `xformOp:orient` + order; identity parts omitted.
fn write_xform_ops(s: &mut String, iso: &na::Isometry3<f64>, indent: &str) {
    write_xform_ops_scaled(s, iso, None, indent);
}

fn write_xform_ops_scaled(
    s: &mut String,
    iso: &na::Isometry3<f64>,
    scale: Option<(f64, f64, f64)>,
    indent: &str,
) {
    let t = iso.translation;
    let q = iso.rotation;
    let has_t = t.x.abs() > 1e-7 || t.y.abs() > 1e-7 || t.z.abs() > 1e-7;
    let has_r =
        (q.w - 1.0).abs() > 1e-7 || q.i.abs() > 1e-7 || q.j.abs() > 1e-7 || q.k.abs() > 1e-7;
    let has_s = scale.is_some();

    if has_t {
        s.push_str(&format!(
            "{indent}double3 xformOp:translate = {}\n",
            fmt_3(t.x, t.y, t.z)
        ));
    }
    if has_r {
        s.push_str(&format!(
            "{indent}quatd xformOp:orient = {}\n",
            fmt_quat(&q)
        ));
    }
    if let Some((sx, sy, sz)) = scale {
        s.push_str(&format!(
            "{indent}double3 xformOp:scale = {}\n",
            fmt_3(sx, sy, sz)
        ));
    }
    if has_t || has_r || has_s {
        let mut ops: Vec<&str> = Vec::new();
        if has_t {
            ops.push("\"xformOp:translate\"");
        }
        if has_r {
            ops.push("\"xformOp:orient\"");
        }
        if has_s {
            ops.push("\"xformOp:scale\"");
        }
        s.push_str(&format!(
            "{indent}uniform token[] xformOpOrder = [{}]\n",
            ops.join(", ")
        ));
    }
}

/// Emit one geometry prim; meshes without host-supplied vertices are
/// emitted without data.
#[allow(clippy::too_many_arguments)]
fn write_geom_prim(
    s: &mut String,
    geom: &mn::Geom,
    origin: &mn::Origin,
    name: &str,
    indent: &str,
    api_schemas: &str,
    material_path: Option<&str>,
    mesh_vertices: Option<Vec<f32>>,
) {
    let iso = origin_iso(origin);
    let inner = format!("{indent}    ");
    let (prim_type, body) = match geom {
        mn::Geom::Box { size } => {
            let mut b = String::new();
            write_xform_ops_scaled(
                &mut b,
                &iso,
                Some((size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)),
                &inner,
            );
            b.push_str(&format!("{inner}double size = 2.0\n"));
            ("Cube", b)
        }
        mn::Geom::Cylinder { radius, length } => {
            let mut b = String::new();
            write_xform_ops(&mut b, &iso, &inner);
            b.push_str(&format!("{inner}double radius = {}\n", fmt(*radius)));
            b.push_str(&format!("{inner}double height = {}\n", fmt(*length)));
            b.push_str(&format!("{inner}token axis = \"Z\"\n"));
            ("Cylinder", b)
        }
        mn::Geom::Sphere { radius } => {
            let mut b = String::new();
            write_xform_ops(&mut b, &iso, &inner);
            b.push_str(&format!("{inner}double radius = {}\n", fmt(*radius)));
            ("Sphere", b)
        }
        mn::Geom::Capsule { radius, length } => {
            let mut b = String::new();
            write_xform_ops(&mut b, &iso, &inner);
            b.push_str(&format!("{inner}double radius = {}\n", fmt(*radius)));
            b.push_str(&format!(
                "{inner}double height = {}\n",
                fmt(length + 2.0 * radius)
            ));
            b.push_str(&format!("{inner}token axis = \"Z\"\n"));
            ("Capsule", b)
        }
        mn::Geom::Mesh { .. } => {
            let mut b = String::new();
            write_xform_ops(&mut b, &iso, &inner);
            if let Some(vertices) = &mesh_vertices {
                write_mesh_data(&mut b, vertices, &inner);
            }
            ("Mesh", b)
        }
    };

    if api_schemas.is_empty() {
        s.push_str(&format!("{indent}def {prim_type} \"{name}\"\n{indent}{{\n"));
    } else {
        s.push_str(&format!("{indent}def {prim_type} \"{name}\" (\n"));
        s.push_str(&format!(
            "{indent}    prepend apiSchemas = [{api_schemas}]\n"
        ));
        s.push_str(&format!("{indent})\n{indent}{{\n"));
    }

    s.push_str(&body);

    if let Some(mat_path) = material_path {
        s.push_str(&format!(
            "{indent}    rel material:binding = <{mat_path}>\n"
        ));
    }

    s.push_str(&format!("{indent}}}\n\n"));
}

/// Write inline mesh data (points, normals, face indices).
fn write_mesh_data(s: &mut String, vertices: &[f32], indent: &str) {
    let num_verts = vertices.len() / 6;
    if num_verts == 0 {
        return;
    }
    let num_faces = num_verts / 3;

    s.push_str(&format!("{indent}point3f[] points = [\n"));
    for i in 0..num_verts {
        let comma = if i + 1 < num_verts { "," } else { "" };
        s.push_str(&format!(
            "{indent}    ({}, {}, {}){comma}\n",
            vertices[i * 6],
            vertices[i * 6 + 1],
            vertices[i * 6 + 2]
        ));
    }
    s.push_str(&format!("{indent}]\n"));

    s.push_str(&format!("{indent}normal3f[] normals = [\n"));
    for i in 0..num_verts {
        let comma = if i + 1 < num_verts { "," } else { "" };
        s.push_str(&format!(
            "{indent}    ({}, {}, {}){comma}\n",
            vertices[i * 6 + 3],
            vertices[i * 6 + 4],
            vertices[i * 6 + 5]
        ));
    }
    s.push_str(&format!("{indent}]\n"));
    s.push_str(&format!(
        "{indent}uniform token normals:interpolation = \"vertex\"\n"
    ));

    s.push_str(&format!("{indent}int[] faceVertexCounts = ["));
    for i in 0..num_faces {
        if i > 0 {
            s.push_str(", ");
        }
        s.push('3');
    }
    s.push_str("]\n");

    s.push_str(&format!("{indent}int[] faceVertexIndices = ["));
    for i in 0..(num_verts as u32) {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&i.to_string());
    }
    s.push_str("]\n");

    s.push_str(&format!(
        "{indent}uniform token subdivisionScheme = \"none\"\n"
    ));
}

/// Write a link prim with physics APIs, visuals, and collisions.
#[allow(clippy::too_many_arguments)]
fn write_link(
    s: &mut String,
    file: &MisaFile,
    link_i: usize,
    rest_tf: &na::Isometry3<f64>,
    robot_path: &str,
    named_materials: &HashMap<&str, [f32; 4]>,
    material_map: &HashMap<u64, usize>,
    indent: &str,
    filter_partners: &[&str],
    refs: &UsdExportRefs,
) {
    let link = &file.link[link_i];
    let link_name = sanitize_name(&link.name);

    s.push_str(&format!("{indent}def Xform \"{link_name}\" (\n"));
    let api_schemas = if filter_partners.is_empty() {
        "[\"PhysicsRigidBodyAPI\", \"PhysicsMassAPI\"]"
    } else {
        "[\"PhysicsRigidBodyAPI\", \"PhysicsMassAPI\", \"PhysicsFilteredPairsAPI\"]"
    };
    s.push_str(&format!("{indent}    prepend apiSchemas = {api_schemas}\n"));
    s.push_str(&format!("{indent})\n{indent}{{\n"));

    let inner = format!("{indent}    ");

    write_xform_ops(s, rest_tf, &inner);

    if !filter_partners.is_empty() {
        let paths: Vec<String> = filter_partners
            .iter()
            .map(|n| format!("<{robot_path}/{}>", sanitize_name(n)))
            .collect();
        s.push_str(&format!(
            "{inner}rel physics:filteredPairs = [{}]\n",
            paths.join(", "),
        ));
    }

    s.push_str(&format!(
        "{inner}float physics:mass = {}\n",
        fmt(link.inertial.mass)
    ));
    s.push_str(&format!(
        "{inner}float3 physics:diagonalInertia = {}\n",
        fmt_3(link.inertial.ixx, link.inertial.iyy, link.inertial.izz)
    ));
    let com = link.inertial.origin.xyz;
    if com[0].abs() > 1e-7 || com[1].abs() > 1e-7 || com[2].abs() > 1e-7 {
        s.push_str(&format!(
            "{inner}point3f physics:centerOfMass = {}\n",
            fmt_3(com[0], com[1], com[2])
        ));
    }

    s.push('\n');

    if !link.visual.is_empty() {
        s.push_str(&format!("{inner}def Scope \"visuals\"\n{inner}{{\n"));
        let vis_indent = format!("{inner}    ");
        for (i, vis) in link.visual.iter().enumerate() {
            let color = resolve_visual_rgba(vis, named_materials);
            let mat_idx = material_map.get(&color_key(&color)).copied().unwrap_or(0);
            let mat_path = format!("{robot_path}/Materials/material_{mat_idx}");
            let vertices = mesh_vertices_for(refs, &link.name, GeomSlot::Visual(i), &vis.geom);
            write_geom_prim(
                s,
                &vis.geom,
                &vis.origin,
                &format!("visual_{i}"),
                &vis_indent,
                "",
                Some(&mat_path),
                vertices,
            );
        }
        s.push_str(&format!("{inner}}}\n\n"));
    }

    if !link.collision.is_empty() {
        s.push_str(&format!("{inner}def Scope \"collisions\"\n{inner}{{\n"));
        let col_indent = format!("{inner}    ");
        for (i, col) in link.collision.iter().enumerate() {
            let vertices = mesh_vertices_for(refs, &link.name, GeomSlot::Collision(i), &col.geom);
            write_geom_prim(
                s,
                &col.geom,
                &col.origin,
                &format!("collision_{i}"),
                &col_indent,
                "\"PhysicsCollisionAPI\"",
                None,
                vertices,
            );
        }
        s.push_str(&format!("{inner}}}\n\n"));
    }

    s.push_str(&format!("{indent}}}\n\n"));
}

fn mesh_vertices_for(
    refs: &UsdExportRefs,
    link: &str,
    slot: GeomSlot,
    geom: &mn::Geom,
) -> Option<Vec<f32>> {
    if !matches!(geom, mn::Geom::Mesh { .. }) {
        return None;
    }
    refs.mesh_vertices.and_then(|f| f(link, slot))
}

/// Write a physics joint prim.
fn write_joint(s: &mut String, joint: &mn::Joint, robot_path: &str, indent: &str) {
    let joint_name = sanitize_name(&joint.name);
    let parent_name = sanitize_name(&joint.parent);
    let child_name = sanitize_name(&joint.child);

    let (usd_prim_type, drive_kind) = match joint.kind {
        mn::JointKind::Revolute | mn::JointKind::Continuous => {
            ("PhysicsRevoluteJoint", Some("angular"))
        }
        mn::JointKind::Prismatic => ("PhysicsPrismaticJoint", Some("linear")),
        // Fixed, and kinds UsdPhysics has no counterpart for (floating /
        // planar), weld the bodies — same as the original exporter.
        _ => ("PhysicsFixedJoint", None),
    };

    if let Some(dk) = drive_kind {
        s.push_str(&format!("{indent}def {usd_prim_type} \"{joint_name}\" (\n"));
        s.push_str(&format!(
            "{indent}    prepend apiSchemas = [\"PhysicsDriveAPI:{dk}\"]\n"
        ));
        s.push_str(&format!("{indent})\n{indent}{{\n"));
    } else {
        s.push_str(&format!(
            "{indent}def {usd_prim_type} \"{joint_name}\"\n{indent}{{\n"
        ));
    }

    let inner = format!("{indent}    ");

    s.push_str(&format!(
        "{inner}rel physics:body0 = <{robot_path}/{parent_name}>\n"
    ));
    s.push_str(&format!(
        "{inner}rel physics:body1 = <{robot_path}/{child_name}>\n"
    ));

    let iso = origin_iso(&joint.origin);
    if drive_kind.is_some() {
        let (usd_axis, extra_rot) = determine_usd_axis(&joint.axis);
        s.push_str(&format!(
            "{inner}uniform token physics:axis = \"{usd_axis}\"\n"
        ));

        let t = iso.translation;
        let local_rot0 = iso.rotation * extra_rot;
        s.push_str(&format!(
            "{inner}point3f physics:localPos0 = {}\n",
            fmt_3(t.x, t.y, t.z)
        ));
        s.push_str(&format!(
            "{inner}quatf physics:localRot0 = {}\n",
            fmt_quat(&local_rot0)
        ));
        s.push_str(&format!("{inner}point3f physics:localPos1 = (0, 0, 0)\n"));
        s.push_str(&format!(
            "{inner}quatf physics:localRot1 = {}\n",
            fmt_quat(&extra_rot)
        ));

        // Limits: revolute in degrees, prismatic in metres, continuous
        // unlimited (±360 marker).
        match joint.kind {
            mn::JointKind::Revolute => {
                s.push_str(&format!(
                    "{inner}float physics:lowerLimit = {}\n",
                    fmt(joint.limit.lower.to_degrees())
                ));
                s.push_str(&format!(
                    "{inner}float physics:upperLimit = {}\n",
                    fmt(joint.limit.upper.to_degrees())
                ));
            }
            mn::JointKind::Continuous => {
                s.push_str(&format!("{inner}float physics:lowerLimit = -360\n"));
                s.push_str(&format!("{inner}float physics:upperLimit = 360\n"));
            }
            mn::JointKind::Prismatic => {
                s.push_str(&format!(
                    "{inner}float physics:lowerLimit = {}\n",
                    fmt(joint.limit.lower)
                ));
                s.push_str(&format!(
                    "{inner}float physics:upperLimit = {}\n",
                    fmt(joint.limit.upper)
                ));
            }
            _ => {}
        }

        if let Some(dk) = drive_kind {
            s.push_str(&format!(
                "{inner}float drive:{dk}:physics:damping = 1000\n"
            ));
            s.push_str(&format!(
                "{inner}float drive:{dk}:physics:stiffness = 10000\n"
            ));
            s.push_str(&format!(
                "{inner}token drive:{dk}:physics:type = \"force\"\n"
            ));
        }
    } else {
        // Fixed joint — just the local transforms.
        let t = iso.translation;
        s.push_str(&format!(
            "{inner}point3f physics:localPos0 = {}\n",
            fmt_3(t.x, t.y, t.z)
        ));
        s.push_str(&format!(
            "{inner}quatf physics:localRot0 = {}\n",
            fmt_quat(&iso.rotation)
        ));
        s.push_str(&format!("{inner}point3f physics:localPos1 = (0, 0, 0)\n"));
        s.push_str(&format!("{inner}quatf physics:localRot1 = (1, 0, 0, 0)\n"));
    }

    s.push_str(&format!("{indent}}}\n\n"));
}

/// Write a UsdPreviewSurface material.
fn write_material(s: &mut String, idx: usize, color: &[f32; 4], robot_path: &str, indent: &str) {
    let mat_name = format!("material_{idx}");
    let inner = format!("{indent}    ");
    let shader_indent = format!("{inner}    ");

    s.push_str(&format!(
        "{indent}def Material \"{mat_name}\"\n{indent}{{\n"
    ));
    s.push_str(&format!(
        "{inner}token outputs:surface.connect = <{robot_path}/Materials/{mat_name}/PBRShader.outputs:surface>\n"
    ));

    s.push_str(&format!("{inner}def Shader \"PBRShader\"\n{inner}{{\n"));
    s.push_str(&format!(
        "{shader_indent}uniform token info:id = \"UsdPreviewSurface\"\n"
    ));
    s.push_str(&format!(
        "{shader_indent}color3f inputs:diffuseColor = ({}, {}, {})\n",
        color[0], color[1], color[2]
    ));
    if (color[3] - 1.0).abs() > 1e-4 {
        s.push_str(&format!(
            "{shader_indent}float inputs:opacity = {}\n",
            color[3]
        ));
    }
    s.push_str(&format!("{shader_indent}float inputs:metallic = 0\n"));
    s.push_str(&format!("{shader_indent}float inputs:roughness = 0.5\n"));
    s.push_str(&format!("{shader_indent}token outputs:surface\n"));
    s.push_str(&format!("{inner}}}\n"));
    s.push_str(&format!("{indent}}}\n\n"));
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A small two-link MisaFile with one revolute joint.
    fn sample_file() -> MisaFile {
        let mut f = MisaFile::new("test_robot", "base_link");
        f.link.push(mn::Link {
            name: "base_link".into(),
            description: String::new(),
            inertial: mn::Inertial {
                mass: 2.5,
                ixx: 0.01,
                iyy: 0.02,
                izz: 0.03,
                origin: mn::Origin {
                    xyz: [0.01, 0.02, 0.03],
                    rpy: None,
                    quat: None,
                },
                ..Default::default()
            },
            visual: vec![mn::Visual {
                origin: mn::Origin::default(),
                geom: mn::Geom::Box {
                    size: [0.1, 0.1, 0.05],
                },
                color: Some(mn::ColorSpec::Rgba([1.0, 0.0, 0.0, 0.8])),
                material: None,
            }],
            collision: vec![],
            collision_enabled: true,
        });
        f.link.push(mn::Link {
            name: "arm_link".into(),
            description: String::new(),
            inertial: mn::Inertial {
                mass: 0.5,
                ixx: 0.001,
                iyy: 0.001,
                izz: 0.001,
                ..Default::default()
            },
            visual: vec![mn::Visual {
                origin: mn::Origin::default(),
                geom: mn::Geom::Cylinder {
                    radius: 0.03,
                    length: 0.3,
                },
                color: None,
                material: None,
            }],
            collision: vec![],
            collision_enabled: true,
        });
        f.joint.push(mn::Joint {
            name: "arm_joint".into(),
            kind: mn::JointKind::Revolute,
            parent: "base_link".into(),
            child: "arm_link".into(),
            axis: [0.0, 0.0, 1.0],
            origin: mn::Origin {
                xyz: [0.0, 0.0, 0.1],
                rpy: None,
                quat: None,
            },
            limit: mn::JointLimit {
                lower: -1.57,
                upper: 1.57,
                effort: 10.0,
                velocity: 2.0,
            },
            dynamics: mn::JointDynamics::default(),
        });
        f
    }

    #[test]
    fn sanitize_various_names() {
        assert_eq!(sanitize_name("base_link"), "base_link");
        assert_eq!(sanitize_name("link-1"), "link_1");
        assert_eq!(sanitize_name("123abc"), "_123abc");
        assert_eq!(sanitize_name(""), "prim");
        assert_eq!(sanitize_name("my link!"), "my_link_");
    }

    #[test]
    fn determine_axis_variants() {
        let (axis, rot) = determine_usd_axis(&[0.0, 0.0, 1.0]);
        assert_eq!(axis, "Z");
        assert!((rot.quaternion().w - 1.0).abs() < 1e-9);
        let (axis, _) = determine_usd_axis(&[1.0, 0.0, 0.0]);
        assert_eq!(axis, "X");
        // Arbitrary axis: Z rotated onto the direction.
        let d = 0.707_f64;
        let (axis, rot) = determine_usd_axis(&[0.0, d, d]);
        assert_eq!(axis, "Z");
        let mapped = rot * na::Vector3::z();
        let target = na::Vector3::new(0.0, d, d).normalize();
        assert!((mapped - target).norm() < 1e-3);
    }

    #[test]
    fn export_structure() {
        let usda = export(&sample_file(), &UsdExportRefs::default());
        assert!(usda.starts_with("#usda 1.0"));
        assert!(usda.contains("defaultPrim = \"World\""));
        assert!(usda.contains("upAxis = \"Z\""));
        assert!(usda.contains("def Xform \"test_robot\""));
        assert!(usda.contains("PhysicsArticulationRootAPI"));
        assert!(usda.contains("def Xform \"base_link\""));
        assert!(usda.contains("PhysicsRigidBodyAPI"));
        assert!(usda.contains("def Cube \"visual_0\""));
        assert!(usda.contains("double size = 2.0"));
        assert!(usda.contains("def Material \"material_0\""));
        assert!(usda.contains("UsdPreviewSurface"));
        assert!(usda.contains("PhysicsRevoluteJoint"));
        assert!(usda.contains("physics:lowerLimit"));
    }

    #[test]
    fn roundtrip_joints_and_limits() {
        let usda = export(&sample_file(), &UsdExportRefs::default());
        let back = import_str(&usda).expect("re-import");
        let f = &back.file;
        assert_eq!(f.robot.name, "test_robot");
        assert_eq!(f.robot.root, "base_link");
        assert_eq!(f.link.len(), 2);
        assert_eq!(f.joint.len(), 1);

        let j = &f.joint[0];
        assert_eq!(j.kind, mn::JointKind::Revolute);
        assert_eq!(j.parent, "base_link");
        assert_eq!(j.child, "arm_link");
        // Degrees → radians round trip.
        assert!(
            (j.limit.lower - (-1.57)).abs() < 0.02,
            "lower = {}",
            j.limit.lower
        );
        assert!((j.limit.upper - 1.57).abs() < 0.02);
        // Origin translation preserved.
        assert!((j.origin.xyz[2] - 0.1).abs() < 1e-3);
        // Axis reconstructed as ~Z.
        assert!((j.axis[2].abs() - 1.0).abs() < 0.1, "axis = {:?}", j.axis);
    }

    #[test]
    fn roundtrip_geometry_and_color() {
        let usda = export(&sample_file(), &UsdExportRefs::default());
        let f = import_str(&usda).unwrap().file;

        match &f.link[0].visual[0].geom {
            mn::Geom::Box { size } => {
                assert!((size[0] - 0.1).abs() < 1e-3);
                assert!((size[2] - 0.05).abs() < 1e-3);
            }
            other => panic!("expected box, got {other:?}"),
        }
        match &f.link[1].visual[0].geom {
            mn::Geom::Cylinder { radius, length } => {
                assert!((radius - 0.03).abs() < 1e-6);
                assert!((length - 0.3).abs() < 1e-6);
            }
            other => panic!("expected cylinder, got {other:?}"),
        }
        // Inline color survives via the material table.
        match &f.link[0].visual[0].color {
            Some(mn::ColorSpec::Rgba(c)) => {
                assert!((c[0] - 1.0).abs() < 0.01);
                assert!((c[3] - 0.8).abs() < 0.01);
            }
            other => panic!("expected rgba, got {other:?}"),
        }
    }

    #[test]
    fn roundtrip_inertial() {
        let usda = export(&sample_file(), &UsdExportRefs::default());
        let f = import_str(&usda).unwrap().file;
        let i = &f.link[0].inertial;
        assert!((i.mass - 2.5).abs() < 0.01);
        assert!((i.ixx - 0.01).abs() < 1e-3);
        assert!((i.iyy - 0.02).abs() < 1e-3);
        assert!((i.izz - 0.03).abs() < 1e-3);
        assert!((i.origin.xyz[0] - 0.01).abs() < 1e-3);
    }

    #[test]
    fn capsule_roundtrips() {
        let mut file = sample_file();
        file.link[1].visual[0].geom = mn::Geom::Capsule {
            radius: 0.02,
            length: 0.2,
        };
        let usda = export(&file, &UsdExportRefs::default());
        // Total height = length + 2·radius.
        assert!(usda.contains("double height = 0.24"), "{usda}");
        let back = import_str(&usda).unwrap().file;
        match &back.link[1].visual[0].geom {
            mn::Geom::Capsule { radius, length } => {
                assert!((radius - 0.02).abs() < 1e-9);
                assert!((length - 0.2).abs() < 1e-9);
            }
            other => panic!("expected capsule, got {other:?}"),
        }
    }

    #[test]
    fn inline_meshes_export_and_reimport() {
        let mut file = sample_file();
        file.link[0].visual[0].geom = mn::Geom::Mesh {
            file: "meshes/trunk.stl".into(),
            scale: [1.0, 1.0, 1.0],
        };
        // One triangle.
        let tri: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, //
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let mesh_fn = move |link: &str, slot: GeomSlot| -> Option<Vec<f32>> {
            (link == "base_link" && slot == GeomSlot::Visual(0)).then(|| tri.clone())
        };
        let refs = UsdExportRefs {
            link_world_tf: None,
            mesh_vertices: Some(&mesh_fn),
        };
        let usda = export(&file, &refs);
        assert!(usda.contains("def Mesh \"visual_0\""));
        assert!(usda.contains("point3f[] points"));

        let back = import_str(&usda).unwrap();
        assert_eq!(back.inline_meshes.len(), 1);
        assert_eq!(back.inline_meshes[0].link, "base_link");
        assert_eq!(back.inline_meshes[0].slot, GeomSlot::Visual(0));
        assert_eq!(back.inline_meshes[0].mesh.num_triangles(), 1);
        match &back.file.link[0].visual[0].geom {
            mn::Geom::Mesh { file, .. } => assert!(file.starts_with(INLINE_MESH_PREFIX)),
            other => panic!("expected mesh, got {other:?}"),
        }
    }

    #[test]
    fn filtered_pairs_roundtrip() {
        let mut file = sample_file();
        file.collision_pair.push(mn::CollisionPair {
            link_a: "arm_link".into(),
            link_b: "base_link".into(),
            enabled: false,
        });
        let usda = export(&file, &UsdExportRefs::default());
        assert!(usda.contains("PhysicsFilteredPairsAPI"));
        assert!(usda.contains("rel physics:filteredPairs"));

        let back = import_str(&usda).unwrap().file;
        assert_eq!(back.collision_pair.len(), 1);
        assert!(!back.collision_pair[0].enabled);
        assert_eq!(back.collision_pair[0].link_a, "arm_link");
        assert_eq!(back.collision_pair[0].link_b, "base_link");
    }

    #[test]
    fn posed_transform_override() {
        let file = sample_file();
        let tf_fn = |link: &str| -> Option<na::Isometry3<f64>> {
            (link == "base_link").then(|| {
                na::Isometry3::from_parts(
                    na::Translation3::new(0.0, 0.0, 0.42),
                    na::UnitQuaternion::identity(),
                )
            })
        };
        let refs = UsdExportRefs {
            link_world_tf: Some(&tf_fn),
            mesh_vertices: None,
        };
        let usda = export(&file, &refs);
        assert!(usda.contains("(0, 0, 0.42)"), "{usda}");
    }
}
