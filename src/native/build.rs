//! Convert a parsed [`MisaFile`] into runtime [`Model`] +
//! [`GeometryModel`]s.
//!
//! Mirrors the post-parse step in [`crate::urdf::load_urdf_string`]:
//! topologically sort joints from the root link, feed them into
//! [`ModelBuilder`], then apply mimic constraints in a second pass.
//! Visual / collision geometries become [`GeometryObject`]s attached to
//! the same joint indices.
//!
//! Material colour resolution: `visual.color` (inline) takes precedence
//! over `visual.material` (named lookup); both are pre-validated by
//! `parse::parse_str` to be mutually exclusive, but the build code
//! defends in depth by treating a missing material name as silent
//! "use default colour".

use std::collections::{HashMap, HashSet, VecDeque};

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};

use crate::geometry::{GeometryModel, GeometryObject, GeometryShape};
use crate::joint::JointType;
use crate::mesh::Material as MeshMaterial;
use crate::model::{LinkInertia, Model, ModelBuilder};
use crate::se3::{self, SE3};

use super::schema::{
    ColorSpec, Geom, JointKind, Material as SchemaMaterial, MisaFile, Origin, Visual,
};
use super::NativeError;

/// Build a runtime [`Model`], visual [`GeometryModel`], and collision
/// [`GeometryModel`] from a parsed `.misa` document.
///
/// The model's link names match `MisaFile.link[*].name`; visual /
/// collision objects are named `<link>_visual_<i>` /
/// `<link>_collision_<i>` (1-based not used — we follow URDF's 0-based
/// numbering for parity).
pub fn build_model(
    file: &MisaFile,
) -> Result<(Model<f64>, GeometryModel, GeometryModel), NativeError> {
    // ── 1. Index links and validate root presence (defence in depth) ────
    let link_idx_map: HashMap<&str, usize> = file
        .link
        .iter()
        .enumerate()
        .map(|(i, l)| (l.name.as_str(), i))
        .collect();
    if !link_idx_map.contains_key(file.robot.root.as_str()) {
        return Err(NativeError::Validation(format!(
            "robot.root '{}' is not in the link list",
            file.robot.root
        )));
    }

    // ── 2. Topological sort joints from root ────────────────────────────
    // We assign each link a `Model` joint index (root link → 0, then BFS
    // order). The URDF parser uses the same convention.
    let mut model_idx: HashMap<&str, usize> = HashMap::new();
    model_idx.insert(file.robot.root.as_str(), 0);

    let mut queue: VecDeque<&str> = VecDeque::new();
    queue.push_back(file.robot.root.as_str());

    let mut ordered_joint_indices: Vec<usize> = Vec::with_capacity(file.joint.len());
    let mut visited_joints: HashSet<usize> = HashSet::new();

    while let Some(parent_link) = queue.pop_front() {
        for (ji, j) in file.joint.iter().enumerate() {
            if j.parent == parent_link
                && !model_idx.contains_key(j.child.as_str())
                && visited_joints.insert(ji)
            {
                let next_idx = model_idx.len();
                model_idx.insert(j.child.as_str(), next_idx);
                ordered_joint_indices.push(ji);
                queue.push_back(j.child.as_str());
            }
        }
    }

    if ordered_joint_indices.len() != file.joint.len() {
        return Err(NativeError::Validation(format!(
            "{} joint(s) could not be reached from root link '{}' \
             (orphaned subtree)",
            file.joint.len() - ordered_joint_indices.len(),
            file.robot.root,
        )));
    }

    // ── 3. Inertia map by link name ─────────────────────────────────────
    // `inertial.origin` is the centre-of-mass / principal-axis frame
    // relative to the link frame. misarta's `LinkInertia` carries
    // `center_of_mass` (a translation) and `rotational_inertia` (a 3×3
    // matrix expressed at the COM in the link frame). We honour the
    // COM translation but currently fold the rotation back into the
    // tensor (rotate the inertia matrix into the link frame), since
    // misarta has no separate inertial-frame rotation.
    let inertia_for_link = |name: &str| -> LinkInertia<f64> {
        let li = file
            .link
            .iter()
            .find(|l| l.name == name)
            .map(|l| &l.inertial);
        match li {
            None => LinkInertia::zero(),
            Some(i) => {
                let com_local = Vector3::new(
                    i.origin.xyz[0],
                    i.origin.xyz[1],
                    i.origin.xyz[2],
                );
                let raw = Matrix3::new(
                    i.ixx, i.ixy, i.ixz, i.ixy, i.iyy, i.iyz, i.ixz, i.iyz, i.izz,
                );
                let rot = origin_rotation(&i.origin);
                let rot_mat = rot.to_rotation_matrix();
                let r = rot_mat.matrix();
                let rotated = r * raw * r.transpose();
                LinkInertia {
                    mass: i.mass,
                    center_of_mass: com_local,
                    rotational_inertia: rotated,
                }
            }
        }
    };

    // ── 4. Build kinematic Model ────────────────────────────────────────
    let mut builder = ModelBuilder::<f64>::new()
        .name(file.robot.name.clone())
        .root_link_name(file.robot.root.clone());

    for &ji in &ordered_joint_indices {
        let j = &file.joint[ji];
        let parent_idx = model_idx[j.parent.as_str()];
        let placement = origin_to_se3(&j.origin);
        let axis = unit_axis(&j.axis);
        let joint_type = match j.kind {
            JointKind::Revolute | JointKind::Continuous => JointType::Revolute { axis },
            JointKind::Prismatic => JointType::Prismatic { axis },
            JointKind::Fixed => JointType::Fixed,
            JointKind::Floating => JointType::FreeFlyer,
            JointKind::Planar => {
                return Err(NativeError::Validation(format!(
                    "joint '{}': type 'planar' is not yet supported by misarta",
                    j.name,
                )));
            }
        };
        let inertia = inertia_for_link(&j.child);
        builder = builder.add_joint_with_link(
            j.name.clone(),
            parent_idx,
            joint_type,
            placement,
            inertia,
            j.child.clone(),
        );
    }

    let model_tmp = builder.build();

    // ── 5. Apply mimic constraints in a second pass ─────────────────────
    let joint_name_to_model_idx: HashMap<&str, usize> = model_tmp
        .joints
        .iter()
        .enumerate()
        .map(|(i, j)| (j.name.as_str(), i))
        .collect();

    let mut builder2 = ModelBuilder::from_model(&model_tmp);
    for m in &file.mimic {
        let slave_idx = *joint_name_to_model_idx.get(m.joint.as_str()).ok_or_else(|| {
            NativeError::Validation(format!(
                "mimic: target joint '{}' not found after build",
                m.joint
            ))
        })?;
        let master_idx = *joint_name_to_model_idx.get(m.source.as_str()).ok_or_else(|| {
            NativeError::Validation(format!(
                "mimic: source joint '{}' not found after build",
                m.source
            ))
        })?;
        // Skip mimics that point at non-1-DoF joints with a clear error
        // rather than tripping the assert! inside `add_mimic`.
        let slave_dof = model_tmp.joints[slave_idx].joint_type.nq();
        let master_dof = model_tmp.joints[master_idx].joint_type.nq();
        if slave_dof != 1 || master_dof != 1 {
            return Err(NativeError::Validation(format!(
                "mimic: joints must be 1-DoF (slave '{}' has {}, master '{}' has {})",
                m.joint, slave_dof, m.source, master_dof,
            )));
        }
        builder2 = builder2.add_mimic(slave_idx, master_idx, m.multiplier, m.offset);
    }
    let model = builder2.build();

    // ── 6. Build visual / collision GeometryModels ──────────────────────
    let material_lookup: HashMap<&str, &SchemaMaterial> = file
        .material
        .iter()
        .map(|m| (m.name.as_str(), m))
        .collect();

    let mut visual = GeometryModel::new();
    let mut collision = GeometryModel::new();

    for (link_local_idx, link) in file.link.iter().enumerate() {
        let parent_joint = match model_idx.get(link.name.as_str()) {
            Some(&i) => i,
            None => {
                // Link wasn't reachable — already returned above, but keep
                // this branch silent to avoid a panic.
                continue;
            }
        };
        let _ = link_local_idx; // currently unused in object naming

        for (vi, v) in link.visual.iter().enumerate() {
            let placement = origin_to_se3(&v.origin);
            let shape = geom_to_shape(&v.geom);
            let (mesh_path, mesh_scale) = mesh_info(&shape);
            let material =
                resolve_material(v, &material_lookup).map(|c| MeshMaterial::from_color(
                    c[0] as f64,
                    c[1] as f64,
                    c[2] as f64,
                    c[3] as f64,
                ));
            visual.add(GeometryObject {
                name: format!("{}_visual_{vi}", link.name),
                parent_joint,
                placement,
                shape,
                mesh_path,
                mesh_scale,
                mesh_data: None,
                material,
            });
        }

        for (ci, c) in link.collision.iter().enumerate() {
            let placement = origin_to_se3(&c.origin);
            let shape = geom_to_shape(&c.geom);
            let (mesh_path, mesh_scale) = mesh_info(&shape);
            collision.add(GeometryObject {
                name: format!("{}_collision_{ci}", link.name),
                parent_joint,
                placement,
                shape,
                mesh_path,
                mesh_scale,
                mesh_data: None,
                material: None,
            });
        }
    }

    Ok((model, visual, collision))
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn origin_to_se3(o: &Origin) -> SE3<f64> {
    let xyz = Vector3::new(o.xyz[0], o.xyz[1], o.xyz[2]);
    let rot = origin_rotation(o);
    let rot_mat = Rotation3::from_matrix_unchecked(rot.to_rotation_matrix().into_inner());
    se3::from_rotation_and_translation(&rot_mat, &xyz)
}

/// Resolve `Origin` rotation to a `UnitQuaternion`, preferring `quat`
/// over `rpy` when both are present (the parser rejects that case, but
/// being permissive here keeps `build_model` callable on hand-built
/// MisaFile instances). Returns identity when neither is set.
fn origin_rotation(o: &Origin) -> UnitQuaternion<f64> {
    if let Some(q) = o.quat {
        // schema stores [x, y, z, w]
        UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(q[3], q[0], q[1], q[2]))
    } else if let Some(rpy) = o.rpy {
        UnitQuaternion::from_euler_angles(rpy[0], rpy[1], rpy[2])
    } else {
        UnitQuaternion::identity()
    }
}

fn unit_axis(a: &[f64; 3]) -> Vector3<f64> {
    let v = Vector3::new(a[0], a[1], a[2]);
    let n = v.norm();
    if n > 1e-12 {
        v / n
    } else {
        Vector3::z()
    }
}

fn geom_to_shape(g: &Geom) -> GeometryShape {
    match g {
        Geom::Box { size } => GeometryShape::Box {
            x: size[0],
            y: size[1],
            z: size[2],
        },
        Geom::Cylinder { radius, length } => GeometryShape::Cylinder {
            radius: *radius,
            length: *length,
        },
        Geom::Sphere { radius } => GeometryShape::Sphere { radius: *radius },
        Geom::Capsule { radius, length } => GeometryShape::Capsule {
            radius: *radius,
            length: *length,
        },
        Geom::Mesh { file, scale } => GeometryShape::Mesh {
            filename: file.clone(),
            scale: Vector3::new(scale[0], scale[1], scale[2]),
        },
    }
}

fn mesh_info(shape: &GeometryShape) -> (Option<String>, Option<Vector3<f64>>) {
    match shape {
        GeometryShape::Mesh { filename, scale } => (Some(filename.clone()), Some(*scale)),
        _ => (None, None),
    }
}

/// Resolve a visual's effective colour as RGBA in 0..1.
///
/// Order: inline `color` first, then named `material` lookup. Returns
/// `None` if neither is present (caller defaults to no material).
fn resolve_material(
    v: &Visual,
    materials: &HashMap<&str, &SchemaMaterial>,
) -> Option<[f32; 4]> {
    if let Some(c) = &v.color {
        return Some(color_spec_to_rgba(c));
    }
    if let Some(name) = &v.material {
        if let Some(m) = materials.get(name.as_str()) {
            return Some(color_spec_to_rgba(&m.color));
        }
    }
    None
}

fn color_spec_to_rgba(c: &ColorSpec) -> [f32; 4] {
    match c {
        ColorSpec::Rgba(v) => *v,
        ColorSpec::Hex(s) => parse_hex_color(s).unwrap_or([0.8, 0.8, 0.8, 1.0]),
    }
}

fn parse_hex_color(s: &str) -> Option<[f32; 4]> {
    let s = s.strip_prefix('#').unwrap_or(s);
    fn byte(s: &str, i: usize) -> Option<f32> {
        let pair = s.get(i..i + 2)?;
        u8::from_str_radix(pair, 16).ok().map(|b| b as f32 / 255.0)
    }
    match s.len() {
        6 => Some([byte(s, 0)?, byte(s, 2)?, byte(s, 4)?, 1.0]),
        8 => Some([byte(s, 0)?, byte(s, 2)?, byte(s, 4)?, byte(s, 6)?]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hex_color_6_digit() {
        let c = parse_hex_color("#ff8040").unwrap();
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1] - 0.50196).abs() < 1e-3);
        assert!((c[2] - 0.25098).abs() < 1e-3);
        assert_eq!(c[3], 1.0);
    }

    #[test]
    fn parse_hex_color_8_digit_with_alpha() {
        let c = parse_hex_color("#ff8040c0").unwrap();
        assert_eq!(c[0], 1.0);
        assert!((c[3] - 0.7529).abs() < 1e-3);
    }

    #[test]
    fn parse_hex_color_no_hash() {
        assert!(parse_hex_color("ff8040").is_some());
    }

    #[test]
    fn parse_hex_color_rejects_invalid() {
        assert!(parse_hex_color("#fff").is_none());
        assert!(parse_hex_color("#xxxxxx").is_none());
    }
}
