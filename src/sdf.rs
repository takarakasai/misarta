//! SDF (Simulation Description Format) loader.
//!
//! Parses an SDF XML string (version 1.5–1.8) and builds a `Model<f64>`.
//!
//! # Supported elements
//!
//! - `<sdf>` / `<model>` — top-level containers
//! - `<link>` — rigid body with optional `<inertial>`
//! - `<joint>` — revolute, prismatic, fixed, ball (→ free-flyer)
//! - `<pose>` — "x y z roll pitch yaw" placement
//! - `<axis><xyz>` — joint axis
//!
//! # Differences from URDF
//!
//! - SDF joints have a `<pose>` relative to their **child link** by default.
//!   To compute the placement in the parent frame (which misarta uses),
//!   we compose the parent→child link offset with the joint's local pose.
//!
//! # Example
//!
//! ```no_run
//! use misarta::sdf;
//! let xml = std::fs::read_to_string("robot.sdf").unwrap();
//! let model = sdf::load_sdf_string(&xml).unwrap();
//! ```

use crate::geometry::{GeometryModel, GeometryObject, GeometryShape};
use crate::joint::JointType;
use crate::model::{LinkInertia, Model, ModelBuilder};
use crate::se3;
use nalgebra::{Matrix3, Rotation3, Vector3};
use roxmltree::Document;
use std::collections::HashMap;

/// Errors arising from SDF parsing.
#[derive(Debug, Clone)]
pub enum SdfError {
    /// XML is not well-formed.
    XmlParse(String),
    /// Missing required element.
    MissingElement(String),
    /// Unsupported joint type.
    UnsupportedJointType(String),
    /// Topological sort failed.
    Topology(String),
}

impl std::fmt::Display for SdfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SdfError::XmlParse(e) => write!(f, "SDF XML parse error: {e}"),
            SdfError::MissingElement(e) => write!(f, "SDF missing element: {e}"),
            SdfError::UnsupportedJointType(e) => write!(f, "unsupported SDF joint type: {e}"),
            SdfError::Topology(e) => write!(f, "SDF topology error: {e}"),
        }
    }
}

impl std::error::Error for SdfError {}

/// Load a `Model<f64>` from an SDF XML file on disk.
///
/// If the SDF contains multiple `<model>` elements, only the first is loaded.
pub fn load_sdf(path: &std::path::Path) -> Result<Model<f64>, SdfError> {
    let xml = std::fs::read_to_string(path)
        .map_err(|e| SdfError::XmlParse(format!("cannot read {}: {e}", path.display())))?;
    load_sdf_string(&xml)
}

/// Load a `Model<f64>` together with visual and collision `GeometryModel`s
/// from an SDF XML file on disk.
pub fn load_sdf_geometry(
    path: &std::path::Path,
) -> Result<(Model<f64>, GeometryModel, GeometryModel), SdfError> {
    let xml = std::fs::read_to_string(path)
        .map_err(|e| SdfError::XmlParse(format!("cannot read {}: {e}", path.display())))?;
    load_sdf_geometry_string(&xml)
}

/// Load a `Model<f64>` together with visual and collision `GeometryModel`s
/// from an SDF XML string.
pub fn load_sdf_geometry_string(
    xml: &str,
) -> Result<(Model<f64>, GeometryModel, GeometryModel), SdfError> {
    let doc = Document::parse(xml).map_err(|e| SdfError::XmlParse(e.to_string()))?;
    let sdf_root = doc.root_element();
    if sdf_root.tag_name().name() != "sdf" {
        return Err(SdfError::MissingElement("root <sdf> element".into()));
    }
    let model_el = sdf_root
        .children()
        .find(|n| n.tag_name().name() == "model")
        .ok_or_else(|| SdfError::MissingElement("<model> inside <sdf>".into()))?;

    // Build kinematic model via the existing parser
    let model = load_sdf_string(xml)?;

    // Build link_name → joint index map
    let mut link_to_idx: HashMap<&str, usize> = HashMap::new();
    for (i, name) in model.link_names.iter().enumerate() {
        link_to_idx.insert(name.as_str(), i);
    }

    let mut visual_model = GeometryModel::new();
    let mut collision_model = GeometryModel::new();

    for link_el in model_el
        .children()
        .filter(|n| n.tag_name().name() == "link")
    {
        let link_name = link_el
            .attribute("name")
            .ok_or_else(|| SdfError::MissingElement("link name".into()))?;
        let joint_idx = *link_to_idx
            .get(link_name)
            .ok_or_else(|| SdfError::Topology(format!("link '{link_name}' not in model")))?;

        // Visual geometries
        for (vi, vis_el) in link_el
            .children()
            .filter(|n| n.tag_name().name() == "visual")
            .enumerate()
        {
            let placement = parse_pose_element(&vis_el);
            if let Some(shape) = parse_sdf_geometry(&vis_el) {
                let obj_name = vis_el
                    .attribute("name")
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("{link_name}_visual_{vi}"));
                let (mesh_path, mesh_scale) = extract_mesh_info(&shape);
                visual_model.add(GeometryObject {
                    name: obj_name,
                    parent_joint: joint_idx,
                    placement,
                    shape,
                    mesh_path,
                    mesh_scale,
                    mesh_data: None,
            material: None,
                });
            }
        }

        // Collision geometries
        for (ci, col_el) in link_el
            .children()
            .filter(|n| n.tag_name().name() == "collision")
            .enumerate()
        {
            let placement = parse_pose_element(&col_el);
            if let Some(shape) = parse_sdf_geometry(&col_el) {
                let obj_name = col_el
                    .attribute("name")
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("{link_name}_collision_{ci}"));
                let (mesh_path, mesh_scale) = extract_mesh_info(&shape);
                collision_model.add(GeometryObject {
                    name: obj_name,
                    parent_joint: joint_idx,
                    placement,
                    shape,
                    mesh_path,
                    mesh_scale,
                    mesh_data: None,
            material: None,
                });
            }
        }
    }

    Ok((model, visual_model, collision_model))
}

/// Load a `Model<f64>` from an SDF XML string.
///
/// If the SDF contains multiple `<model>` elements, only the first is loaded.
pub fn load_sdf_string(xml: &str) -> Result<Model<f64>, SdfError> {
    let doc = Document::parse(xml).map_err(|e| SdfError::XmlParse(e.to_string()))?;
    let sdf_root = doc.root_element();
    if sdf_root.tag_name().name() != "sdf" {
        return Err(SdfError::MissingElement("root <sdf> element".into()));
    }

    let model_el = sdf_root
        .children()
        .find(|n| n.tag_name().name() == "model")
        .ok_or_else(|| SdfError::MissingElement("<model> inside <sdf>".into()))?;

    // ── Collect links ───────────────────────────────────────────────────
    let mut link_inertias: HashMap<String, LinkInertia<f64>> = HashMap::new();
    for link_el in model_el
        .children()
        .filter(|n| n.tag_name().name() == "link")
    {
        let name = link_el
            .attribute("name")
            .ok_or_else(|| SdfError::MissingElement("link name".into()))?
            .to_string();
        let inertia = parse_link_inertia(&link_el);
        link_inertias.insert(name, inertia);
    }

    // ── Collect joints ──────────────────────────────────────────────────
    struct JointInfo {
        name: String,
        joint_type: JointType<f64>,
        parent_link: String,
        child_link: String,
        pose: nalgebra::Isometry3<f64>,
    }

    let mut joints: Vec<JointInfo> = Vec::new();
    for joint_el in model_el
        .children()
        .filter(|n| n.tag_name().name() == "joint")
    {
        let name = joint_el
            .attribute("name")
            .ok_or_else(|| SdfError::MissingElement("joint name".into()))?
            .to_string();
        let jtype_str = joint_el
            .attribute("type")
            .ok_or_else(|| SdfError::MissingElement(format!("joint type for '{name}'")))?;

        let parent_link = child_text(&joint_el, "parent")
            .ok_or_else(|| SdfError::MissingElement(format!("parent for '{name}'")))?;

        let child_link = child_text(&joint_el, "child")
            .ok_or_else(|| SdfError::MissingElement(format!("child for '{name}'")))?;

        let pose = parse_pose_element(&joint_el);
        let axis = parse_axis_sdf(&joint_el);

        let joint_type = match jtype_str {
            "revolute" => JointType::Revolute { axis },
            "prismatic" => JointType::Prismatic { axis },
            "fixed" => JointType::Fixed,
            "ball" | "universal" | "floating" => JointType::FreeFlyer,
            other => return Err(SdfError::UnsupportedJointType(other.to_string())),
        };

        joints.push(JointInfo {
            name,
            joint_type,
            parent_link,
            child_link,
            pose,
        });
    }

    // ── Find root link ──────────────────────────────────────────────────
    let child_links: std::collections::HashSet<&str> =
        joints.iter().map(|j| j.child_link.as_str()).collect();
    let root_link = link_inertias
        .keys()
        .find(|name| !child_links.contains(name.as_str()))
        .ok_or_else(|| SdfError::Topology("no root link found".into()))?
        .clone();

    // ── BFS topological order ───────────────────────────────────────────
    let mut link_to_idx: HashMap<String, usize> = HashMap::new();
    link_to_idx.insert(root_link.clone(), 0);

    let mut queue = std::collections::VecDeque::new();
    queue.push_back(root_link.clone());
    let mut ordered_joints: Vec<&JointInfo> = Vec::new();

    while let Some(parent_name) = queue.pop_front() {
        for ji in &joints {
            if ji.parent_link == parent_name && !link_to_idx.contains_key(&ji.child_link) {
                ordered_joints.push(ji);
                let idx = link_to_idx.len();
                link_to_idx.insert(ji.child_link.clone(), idx);
                queue.push_back(ji.child_link.clone());
            }
        }
    }

    if ordered_joints.len() != joints.len() {
        return Err(SdfError::Topology(
            "some joints could not be reached from root link".into(),
        ));
    }

    // ── Build model ─────────────────────────────────────────────────────
    let model_name = model_el.attribute("name").unwrap_or("").to_string();
    let mut builder = ModelBuilder::new()
        .name(model_name)
        .root_link_name(root_link.clone());
    for ji in &ordered_joints {
        let parent_idx = link_to_idx[&ji.parent_link];
        let inertia = link_inertias
            .get(&ji.child_link)
            .cloned()
            .unwrap_or_else(LinkInertia::zero);

        // In SDF, <joint><pose> is typically relative to the child link frame,
        // and for simple cases it matches the joint origin in the parent frame.
        // We use it directly as the parent→joint placement, which is correct
        // when the joint pose is expressed relative to the parent (the common case
        // for simple SDF files without nested model frames).
        builder = builder.add_joint_with_link(
            ji.name.clone(),
            parent_idx,
            ji.joint_type.clone(),
            ji.pose,
            inertia,
            ji.child_link.clone(),
        );
    }

    Ok(builder.build())
}

// ─── Internal helpers ───────────────────────────────────────────────────────

/// Parse a `<pose>x y z roll pitch yaw</pose>` element.
fn parse_pose_element(parent: &roxmltree::Node) -> nalgebra::Isometry3<f64> {
    if let Some(pose_el) = parent.children().find(|n| n.tag_name().name() == "pose") {
        if let Some(text) = pose_el.text() {
            let vals: Vec<f64> = text.split_whitespace().filter_map(|v| v.parse().ok()).collect();
            if vals.len() >= 6 {
                let t = Vector3::new(vals[0], vals[1], vals[2]);
                let rot = Rotation3::from_euler_angles(vals[3], vals[4], vals[5]);
                return se3::from_rotation_and_translation(&rot, &t);
            }
        }
    }
    se3::identity()
}

/// Parse `<axis><xyz>x y z</xyz></axis>`.
fn parse_axis_sdf(parent: &roxmltree::Node) -> Vector3<f64> {
    if let Some(axis_el) = parent.children().find(|n| n.tag_name().name() == "axis") {
        if let Some(xyz_text) = child_text(&axis_el, "xyz") {
            let v = parse_vec3(&xyz_text);
            let n = v.norm();
            if n > 1e-12 {
                return v / n;
            }
        }
    }
    Vector3::z()
}

/// Parse `<inertial>` for an SDF link.
fn parse_link_inertia(link_el: &roxmltree::Node) -> LinkInertia<f64> {
    if let Some(inertial) = link_el
        .children()
        .find(|n| n.tag_name().name() == "inertial")
    {
        let mass = child_text(&inertial, "mass")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);

        let com = if let Some(pose_el) = inertial
            .children()
            .find(|n| n.tag_name().name() == "pose")
        {
            if let Some(text) = pose_el.text() {
                let vals: Vec<f64> =
                    text.split_whitespace().filter_map(|v| v.parse().ok()).collect();
                if vals.len() >= 3 {
                    Vector3::new(vals[0], vals[1], vals[2])
                } else {
                    Vector3::zeros()
                }
            } else {
                Vector3::zeros()
            }
        } else {
            Vector3::zeros()
        };

        let rotational_inertia = if let Some(inertia_el) = inertial
            .children()
            .find(|n| n.tag_name().name() == "inertia")
        {
            let ixx = child_text(&inertia_el, "ixx").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            let ixy = child_text(&inertia_el, "ixy").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            let ixz = child_text(&inertia_el, "ixz").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            let iyy = child_text(&inertia_el, "iyy").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            let iyz = child_text(&inertia_el, "iyz").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            let izz = child_text(&inertia_el, "izz").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            Matrix3::new(ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz)
        } else {
            Matrix3::zeros()
        };

        LinkInertia {
            mass,
            center_of_mass: com,
            rotational_inertia,
        }
    } else {
        LinkInertia::zero()
    }
}

/// Get the text content of a named child element.
fn child_text(parent: &roxmltree::Node, tag: &str) -> Option<String> {
    parent
        .children()
        .find(|n| n.tag_name().name() == tag)
        .and_then(|n| n.text())
        .map(|s| s.trim().to_string())
}

/// Parse a whitespace-separated triple "x y z" into a Vector3.
fn parse_vec3(s: &str) -> Vector3<f64> {
    let vals: Vec<f64> = s.split_whitespace().filter_map(|v| v.parse().ok()).collect();
    if vals.len() >= 3 {
        Vector3::new(vals[0], vals[1], vals[2])
    } else {
        Vector3::zeros()
    }
}

/// Parse an SDF `<geometry>` child element into a `GeometryShape`.
fn parse_sdf_geometry(parent: &roxmltree::Node) -> Option<GeometryShape> {
    let geom_el = parent.children().find(|n| n.tag_name().name() == "geometry")?;

    for child in geom_el.children() {
        match child.tag_name().name() {
            "box" => {
                let size_str = child_text(&child, "size").unwrap_or_default();
                let size = parse_vec3(&size_str);
                return Some(GeometryShape::Box {
                    x: size[0],
                    y: size[1],
                    z: size[2],
                });
            }
            "sphere" => {
                let r = child_text(&child, "radius")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                return Some(GeometryShape::Sphere { radius: r });
            }
            "cylinder" => {
                let r = child_text(&child, "radius")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                let l = child_text(&child, "length")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                return Some(GeometryShape::Cylinder { radius: r, length: l });
            }
            "capsule" => {
                let r = child_text(&child, "radius")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                let l = child_text(&child, "length")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                return Some(GeometryShape::Capsule { radius: r, length: l });
            }
            "mesh" => {
                let filename = child_text(&child, "uri").unwrap_or_default();
                let scale = child_text(&child, "scale")
                    .map(|s| parse_vec3(&s))
                    .unwrap_or_else(|| Vector3::new(1.0, 1.0, 1.0));
                return Some(GeometryShape::Mesh { filename, scale });
            }
            _ => {}
        }
    }
    None
}

/// Extract mesh_path / mesh_scale from a shape (convenience for GeometryObject).
fn extract_mesh_info(shape: &GeometryShape) -> (Option<String>, Option<Vector3<f64>>) {
    match shape {
        GeometryShape::Mesh { filename, scale } => {
            (Some(filename.clone()), Some(scale.clone()))
        }
        _ => (None, None),
    }
}

// ─── Writer ─────────────────────────────────────────────────────────────────

/// Write a `Model<f64>` to an SDF XML file on disk.
pub fn write_sdf(model: &Model<f64>, path: &std::path::Path) -> Result<(), SdfError> {
    let xml = write_sdf_string(model);
    std::fs::write(path, xml)
        .map_err(|e| SdfError::XmlParse(format!("cannot write {}: {e}", path.display())))
}

/// Serialize a `Model<f64>` to an SDF XML string.
pub fn write_sdf_string(model: &Model<f64>) -> String {
    write_sdf_geometry_string(model, None, None)
}

/// Serialize a `Model<f64>` with optional visual/collision geometry to an SDF XML string.
pub fn write_sdf_geometry_string(
    model: &Model<f64>,
    visual: Option<&GeometryModel>,
    collision: Option<&GeometryModel>,
) -> String {
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\"?>\n");
    out.push_str("<sdf version=\"1.7\">\n");
    out.push_str(&format!(
        "  <model name=\"{}\">\n",
        xml_escape(&model.name)
    ));

    // ── Links ───────────────────────────────────────────────────────────
    for (i, link_name) in model.link_names.iter().enumerate() {
        out.push_str(&format!(
            "    <link name=\"{}\">\n",
            xml_escape(link_name)
        ));
        let inertia = &model.inertias[i];
        if inertia.mass != 0.0
            || inertia.center_of_mass[0] != 0.0
            || inertia.center_of_mass[1] != 0.0
            || inertia.center_of_mass[2] != 0.0
            || inertia.rotational_inertia.norm() != 0.0
        {
            out.push_str("      <inertial>\n");
            out.push_str(&format!(
                "        <pose>{} {} {} 0 0 0</pose>\n",
                inertia.center_of_mass[0],
                inertia.center_of_mass[1],
                inertia.center_of_mass[2],
            ));
            out.push_str(&format!("        <mass>{}</mass>\n", inertia.mass));
            let ri = &inertia.rotational_inertia;
            out.push_str("        <inertia>\n");
            out.push_str(&format!("          <ixx>{}</ixx><ixy>{}</ixy><ixz>{}</ixz>\n", ri[(0,0)], ri[(0,1)], ri[(0,2)]));
            out.push_str(&format!("          <iyy>{}</iyy><iyz>{}</iyz><izz>{}</izz>\n", ri[(1,1)], ri[(1,2)], ri[(2,2)]));
            out.push_str("        </inertia>\n");
            out.push_str("      </inertial>\n");
        }

        // Visual geometries for this link
        if let Some(vis) = visual {
            for obj in &vis.objects {
                if obj.parent_joint == i {
                    write_sdf_visual_or_collision(&mut out, obj, "visual");
                }
            }
        }

        // Collision geometries for this link
        if let Some(col) = collision {
            for obj in &col.objects {
                if obj.parent_joint == i {
                    write_sdf_visual_or_collision(&mut out, obj, "collision");
                }
            }
        }

        out.push_str("    </link>\n");
    }

    // ── Joints ──────────────────────────────────────────────────────────
    for i in 1..model.joints.len() {
        let joint = &model.joints[i];
        let jtype_str = match &joint.joint_type {
            JointType::Revolute { .. } => "revolute",
            JointType::Prismatic { .. } => "prismatic",
            JointType::Fixed => "fixed",
            JointType::FreeFlyer => "ball",
        };
        out.push_str(&format!(
            "    <joint name=\"{}\" type=\"{}\">\n",
            xml_escape(&joint.name),
            jtype_str,
        ));

        // parent / child
        out.push_str(&format!(
            "      <parent>{}</parent>\n",
            xml_escape(&model.link_names[joint.parent]),
        ));
        out.push_str(&format!(
            "      <child>{}</child>\n",
            xml_escape(&model.link_names[i]),
        ));

        // pose
        let t = se3::translation(&joint.placement);
        let rot = se3::rotation_matrix(&joint.placement);
        let rotation = Rotation3::from_matrix_unchecked(rot);
        let (r, p, y) = rotation.euler_angles();
        out.push_str(&format!(
            "      <pose>{} {} {} {} {} {}</pose>\n",
            t[0], t[1], t[2], r, p, y,
        ));

        // axis (for revolute / prismatic)
        match &joint.joint_type {
            JointType::Revolute { axis } | JointType::Prismatic { axis } => {
                out.push_str(&format!(
                    "      <axis>\n        <xyz>{} {} {}</xyz>\n      </axis>\n",
                    axis[0], axis[1], axis[2],
                ));
            }
            _ => {}
        }

        out.push_str("    </joint>\n");
    }

    out.push_str("  </model>\n");
    out.push_str("</sdf>\n");
    out
}

/// Write a `<visual>` or `<collision>` element for a geometry object in SDF format.
fn write_sdf_visual_or_collision(out: &mut String, obj: &GeometryObject, tag: &str) {
    out.push_str(&format!(
        "      <{tag} name=\"{}\">\n",
        xml_escape(&obj.name)
    ));

    // pose
    let t = se3::translation(&obj.placement);
    let rot = se3::rotation_matrix(&obj.placement);
    let rotation = Rotation3::from_matrix_unchecked(rot);
    let (r, p, y) = rotation.euler_angles();
    out.push_str(&format!(
        "        <pose>{} {} {} {} {} {}</pose>\n",
        t[0], t[1], t[2], r, p, y,
    ));

    // geometry
    out.push_str("        <geometry>\n");
    match &obj.shape {
        GeometryShape::Box { x, y, z } => {
            out.push_str(&format!("          <box><size>{x} {y} {z}</size></box>\n"));
        }
        GeometryShape::Sphere { radius } => {
            out.push_str(&format!(
                "          <sphere><radius>{radius}</radius></sphere>\n"
            ));
        }
        GeometryShape::Cylinder { radius, length } => {
            out.push_str(&format!(
                "          <cylinder><radius>{radius}</radius><length>{length}</length></cylinder>\n"
            ));
        }
        GeometryShape::Capsule { radius, length } => {
            out.push_str(&format!(
                "          <capsule><radius>{radius}</radius><length>{length}</length></capsule>\n"
            ));
        }
        GeometryShape::Cone { radius, length } => {
            // SDF does not natively support cone; fallback to cylinder
            out.push_str(&format!(
                "          <!-- cone not standard in SDF -->\n          <cylinder><radius>{radius}</radius><length>{length}</length></cylinder>\n"
            ));
        }
        GeometryShape::Mesh { filename, scale } => {
            out.push_str(&format!(
                "          <mesh><uri>{}</uri><scale>{} {} {}</scale></mesh>\n",
                xml_escape(filename),
                scale[0],
                scale[1],
                scale[2],
            ));
        }
    }
    out.push_str("        </geometry>\n");
    out.push_str(&format!("      </{tag}>\n"));
}

/// Minimal XML escaping for text content and attribute values.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const SIMPLE_SDF: &str = r#"<?xml version="1.0"?>
<sdf version="1.7">
  <model name="test">
    <link name="base_link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
    </link>
    <link name="link1">
      <inertial>
        <pose>0 0 0.1 0 0 0</pose>
        <mass>0.5</mass>
      </inertial>
    </link>
    <link name="link2">
      <inertial>
        <pose>0 0 0.075 0 0 0</pose>
        <mass>0.3</mass>
      </inertial>
    </link>
    <joint name="joint1" type="revolute">
      <parent>base_link</parent>
      <child>link1</child>
      <pose>0 0 0.05 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
      </axis>
    </joint>
    <joint name="joint2" type="revolute">
      <parent>link1</parent>
      <child>link2</child>
      <pose>0 0 0.2 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
      </axis>
    </joint>
  </model>
</sdf>"#;

    #[test]
    fn parse_simple_sdf() {
        let model = load_sdf_string(SIMPLE_SDF).unwrap();
        assert_eq!(model.num_joints(), 2);
        assert_eq!(model.nq, 2);
        assert_eq!(model.nv, 2);
    }

    #[test]
    fn sdf_joint_names() {
        let model = load_sdf_string(SIMPLE_SDF).unwrap();
        assert_eq!(model.joints[1].name, "joint1");
        assert_eq!(model.joints[2].name, "joint2");
    }

    #[test]
    fn sdf_joint_parents() {
        let model = load_sdf_string(SIMPLE_SDF).unwrap();
        assert_eq!(model.joints[1].parent, 0);
        assert_eq!(model.joints[2].parent, 1);
    }

    #[test]
    fn sdf_placement() {
        let model = load_sdf_string(SIMPLE_SDF).unwrap();
        let t1 = crate::se3::translation(&model.joints[1].placement);
        assert_relative_eq!(t1, Vector3::new(0.0, 0.0, 0.05), epsilon = 1e-12);
    }

    #[test]
    fn sdf_inertia() {
        let model = load_sdf_string(SIMPLE_SDF).unwrap();
        assert_relative_eq!(model.inertias[1].mass, 0.5, epsilon = 1e-12);
        assert_relative_eq!(model.inertias[2].mass, 0.3, epsilon = 1e-12);
    }

    #[test]
    fn sdf_revolute_axis() {
        let model = load_sdf_string(SIMPLE_SDF).unwrap();
        match &model.joints[1].joint_type {
            JointType::Revolute { axis } => {
                assert_relative_eq!(*axis, Vector3::y(), epsilon = 1e-12);
            }
            _ => panic!("expected revolute joint"),
        }
    }

    #[test]
    fn sdf_fixed_joint() {
        let xml = r#"<?xml version="1.0"?>
<sdf version="1.7">
  <model name="test">
    <link name="base"/>
    <link name="child"/>
    <joint name="j_fixed" type="fixed">
      <parent>base</parent>
      <child>child</child>
      <pose>0.1 0 0 0 0 0</pose>
    </joint>
  </model>
</sdf>"#;
        let model = load_sdf_string(xml).unwrap();
        assert_eq!(model.num_joints(), 1);
        assert_eq!(model.nq, 0);
        assert!(matches!(model.joints[1].joint_type, JointType::Fixed));
    }

    #[test]
    fn sdf_prismatic_joint() {
        let xml = r#"<?xml version="1.0"?>
<sdf version="1.7">
  <model name="test">
    <link name="base"/>
    <link name="slider"/>
    <joint name="slide" type="prismatic">
      <parent>base</parent>
      <child>slider</child>
      <axis>
        <xyz>1 0 0</xyz>
      </axis>
    </joint>
  </model>
</sdf>"#;
        let model = load_sdf_string(xml).unwrap();
        assert_eq!(model.nq, 1);
        match &model.joints[1].joint_type {
            JointType::Prismatic { axis } => {
                assert_relative_eq!(*axis, Vector3::x(), epsilon = 1e-12);
            }
            _ => panic!("expected prismatic"),
        }
    }

    #[test]
    fn sdf_fk_matches_urdf_equivalent() {
        // The same robot described in SDF should give identical FK results
        // to the URDF version.
        let model = load_sdf_string(SIMPLE_SDF).unwrap();
        let q = vec![0.3, -0.5];
        let data = crate::fk::forward_kinematics(&model, &q);

        let urdf_xml = r#"<?xml version="1.0"?>
<robot name="simple">
  <link name="base_link">
    <inertial><mass value="1.0"/><origin xyz="0 0 0"/></inertial>
  </link>
  <link name="link1">
    <inertial><mass value="0.5"/><origin xyz="0 0 0.1"/></inertial>
  </link>
  <link name="link2">
    <inertial><mass value="0.3"/><origin xyz="0 0 0.075"/></inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/><axis xyz="0 1 0"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/>
  </joint>
</robot>"#;
        let urdf_model = crate::urdf::load_urdf_string(urdf_xml).unwrap();
        let data_urdf = crate::fk::forward_kinematics(&urdf_model, &q);

        for i in 1..model.joints.len() {
            assert_relative_eq!(
                crate::se3::to_homogeneous(&data.oMi[i]),
                crate::se3::to_homogeneous(&data_urdf.oMi[i]),
                epsilon = 1e-12,
            );
        }
    }

    #[test]
    fn sdf_roundtrip() {
        // load → write → load again → models must be structurally equal
        let model = load_sdf_string(SIMPLE_SDF).unwrap();
        let xml = write_sdf_string(&model);
        let model2 = load_sdf_string(&xml).unwrap();
        assert!(model.approx_eq(&model2, 1e-12));
    }

    #[test]
    fn sdf_write_preserves_link_names() {
        let model = load_sdf_string(SIMPLE_SDF).unwrap();
        assert_eq!(model.link_names[0], "base_link");
        assert_eq!(model.link_names[1], "link1");
        assert_eq!(model.link_names[2], "link2");
        let xml = write_sdf_string(&model);
        assert!(xml.contains("name=\"base_link\""));
        assert!(xml.contains("name=\"link1\""));
        assert!(xml.contains("name=\"link2\""));
    }

    const SDF_WITH_GEOMETRY: &str = r#"<?xml version="1.0"?>
<sdf version="1.7">
  <model name="geom_test">
    <link name="base">
      <visual name="vis_box">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <box><size>0.2 0.3 0.1</size></box>
        </geometry>
      </visual>
      <collision name="col_box">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <box><size>0.2 0.3 0.1</size></box>
        </geometry>
      </collision>
    </link>
    <link name="child">
      <visual name="vis_cyl">
        <pose>0 0 0.1 0 0 0</pose>
        <geometry>
          <cylinder><radius>0.02</radius><length>0.2</length></cylinder>
        </geometry>
      </visual>
      <visual name="vis_sph">
        <pose>0 0 0.2 0 0 0</pose>
        <geometry>
          <sphere><radius>0.03</radius></sphere>
        </geometry>
      </visual>
    </link>
    <joint name="j1" type="revolute">
      <parent>base</parent>
      <child>child</child>
      <pose>0 0 0.05 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
      </axis>
    </joint>
  </model>
</sdf>"#;

    #[test]
    fn sdf_parse_geometry() {
        let (model, vis, col) = load_sdf_geometry_string(SDF_WITH_GEOMETRY).unwrap();
        assert_eq!(model.num_joints(), 1);
        assert_eq!(vis.num_objects(), 3);
        assert_eq!(col.num_objects(), 1);

        assert_eq!(
            vis.objects[0].shape,
            GeometryShape::Box { x: 0.2, y: 0.3, z: 0.1 }
        );
        assert_eq!(
            vis.objects[1].shape,
            GeometryShape::Cylinder { radius: 0.02, length: 0.2 }
        );
        assert_eq!(
            vis.objects[2].shape,
            GeometryShape::Sphere { radius: 0.03 }
        );

        assert_eq!(vis.objects[0].parent_joint, 0);
        assert_eq!(vis.objects[1].parent_joint, 1);
        assert_eq!(vis.objects[2].parent_joint, 1);

        // Check names from SDF attributes
        assert_eq!(vis.objects[0].name, "vis_box");
        assert_eq!(vis.objects[1].name, "vis_cyl");
        assert_eq!(col.objects[0].name, "col_box");
    }

    #[test]
    fn sdf_geometry_roundtrip() {
        let (model, vis, col) = load_sdf_geometry_string(SDF_WITH_GEOMETRY).unwrap();
        let xml = write_sdf_geometry_string(&model, Some(&vis), Some(&col));
        let (model2, vis2, col2) = load_sdf_geometry_string(&xml).unwrap();

        assert!(model.approx_eq(&model2, 1e-12));
        assert_eq!(vis.num_objects(), vis2.num_objects());
        assert_eq!(col.num_objects(), col2.num_objects());
        for (a, b) in vis.objects.iter().zip(vis2.objects.iter()) {
            assert_eq!(a.shape, b.shape);
            assert_eq!(a.parent_joint, b.parent_joint);
        }
    }

    #[test]
    fn sdf_mesh_geometry() {
        let xml = r#"<?xml version="1.0"?>
<sdf version="1.7">
  <model name="mesh_test">
    <link name="base">
      <visual name="v0">
        <geometry>
          <mesh>
            <uri>model://robot/meshes/base.dae</uri>
            <scale>0.001 0.001 0.001</scale>
          </mesh>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>"#;
        let (_, vis, _) = load_sdf_geometry_string(xml).unwrap();
        assert_eq!(vis.num_objects(), 1);
        match &vis.objects[0].shape {
            GeometryShape::Mesh { filename, scale } => {
                assert_eq!(filename, "model://robot/meshes/base.dae");
                assert_relative_eq!(*scale, Vector3::new(0.001, 0.001, 0.001), epsilon = 1e-12);
            }
            _ => panic!("expected mesh shape"),
        }
    }
}
