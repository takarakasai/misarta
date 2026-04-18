//! URDF (Unified Robot Description Format) loader.
//!
//! Parses a URDF XML string and builds a `Model<f64>` using `ModelBuilder`.
//!
//! # Supported elements
//!
//! - `<robot>` — top-level container
//! - `<link>` — rigid body with optional `<inertial>`
//! - `<joint>` — revolute, prismatic, continuous, fixed, floating
//! - `<origin xyz="..." rpy="..."/>` — placement
//! - `<axis xyz="..."/>` — joint axis
//!
//! # Example
//!
//! ```no_run
//! use misarta::urdf;
//! let xml = std::fs::read_to_string("robot.urdf").unwrap();
//! let model = urdf::load_urdf_string(&xml).unwrap();
//! ```

use crate::geometry::{GeometryModel, GeometryObject, GeometryShape};
use crate::joint::JointType;
use crate::model::{LinkInertia, Model, ModelBuilder};
use crate::se3;
use nalgebra::{Matrix3, Rotation3, Vector3};
use roxmltree::Document;
use std::collections::HashMap;

/// Errors arising from URDF parsing.
#[derive(Debug, Clone)]
pub enum UrdfError {
    /// XML is not well-formed.
    XmlParse(String),
    /// Missing required element or attribute.
    MissingElement(String),
    /// Unsupported joint type.
    UnsupportedJointType(String),
    /// Topological sort failed (cycle or missing link).
    Topology(String),
}

impl std::fmt::Display for UrdfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UrdfError::XmlParse(e) => write!(f, "URDF XML parse error: {e}"),
            UrdfError::MissingElement(e) => write!(f, "URDF missing element: {e}"),
            UrdfError::UnsupportedJointType(e) => write!(f, "unsupported joint type: {e}"),
            UrdfError::Topology(e) => write!(f, "URDF topology error: {e}"),
        }
    }
}

impl std::error::Error for UrdfError {}

/// Load a `Model<f64>` from a URDF XML file on disk.
pub fn load_urdf(path: &std::path::Path) -> Result<Model<f64>, UrdfError> {
    let xml = std::fs::read_to_string(path)
        .map_err(|e| UrdfError::XmlParse(format!("cannot read {}: {e}", path.display())))?;
    load_urdf_string(&xml)
}

/// Load a `Model<f64>` together with visual and collision `GeometryModel`s
/// from a URDF XML file on disk.
pub fn load_urdf_geometry(
    path: &std::path::Path,
) -> Result<(Model<f64>, GeometryModel, GeometryModel), UrdfError> {
    let xml = std::fs::read_to_string(path)
        .map_err(|e| UrdfError::XmlParse(format!("cannot read {}: {e}", path.display())))?;
    load_urdf_geometry_string(&xml)
}

/// Load a `Model<f64>` together with visual and collision `GeometryModel`s
/// from a URDF XML string.
pub fn load_urdf_geometry_string(
    xml: &str,
) -> Result<(Model<f64>, GeometryModel, GeometryModel), UrdfError> {
    let doc = Document::parse(xml).map_err(|e| UrdfError::XmlParse(e.to_string()))?;
    let robot = doc.root_element();
    if robot.tag_name().name() != "robot" {
        return Err(UrdfError::MissingElement("root <robot> element".into()));
    }

    // Build kinematic model via the existing parser (re-parse is cheap)
    let model = load_urdf_string(xml)?;

    // Build link_name → joint index map from the model
    let mut link_to_idx: HashMap<&str, usize> = HashMap::new();
    for (i, name) in model.link_names.iter().enumerate() {
        link_to_idx.insert(name.as_str(), i);
    }

    let mut visual_model = GeometryModel::new();
    let mut collision_model = GeometryModel::new();

    for link_el in robot.children().filter(|n| n.tag_name().name() == "link") {
        let link_name = link_el
            .attribute("name")
            .ok_or_else(|| UrdfError::MissingElement("link name".into()))?;
        let joint_idx = *link_to_idx
            .get(link_name)
            .ok_or_else(|| UrdfError::Topology(format!("link '{link_name}' not in model")))?;

        // Visual geometries
        for (vi, vis_el) in link_el
            .children()
            .filter(|n| n.tag_name().name() == "visual")
            .enumerate()
        {
            let placement = parse_origin_element(&vis_el);
            if let Some(shape) = parse_urdf_geometry(&vis_el) {
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
            let placement = parse_origin_element(&col_el);
            if let Some(shape) = parse_urdf_geometry(&col_el) {
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

/// Load a `Model<f64>` from a URDF XML string.
pub fn load_urdf_string(xml: &str) -> Result<Model<f64>, UrdfError> {
    let doc = Document::parse(xml).map_err(|e| UrdfError::XmlParse(e.to_string()))?;
    let robot = doc
        .root_element();
    if robot.tag_name().name() != "robot" {
        return Err(UrdfError::MissingElement("root <robot> element".into()));
    }

    // ── Collect links ───────────────────────────────────────────────────
    // Map link_name → LinkInertia
    let mut link_inertias: HashMap<String, LinkInertia<f64>> = HashMap::new();
    for link_el in robot.children().filter(|n| n.tag_name().name() == "link") {
        let name = link_el
            .attribute("name")
            .ok_or_else(|| UrdfError::MissingElement("link name".into()))?
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
        origin: nalgebra::Isometry3<f64>,
        /// URDF `<mimic joint="..." multiplier="..." offset="..."/>`
        mimic: Option<(String, f64, f64)>,
    }

    let mut joints: Vec<JointInfo> = Vec::new();
    for joint_el in robot.children().filter(|n| n.tag_name().name() == "joint") {
        let name = joint_el
            .attribute("name")
            .ok_or_else(|| UrdfError::MissingElement("joint name".into()))?
            .to_string();
        let jtype_str = joint_el
            .attribute("type")
            .ok_or_else(|| UrdfError::MissingElement(format!("joint type for '{name}'")))?;

        let parent_link = joint_el
            .children()
            .find(|n| n.tag_name().name() == "parent")
            .and_then(|n| n.attribute("link"))
            .ok_or_else(|| UrdfError::MissingElement(format!("parent link for '{name}'")))?
            .to_string();

        let child_link = joint_el
            .children()
            .find(|n| n.tag_name().name() == "child")
            .and_then(|n| n.attribute("link"))
            .ok_or_else(|| UrdfError::MissingElement(format!("child link for '{name}'")))?
            .to_string();

        let origin = parse_origin_element(&joint_el);

        let axis = parse_axis_element(&joint_el);

        let joint_type = match jtype_str {
            "revolute" | "continuous" => JointType::Revolute { axis },
            "prismatic" => JointType::Prismatic { axis },
            "fixed" => JointType::Fixed,
            "floating" => JointType::FreeFlyer,
            other => return Err(UrdfError::UnsupportedJointType(other.to_string())),
        };

        // Parse optional <mimic joint="..." multiplier="..." offset="..."/>
        let mimic = joint_el
            .children()
            .find(|n| n.tag_name().name() == "mimic")
            .map(|mimic_el| {
                let master_name = mimic_el
                    .attribute("joint")
                    .unwrap_or("")
                    .to_string();
                let multiplier: f64 = mimic_el
                    .attribute("multiplier")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1.0);
                let offset: f64 = mimic_el
                    .attribute("offset")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                (master_name, multiplier, offset)
            });

        joints.push(JointInfo {
            name,
            joint_type,
            parent_link,
            child_link,
            origin,
            mimic,
        });
    }

    // ── Find root link (not a child of any joint) ───────────────────────
    let child_links: std::collections::HashSet<&str> =
        joints.iter().map(|j| j.child_link.as_str()).collect();
    let root_link = link_inertias
        .keys()
        .find(|name| !child_links.contains(name.as_str()))
        .ok_or_else(|| UrdfError::Topology("no root link found".into()))?
        .clone();

    // ── Build model via topological traversal ───────────────────────────
    // link_name → model joint index
    let mut link_to_idx: HashMap<String, usize> = HashMap::new();
    link_to_idx.insert(root_link.clone(), 0); // root link ↔ universe joint

    // BFS / topological order
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(root_link.clone());
    let mut ordered_joints: Vec<&JointInfo> = Vec::new();

    while let Some(parent_name) = queue.pop_front() {
        for ji in &joints {
            if ji.parent_link == parent_name && !link_to_idx.contains_key(&ji.child_link) {
                ordered_joints.push(ji);
                // Reserve an index for this child link
                let idx = link_to_idx.len();
                link_to_idx.insert(ji.child_link.clone(), idx);
                queue.push_back(ji.child_link.clone());
            }
        }
    }

    // Check all joints were visited
    if ordered_joints.len() != joints.len() {
        return Err(UrdfError::Topology(
            "some joints could not be reached from root link".into(),
        ));
    }

    let robot_name = robot.attribute("name").unwrap_or("").to_string();
    let mut builder = ModelBuilder::new()
        .name(robot_name)
        .root_link_name(root_link.clone());
    for ji in &ordered_joints {
        let parent_idx = link_to_idx[&ji.parent_link];
        let inertia = link_inertias
            .get(&ji.child_link)
            .cloned()
            .unwrap_or_else(LinkInertia::zero);

        builder = builder.add_joint_with_link(
            ji.name.clone(),
            parent_idx,
            ji.joint_type.clone(),
            ji.origin,
            inertia,
            ji.child_link.clone(),
        );
    }

    // ── Apply mimic constraints ─────────────────────────────────────────
    // We need to resolve master joint names to model indices.
    // Build joint_name → model index map.
    let model_tmp = builder.build();
    let joint_name_to_idx: HashMap<&str, usize> = model_tmp
        .joints
        .iter()
        .enumerate()
        .map(|(i, j)| (j.name.as_str(), i))
        .collect();

    let mut builder2 = ModelBuilder::from_model(&model_tmp);
    for ji in &ordered_joints {
        if let Some((ref master_name, multiplier, offset)) = ji.mimic {
            let slave_idx = *joint_name_to_idx.get(ji.name.as_str()).ok_or_else(|| {
                UrdfError::Topology(format!("mimic slave joint '{}' not found", ji.name))
            })?;
            let master_idx = *joint_name_to_idx.get(master_name.as_str()).ok_or_else(|| {
                UrdfError::Topology(format!(
                    "mimic master joint '{}' (referenced by '{}') not found",
                    master_name, ji.name
                ))
            })?;
            builder2 = builder2.add_mimic(slave_idx, master_idx, multiplier, offset);
        }
    }

    Ok(builder2.build())
}

// ─── Internal helpers ───────────────────────────────────────────────────────

/// Parse an `<origin xyz="..." rpy="..."/>` beneath a parent node.
fn parse_origin_element(parent: &roxmltree::Node) -> nalgebra::Isometry3<f64> {
    if let Some(origin_el) = parent.children().find(|n| n.tag_name().name() == "origin") {
        let xyz = parse_vec3(origin_el.attribute("xyz").unwrap_or("0 0 0"));
        let rpy = parse_vec3(origin_el.attribute("rpy").unwrap_or("0 0 0"));
        let rot = Rotation3::from_euler_angles(rpy[0], rpy[1], rpy[2]);
        se3::from_rotation_and_translation(&rot, &xyz)
    } else {
        se3::identity()
    }
}

/// Parse an `<axis xyz="..."/>` element, defaulting to Z.
fn parse_axis_element(parent: &roxmltree::Node) -> Vector3<f64> {
    if let Some(axis_el) = parent.children().find(|n| n.tag_name().name() == "axis") {
        let v = parse_vec3(axis_el.attribute("xyz").unwrap_or("0 0 1"));
        let n = v.norm();
        if n > 1e-12 {
            v / n
        } else {
            Vector3::z()
        }
    } else {
        Vector3::z()
    }
}

/// Parse `<inertial>` for a link.
fn parse_link_inertia(link_el: &roxmltree::Node) -> LinkInertia<f64> {
    if let Some(inertial) = link_el.children().find(|n| n.tag_name().name() == "inertial") {
        let mass = inertial
            .children()
            .find(|n| n.tag_name().name() == "mass")
            .and_then(|n| n.attribute("value"))
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);

        let com = if let Some(origin) = inertial
            .children()
            .find(|n| n.tag_name().name() == "origin")
        {
            parse_vec3(origin.attribute("xyz").unwrap_or("0 0 0"))
        } else {
            Vector3::zeros()
        };

        let rotational_inertia = if let Some(inertia_el) = inertial
            .children()
            .find(|n| n.tag_name().name() == "inertia")
        {
            let ixx = inertia_el.attribute("ixx").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            let ixy = inertia_el.attribute("ixy").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            let ixz = inertia_el.attribute("ixz").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            let iyy = inertia_el.attribute("iyy").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            let iyz = inertia_el.attribute("iyz").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
            let izz = inertia_el.attribute("izz").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
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

/// Parse a whitespace-separated triple "x y z" into a Vector3.
fn parse_vec3(s: &str) -> Vector3<f64> {
    let vals: Vec<f64> = s.split_whitespace().filter_map(|v| v.parse().ok()).collect();
    if vals.len() >= 3 {
        Vector3::new(vals[0], vals[1], vals[2])
    } else {
        Vector3::zeros()
    }
}

/// Parse a URDF `<geometry>` child element into a `GeometryShape`.
fn parse_urdf_geometry(parent: &roxmltree::Node) -> Option<GeometryShape> {
    let geom_el = parent.children().find(|n| n.tag_name().name() == "geometry")?;

    for child in geom_el.children() {
        match child.tag_name().name() {
            "box" => {
                let size = parse_vec3(child.attribute("size").unwrap_or("0 0 0"));
                return Some(GeometryShape::Box {
                    x: size[0],
                    y: size[1],
                    z: size[2],
                });
            }
            "sphere" => {
                let r = child
                    .attribute("radius")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                return Some(GeometryShape::Sphere { radius: r });
            }
            "cylinder" => {
                let r = child
                    .attribute("radius")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                let l = child
                    .attribute("length")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                return Some(GeometryShape::Cylinder { radius: r, length: l });
            }
            "capsule" => {
                let r = child
                    .attribute("radius")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                let l = child
                    .attribute("length")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                return Some(GeometryShape::Capsule { radius: r, length: l });
            }
            "mesh" => {
                let filename = child
                    .attribute("filename")
                    .unwrap_or("")
                    .to_string();
                let scale = child
                    .attribute("scale")
                    .map(|s| parse_vec3(s))
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

/// Write a `Model<f64>` to a URDF XML file on disk.
pub fn write_urdf(model: &Model<f64>, path: &std::path::Path) -> Result<(), UrdfError> {
    let xml = write_urdf_string(model);
    std::fs::write(path, xml)
        .map_err(|e| UrdfError::XmlParse(format!("cannot write {}: {e}", path.display())))
}

/// Serialize a `Model<f64>` to a URDF XML string.
pub fn write_urdf_string(model: &Model<f64>) -> String {
    write_urdf_geometry_string(model, None, None)
}

/// Serialize a `Model<f64>` with optional visual/collision geometry to a URDF XML string.
pub fn write_urdf_geometry_string(
    model: &Model<f64>,
    visual: Option<&GeometryModel>,
    collision: Option<&GeometryModel>,
) -> String {
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\"?>\n");
    out.push_str(&format!("<robot name=\"{}\">\n", xml_escape(&model.name)));

    // ── Links ───────────────────────────────────────────────────────────
    for (i, link_name) in model.link_names.iter().enumerate() {
        out.push_str(&format!("  <link name=\"{}\">\n", xml_escape(link_name)));
        let inertia = &model.inertias[i];
        if inertia.mass != 0.0
            || inertia.center_of_mass[0] != 0.0
            || inertia.center_of_mass[1] != 0.0
            || inertia.center_of_mass[2] != 0.0
            || inertia.rotational_inertia.norm() != 0.0
        {
            out.push_str("    <inertial>\n");
            out.push_str(&format!("      <mass value=\"{}\"/>\n", inertia.mass));
            out.push_str(&format!(
                "      <origin xyz=\"{} {} {}\"/>\n",
                inertia.center_of_mass[0],
                inertia.center_of_mass[1],
                inertia.center_of_mass[2],
            ));
            let ri = &inertia.rotational_inertia;
            out.push_str(&format!(
                "      <inertia ixx=\"{}\" ixy=\"{}\" ixz=\"{}\" iyy=\"{}\" iyz=\"{}\" izz=\"{}\"/>\n",
                ri[(0, 0)], ri[(0, 1)], ri[(0, 2)], ri[(1, 1)], ri[(1, 2)], ri[(2, 2)],
            ));
            out.push_str("    </inertial>\n");
        }

        // Visual geometries for this link
        if let Some(vis) = visual {
            for obj in &vis.objects {
                if obj.parent_joint == i {
                    write_urdf_visual_or_collision(&mut out, obj, "visual");
                }
            }
        }

        // Collision geometries for this link
        if let Some(col) = collision {
            for obj in &col.objects {
                if obj.parent_joint == i {
                    write_urdf_visual_or_collision(&mut out, obj, "collision");
                }
            }
        }

        out.push_str("  </link>\n");
    }

    // ── Joints ──────────────────────────────────────────────────────────
    for i in 1..model.joints.len() {
        let joint = &model.joints[i];
        let jtype_str = match &joint.joint_type {
            JointType::Revolute { .. } => "revolute",
            JointType::Prismatic { .. } => "prismatic",
            JointType::Fixed => "fixed",
            JointType::FreeFlyer => "floating",
        };
        out.push_str(&format!(
            "  <joint name=\"{}\" type=\"{}\">\n",
            xml_escape(&joint.name),
            jtype_str,
        ));

        // parent / child
        out.push_str(&format!(
            "    <parent link=\"{}\"/>\n",
            xml_escape(&model.link_names[joint.parent]),
        ));
        out.push_str(&format!(
            "    <child link=\"{}\"/>\n",
            xml_escape(&model.link_names[i]),
        ));

        // origin
        let t = se3::translation(&joint.placement);
        let rot = se3::rotation_matrix(&joint.placement);
        let rotation = Rotation3::from_matrix_unchecked(rot);
        let (r, p, y) = rotation.euler_angles();
        out.push_str(&format!(
            "    <origin xyz=\"{} {} {}\" rpy=\"{} {} {}\"/>\n",
            t[0], t[1], t[2], r, p, y,
        ));

        // axis (for revolute / prismatic)
        match &joint.joint_type {
            JointType::Revolute { axis } | JointType::Prismatic { axis } => {
                out.push_str(&format!(
                    "    <axis xyz=\"{} {} {}\"/>\n",
                    axis[0], axis[1], axis[2],
                ));
            }
            _ => {}
        }

        out.push_str("  </joint>\n");
    }

    out.push_str("</robot>\n");
    out
}

/// Write a `<visual>` or `<collision>` element for a geometry object.
fn write_urdf_visual_or_collision(out: &mut String, obj: &GeometryObject, tag: &str) {
    out.push_str(&format!("    <{tag}>\n"));

    // origin
    let t = se3::translation(&obj.placement);
    let rot = se3::rotation_matrix(&obj.placement);
    let rotation = Rotation3::from_matrix_unchecked(rot);
    let (r, p, y) = rotation.euler_angles();
    out.push_str(&format!(
        "      <origin xyz=\"{} {} {}\" rpy=\"{} {} {}\"/>\n",
        t[0], t[1], t[2], r, p, y,
    ));

    // geometry
    out.push_str("      <geometry>\n");
    match &obj.shape {
        GeometryShape::Box { x, y, z } => {
            out.push_str(&format!("        <box size=\"{x} {y} {z}\"/>\n"));
        }
        GeometryShape::Sphere { radius } => {
            out.push_str(&format!("        <sphere radius=\"{radius}\"/>\n"));
        }
        GeometryShape::Cylinder { radius, length } => {
            out.push_str(&format!(
                "        <cylinder radius=\"{radius}\" length=\"{length}\"/>\n"
            ));
        }
        GeometryShape::Capsule { radius, length } => {
            out.push_str(&format!(
                "        <capsule radius=\"{radius}\" length=\"{length}\"/>\n"
            ));
        }
        GeometryShape::Cone { radius, length } => {
            // URDF does not natively support cone; write as a comment + cylinder fallback
            out.push_str(&format!(
                "        <!-- cone not standard in URDF -->\n        <cylinder radius=\"{radius}\" length=\"{length}\"/>\n"
            ));
        }
        GeometryShape::Mesh { filename, scale } => {
            out.push_str(&format!(
                "        <mesh filename=\"{}\" scale=\"{} {} {}\"/>\n",
                xml_escape(filename),
                scale[0],
                scale[1],
                scale[2],
            ));
        }
    }
    out.push_str("      </geometry>\n");
    out.push_str(&format!("    </{tag}>\n"));
}

/// Minimal XML escaping for attribute values.
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

    const SIMPLE_URDF: &str = r#"<?xml version="1.0"?>
<robot name="simple">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
    </inertial>
  </link>
  <link name="link1">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0.1"/>
    </inertial>
  </link>
  <link name="link2">
    <inertial>
      <mass value="0.3"/>
      <origin xyz="0 0 0.075"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>"#;

    #[test]
    fn parse_simple_urdf() {
        let model = load_urdf_string(SIMPLE_URDF).unwrap();
        assert_eq!(model.num_joints(), 2);
        assert_eq!(model.nq, 2);
        assert_eq!(model.nv, 2);
    }

    #[test]
    fn urdf_joint_names() {
        let model = load_urdf_string(SIMPLE_URDF).unwrap();
        assert_eq!(model.joints[1].name, "joint1");
        assert_eq!(model.joints[2].name, "joint2");
    }

    #[test]
    fn urdf_joint_parents() {
        let model = load_urdf_string(SIMPLE_URDF).unwrap();
        assert_eq!(model.joints[1].parent, 0); // parent = universe (base_link)
        assert_eq!(model.joints[2].parent, 1); // parent = joint1 (link1)
    }

    #[test]
    fn urdf_placement() {
        let model = load_urdf_string(SIMPLE_URDF).unwrap();
        let t1 = crate::se3::translation(&model.joints[1].placement);
        assert_relative_eq!(t1, Vector3::new(0.0, 0.0, 0.05), epsilon = 1e-12);

        let t2 = crate::se3::translation(&model.joints[2].placement);
        assert_relative_eq!(t2, Vector3::new(0.0, 0.0, 0.2), epsilon = 1e-12);
    }

    #[test]
    fn urdf_inertia() {
        let model = load_urdf_string(SIMPLE_URDF).unwrap();
        assert_relative_eq!(model.inertias[1].mass, 0.5, epsilon = 1e-12);
        assert_relative_eq!(model.inertias[2].mass, 0.3, epsilon = 1e-12);
    }

    #[test]
    fn urdf_revolute_axis() {
        let model = load_urdf_string(SIMPLE_URDF).unwrap();
        match &model.joints[1].joint_type {
            JointType::Revolute { axis } => {
                assert_relative_eq!(*axis, Vector3::y(), epsilon = 1e-12);
            }
            _ => panic!("expected revolute joint"),
        }
    }

    #[test]
    fn urdf_fixed_joint() {
        let xml = r#"<?xml version="1.0"?>
<robot name="fixed_test">
  <link name="base"/>
  <link name="child"/>
  <joint name="j_fixed" type="fixed">
    <parent link="base"/>
    <child link="child"/>
    <origin xyz="0.1 0 0"/>
  </joint>
</robot>"#;
        let model = load_urdf_string(xml).unwrap();
        assert_eq!(model.num_joints(), 1);
        assert_eq!(model.nq, 0);
        assert_eq!(model.nv, 0);
        assert!(matches!(model.joints[1].joint_type, JointType::Fixed));
    }

    #[test]
    fn urdf_prismatic_joint() {
        let xml = r#"<?xml version="1.0"?>
<robot name="prismatic_test">
  <link name="base"/>
  <link name="slider"/>
  <joint name="slide" type="prismatic">
    <parent link="base"/>
    <child link="slider"/>
    <axis xyz="1 0 0"/>
  </joint>
</robot>"#;
        let model = load_urdf_string(xml).unwrap();
        assert_eq!(model.nq, 1);
        match &model.joints[1].joint_type {
            JointType::Prismatic { axis } => {
                assert_relative_eq!(*axis, Vector3::x(), epsilon = 1e-12);
            }
            _ => panic!("expected prismatic joint"),
        }
    }

    #[test]
    fn urdf_continuous_parsed_as_revolute() {
        let xml = r#"<?xml version="1.0"?>
<robot name="cont_test">
  <link name="base"/>
  <link name="wheel"/>
  <joint name="wheel_joint" type="continuous">
    <parent link="base"/>
    <child link="wheel"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>"#;
        let model = load_urdf_string(xml).unwrap();
        assert!(matches!(
            model.joints[1].joint_type,
            JointType::Revolute { .. }
        ));
    }

    #[test]
    fn urdf_fk_matches_manual() {
        // FK from URDF should match a manually-built model.
        let model = load_urdf_string(SIMPLE_URDF).unwrap();
        let q = vec![0.3, -0.5];
        let data = crate::fk::forward_kinematics(&model, &q);

        // Build the equivalent model by hand.
        let offset1 = crate::se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(0.0, 0.0, 0.05),
        );
        let offset2 = crate::se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(0.0, 0.0, 0.2),
        );
        let manual = ModelBuilder::new()
            .add_joint(
                "j1",
                0,
                JointType::Revolute {
                    axis: Vector3::y(),
                },
                offset1,
                LinkInertia::zero(),
            )
            .add_joint(
                "j2",
                1,
                JointType::Revolute {
                    axis: Vector3::y(),
                },
                offset2,
                LinkInertia::zero(),
            )
            .build();
        let data_manual = crate::fk::forward_kinematics(&manual, &q);

        for i in 1..model.joints.len() {
            assert_relative_eq!(
                crate::se3::to_homogeneous(&data.oMi[i]),
                crate::se3::to_homogeneous(&data_manual.oMi[i]),
                epsilon = 1e-12,
            );
        }
    }

    #[test]
    fn urdf_roundtrip() {
        // load → write → load again → models must be structurally equal
        let model = load_urdf_string(SIMPLE_URDF).unwrap();
        let xml = write_urdf_string(&model);
        let model2 = load_urdf_string(&xml).unwrap();
        assert!(model.approx_eq(&model2, 1e-12));
    }

    #[test]
    fn urdf_write_preserves_link_names() {
        let model = load_urdf_string(SIMPLE_URDF).unwrap();
        assert_eq!(model.link_names[0], "base_link");
        assert_eq!(model.link_names[1], "link1");
        assert_eq!(model.link_names[2], "link2");
        let xml = write_urdf_string(&model);
        assert!(xml.contains("name=\"base_link\""));
        assert!(xml.contains("name=\"link1\""));
        assert!(xml.contains("name=\"link2\""));
    }

    const URDF_WITH_GEOMETRY: &str = r#"<?xml version="1.0"?>
<robot name="geom_test">
  <link name="base">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.3 0.1"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.3 0.1"/>
      </geometry>
    </collision>
  </link>
  <link name="child">
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.02" length="0.2"/>
      </geometry>
    </visual>
    <visual>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.03"/>
      </geometry>
    </visual>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="child"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>"#;

    #[test]
    fn urdf_parse_geometry() {
        let (model, vis, col) = load_urdf_geometry_string(URDF_WITH_GEOMETRY).unwrap();
        assert_eq!(model.num_joints(), 1);
        assert_eq!(vis.num_objects(), 3);  // 1 box + 1 cylinder + 1 sphere
        assert_eq!(col.num_objects(), 1);  // 1 box

        // Check shapes
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

        // Check parent joints
        assert_eq!(vis.objects[0].parent_joint, 0); // base
        assert_eq!(vis.objects[1].parent_joint, 1); // child
        assert_eq!(vis.objects[2].parent_joint, 1); // child
    }

    #[test]
    fn urdf_geometry_roundtrip() {
        let (model, vis, col) = load_urdf_geometry_string(URDF_WITH_GEOMETRY).unwrap();
        let xml = write_urdf_geometry_string(&model, Some(&vis), Some(&col));
        let (model2, vis2, col2) = load_urdf_geometry_string(&xml).unwrap();

        assert!(model.approx_eq(&model2, 1e-12));
        assert_eq!(vis.num_objects(), vis2.num_objects());
        assert_eq!(col.num_objects(), col2.num_objects());
        for (a, b) in vis.objects.iter().zip(vis2.objects.iter()) {
            assert_eq!(a.shape, b.shape);
            assert_eq!(a.parent_joint, b.parent_joint);
        }
        for (a, b) in col.objects.iter().zip(col2.objects.iter()) {
            assert_eq!(a.shape, b.shape);
            assert_eq!(a.parent_joint, b.parent_joint);
        }
    }

    #[test]
    fn urdf_mesh_geometry() {
        let xml = r#"<?xml version="1.0"?>
<robot name="mesh_test">
  <link name="base">
    <visual>
      <geometry>
        <mesh filename="package://robot/meshes/base.stl" scale="0.001 0.001 0.001"/>
      </geometry>
    </visual>
  </link>
</robot>"#;
        let (_, vis, _) = load_urdf_geometry_string(xml).unwrap();
        assert_eq!(vis.num_objects(), 1);
        match &vis.objects[0].shape {
            GeometryShape::Mesh { filename, scale } => {
                assert_eq!(filename, "package://robot/meshes/base.stl");
                assert_relative_eq!(*scale, Vector3::new(0.001, 0.001, 0.001), epsilon = 1e-12);
            }
            _ => panic!("expected mesh shape"),
        }
        assert_eq!(
            vis.objects[0].mesh_path.as_deref(),
            Some("package://robot/meshes/base.stl")
        );
    }
}
