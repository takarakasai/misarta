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

use crate::joint::JointType;
use crate::model::{LinkInertia, Model, ModelBuilder};
use crate::se3;
use nalgebra::{Rotation3, Vector3};
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

        joints.push(JointInfo {
            name,
            joint_type,
            parent_link,
            child_link,
            origin,
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

    Ok(builder.build())
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

        LinkInertia {
            mass,
            center_of_mass: com,
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

// ─── Writer ─────────────────────────────────────────────────────────────────

/// Write a `Model<f64>` to a URDF XML file on disk.
pub fn write_urdf(model: &Model<f64>, path: &std::path::Path) -> Result<(), UrdfError> {
    let xml = write_urdf_string(model);
    std::fs::write(path, xml)
        .map_err(|e| UrdfError::XmlParse(format!("cannot write {}: {e}", path.display())))
}

/// Serialize a `Model<f64>` to a URDF XML string.
pub fn write_urdf_string(model: &Model<f64>) -> String {
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
        {
            out.push_str("    <inertial>\n");
            out.push_str(&format!("      <mass value=\"{}\"/>\n", inertia.mass));
            out.push_str(&format!(
                "      <origin xyz=\"{} {} {}\"/>\n",
                inertia.center_of_mass[0],
                inertia.center_of_mass[1],
                inertia.center_of_mass[2],
            ));
            out.push_str("    </inertial>\n");
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
}
