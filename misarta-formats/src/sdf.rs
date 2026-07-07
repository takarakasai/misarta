//! SDF (Simulation Description Format) loader — standalone misarta layer.
//!
//! Parses an SDF XML string (version 1.5–1.8) and builds a `Model<f64>`
//! plus optional `GeometryModel`s. Targeted at downstream consumers of
//! misarta who don't depend on articara — e.g. headless analysis tools,
//! research code, or alternative front-ends.
//!
//! # Layering
//!
//! The single SDF parser is [`import_str`] (SDF → [`MisaFile`]); both
//! this module's `Model` loaders and articara's `RobotModel` import go
//! through it, so structural parsing has exactly one implementation
//! (the historical dual-parser setup was unified in A5, see articara
//! `doc/refactor_20260702.md` §4.7).
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
//! use misarta_formats::sdf;
//! let xml = std::fs::read_to_string("robot.sdf").unwrap();
//! let model = sdf::load_sdf_string(&xml).unwrap();
//! ```

use misarta::geometry::{GeometryModel, GeometryObject, GeometryShape};
use misarta::joint::JointType;
use misarta::model::Model;
use misarta::se3;
use nalgebra::Rotation3;
#[cfg(test)]
use nalgebra::Vector3;
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
///
/// Thin wrapper over the `.misa` conversion pipeline: the SDF is first
/// converted to a [`MisaFile`] via [`import_str`], then realised with
/// [`misarta::native::build_model`] — the same path the hosts use, so
/// the tree has exactly one SDF parser. Two consequences vs. the old
/// dedicated loader: geometry objects follow the native
/// `<link>_visual_<i>` / `<link>_collision_<i>` naming (SDF `name`
/// attributes are not part of the `.misa` schema), and mesh `<uri>`s
/// are normalised to SDF-directory-relative paths (`model://x` → `x`,
/// `package://pkg/rel` → `../rel`, `file:///abs` → `/abs`).
pub fn load_sdf_geometry_string(
    xml: &str,
) -> Result<(Model<f64>, GeometryModel, GeometryModel), SdfError> {
    let import = import_str(xml).map_err(import_err_to_sdf)?;
    misarta::native::build_model(&import.file)
        .map_err(|e| SdfError::Topology(e.to_string()))
}

/// Load a `Model<f64>` from an SDF XML string. See
/// [`load_sdf_geometry_string`] for the conversion pipeline.
///
/// If the SDF contains multiple `<model>` elements, only the first is loaded.
pub fn load_sdf_string(xml: &str) -> Result<Model<f64>, SdfError> {
    Ok(load_sdf_geometry_string(xml)?.0)
}

/// Map an [`import_str`] error string onto the closest [`SdfError`]
/// variant so the loader API keeps its typed errors.
fn import_err_to_sdf(e: String) -> SdfError {
    if let Some(msg) = e.strip_prefix("Parse SDF XML: ") {
        SdfError::XmlParse(msg.to_string())
    } else if e.starts_with("No <model>") {
        SdfError::MissingElement("<model> element".into())
    } else {
        SdfError::Topology(e)
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
        let t1 = misarta::se3::translation(&model.joints[1].placement);
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
        let data = misarta::fk::forward_kinematics(&model, &q);

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
        let data_urdf = misarta::fk::forward_kinematics(&urdf_model, &q);

        for i in 1..model.joints.len() {
            assert_relative_eq!(
                misarta::se3::to_homogeneous(&data.oMi[i]),
                misarta::se3::to_homogeneous(&data_urdf.oMi[i]),
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

        // Unified pipeline: objects use the native `<link>_visual_<i>`
        // naming — SDF `name` attributes are not part of the `.misa`
        // schema, so they are not preserved.
        assert_eq!(vis.objects[0].name, "base_visual_0");
        assert_eq!(vis.objects[1].name, "child_visual_0");
        assert_eq!(col.objects[0].name, "base_collision_0");
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
                // `model://` URIs are normalised to SDF-relative paths.
                assert_eq!(filename, "robot/meshes/base.dae");
                assert_relative_eq!(*scale, Vector3::new(0.001, 0.001, 0.001), epsilon = 1e-12);
            }
            _ => panic!("expected mesh shape"),
        }
    }
}

// ═══════════════════ MisaFile conversion (A5) ═══════════════════════════
//
// Ported from articara's `src/sdf.rs`: a richer SDF layer that converts
// to / from the `.misa` master schema (sensors, mimics, materials),
// complementing the direct `Model<f64>` loader above. Mesh files are
// never touched — `<uri>` references are normalised to paths relative to
// the SDF file's directory (`package://pkg/rel` → `../rel`, `model://x`
// → `x`, `file:///abs` → `/abs`) so the standard `.misa` base-directory
// rule resolves them; emission takes `Geom::Mesh.file` verbatim (path
// policy is the host's concern).

use misarta::native as mn;
use misarta::native::MisaFile;
use std::path::Path;

use crate::util::{fmt, origin_rotation, parse_f64_list, resolve_visual_rgba};

/// Result of a successful SDF import: the converted [`MisaFile`] plus
/// non-fatal conversion notes (approximated joint kinds).
#[derive(Debug, Clone)]
pub struct SdfImport {
    pub file: MisaFile,
    pub warnings: Vec<String>,
}

/// Parse an SDF file on disk into a [`MisaFile`].
pub fn import(path: &Path) -> Result<SdfImport, String> {
    let xml = std::fs::read_to_string(path).map_err(|e| format!("Read SDF: {e}"))?;
    import_str(&xml)
}

/// Parse SDF XML text into a [`MisaFile`].
pub fn import_str(xml: &str) -> Result<SdfImport, String> {
    let doc = Document::parse(xml).map_err(|e| format!("Parse SDF XML: {e}"))?;

    let model_el = doc
        .descendants()
        .find(|n| n.tag_name().name() == "model")
        .ok_or("No <model> element found in SDF")?;

    let mut file = MisaFile::new(model_el.attribute("name").unwrap_or("sdf_model"), "");
    let mut warnings: Vec<String> = Vec::new();
    let mut child_links: std::collections::HashSet<String> = std::collections::HashSet::new();

    // ── Links ───────────────────────────────────────────────────────────
    for link_el in model_el
        .children()
        .filter(|n| n.tag_name().name() == "link")
    {
        let name = link_el.attribute("name").unwrap_or("link").to_string();

        let visual: Vec<mn::Visual> = link_el
            .children()
            .filter(|n| n.tag_name().name() == "visual")
            .map(|v| mn::Visual {
                origin: parse_pose_origin(v),
                geom: parse_geometry(v),
                color: parse_visual_color(v),
                material: None,
            })
            .collect();

        let collision: Vec<mn::Collision> = link_el
            .children()
            .filter(|n| n.tag_name().name() == "collision")
            .map(|c| mn::Collision {
                origin: parse_pose_origin(c),
                geom: parse_geometry(c),
                physics: None,
            })
            .collect();

        for sensor_el in link_el
            .children()
            .filter(|n| n.tag_name().name() == "sensor")
        {
            if let Some(s) = parse_sensor(sensor_el, &name) {
                file.sensor.push(s);
            }
        }

        file.link.push(mn::Link {
            name,
            description: String::new(),
            inertial: parse_inertial(link_el),
            visual,
            collision,
            collision_enabled: true,
        });
    }

    // ── Joints ──────────────────────────────────────────────────────────
    for joint_el in model_el
        .children()
        .filter(|n| n.tag_name().name() == "joint")
    {
        let name = joint_el.attribute("name").unwrap_or("joint").to_string();
        let jtype = joint_el.attribute("type").unwrap_or("fixed");
        let kind = match jtype {
            "revolute" => mn::JointKind::Revolute,
            "continuous" => mn::JointKind::Continuous,
            "prismatic" => mn::JointKind::Prismatic,
            "fixed" => mn::JointKind::Fixed,
            "floating" => mn::JointKind::Floating,
            "planar" => mn::JointKind::Planar,
            "ball" | "universal" => {
                warnings.push(format!(
                    "joint '{name}': {jtype} joint approximated as 'floating' \
                     (.misa schema has no spherical / universal kind)"
                ));
                mn::JointKind::Floating
            }
            other => {
                warnings.push(format!(
                    "joint '{name}': unsupported SDF joint type '{other}', \
                     treating as 'fixed'"
                ));
                mn::JointKind::Fixed
            }
        };

        let parent = joint_el
            .children()
            .find(|n| n.tag_name().name() == "parent")
            .and_then(|n| n.text())
            .unwrap_or("world")
            .to_string();
        let child = joint_el
            .children()
            .find(|n| n.tag_name().name() == "child")
            .and_then(|n| n.text())
            .unwrap_or("link")
            .to_string();

        let mut axis = [0.0, 0.0, 1.0];
        let mut limit = mn::JointLimit::default();
        if let Some(axis_el) = joint_el.children().find(|n| n.tag_name().name() == "axis") {
            if let Some(xyz) = axis_el.children().find(|n| n.tag_name().name() == "xyz") {
                let v = parse_f64_list(xyz.text().unwrap_or("0 0 1"));
                if v.len() >= 3 {
                    axis = [v[0], v[1], v[2]];
                }
            }
            if let Some(limit_el) = axis_el.children().find(|n| n.tag_name().name() == "limit") {
                limit = mn::JointLimit {
                    lower: child_f64(limit_el, "lower"),
                    upper: child_f64(limit_el, "upper"),
                    effort: child_f64(limit_el, "effort"),
                    velocity: child_f64(limit_el, "velocity"),
                };
            }
            // <mimic joint="src" multiplier=".." offset=".."/> (SDF 1.7+).
            if let Some(mimic_el) = axis_el.children().find(|n| n.tag_name().name() == "mimic") {
                if let Some(src) = mimic_el.attribute("joint") {
                    file.mimic.push(mn::Mimic {
                        joint: name.clone(),
                        source: src.to_string(),
                        multiplier: mimic_el
                            .attribute("multiplier")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(1.0),
                        offset: mimic_el
                            .attribute("offset")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0.0),
                    });
                }
            }
        }

        child_links.insert(child.clone());
        file.joint.push(mn::Joint {
            name,
            kind,
            parent,
            child,
            axis,
            origin: parse_pose_origin(joint_el),
            limit,
            dynamics: mn::JointDynamics::default(),
        });
    }

    file.robot.root = file
        .link
        .iter()
        .find(|l| !child_links.contains(&l.name))
        .map(|l| l.name.clone())
        .unwrap_or_default();

    Ok(SdfImport { file, warnings })
}

/// Parse `<pose>x y z r p y</pose>` into an [`mn::Origin`].
fn parse_pose_origin(node: roxmltree::Node) -> mn::Origin {
    if let Some(pose) = node.children().find(|n| n.tag_name().name() == "pose") {
        let v = parse_f64_list(pose.text().unwrap_or("0 0 0 0 0 0"));
        if v.len() >= 6 {
            let rpy = [v[3], v[4], v[5]];
            return mn::Origin {
                xyz: [v[0], v[1], v[2]],
                rpy: if rpy == [0.0; 3] { None } else { Some(rpy) },
                quat: None,
            };
        }
    }
    mn::Origin::default()
}

fn child_f64(node: roxmltree::Node, tag: &str) -> f64 {
    node.children()
        .find(|n| n.tag_name().name() == tag)
        .and_then(|n| n.text())
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(0.0)
}

fn child_u32(node: roxmltree::Node, tag: &str) -> Option<u32> {
    node.children()
        .find(|n| n.tag_name().name() == tag)
        .and_then(|n| n.text())
        .and_then(|s| s.trim().parse::<u32>().ok())
}

fn parse_inertial(link_el: roxmltree::Node) -> mn::Inertial {
    let Some(inertial) = link_el
        .children()
        .find(|n| n.tag_name().name() == "inertial")
    else {
        return mn::Inertial::default();
    };
    let (ixx, ixy, ixz, iyy, iyz, izz) = if let Some(i) = inertial
        .children()
        .find(|n| n.tag_name().name() == "inertia")
    {
        (
            child_f64(i, "ixx"),
            child_f64(i, "ixy"),
            child_f64(i, "ixz"),
            child_f64(i, "iyy"),
            child_f64(i, "iyz"),
            child_f64(i, "izz"),
        )
    } else {
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    };
    mn::Inertial {
        mass: child_f64(inertial, "mass"),
        ixx,
        iyy,
        izz,
        ixy,
        ixz,
        iyz,
        origin: parse_pose_origin(inertial),
    }
}

fn parse_visual_color(node: roxmltree::Node) -> Option<mn::ColorSpec> {
    let mat = node.children().find(|n| n.tag_name().name() == "material")?;
    for child in mat.children() {
        let tag = child.tag_name().name();
        if tag == "ambient" || tag == "diffuse" {
            let v: Vec<f32> = child
                .text()
                .unwrap_or("")
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if v.len() >= 3 {
                return Some(mn::ColorSpec::Rgba([
                    v[0],
                    v[1],
                    v[2],
                    v.get(3).copied().unwrap_or(1.0),
                ]));
            }
        }
    }
    None
}

fn parse_geometry(node: roxmltree::Node) -> mn::Geom {
    if let Some(geom) = node.children().find(|n| n.tag_name().name() == "geometry") {
        for child in geom.children() {
            match child.tag_name().name() {
                "box" => {
                    if let Some(size) = child.children().find(|n| n.tag_name().name() == "size") {
                        let v = parse_f64_list(size.text().unwrap_or("0.1 0.1 0.1"));
                        if v.len() >= 3 {
                            return mn::Geom::Box {
                                size: [v[0], v[1], v[2]],
                            };
                        }
                    }
                }
                "cylinder" => {
                    return mn::Geom::Cylinder {
                        radius: child_f64(child, "radius"),
                        length: child_f64(child, "length"),
                    };
                }
                "sphere" => {
                    return mn::Geom::Sphere {
                        radius: child_f64(child, "radius"),
                    };
                }
                "capsule" => {
                    return mn::Geom::Capsule {
                        radius: child_f64(child, "radius"),
                        length: child_f64(child, "length"),
                    };
                }
                "mesh" => {
                    if let Some(uri) = child.children().find(|n| n.tag_name().name() == "uri") {
                        let scale = child
                            .children()
                            .find(|n| n.tag_name().name() == "scale")
                            .and_then(|n| n.text())
                            .map(|t| parse_f64_list(t))
                            .filter(|v| v.len() >= 3)
                            .map(|v| [v[0], v[1], v[2]])
                            .unwrap_or([1.0, 1.0, 1.0]);
                        return mn::Geom::Mesh {
                            file: normalise_mesh_uri(uri.text().unwrap_or("")),
                            scale,
                        };
                    }
                }
                _ => {}
            }
        }
    }
    mn::Geom::Box {
        size: [0.02, 0.02, 0.02],
    }
}

/// Normalise an SDF mesh `<uri>` to a path relative to the SDF file's
/// directory, matching how the old resolver located the file on disk:
/// `package://pkg/rel` resolved against the package root (the SDF dir's
/// parent) → `../rel`; `model://x` → `x`; `file:///abs` → `/abs`.
fn normalise_mesh_uri(uri: &str) -> String {
    if let Some(rest) = uri.strip_prefix("package://") {
        match rest.find('/') {
            Some(i) => format!("../{}", &rest[i + 1..]),
            None => "..".to_string(),
        }
    } else if let Some(rest) = uri.strip_prefix("model://") {
        rest.to_string()
    } else if let Some(rest) = uri.strip_prefix("file://") {
        rest.to_string()
    } else {
        uri.to_string()
    }
}

/// Parse one SDF `<sensor>` element into a master [`mn::Sensor`].
fn parse_sensor(el: roxmltree::Node, link_name: &str) -> Option<mn::Sensor> {
    let name = el.attribute("name")?.to_string();
    let stype = el.attribute("type").unwrap_or("");
    let update_rate = el
        .children()
        .find(|n| n.tag_name().name() == "update_rate")
        .and_then(|n| n.text())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);

    let kind = match stype {
        "camera" | "depth_camera" => {
            let cam = el.children().find(|n| n.tag_name().name() == "camera");
            let mut fov = 1.047;
            let mut width = 320u32;
            let mut height = 240u32;
            let mut near = 0.05;
            let mut far = 100.0;
            if let Some(c) = cam {
                if let Some(h) = c
                    .children()
                    .find(|n| n.tag_name().name() == "horizontal_fov")
                {
                    fov = h.text().and_then(|s| s.parse().ok()).unwrap_or(fov);
                }
                if let Some(img) = c.children().find(|n| n.tag_name().name() == "image") {
                    width = child_u32(img, "width").unwrap_or(width);
                    height = child_u32(img, "height").unwrap_or(height);
                }
                if let Some(clip) = c.children().find(|n| n.tag_name().name() == "clip") {
                    near = child_f64(clip, "near");
                    far = child_f64(clip, "far");
                }
            }
            mn::SensorKind::Camera {
                fov,
                width,
                height,
                near,
                far,
            }
        }
        "ray" | "lidar" | "gpu_lidar" => {
            let ray = el.children().find(|n| n.tag_name().name() == "ray");
            let mut range_min = 0.05;
            let mut range_max = 30.0;
            let mut h_fov = std::f64::consts::TAU;
            let mut h_samples = 360u32;
            let mut v_fov = 0.0;
            let mut v_samples = 1u32;
            if let Some(r) = ray {
                if let Some(scan) = r.children().find(|n| n.tag_name().name() == "scan") {
                    if let Some(h) = scan
                        .children()
                        .find(|n| n.tag_name().name() == "horizontal")
                    {
                        h_samples = child_u32(h, "samples").unwrap_or(h_samples);
                        let min = child_f64(h, "min_angle");
                        let max = child_f64(h, "max_angle");
                        if max > min {
                            h_fov = max - min;
                        }
                    }
                    if let Some(v) = scan.children().find(|n| n.tag_name().name() == "vertical") {
                        v_samples = child_u32(v, "samples").unwrap_or(v_samples);
                        let min = child_f64(v, "min_angle");
                        let max = child_f64(v, "max_angle");
                        if max > min {
                            v_fov = max - min;
                        }
                    }
                }
                if let Some(range) = r.children().find(|n| n.tag_name().name() == "range") {
                    range_min = child_f64(range, "min");
                    range_max = child_f64(range, "max");
                }
            }
            mn::SensorKind::Lidar {
                range_min,
                range_max,
                h_fov,
                h_samples,
                v_fov,
                v_samples,
            }
        }
        "imu" => mn::SensorKind::Imu {
            gyro_noise: 0.0,
            accel_noise: 0.0,
        },
        "force_torque" => mn::SensorKind::ForceTorque { joint: None },
        "contact" => mn::SensorKind::Contact { partner: None },
        other => mn::SensorKind::Generic {
            kind: other.to_string(),
            params: std::collections::BTreeMap::new(),
        },
    };

    Some(mn::Sensor {
        name,
        link: link_name.to_string(),
        origin: parse_pose_origin(el),
        update_rate,
        kind,
    })
}

// ─── Export ─────────────────────────────────────────────────────────────

/// Export a [`MisaFile`] to SDF XML text. `Geom::Mesh.file` strings are
/// emitted into `<uri>` verbatim — the host applies its path policy first.
pub fn export(file: &MisaFile) -> String {
    let materials: HashMap<&str, [f32; 4]> = file
        .material
        .iter()
        .map(|m| (m.name.as_str(), crate::util::color_spec_to_rgba(&m.color)))
        .collect();

    let mut s = String::new();
    s.push_str("<?xml version=\"1.0\"?>\n");
    s.push_str("<sdf version=\"1.7\">\n");
    s.push_str(&format!("  <model name=\"{}\">\n", file.robot.name));

    for link in &file.link {
        s.push_str(&format!("    <link name=\"{}\">\n", link.name));

        // Inertial
        s.push_str("      <inertial>\n");
        write_pose(&mut s, &link.inertial.origin, 8);
        s.push_str(&format!("        <mass>{}</mass>\n", fmt(link.inertial.mass)));
        s.push_str("        <inertia>\n");
        s.push_str(&format!("          <ixx>{}</ixx>\n", fmt(link.inertial.ixx)));
        s.push_str(&format!("          <ixy>{}</ixy>\n", fmt(link.inertial.ixy)));
        s.push_str(&format!("          <ixz>{}</ixz>\n", fmt(link.inertial.ixz)));
        s.push_str(&format!("          <iyy>{}</iyy>\n", fmt(link.inertial.iyy)));
        s.push_str(&format!("          <iyz>{}</iyz>\n", fmt(link.inertial.iyz)));
        s.push_str(&format!("          <izz>{}</izz>\n", fmt(link.inertial.izz)));
        s.push_str("        </inertia>\n");
        s.push_str("      </inertial>\n");

        for (vi, vis) in link.visual.iter().enumerate() {
            s.push_str(&format!("      <visual name=\"visual_{vi}\">\n"));
            write_pose(&mut s, &vis.origin, 8);
            write_geometry(&mut s, &vis.geom, 8);
            let c = resolve_visual_rgba(vis, &materials);
            s.push_str(&format!(
                "        <material>\n          <ambient>{} {} {} {}</ambient>\n        </material>\n",
                c[0], c[1], c[2], c[3]
            ));
            s.push_str("      </visual>\n");
        }

        for (ci, col) in link.collision.iter().enumerate() {
            s.push_str(&format!("      <collision name=\"collision_{ci}\">\n"));
            write_pose(&mut s, &col.origin, 8);
            write_geometry(&mut s, &col.geom, 8);
            s.push_str("      </collision>\n");
        }

        for sensor in file.sensor.iter().filter(|sn| sn.link == link.name) {
            write_sensor(&mut s, sensor);
        }

        s.push_str("    </link>\n");
    }

    for joint in &file.joint {
        let jtype = match joint.kind {
            mn::JointKind::Revolute => "revolute",
            mn::JointKind::Continuous => "continuous",
            mn::JointKind::Prismatic => "prismatic",
            mn::JointKind::Fixed => "fixed",
            mn::JointKind::Floating => "floating",
            mn::JointKind::Planar => "planar",
        };
        s.push_str(&format!(
            "    <joint name=\"{}\" type=\"{jtype}\">\n",
            joint.name
        ));
        s.push_str(&format!("      <parent>{}</parent>\n", joint.parent));
        s.push_str(&format!("      <child>{}</child>\n", joint.child));
        write_pose(&mut s, &joint.origin, 6);
        s.push_str("      <axis>\n");
        s.push_str(&format!(
            "        <xyz>{} {} {}</xyz>\n",
            fmt(joint.axis[0]),
            fmt(joint.axis[1]),
            fmt(joint.axis[2])
        ));
        s.push_str("        <limit>\n");
        s.push_str(&format!("          <lower>{}</lower>\n", fmt(joint.limit.lower)));
        s.push_str(&format!("          <upper>{}</upper>\n", fmt(joint.limit.upper)));
        s.push_str(&format!("          <effort>{}</effort>\n", fmt(joint.limit.effort)));
        s.push_str(&format!(
            "          <velocity>{}</velocity>\n",
            fmt(joint.limit.velocity)
        ));
        s.push_str("        </limit>\n");
        if let Some(m) = file.mimic.iter().find(|m| m.joint == joint.name) {
            s.push_str(&format!(
                "        <mimic joint=\"{}\" multiplier=\"{}\" offset=\"{}\"/>\n",
                m.source,
                fmt(m.multiplier),
                fmt(m.offset),
            ));
        }
        s.push_str("      </axis>\n");
        s.push_str("    </joint>\n");
    }

    s.push_str("  </model>\n");
    s.push_str("</sdf>\n");
    s
}

fn write_pose(s: &mut String, origin: &mn::Origin, indent: usize) {
    let pad: String = " ".repeat(indent);
    let (r, p, y) = origin_rotation(origin).euler_angles();
    s.push_str(&format!(
        "{pad}<pose>{} {} {} {} {} {}</pose>\n",
        fmt(origin.xyz[0]),
        fmt(origin.xyz[1]),
        fmt(origin.xyz[2]),
        fmt(r),
        fmt(p),
        fmt(y)
    ));
}

fn write_geometry(s: &mut String, geom: &mn::Geom, indent: usize) {
    let pad: String = " ".repeat(indent);
    s.push_str(&format!("{pad}<geometry>\n"));
    match geom {
        mn::Geom::Box { size } => {
            s.push_str(&format!(
                "{pad}  <box><size>{} {} {}</size></box>\n",
                fmt(size[0]),
                fmt(size[1]),
                fmt(size[2])
            ));
        }
        mn::Geom::Cylinder { radius, length } => {
            s.push_str(&format!(
                "{pad}  <cylinder><radius>{}</radius><length>{}</length></cylinder>\n",
                fmt(*radius),
                fmt(*length)
            ));
        }
        mn::Geom::Sphere { radius } => {
            s.push_str(&format!(
                "{pad}  <sphere><radius>{}</radius></sphere>\n",
                fmt(*radius)
            ));
        }
        mn::Geom::Capsule { radius, length } => {
            s.push_str(&format!(
                "{pad}  <capsule><radius>{}</radius><length>{}</length></capsule>\n",
                fmt(*radius),
                fmt(*length)
            ));
        }
        mn::Geom::Mesh { file, scale } => {
            s.push_str(&format!("{pad}  <mesh>\n"));
            s.push_str(&format!("{pad}    <uri>{file}</uri>\n"));
            let unit = scale == &[1.0, 1.0, 1.0];
            if !unit {
                s.push_str(&format!(
                    "{pad}    <scale>{} {} {}</scale>\n",
                    fmt(scale[0]),
                    fmt(scale[1]),
                    fmt(scale[2])
                ));
            }
            s.push_str(&format!("{pad}  </mesh>\n"));
        }
    }
    s.push_str(&format!("{pad}</geometry>\n"));
}

/// Emit one `<sensor>` block. Each kind maps to the SDF type it
/// represents; `Generic` kinds are emitted verbatim so they round-trip.
fn write_sensor(s: &mut String, sensor: &mn::Sensor) {
    let stype = match &sensor.kind {
        mn::SensorKind::Camera { .. } => "camera",
        mn::SensorKind::Lidar { .. } => "ray",
        mn::SensorKind::Imu { .. } => "imu",
        mn::SensorKind::ForceTorque { .. } => "force_torque",
        mn::SensorKind::Contact { .. } => "contact",
        mn::SensorKind::Generic { kind, .. } => kind.as_str(),
    };
    s.push_str(&format!(
        "      <sensor name=\"{}\" type=\"{stype}\">\n",
        sensor.name,
    ));
    write_pose(s, &sensor.origin, 8);
    if sensor.update_rate > 0.0 {
        s.push_str(&format!(
            "        <update_rate>{}</update_rate>\n",
            fmt(sensor.update_rate),
        ));
    }
    match &sensor.kind {
        mn::SensorKind::Camera {
            fov,
            width,
            height,
            near,
            far,
        } => {
            s.push_str("        <camera>\n");
            s.push_str(&format!(
                "          <horizontal_fov>{}</horizontal_fov>\n",
                fmt(*fov)
            ));
            s.push_str(&format!(
                "          <image><width>{width}</width><height>{height}</height></image>\n",
            ));
            s.push_str(&format!(
                "          <clip><near>{}</near><far>{}</far></clip>\n",
                fmt(*near),
                fmt(*far),
            ));
            s.push_str("        </camera>\n");
        }
        mn::SensorKind::Lidar {
            range_min,
            range_max,
            h_fov,
            h_samples,
            v_fov,
            v_samples,
        } => {
            s.push_str("        <ray>\n");
            s.push_str("          <scan>\n");
            s.push_str(&format!(
                "            <horizontal><samples>{}</samples><min_angle>{}</min_angle><max_angle>{}</max_angle></horizontal>\n",
                h_samples,
                fmt(-h_fov / 2.0),
                fmt(h_fov / 2.0),
            ));
            if *v_samples > 1 || *v_fov > 0.0 {
                s.push_str(&format!(
                    "            <vertical><samples>{}</samples><min_angle>{}</min_angle><max_angle>{}</max_angle></vertical>\n",
                    v_samples,
                    fmt(-v_fov / 2.0),
                    fmt(v_fov / 2.0),
                ));
            }
            s.push_str("          </scan>\n");
            s.push_str(&format!(
                "          <range><min>{}</min><max>{}</max></range>\n",
                fmt(*range_min),
                fmt(*range_max),
            ));
            s.push_str("        </ray>\n");
        }
        mn::SensorKind::Imu { .. }
        | mn::SensorKind::ForceTorque { .. }
        | mn::SensorKind::Contact { .. }
        | mn::SensorKind::Generic { .. } => {
            // Default SDF behaviour for these is fine without inner blocks.
        }
    }
    s.push_str("      </sensor>\n");
}

#[cfg(test)]
mod misa_tests {
    use super::*;

    const FIXTURE: &str = r#"<?xml version="1.0"?>
<sdf version="1.7">
  <model name="test_sdf_robot">
    <link name="base_link">
      <inertial>
        <pose>0 0 0 0 0 0</pose>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>0.01</iyy><iyz>0</iyz><izz>0.01</izz>
        </inertia>
      </inertial>
      <visual name="visual_0">
        <geometry><box><size>0.2 0.2 0.1</size></box></geometry>
        <material><ambient>0.5 0.5 0.5 1</ambient></material>
      </visual>
      <collision name="collision_0">
        <geometry><box><size>0.2 0.2 0.1</size></box></geometry>
      </collision>
      <sensor name="body_imu" type="imu">
        <pose>0.01 0 0.02 0 0 0</pose>
        <update_rate>200</update_rate>
      </sensor>
    </link>
    <link name="link1">
      <visual name="visual_0">
        <geometry><cylinder><radius>0.02</radius><length>0.2</length></cylinder></geometry>
        <material><ambient>1 0 0 1</ambient></material>
      </visual>
    </link>
    <joint name="joint1" type="revolute">
      <parent>base_link</parent>
      <child>link1</child>
      <pose>0 0 0.05 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <limit><lower>-1.57</lower><upper>1.57</upper><effort>10</effort><velocity>2</velocity></limit>
        <mimic joint="joint0" multiplier="2" offset="0.1"/>
      </axis>
    </joint>
  </model>
</sdf>"#;

    #[test]
    fn import_basic_structure() {
        let out = import_str(FIXTURE).expect("import");
        let f = &out.file;
        assert!(out.warnings.is_empty(), "{:?}", out.warnings);
        assert_eq!(f.robot.name, "test_sdf_robot");
        assert_eq!(f.robot.root, "base_link");
        assert_eq!(f.link.len(), 2);
        assert_eq!(f.joint.len(), 1);

        let base = &f.link[0];
        assert!((base.inertial.mass - 1.0).abs() < 1e-12);
        assert!(matches!(
            base.visual[0].geom,
            mn::Geom::Box { size } if size == [0.2, 0.2, 0.1]
        ));
        assert!(matches!(
            f.link[1].visual[0].geom,
            mn::Geom::Cylinder { radius, length }
                if (radius - 0.02).abs() < 1e-12 && (length - 0.2).abs() < 1e-12
        ));

        let j = &f.joint[0];
        assert_eq!(j.kind, mn::JointKind::Revolute);
        assert_eq!(j.axis, [0.0, 1.0, 0.0]);
        assert_eq!(j.origin.xyz, [0.0, 0.0, 0.05]);
        assert_eq!(
            (j.limit.lower, j.limit.upper, j.limit.effort, j.limit.velocity),
            (-1.57, 1.57, 10.0, 2.0)
        );

        assert_eq!(f.mimic.len(), 1);
        assert_eq!(f.mimic[0].source, "joint0");
        assert_eq!(f.mimic[0].multiplier, 2.0);

        assert_eq!(f.sensor.len(), 1);
        assert!(matches!(f.sensor[0].kind, mn::SensorKind::Imu { .. }));
        assert_eq!(f.sensor[0].update_rate, 200.0);
        assert_eq!(f.sensor[0].origin.xyz, [0.01, 0.0, 0.02]);
    }

    #[test]
    fn mesh_uris_normalise_to_sdf_relative_paths() {
        let xml = r#"<sdf version="1.7"><model name="m">
  <link name="root">
    <visual name="v">
      <geometry><mesh><uri>package://my_pkg/meshes/arm.stl</uri><scale>0.001 0.001 0.001</scale></mesh></geometry>
    </visual>
    <collision name="c">
      <geometry><mesh><uri>model://parts/leg.obj</uri></mesh></geometry>
    </collision>
  </link>
</model></sdf>"#;
        let f = import_str(xml).unwrap().file;
        match &f.link[0].visual[0].geom {
            mn::Geom::Mesh { file, scale } => {
                assert_eq!(file, "../meshes/arm.stl");
                assert_eq!(*scale, [0.001, 0.001, 0.001]);
            }
            other => panic!("expected mesh, got {other:?}"),
        }
        match &f.link[0].collision[0].geom {
            mn::Geom::Mesh { file, .. } => assert_eq!(file, "parts/leg.obj"),
            other => panic!("expected mesh, got {other:?}"),
        }
    }

    #[test]
    fn export_roundtrip_preserves_structure() {
        let out = import_str(FIXTURE).unwrap();
        let xml = export(&out.file);
        assert!(xml.contains("<model name=\"test_sdf_robot\">"));
        assert!(xml.contains("<joint name=\"joint1\" type=\"revolute\">"));
        assert!(xml.contains("<mimic joint=\"joint0\" multiplier=\"2\" offset=\"0.1\"/>"));
        assert!(xml.contains("<sensor name=\"body_imu\" type=\"imu\">"));
        assert!(xml.contains("<box><size>0.2 0.2 0.1</size></box>"));

        let back = import_str(&xml).unwrap().file;
        assert_eq!(back.link.len(), out.file.link.len());
        assert_eq!(back.joint.len(), out.file.joint.len());
        assert_eq!(back.mimic.len(), 1);
        assert_eq!(back.sensor.len(), 1);
        let (a, b) = (&out.file.joint[0], &back.joint[0]);
        assert_eq!(a.origin.xyz, b.origin.xyz);
        assert_eq!(a.limit.lower, b.limit.lower);
    }

    #[test]
    fn ball_joint_warns_and_approximates() {
        let xml = r#"<sdf version="1.7"><model name="m">
  <link name="a"/><link name="b"/>
  <joint name="bj" type="ball"><parent>a</parent><child>b</child></joint>
</model></sdf>"#;
        let out = import_str(xml).unwrap();
        assert_eq!(out.warnings.len(), 1);
        assert_eq!(out.file.joint[0].kind, mn::JointKind::Floating);
    }
}
