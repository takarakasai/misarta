//! URDF (Unified Robot Description Format) ⇄ [`MisaFile`] conversion and
//! `Model<f64>` loader.
//!
//! The single URDF parser lives in [`import_str`], which converts URDF
//! XML into the `.misa` master schema ([`MisaFile`]). The classic loader
//! API ([`load_urdf_string`] / [`load_urdf_geometry_string`]) is a thin
//! wrapper over `import_str` + [`misarta::native::build_model`] — the
//! same pipeline MJCF and SDF use, so every robot-description format
//! enters through one MisaFile door.
//!
//! # Supported elements
//!
//! - `<robot>` — top-level container (plus top-level `<material>`)
//! - `<link>` — rigid body with optional `<inertial>`, `<visual>`,
//!   `<collision>` (box / cylinder / sphere / capsule / mesh)
//! - `<joint>` — revolute, continuous, prismatic, fixed, floating,
//!   planar; `<limit>`, `<dynamics>`, `<mimic>` are captured into the
//!   schema
//! - `<origin xyz="..." rpy="..."/>` — placement
//! - `<axis xyz="..."/>` — joint axis
//!
//! # Example
//!
//! ```no_run
//! use misarta_formats::urdf;
//! let xml = std::fs::read_to_string("robot.urdf").unwrap();
//! let model = urdf::load_urdf_string(&xml).unwrap();
//! ```
//!
//! # Loading meshes via [`AssetSource`](misarta::native::AssetSource)
//!
//! [`load_urdf_geometry_string`] populates each `GeometryObject` with
//! its `mesh_path` and leaves `mesh_data` empty. Mesh references are
//! normalised to URDF-directory-relative paths at parse time
//! (`package://pkg/rel` → `../rel` for the conventional
//! `<pkg>/urdf/robot.urdf` layout, `file:///abs` → `/abs`), matching
//! the `.misa` convention of "relative to the model file".
//! [`misarta::native::load_meshes`] resolves them through any
//! [`AssetSource`](misarta::native::AssetSource).
//!
//! ```no_run
//! use misarta_formats::urdf;
//! use misarta::native;
//! let xml = std::fs::read_to_string("robot.urdf").unwrap();
//! let (model, mut visual, mut collision) =
//!     urdf::load_urdf_geometry_string(&xml).unwrap();
//!
//! // Resolve mesh files relative to the URDF's own directory.
//! let assets = native::FileSystemSource::new("path/to/package_root/urdf");
//! let _vrep = native::load_meshes(&mut visual, &assets).unwrap();
//! let _crep = native::load_meshes(&mut collision, &assets).unwrap();
//! ```

use std::path::Path;

use misarta::geometry::{GeometryModel, GeometryShape};
use misarta::model::Model;
use misarta::native as mn;
use misarta::native::MisaFile;
use misarta::se3;
use nalgebra::Rotation3;
use roxmltree::Document;

use crate::util::parse_vec3_or;

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
///
/// Thin wrapper over the `.misa` conversion pipeline: the URDF is first
/// converted to a [`MisaFile`] via [`import_str`], then realised with
/// [`misarta::native::build_model`] — the same path the hosts use, so
/// the tree has exactly one URDF parser. Two consequences vs. the old
/// dedicated loader: geometry objects follow the native
/// `<link>_visual_<i>` / `<link>_collision_<i>` naming (URDF `name`
/// attributes on `<visual>` are not part of the `.misa` schema), and
/// mesh `filename`s are normalised to URDF-directory-relative paths
/// (`package://pkg/rel` → `../rel`, `file:///abs` → `/abs`).
pub fn load_urdf_geometry_string(
    xml: &str,
) -> Result<(Model<f64>, GeometryModel, GeometryModel), UrdfError> {
    let import = import_str(xml).map_err(import_err_to_urdf)?;
    misarta::native::build_model(&import.file)
        .map_err(|e| UrdfError::Topology(e.to_string()))
}

/// Load a `Model<f64>` from a URDF XML string. See
/// [`load_urdf_geometry_string`] for the conversion pipeline.
pub fn load_urdf_string(xml: &str) -> Result<Model<f64>, UrdfError> {
    Ok(load_urdf_geometry_string(xml)?.0)
}

/// Map an [`import_str`] error string onto the closest [`UrdfError`]
/// variant so the loader API keeps its typed errors.
fn import_err_to_urdf(e: String) -> UrdfError {
    if let Some(msg) = e.strip_prefix("Parse URDF XML: ") {
        UrdfError::XmlParse(msg.to_string())
    } else if e.starts_with("No <robot>") {
        UrdfError::MissingElement("root <robot> element".into())
    } else {
        UrdfError::Topology(e)
    }
}

// ═════════════════════════ MisaFile conversion ══════════════════════════

/// Result of a successful URDF import: the converted [`MisaFile`] plus
/// non-fatal conversion notes (degraded joint kinds, dropped extras).
/// Hosts should surface the warnings to the user.
#[derive(Debug, Clone)]
pub struct UrdfImport {
    pub file: MisaFile,
    pub warnings: Vec<String>,
}

/// Parse a URDF file on disk into a [`MisaFile`].
pub fn import(path: &Path) -> Result<UrdfImport, String> {
    let xml = std::fs::read_to_string(path).map_err(|e| format!("Read URDF: {e}"))?;
    import_str(&xml)
}

/// Parse URDF XML text into a [`MisaFile`].
///
/// Mesh `filename`s are normalised to URDF-directory-relative paths so
/// the `.misa` base-directory resolution rule finds the same files the
/// classic `package://` resolver did (see module docs). `<limit>`,
/// `<dynamics>`, `<mimic>` and materials are captured into the schema;
/// the runtime `Model` ignores limits but hosts round-trip them.
pub fn import_str(xml: &str) -> Result<UrdfImport, String> {
    let doc = Document::parse(xml).map_err(|e| format!("Parse URDF XML: {e}"))?;
    let robot = doc.root_element();
    if robot.tag_name().name() != "robot" {
        return Err("No <robot> root element found in URDF".into());
    }

    let mut file = MisaFile::new(robot.attribute("name").unwrap_or(""), "");
    let mut warnings: Vec<String> = Vec::new();

    // ── Top-level materials ─────────────────────────────────────────────
    // `<material name="x"><color rgba="..."/></material>` becomes a
    // named [[material]] entry; visuals reference it by name.
    for mat_el in robot
        .children()
        .filter(|n| n.tag_name().name() == "material")
    {
        let Some(name) = mat_el.attribute("name") else {
            continue;
        };
        if let Some(rgba) = material_rgba(&mat_el) {
            file.material.push(mn::Material {
                name: name.to_string(),
                color: mn::ColorSpec::Rgba(rgba),
            });
        }
    }

    // ── Links ───────────────────────────────────────────────────────────
    for link_el in robot.children().filter(|n| n.tag_name().name() == "link") {
        let name = link_el
            .attribute("name")
            .map(str::to_string)
            .unwrap_or_else(|| format!("link_{}", file.link.len()));

        let visual: Vec<mn::Visual> = link_el
            .children()
            .filter(|n| n.tag_name().name() == "visual")
            .filter_map(|v| {
                let (color, material) = visual_color_or_material(&v);
                Some(mn::Visual {
                    origin: parse_origin(&v),
                    geom: parse_geometry(&v)?,
                    color,
                    material,
                })
            })
            .collect();

        let collision: Vec<mn::Collision> = link_el
            .children()
            .filter(|n| n.tag_name().name() == "collision")
            .filter_map(|c| {
                Some(mn::Collision {
                    origin: parse_origin(&c),
                    geom: parse_geometry(&c)?,
                    physics: None,
                })
            })
            .collect();

        file.link.push(mn::Link {
            name,
            description: String::new(),
            inertial: parse_inertial(&link_el),
            visual,
            collision,
            collision_enabled: true,
        });
    }

    // ── Joints ──────────────────────────────────────────────────────────
    let mut child_links: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for joint_el in robot.children().filter(|n| n.tag_name().name() == "joint") {
        let name = joint_el
            .attribute("name")
            .map(str::to_string)
            .unwrap_or_else(|| format!("joint_{}", file.joint.len()));

        let jtype = joint_el.attribute("type").unwrap_or_else(|| {
            warnings.push(format!("joint '{name}': missing type, treating as 'fixed'"));
            "fixed"
        });
        let kind = match jtype {
            "revolute" => mn::JointKind::Revolute,
            "continuous" => mn::JointKind::Continuous,
            "prismatic" => mn::JointKind::Prismatic,
            "fixed" => mn::JointKind::Fixed,
            "floating" => mn::JointKind::Floating,
            "planar" => mn::JointKind::Planar,
            other => {
                warnings.push(format!(
                    "joint '{name}': unsupported URDF joint type '{other}', \
                     treating as 'fixed'"
                ));
                mn::JointKind::Fixed
            }
        };

        let parent = joint_el
            .children()
            .find(|n| n.tag_name().name() == "parent")
            .and_then(|n| n.attribute("link"))
            .unwrap_or("world")
            .to_string();
        let child = joint_el
            .children()
            .find(|n| n.tag_name().name() == "child")
            .and_then(|n| n.attribute("link"))
            .unwrap_or("link")
            .to_string();
        child_links.insert(child.clone());

        let axis = joint_el
            .children()
            .find(|n| n.tag_name().name() == "axis")
            .and_then(|n| n.attribute("xyz"))
            .map(|s| parse_vec3_or(s, [0.0, 0.0, 1.0]))
            .unwrap_or([0.0, 0.0, 1.0]);

        let limit = joint_el
            .children()
            .find(|n| n.tag_name().name() == "limit")
            .map(|l| mn::JointLimit {
                lower: attr_f64(&l, "lower"),
                upper: attr_f64(&l, "upper"),
                effort: attr_f64(&l, "effort"),
                velocity: attr_f64(&l, "velocity"),
            })
            .unwrap_or_default();

        let dynamics = joint_el
            .children()
            .find(|n| n.tag_name().name() == "dynamics")
            .map(|d| mn::JointDynamics {
                armature: 0.0,
                damping: attr_f64(&d, "damping"),
                friction: attr_f64(&d, "friction"),
            })
            .unwrap_or_default();

        // `<mimic joint="..." multiplier="..." offset="..."/>`
        if let Some(mimic_el) = joint_el
            .children()
            .find(|n| n.tag_name().name() == "mimic")
        {
            if let Some(source) = mimic_el.attribute("joint") {
                file.mimic.push(mn::Mimic {
                    joint: name.clone(),
                    source: source.to_string(),
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

        file.joint.push(mn::Joint {
            name,
            kind,
            parent,
            child,
            axis,
            origin: parse_origin(&joint_el),
            limit,
            dynamics,
        });
    }

    // ── Root link = not a child of any joint ────────────────────────────
    file.robot.root = file
        .link
        .iter()
        .find(|l| !child_links.contains(&l.name))
        .map(|l| l.name.clone())
        .ok_or("no root link found in URDF (every link is some joint's child)")?;

    Ok(UrdfImport { file, warnings })
}

// ─── Import helpers ─────────────────────────────────────────────────────

/// Parse an `<origin xyz="..." rpy="..."/>` beneath a parent node into
/// an [`mn::Origin`].
fn parse_origin(parent: &roxmltree::Node) -> mn::Origin {
    let Some(origin_el) = parent.children().find(|n| n.tag_name().name() == "origin") else {
        return mn::Origin::default();
    };
    let xyz = origin_el
        .attribute("xyz")
        .map(|s| parse_vec3_or(s, [0.0; 3]))
        .unwrap_or([0.0; 3]);
    let rpy = origin_el
        .attribute("rpy")
        .map(|s| parse_vec3_or(s, [0.0; 3]))
        .filter(|r| r.iter().any(|v| *v != 0.0));
    mn::Origin { xyz, rpy, quat: None }
}

/// Parse a link's `<inertial>` into an [`mn::Inertial`]. The inertial
/// `<origin>` keeps both the COM translation and the principal-axis
/// rotation (`rpy`) — `build_model` rotates the tensor into the link
/// frame, which the old dedicated loader silently skipped.
fn parse_inertial(link_el: &roxmltree::Node) -> mn::Inertial {
    let Some(inertial) = link_el
        .children()
        .find(|n| n.tag_name().name() == "inertial")
    else {
        return mn::Inertial::default();
    };
    let mass = inertial
        .children()
        .find(|n| n.tag_name().name() == "mass")
        .map(|m| attr_f64(&m, "value"))
        .unwrap_or(0.0);
    let (ixx, ixy, ixz, iyy, iyz, izz) = inertial
        .children()
        .find(|n| n.tag_name().name() == "inertia")
        .map(|i| {
            (
                attr_f64(&i, "ixx"),
                attr_f64(&i, "ixy"),
                attr_f64(&i, "ixz"),
                attr_f64(&i, "iyy"),
                attr_f64(&i, "iyz"),
                attr_f64(&i, "izz"),
            )
        })
        .unwrap_or((0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    mn::Inertial {
        mass,
        ixx,
        iyy,
        izz,
        ixy,
        ixz,
        iyz,
        origin: parse_origin(&inertial),
    }
}

fn attr_f64(node: &roxmltree::Node, attr: &str) -> f64 {
    node.attribute(attr)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0)
}

/// Parse a URDF `<geometry>` child element. Returns `None` when the
/// element is missing or contains no recognised shape (matching the old
/// loader, which skipped such visuals).
fn parse_geometry(parent: &roxmltree::Node) -> Option<mn::Geom> {
    let geom_el = parent
        .children()
        .find(|n| n.tag_name().name() == "geometry")?;
    for child in geom_el.children() {
        match child.tag_name().name() {
            "box" => {
                let size = parse_vec3_or(child.attribute("size").unwrap_or("0 0 0"), [0.0; 3]);
                return Some(mn::Geom::Box { size });
            }
            "sphere" => {
                return Some(mn::Geom::Sphere {
                    radius: attr_f64(&child, "radius"),
                });
            }
            "cylinder" => {
                return Some(mn::Geom::Cylinder {
                    radius: attr_f64(&child, "radius"),
                    length: attr_f64(&child, "length"),
                });
            }
            "capsule" => {
                return Some(mn::Geom::Capsule {
                    radius: attr_f64(&child, "radius"),
                    length: attr_f64(&child, "length"),
                });
            }
            "mesh" => {
                return Some(mn::Geom::Mesh {
                    file: normalise_mesh_uri(child.attribute("filename").unwrap_or("")),
                    scale: child
                        .attribute("scale")
                        .map(|s| parse_vec3_or(s, [1.0, 1.0, 1.0]))
                        .unwrap_or([1.0, 1.0, 1.0]),
                });
            }
            _ => {}
        }
    }
    None
}

/// Normalise a URDF mesh reference to a URDF-directory-relative path,
/// matching the `.misa` convention of "relative to the model file":
/// `package://pkg/rel` → `../rel` (the conventional
/// `<pkg>/urdf/robot.urdf` layout puts the package root one level up),
/// `file:///abs` → `/abs`, anything else verbatim.
fn normalise_mesh_uri(uri: &str) -> String {
    if let Some(rest) = uri.strip_prefix("package://") {
        match rest.find('/') {
            Some(i) => format!("../{}", &rest[i + 1..]),
            None => "..".to_string(),
        }
    } else if let Some(rest) = uri.strip_prefix("file://") {
        rest.to_string()
    } else {
        uri.to_string()
    }
}

/// Extract an RGBA colour from a `<material>` element's `<color rgba>`.
fn material_rgba(mat_el: &roxmltree::Node) -> Option<[f32; 4]> {
    let color_el = mat_el.children().find(|n| n.tag_name().name() == "color")?;
    let v: Vec<f32> = color_el
        .attribute("rgba")?
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    Some([
        v.first().copied().unwrap_or(0.8),
        v.get(1).copied().unwrap_or(0.8),
        v.get(2).copied().unwrap_or(0.8),
        v.get(3).copied().unwrap_or(1.0),
    ])
}

/// Resolve a `<visual>`'s material: an inline `<color>` wins, otherwise
/// a named reference to a top-level `[[material]]`.
fn visual_color_or_material(
    vis_el: &roxmltree::Node,
) -> (Option<mn::ColorSpec>, Option<String>) {
    let Some(mat_el) = vis_el.children().find(|n| n.tag_name().name() == "material") else {
        return (None, None);
    };
    if let Some(rgba) = material_rgba(&mat_el) {
        return (Some(mn::ColorSpec::Rgba(rgba)), None);
    }
    (None, mat_el.attribute("name").map(str::to_string))
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
    use misarta::joint::JointType;

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
fn write_urdf_visual_or_collision(
    out: &mut String,
    obj: &misarta::geometry::GeometryObject,
    tag: &str,
) {
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
    use misarta::joint::JointType;
    use misarta::model::{LinkInertia, ModelBuilder};
    use nalgebra::Vector3;

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
        let t1 = misarta::se3::translation(&model.joints[1].placement);
        assert_relative_eq!(t1, Vector3::new(0.0, 0.0, 0.05), epsilon = 1e-12);

        let t2 = misarta::se3::translation(&model.joints[2].placement);
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
        // ...while the MisaFile keeps the distinct Continuous kind.
        let import = import_str(xml).unwrap();
        assert_eq!(import.file.joint[0].kind, mn::JointKind::Continuous);
    }

    #[test]
    fn urdf_fk_matches_manual() {
        // FK from URDF should match a manually-built model.
        let model = load_urdf_string(SIMPLE_URDF).unwrap();
        let q = vec![0.3, -0.5];
        let data = misarta::fk::forward_kinematics(&model, &q);

        // Build the equivalent model by hand.
        let offset1 = misarta::se3::from_rotation_and_translation(
            &Rotation3::identity(),
            &Vector3::new(0.0, 0.0, 0.05),
        );
        let offset2 = misarta::se3::from_rotation_and_translation(
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
        let data_manual = misarta::fk::forward_kinematics(&manual, &q);

        for i in 1..model.joints.len() {
            assert_relative_eq!(
                misarta::se3::to_homogeneous(&data.oMi[i]),
                misarta::se3::to_homogeneous(&data_manual.oMi[i]),
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

        // Geometry objects follow the native naming convention.
        assert_eq!(vis.objects[0].name, "base_visual_0");
        assert_eq!(col.objects[0].name, "base_collision_0");
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
        // package://robot/... is normalised to a URDF-directory-relative
        // path (package root = one level above the urdf/ dir).
        match &vis.objects[0].shape {
            GeometryShape::Mesh { filename, scale } => {
                assert_eq!(filename, "../meshes/base.stl");
                assert_relative_eq!(*scale, Vector3::new(0.001, 0.001, 0.001), epsilon = 1e-12);
            }
            _ => panic!("expected mesh shape"),
        }
        assert_eq!(
            vis.objects[0].mesh_path.as_deref(),
            Some("../meshes/base.stl")
        );
    }

    // ─── MisaFile import ────────────────────────────────────────────────

    #[test]
    fn import_captures_limits_dynamics_and_materials() {
        let xml = r#"<?xml version="1.0"?>
<robot name="rich">
  <material name="red"><color rgba="1 0 0 1"/></material>
  <link name="base">
    <visual>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
      <material name="red"/>
    </visual>
    <visual>
      <geometry><sphere radius="0.05"/></geometry>
      <material name="inline"><color rgba="0 1 0 0.5"/></material>
    </visual>
  </link>
  <link name="arm"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="30" velocity="10"/>
    <dynamics damping="0.2" friction="0.05"/>
  </joint>
</robot>"#;
        let import = import_str(xml).unwrap();
        let f = &import.file;
        assert!(import.warnings.is_empty(), "{:?}", import.warnings);
        assert_eq!(f.robot.root, "base");

        assert_eq!(f.material.len(), 1);
        assert_eq!(f.material[0].name, "red");
        assert_eq!(f.link[0].visual[0].material.as_deref(), Some("red"));
        assert!(f.link[0].visual[0].color.is_none());
        assert!(matches!(
            f.link[0].visual[1].color,
            Some(mn::ColorSpec::Rgba(c)) if c == [0.0, 1.0, 0.0, 0.5]
        ));

        let j = &f.joint[0];
        assert_eq!((j.limit.lower, j.limit.upper), (-1.5, 1.5));
        assert_eq!((j.limit.effort, j.limit.velocity), (30.0, 10.0));
        assert_eq!((j.dynamics.damping, j.dynamics.friction), (0.2, 0.05));
    }

    #[test]
    fn import_captures_mimic() {
        let xml = r#"<robot name="m">
  <link name="a"/><link name="b"/><link name="c"/>
  <joint name="j1" type="revolute">
    <parent link="a"/><child link="b"/><axis xyz="0 0 1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="a"/><child link="c"/><axis xyz="0 0 1"/>
    <mimic joint="j1" multiplier="-2" offset="0.1"/>
  </joint>
</robot>"#;
        let import = import_str(xml).unwrap();
        assert_eq!(import.file.mimic.len(), 1);
        let m = &import.file.mimic[0];
        assert_eq!((m.joint.as_str(), m.source.as_str()), ("j2", "j1"));
        assert_eq!((m.multiplier, m.offset), (-2.0, 0.1));
        // ...and the built Model applies it.
        let model = load_urdf_string(xml).unwrap();
        assert_eq!(model.mimic.len(), 1);
    }

    #[test]
    fn import_degrades_unknown_joint_type_with_warning() {
        let xml = r#"<robot name="u">
  <link name="a"/><link name="b"/>
  <joint name="weird" type="gearbox">
    <parent link="a"/><child link="b"/>
  </joint>
</robot>"#;
        let import = import_str(xml).unwrap();
        assert_eq!(import.file.joint[0].kind, mn::JointKind::Fixed);
        assert_eq!(import.warnings.len(), 1);
        assert!(import.warnings[0].contains("gearbox"));
    }

    #[test]
    fn import_rejects_rootless_and_non_robot() {
        // Cyclic (no root) URDF is an import error, mapped to Topology.
        let cyclic = r#"<robot name="c">
  <link name="a"/><link name="b"/>
  <joint name="j1" type="fixed"><parent link="a"/><child link="b"/></joint>
  <joint name="j2" type="fixed"><parent link="b"/><child link="a"/></joint>
</robot>"#;
        assert!(matches!(
            load_urdf_string(cyclic),
            Err(UrdfError::Topology(_))
        ));
        assert!(matches!(
            load_urdf_string("<sdf version=\"1.7\"/>"),
            Err(UrdfError::MissingElement(_))
        ));
    }

    #[test]
    fn import_inertial_rpy_rotates_tensor() {
        // A 90° roll on the inertial frame swaps the yy / zz moments in
        // the link frame (the old loader silently dropped the rotation).
        // The inertial sits on a child link — root-link inertia is not
        // representable in the joint-indexed Model (same as before).
        let xml = r#"<robot name="i">
  <link name="base"/>
  <link name="a">
    <inertial>
      <mass value="1"/>
      <origin xyz="0 0 0" rpy="1.5707963267948966 0 0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="2" iyz="0" izz="3"/>
    </inertial>
  </link>
  <joint name="j" type="fixed"><parent link="base"/><child link="a"/></joint>
</robot>"#;
        let model = load_urdf_string(xml).unwrap();
        let ri = &model.inertias[1].rotational_inertia;
        assert_relative_eq!(ri[(0, 0)], 1.0, epsilon = 1e-9);
        assert_relative_eq!(ri[(1, 1)], 3.0, epsilon = 1e-9);
        assert_relative_eq!(ri[(2, 2)], 2.0, epsilon = 1e-9);
    }
}
