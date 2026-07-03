//! MJCF (MuJoCo XML) ⇄ [`MisaFile`] conversion.
//!
//! MJCF uses a nested body hierarchy rather than flat link/joint lists.
//! The importer flattens that hierarchy into the `.misa` master schema;
//! the exporter walks the joint topology back into nested `<body>`
//! elements. Neither direction touches the filesystem for meshes —
//! `Geom::Mesh.file` carries the (meshdir-composed) reference verbatim
//! and mesh loading / path policy is the host's concern.
//!
//! Ported from articara's `src/mjcf.rs` (A4, see articara
//! `doc/refactor_20260702.md` §4.7). Two importer bugs were fixed in the
//! port: `fullinertia` now follows MuJoCo's `Ixx Iyy Izz Ixy Ixz Iyz`
//! order (the old reader assumed `Ixx Ixy Ixz Iyy Iyz Izz`), and
//! `euler` attributes now respect `<compiler angle="degree">` (the old
//! reader always assumed radians).

use std::collections::{HashMap, HashSet};
use std::path::Path;

use misarta::native as mn;
use misarta::native::MisaFile;

use crate::util::{
    color_spec_to_rgba, config_iso, fmt, origin_rotation, parse_f64_list, parse_vec3_or,
    resolve_visual_rgba,
};

// ═══════════════════════════════ Import ════════════════════════════════

/// Result of a successful MJCF import: the converted [`MisaFile`] plus
/// non-fatal conversion notes (approximated joint kinds, skipped
/// elements). Hosts should surface the warnings to the user.
#[derive(Debug, Clone)]
pub struct MjcfImport {
    pub file: MisaFile,
    pub warnings: Vec<String>,
}

/// Flattened `<default>` class table.
///
/// Outer key: class name (the unnamed top-level `<default>` becomes
/// `"main"`). Inner: element tag (e.g. `joint`, `geom`, `motor`,
/// `position`) → attribute name → value. Each class's map already has
/// its parent class's defaults merged in, so a single lookup gives the
/// fully resolved attribute.
type ClassTable = HashMap<String, HashMap<String, HashMap<String, String>>>;

/// Resolve `attr` on `el`, falling back to the class system.
///
/// MuJoCo's resolution order is:
/// 1. explicit attribute on the element
/// 2. element's own `class="X"` defaults
/// 3. the most-recent ancestor `<body>`'s `childclass`
/// 4. the unnamed top-level `<default>` (`"main"`)
fn class_attr(
    el: roxmltree::Node,
    tag: &str,
    attr: &str,
    body_childclass: &str,
    table: &ClassTable,
) -> Option<String> {
    if let Some(v) = el.attribute(attr) {
        return Some(v.to_string());
    }
    // Inheritance fallback: class on the element wins over the body's
    // childclass; both fall back to "main".
    let candidates = [el.attribute("class").unwrap_or(""), body_childclass, "main"];
    for cls in candidates.iter().filter(|c| !c.is_empty()) {
        if let Some(v) = table
            .get(*cls)
            .and_then(|tags| tags.get(tag))
            .and_then(|attrs| attrs.get(attr))
        {
            return Some(v.clone());
        }
    }
    None
}

/// Walk a `<default>` element, merging its declared element defaults
/// onto `parent_class`'s already-flattened defaults, then recurse into
/// any nested `<default class="X">`.
fn walk_default(el: roxmltree::Node, parent_class: Option<&str>, table: &mut ClassTable) {
    // The outermost <default> is unnamed and becomes "main"; named
    // siblings/children carry their own class= attribute.
    let this_class = el.attribute("class").unwrap_or("main").to_string();

    // Start from the parent class's flattened map (deep clone) so each
    // class can be queried in one step.
    let mut my_defaults: HashMap<String, HashMap<String, String>> = parent_class
        .and_then(|pc| table.get(pc).cloned())
        .unwrap_or_default();

    for child in el.children().filter(|n| n.is_element()) {
        let tag = child.tag_name().name();
        if tag == "default" {
            // Nested class — handled in the recursion below.
            continue;
        }
        let entry = my_defaults.entry(tag.to_string()).or_default();
        for attr in child.attributes() {
            // Skip `class` itself — it's metadata, not a per-element default.
            if attr.name() == "class" {
                continue;
            }
            entry.insert(attr.name().to_string(), attr.value().to_string());
        }
    }

    table.insert(this_class.clone(), my_defaults);

    for child in el.children().filter(|n| n.tag_name().name() == "default") {
        walk_default(child, Some(&this_class), table);
    }
}

/// Build the full class table for a `<mujoco>` element.
fn parse_class_table(mujoco_el: roxmltree::Node) -> ClassTable {
    let mut table: ClassTable = HashMap::new();
    // The root <default> is optional; if missing, "main" is just empty.
    table.insert("main".to_string(), HashMap::new());
    for top in mujoco_el
        .children()
        .filter(|n| n.tag_name().name() == "default")
    {
        walk_default(top, None, &mut table);
    }
    table
}

/// Parse an MJCF file on disk into a [`MisaFile`].
pub fn import(path: &Path) -> Result<MjcfImport, String> {
    let xml = std::fs::read_to_string(path).map_err(|e| format!("Read MJCF: {e}"))?;
    import_str(&xml)
}

/// Parse MJCF XML text into a [`MisaFile`].
///
/// Mesh references keep their `meshdir`-composed relative form, so the
/// caller must resolve them against the MJCF file's directory (the same
/// base directory rule as `.misa` itself).
pub fn import_str(xml: &str) -> Result<MjcfImport, String> {
    let doc = roxmltree::Document::parse(xml).map_err(|e| format!("Parse MJCF XML: {e}"))?;

    let mujoco_el = doc
        .descendants()
        .find(|n| n.tag_name().name() == "mujoco")
        .ok_or("No <mujoco> element found")?;

    let robot_name = mujoco_el.attribute("model").unwrap_or("mjcf_model");

    // Angle unit: MJCF default is degree; Menagerie-style files set radian.
    let compiler_el = mujoco_el
        .descendants()
        .find(|n| n.tag_name().name() == "compiler");
    let angle_deg = compiler_el
        .and_then(|c| c.attribute("angle"))
        .map(|a| a == "degree")
        .unwrap_or(true);

    // Collect mesh assets: name -> (file, scale).
    //
    // Compose `meshdir` onto every asset path so the stored reference
    // resolves relative to the MJCF's own directory — both at load time
    // and when a host re-emits the file for in-process MuJoCo.
    let mut mesh_assets: HashMap<String, (String, [f64; 3])> = HashMap::new();
    if let Some(asset) = mujoco_el
        .children()
        .find(|n| n.tag_name().name() == "asset")
    {
        let meshdir_rel = compiler_el.and_then(|c| c.attribute("meshdir"));
        for mesh_el in asset.children().filter(|n| n.tag_name().name() == "mesh") {
            // MJCF allows `<mesh file="foo.obj"/>` with no explicit `name=`;
            // the asset name is then the file's basename without extension.
            // Menagerie (e.g. Unitree Go2) relies on this.
            let Some(file) = mesh_el.attribute("file") else {
                continue;
            };
            let name = mesh_el
                .attribute("name")
                .map(str::to_string)
                .unwrap_or_else(|| {
                    Path::new(file)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or(file)
                        .to_string()
                });
            // Skip the meshdir prefix when file= is already absolute or the
            // author wrote the composed form — keeps the import idempotent.
            let file_path = Path::new(file);
            let stored = if file_path.is_absolute() {
                file.to_string()
            } else if let Some(md) = meshdir_rel {
                Path::new(md).join(file).to_string_lossy().into_owned()
            } else {
                file.to_string()
            };
            let scale = mesh_el
                .attribute("scale")
                .map(|s| parse_vec3_or(s, [1.0, 1.0, 1.0]))
                .unwrap_or([1.0, 1.0, 1.0]);
            mesh_assets.insert(name, (stored, scale));
        }
    }

    let mut file = MisaFile::new(robot_name, "");
    let mut warnings: Vec<String> = Vec::new();
    let mut child_links: HashSet<String> = HashSet::new();

    // Build the <default> class table before walking bodies so each
    // `<joint>` / `<geom>` / `<motor>` element can fall back to its class
    // chain. Without this every joint axis / range / damping silently
    // collapses to MJCF's hard-coded defaults (Unitree Go2 declares all
    // joint axes in `<default class="abduction|hip|knee">` blocks).
    let class_table = parse_class_table(mujoco_el);

    let worldbody = mujoco_el
        .children()
        .find(|n| n.tag_name().name() == "worldbody")
        .ok_or("No <worldbody> element found")?;

    let ctx = ImportCtx {
        class_table: &class_table,
        mesh_assets: &mesh_assets,
        angle_deg,
    };
    walk_bodies(
        worldbody,
        None,
        "main",
        &ctx,
        &mut file,
        &mut child_links,
        &mut warnings,
    );

    // <actuator> after bodies so joint references exist.
    parse_actuators(mujoco_el, &class_table, &mut file, &mut warnings);

    parse_equality(mujoco_el, &mut file);
    parse_sensors(mujoco_el, &mut file);
    parse_contact_excludes(mujoco_el, &mut file);

    // Root link = not a child of any joint.
    file.robot.root = file
        .link
        .iter()
        .find(|l| !child_links.contains(&l.name))
        .map(|l| l.name.clone())
        .unwrap_or_default();

    Ok(MjcfImport { file, warnings })
}

struct ImportCtx<'a> {
    class_table: &'a ClassTable,
    mesh_assets: &'a HashMap<String, (String, [f64; 3])>,
    angle_deg: bool,
}

fn walk_bodies(
    parent_node: roxmltree::Node,
    parent_link: Option<&str>,
    parent_childclass: &str,
    ctx: &ImportCtx,
    file: &mut MisaFile,
    child_links: &mut HashSet<String>,
    warnings: &mut Vec<String>,
) {
    for body_el in parent_node
        .children()
        .filter(|n| n.tag_name().name() == "body")
    {
        // `childclass` propagates: a body's childclass becomes the default
        // class for all elements inside it (and its descendants) until
        // another body overrides it.
        let body_childclass = body_el
            .attribute("childclass")
            .unwrap_or(parent_childclass)
            .to_string();

        let body_name = body_el
            .attribute("name")
            .unwrap_or(&format!("body_{}", file.link.len()))
            .to_string();
        let body_origin = origin_from_el(body_el, ctx.angle_deg);

        let inertial = parse_inertial(body_el, ctx.angle_deg);

        // MJCF uses <geom> for both visual and collision; mirror each geom
        // into both lists so the host can prune either side afterwards.
        let mut visuals: Vec<mn::Visual> = Vec::new();
        let mut collisions: Vec<mn::Collision> = Vec::new();
        for geom_el in body_el
            .children()
            .filter(|n| n.tag_name().name() == "geom")
        {
            let geom = parse_geom(geom_el, &body_childclass, ctx);
            let origin = origin_from_el(geom_el, ctx.angle_deg);
            let color = class_attr(geom_el, "geom", "rgba", &body_childclass, ctx.class_table)
                .map(|s| {
                    let v: Vec<f32> =
                        s.split_whitespace().filter_map(|t| t.parse().ok()).collect();
                    mn::ColorSpec::Rgba([
                        v.first().copied().unwrap_or(0.8),
                        v.get(1).copied().unwrap_or(0.8),
                        v.get(2).copied().unwrap_or(0.8),
                        v.get(3).copied().unwrap_or(1.0),
                    ])
                });
            visuals.push(mn::Visual {
                origin: origin.clone(),
                geom: geom.clone(),
                color,
                material: None,
            });
            collisions.push(mn::Collision {
                origin,
                geom,
                physics: parse_geom_physics(geom_el),
            });
        }

        file.link.push(mn::Link {
            name: body_name.clone(),
            description: String::new(),
            inertial,
            visual: visuals,
            collision: collisions,
            collision_enabled: true,
        });

        // Joint(s) between parent and this body.
        if let Some(parent_name) = parent_link {
            let joint_els: Vec<_> = body_el
                .children()
                .filter(|n| n.tag_name().name() == "joint")
                .collect();

            if joint_els.is_empty() {
                // No <joint> → the body is welded to its parent.
                // (`<freejoint/>` on non-root bodies is not modelled; MJCF
                // only allows it directly under <worldbody> children anyway.)
                child_links.insert(body_name.clone());
                file.joint.push(mn::Joint {
                    name: format!("{body_name}_fixed"),
                    kind: mn::JointKind::Fixed,
                    parent: parent_name.to_string(),
                    child: body_name.clone(),
                    axis: [0.0, 0.0, 1.0],
                    origin: body_origin.clone(),
                    limit: mn::JointLimit::default(),
                    dynamics: mn::JointDynamics::default(),
                });
            } else {
                for joint_el in joint_els {
                    let jname = joint_el
                        .attribute("name")
                        .unwrap_or(&format!("joint_{}", file.joint.len()))
                        .to_string();

                    // `type` itself can come from <default><joint type="..."/>.
                    // MJCF's default is "hinge".
                    let jtype_raw =
                        class_attr(joint_el, "joint", "type", &body_childclass, ctx.class_table)
                            .unwrap_or_else(|| "hinge".to_string());
                    let kind = match jtype_raw.as_str() {
                        "hinge" => mn::JointKind::Revolute,
                        "slide" => mn::JointKind::Prismatic,
                        "free" => mn::JointKind::Floating,
                        "ball" => {
                            warnings.push(format!(
                                "joint '{jname}': ball joint approximated as 'floating' \
                                 (.misa schema has no spherical kind)"
                            ));
                            mn::JointKind::Floating
                        }
                        other => {
                            warnings.push(format!(
                                "joint '{jname}': unknown MJCF joint type '{other}', \
                                 treating as 'revolute'"
                            ));
                            mn::JointKind::Revolute
                        }
                    };

                    let axis =
                        class_attr(joint_el, "joint", "axis", &body_childclass, ctx.class_table)
                            .map(|s| parse_vec3_or(&s, [0.0, 0.0, 1.0]))
                            .unwrap_or([0.0, 0.0, 1.0]);

                    let (lower, upper) = if let Some(range) =
                        class_attr(joint_el, "joint", "range", &body_childclass, ctx.class_table)
                    {
                        let vals = parse_f64_list(&range);
                        let lo = vals.first().copied().unwrap_or(0.0);
                        let hi = vals.get(1).copied().unwrap_or(0.0);
                        if ctx.angle_deg && kind == mn::JointKind::Revolute {
                            (lo.to_radians(), hi.to_radians())
                        } else {
                            (lo, hi)
                        }
                    } else {
                        (0.0, 0.0)
                    };

                    let armature = class_attr(
                        joint_el, "joint", "armature", &body_childclass, ctx.class_table,
                    )
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                    let damping = class_attr(
                        joint_el, "joint", "damping", &body_childclass, ctx.class_table,
                    )
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);

                    child_links.insert(body_name.clone());
                    file.joint.push(mn::Joint {
                        name: jname,
                        kind,
                        parent: parent_name.to_string(),
                        child: body_name.clone(),
                        axis,
                        origin: body_origin.clone(),
                        limit: mn::JointLimit {
                            lower,
                            upper,
                            effort: 0.0,
                            velocity: 0.0,
                        },
                        dynamics: mn::JointDynamics {
                            armature,
                            damping,
                            friction: 0.0,
                        },
                    });
                }
            }
        }

        // Recurse into child bodies — propagate this body's effective
        // childclass so nested joints/geoms keep the right class chain.
        walk_bodies(
            body_el,
            Some(&body_name),
            &body_childclass,
            ctx,
            file,
            child_links,
            warnings,
        );
    }
}

/// Parse the top-level `<actuator>` block into `[[actuator]]` entries and
/// fold `ctrlrange` / `forcerange` onto the driven joint's effort limit.
///
/// MJCF element tags map to actuator modes:
/// - `<motor>`    → `Torque`
/// - `<position>` → `Position` (kp attr, kv defaults to 0)
/// - `<velocity>` → `Velocity` (kv attr, kp defaults to 0)
/// - `<general>`  → best-effort `Torque`
///
/// Unmapped tags (intvelocity, adhesion, muscle, …) are skipped. Class
/// inheritance resolves against the top-level `<default>` table only —
/// bodies aren't involved for actuators.
fn parse_actuators(
    mujoco_el: roxmltree::Node,
    class_table: &ClassTable,
    file: &mut MisaFile,
    warnings: &mut Vec<String>,
) {
    let Some(actuator_el) = mujoco_el
        .children()
        .find(|n| n.tag_name().name() == "actuator")
    else {
        return;
    };

    let joint_idx: HashMap<String, usize> = file
        .joint
        .iter()
        .enumerate()
        .map(|(i, j)| (j.name.clone(), i))
        .collect();

    for el in actuator_el.children().filter(|n| n.is_element()) {
        let tag = el.tag_name().name();
        // Default gains mirror the host-side JointData defaults so a
        // round-trip through the actuator table is behaviour-neutral.
        let (mode, default_kp, default_kv) = match tag {
            "motor" => (mn::ActuatorMode::Torque, 50.0, 5.0),
            "position" => (mn::ActuatorMode::Position, 50.0, 0.0),
            "velocity" => (mn::ActuatorMode::Velocity, 0.0, 5.0),
            "general" => (mn::ActuatorMode::Torque, 50.0, 5.0),
            _ => continue,
        };

        let Some(joint_name) = el.attribute("joint") else {
            continue;
        };
        let Some(&ji) = joint_idx.get(joint_name) else {
            warnings.push(format!(
                "actuator '{}' references unknown joint '{joint_name}' — skipped",
                el.attribute("name").unwrap_or(tag),
            ));
            continue;
        };

        let kp = class_attr(el, tag, "kp", "", class_table)
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(default_kp);
        let kv = class_attr(el, tag, "kv", "", class_table)
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(default_kv);
        let gear = class_attr(el, tag, "gear", "", class_table)
            .map(|s| parse_f64_list(&s))
            .and_then(|v| v.first().copied())
            .unwrap_or(1.0);

        // Effort: prefer forcerange (limit on output force), fall back to
        // ctrlrange (same units for motor-class actuators when gear == 1).
        if let Some(fr) = class_attr(el, tag, "forcerange", "", class_table)
            .or_else(|| class_attr(el, tag, "ctrlrange", "", class_table))
        {
            let vals = parse_f64_list(&fr);
            if vals.len() == 2 {
                let max_abs = vals[0].abs().max(vals[1].abs());
                if max_abs > 0.0 {
                    file.joint[ji].limit.effort = max_abs;
                }
            }
        }

        let name = el
            .attribute("name")
            .map(str::to_string)
            .unwrap_or_else(|| format!("{joint_name}_motor"));
        file.actuator.push(mn::Actuator {
            name,
            mode,
            joints: vec![mn::ActuatorJointRef {
                name: joint_name.to_string(),
                gear,
            }],
            kp,
            kv,
        });
    }
}

/// Parse `<equality>`: `<joint>` → mimic, `<connect>` → 3-DoF loop
/// closure, `<weld>` → 6-DoF loop closure.
fn parse_equality(mujoco_el: roxmltree::Node, file: &mut MisaFile) {
    let Some(eq) = mujoco_el
        .children()
        .find(|n| n.tag_name().name() == "equality")
    else {
        return;
    };

    for je in eq.children().filter(|n| n.tag_name().name() == "joint") {
        let (Some(j1), Some(j2)) = (je.attribute("joint1"), je.attribute("joint2")) else {
            continue;
        };
        // polycoef = "offset multiplier c2 c3 c4"; we keep the linear part.
        let mut multiplier = 1.0;
        let mut offset = 0.0;
        if let Some(poly) = je.attribute("polycoef") {
            let vals = parse_f64_list(poly);
            offset = vals.first().copied().unwrap_or(0.0);
            multiplier = vals.get(1).copied().unwrap_or(1.0);
        }
        file.mimic.push(mn::Mimic {
            joint: j1.to_string(),
            source: j2.to_string(),
            multiplier,
            offset,
        });
    }

    // <connect body1=… body2=… anchor="x y z"> → 3-DoF loop closure.
    for ce in eq.children().filter(|n| n.tag_name().name() == "connect") {
        let (Some(b1), Some(b2)) = (ce.attribute("body1"), ce.attribute("body2")) else {
            continue;
        };
        let anchor = ce
            .attribute("anchor")
            .map(|s| parse_vec3_or(s, [0.0; 3]))
            .unwrap_or([0.0; 3]);
        file.loop_closure.push(mn::LoopClosure {
            name: ce
                .attribute("name")
                .unwrap_or(&format!("{b1}_{b2}_loop"))
                .to_string(),
            link_a: b1.to_string(),
            offset_a: anchor,
            rot_a: [0.0, 0.0, 0.0, 1.0],
            link_b: b2.to_string(),
            offset_b: [0.0; 3],
            rot_b: [0.0, 0.0, 0.0, 1.0],
            pose_6dof: false,
        });
    }

    // <weld body1=… body2=… relpose="x y z qw qx qy qz"> → 6-DoF. The
    // relative pose is stored on the A side (B-side offset = identity),
    // matching how the exporter reconstructs `relpose = A · B⁻¹`.
    for we in eq.children().filter(|n| n.tag_name().name() == "weld") {
        let (Some(b1), Some(b2)) = (we.attribute("body1"), we.attribute("body2")) else {
            continue;
        };
        let mut t = [0.0; 3];
        let mut q_xyzw = [0.0, 0.0, 0.0, 1.0];
        if let Some(rp) = we.attribute("relpose") {
            let v = parse_f64_list(rp);
            if v.len() >= 7 {
                t = [v[0], v[1], v[2]];
                // relpose order is (w, x, y, z); config order is (x, y, z, w).
                q_xyzw = [v[4], v[5], v[6], v[3]];
            }
        }
        file.loop_closure.push(mn::LoopClosure {
            name: we
                .attribute("name")
                .unwrap_or(&format!("{b1}_{b2}_weld"))
                .to_string(),
            link_a: b1.to_string(),
            offset_a: t,
            rot_a: q_xyzw,
            link_b: b2.to_string(),
            offset_b: [0.0; 3],
            rot_b: [0.0, 0.0, 0.0, 1.0],
            pose_6dof: true,
        });
    }
}

/// Parse the top-level `<sensor>` block. Known types map to master
/// [`mn::SensorKind`]s; everything else round-trips as `Generic`.
fn parse_sensors(mujoco_el: roxmltree::Node, file: &mut MisaFile) {
    let Some(snode) = mujoco_el
        .children()
        .find(|n| n.tag_name().name() == "sensor")
    else {
        return;
    };
    for el in snode.children().filter(|n| n.is_element()) {
        let kind_str = el.tag_name().name();
        let name = el.attribute("name").unwrap_or(kind_str).to_string();
        // "site" is the usual mount attribute; fall back to "body" /
        // "objname" for the other sensor flavours.
        let link = el
            .attribute("site")
            .or_else(|| el.attribute("body"))
            .or_else(|| el.attribute("objname"))
            .unwrap_or("")
            .to_string();
        let kind = match kind_str {
            "accelerometer" | "gyro" | "velocimeter" => mn::SensorKind::Imu {
                gyro_noise: 0.0,
                accel_noise: 0.0,
            },
            "touch" => mn::SensorKind::Contact { partner: None },
            "force" | "torque" | "jointactuatorfrc" | "force_torque" => {
                mn::SensorKind::ForceTorque {
                    joint: el.attribute("joint").map(str::to_string),
                }
            }
            _ => mn::SensorKind::Generic {
                kind: kind_str.to_string(),
                params: el
                    .attributes()
                    .map(|a| (a.name().to_string(), a.value().to_string()))
                    .collect(),
            },
        };
        file.sensor.push(mn::Sensor {
            name,
            link,
            origin: mn::Origin::default(),
            update_rate: 0.0,
            kind,
        });
    }
}

/// Parse `<contact><exclude body1 body2/>` into disabled collision pairs.
fn parse_contact_excludes(mujoco_el: roxmltree::Node, file: &mut MisaFile) {
    let Some(cnode) = mujoco_el
        .children()
        .find(|n| n.tag_name().name() == "contact")
    else {
        return;
    };
    for el in cnode.children().filter(|n| n.tag_name().name() == "exclude") {
        let (Some(b1), Some(b2)) = (el.attribute("body1"), el.attribute("body2")) else {
            continue;
        };
        file.collision_pair.push(mn::CollisionPair {
            link_a: b1.to_string(),
            link_b: b2.to_string(),
            enabled: false,
        });
    }
}

// ─── Import helpers ─────────────────────────────────────────────────────

/// Build an [`mn::Origin`] from an element's `pos` and `quat`/`euler`
/// attributes. MJCF quat order is `w x y z`; the schema stores `x y z w`.
/// `euler` obeys the compiler's angle unit.
fn origin_from_el(node: roxmltree::Node, angle_deg: bool) -> mn::Origin {
    let xyz = node
        .attribute("pos")
        .map(|s| parse_vec3_or(s, [0.0; 3]))
        .unwrap_or([0.0; 3]);
    if let Some(qs) = node.attribute("quat") {
        let v = parse_f64_list(qs);
        if v.len() >= 4 {
            return mn::Origin {
                xyz,
                rpy: None,
                quat: Some([v[1], v[2], v[3], v[0]]),
            };
        }
    }
    if let Some(es) = node.attribute("euler") {
        let mut e = parse_vec3_or(es, [0.0; 3]);
        if angle_deg {
            e = [e[0].to_radians(), e[1].to_radians(), e[2].to_radians()];
        }
        return mn::Origin {
            xyz,
            rpy: Some(e),
            quat: None,
        };
    }
    mn::Origin {
        xyz,
        rpy: None,
        quat: None,
    }
}

fn parse_inertial(body_el: roxmltree::Node, angle_deg: bool) -> mn::Inertial {
    let Some(i) = body_el
        .children()
        .find(|n| n.tag_name().name() == "inertial")
    else {
        return mn::Inertial::default();
    };
    let mass = i.attribute("mass").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let origin = origin_from_el(i, angle_deg);

    let (ixx, iyy, izz, ixy, ixz, iyz) = if let Some(diag) = i.attribute("diaginertia") {
        let v = parse_f64_list(diag);
        (
            v.first().copied().unwrap_or(0.0),
            v.get(1).copied().unwrap_or(0.0),
            v.get(2).copied().unwrap_or(0.0),
            0.0,
            0.0,
            0.0,
        )
    } else if let Some(full) = i.attribute("fullinertia") {
        // MuJoCo order: Ixx Iyy Izz Ixy Ixz Iyz.
        let v = parse_f64_list(full);
        (
            v.first().copied().unwrap_or(0.0),
            v.get(1).copied().unwrap_or(0.0),
            v.get(2).copied().unwrap_or(0.0),
            v.get(3).copied().unwrap_or(0.0),
            v.get(4).copied().unwrap_or(0.0),
            v.get(5).copied().unwrap_or(0.0),
        )
    } else {
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    };

    mn::Inertial {
        mass,
        ixx,
        iyy,
        izz,
        ixy,
        ixz,
        iyz,
        origin,
    }
}

fn parse_geom(geom_el: roxmltree::Node, body_childclass: &str, ctx: &ImportCtx) -> mn::Geom {
    // `type` and `mesh` must resolve through the <default> chain: the
    // Menagerie convention declares `<geom type="mesh"/>` in a
    // `class="visual"` block, so the per-geom element carries only
    // `mesh="…" class="visual"`.
    let geom_type = class_attr(geom_el, "geom", "type", body_childclass, ctx.class_table)
        .unwrap_or_else(|| {
            // MJCF's true default is `type="mesh"` when a mesh is referenced,
            // otherwise `type="sphere"`.
            if class_attr(geom_el, "geom", "mesh", body_childclass, ctx.class_table).is_some() {
                "mesh".to_string()
            } else {
                "sphere".to_string()
            }
        });
    let size = class_attr(geom_el, "geom", "size", body_childclass, ctx.class_table)
        .map(|s| parse_f64_list(&s))
        .unwrap_or_default();

    // MJCF size fields are half-extents / half-lengths; the schema stores
    // full dimensions (URDF convention).
    match geom_type.as_str() {
        "box" => {
            let hx = size.first().copied().unwrap_or(0.05);
            let hy = size.get(1).copied().unwrap_or(hx);
            let hz = size.get(2).copied().unwrap_or(hy);
            mn::Geom::Box {
                size: [hx * 2.0, hy * 2.0, hz * 2.0],
            }
        }
        "cylinder" => mn::Geom::Cylinder {
            radius: size.first().copied().unwrap_or(0.05),
            length: size.get(1).copied().unwrap_or(0.1) * 2.0,
        },
        "capsule" => mn::Geom::Capsule {
            radius: size.first().copied().unwrap_or(0.05),
            length: size.get(1).copied().unwrap_or(0.1) * 2.0,
        },
        "sphere" => mn::Geom::Sphere {
            radius: size.first().copied().unwrap_or(0.05),
        },
        "mesh" => {
            if let Some(mesh_name) =
                class_attr(geom_el, "geom", "mesh", body_childclass, ctx.class_table)
            {
                let (file, scale) = ctx
                    .mesh_assets
                    .get(&mesh_name)
                    .cloned()
                    .unwrap_or_else(|| (format!("{mesh_name}.stl"), [1.0, 1.0, 1.0]));
                mn::Geom::Mesh { file, scale }
            } else {
                mn::Geom::Box {
                    size: [0.02, 0.02, 0.02],
                }
            }
        }
        _ => mn::Geom::Sphere {
            radius: size.first().copied().unwrap_or(0.05),
        },
    }
}

/// Read per-geom contact-physics attributes. Inline attributes only —
/// resolving these through the class chain would stamp an exporter's
/// `<default><geom friction=…/></default>` onto every collision, turning
/// the file-wide default into thousands of per-geom overrides.
fn parse_geom_physics(geom_el: roxmltree::Node) -> Option<mn::MjcfPhysics> {
    let friction = geom_el.attribute("friction").map(|s| {
        let v = parse_f64_list(s);
        [
            v.first().copied().unwrap_or(1.0),
            v.get(1).copied().unwrap_or(0.005),
            v.get(2).copied().unwrap_or(0.0001),
        ]
    });
    let condim = geom_el.attribute("condim").and_then(|s| s.parse().ok());
    let priority = geom_el.attribute("priority").and_then(|s| s.parse().ok());
    let solimp = geom_el.attribute("solimp").map(|s| {
        let v = parse_f64_list(s);
        [
            v.first().copied().unwrap_or(0.9),
            v.get(1).copied().unwrap_or(0.95),
            v.get(2).copied().unwrap_or(0.001),
        ]
    });
    let margin = geom_el.attribute("margin").and_then(|s| s.parse().ok());

    if friction.is_none()
        && condim.is_none()
        && priority.is_none()
        && solimp.is_none()
        && margin.is_none()
    {
        return None;
    }
    Some(mn::MjcfPhysics {
        friction,
        condim,
        priority,
        solimp,
        margin,
    })
}

// ═══════════════════════════════ Export ════════════════════════════════

/// World-frame ground plane to embed in exported MJCF (for MuJoCo sim).
#[derive(Clone, Copy, Debug)]
pub struct GroundPlaneCfg {
    /// Z height of the plane (world frame).
    pub z: f64,
    /// Half-extent (rendering hint; the plane is mathematically infinite).
    pub half_size: f64,
    /// Rotation about the X axis (radians).
    pub roll: f64,
    /// Rotation about the Y axis (radians).
    pub pitch: f64,
}

/// Options controlling how a [`MisaFile`] is exported to MJCF XML.
///
/// Host-side policies stay with the host: the floating-base position is
/// taken verbatim (auto-lift against loaded meshes is an editor concern),
/// and `Geom::Mesh.file` strings are emitted verbatim (path style /
/// mesh copying is the host's `MeshPathStyle` policy).
#[derive(Clone, Debug)]
pub struct MjcfExportOptions {
    /// World position of the root body.
    pub base_pos: [f64; 3],
    /// Embed a collidable ground plane geom at the given configuration.
    pub ground_plane: Option<GroundPlaneCfg>,
    /// When true, emit `<motor>` actuators (named `motor_<joint>`) for each
    /// non-fixed joint so torques can be applied via `data.ctrl`.
    pub add_actuators: bool,
    /// Per-axis locks on the floating base, ordered `[TX, TY, TZ, RX, RY, RZ]`.
    /// `true` = axis locked (no DoF), `false` = axis free.
    ///
    /// - All `false` → emit `<freejoint/>` (full 6-DoF base, the default)
    /// - All `true`  → emit no joint (base welded to the world at `base_pos`)
    /// - Mixed       → emit individual `<joint type="slide"/>` / hinge
    ///   elements only for the unlocked axes
    pub base_locked_axes: [bool; 6],
    /// When true, `<motor>` actuators carry `forcelimited="true"
    /// forcerange="-effort effort"` from the joint's effort limit.
    pub bake_actuator_limits: bool,
    /// When true, joints carry `range="lower upper"` so MuJoCo enforces
    /// position limits; false omits the range (limits off).
    pub bake_joint_position_limits: bool,
    /// Default contact friction for every emitted geom, ordered
    /// `[sliding, torsional, rolling]`. Emitted into
    /// `<default><geom friction="…"/></default>` so the ground plane and
    /// every collider inherit the same μ (MuJoCo combines pairs by
    /// per-axis max).
    pub default_friction: [f64; 3],
}

impl Default for MjcfExportOptions {
    fn default() -> Self {
        Self {
            base_pos: [0.0, 0.0, 0.0],
            ground_plane: None,
            add_actuators: false,
            base_locked_axes: [false; 6],
            bake_actuator_limits: true,
            bake_joint_position_limits: true,
            default_friction: [0.7, 0.005, 0.0001],
        }
    }
}

/// Slot key for mesh-asset lookup during body emission.
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
enum GeomSlot {
    Visual(usize, usize),
    Collision(usize, usize),
}

struct ExportCtx<'a> {
    file: &'a MisaFile,
    /// link name → index into `file.link`.
    link_idx: HashMap<&'a str, usize>,
    /// parent link name → joint indices, in file order.
    children: HashMap<&'a str, Vec<usize>>,
    /// child link name → joint index.
    parent_joint: HashMap<&'a str, usize>,
    joint_names: HashSet<&'a str>,
    /// First-wins actuator mode per joint (mirrors the host's rule).
    actuator_mode: HashMap<&'a str, mn::ActuatorMode>,
    /// material name → RGBA.
    materials: HashMap<&'a str, [f32; 4]>,
    /// geom slot → emitted `<mesh name=…>`.
    mesh_name: HashMap<GeomSlot, String>,
}

/// Export a [`MisaFile`] to MJCF XML text.
pub fn export(file: &MisaFile, opts: &MjcfExportOptions) -> String {
    let mut ctx = ExportCtx {
        file,
        link_idx: file
            .link
            .iter()
            .enumerate()
            .map(|(i, l)| (l.name.as_str(), i))
            .collect(),
        children: HashMap::new(),
        parent_joint: HashMap::new(),
        joint_names: file.joint.iter().map(|j| j.name.as_str()).collect(),
        actuator_mode: HashMap::new(),
        materials: file
            .material
            .iter()
            .map(|m| (m.name.as_str(), color_spec_to_rgba(&m.color)))
            .collect(),
        mesh_name: HashMap::new(),
    };
    for (i, j) in file.joint.iter().enumerate() {
        ctx.children.entry(j.parent.as_str()).or_default().push(i);
        ctx.parent_joint.insert(j.child.as_str(), i);
    }
    for a in &file.actuator {
        for jr in &a.joints {
            ctx.actuator_mode.entry(jr.name.as_str()).or_insert(a.mode);
        }
    }

    let mut s = String::new();
    s.push_str(&format!("<mujoco model=\"{}\">\n", file.robot.name));
    s.push_str("  <compiler angle=\"radian\"/>\n\n");

    // Sim-side default contact friction. MuJoCo's built-in default
    // (μ_sliding = 1.0) overshoots typical rubber-on-lab-floor values;
    // pairs combine via per-axis max, so setting it here gives every
    // contact the requested μ.
    s.push_str("  <default>\n");
    s.push_str(&format!(
        "    <geom friction=\"{} {} {}\"/>\n",
        fmt(opts.default_friction[0]),
        fmt(opts.default_friction[1]),
        fmt(opts.default_friction[2]),
    ));
    s.push_str("  </default>\n\n");

    // Mesh assets. One `<mesh>` per mesh geom occurrence, named mesh_N in
    // link order (visuals before collisions per link). `file=` strings are
    // emitted verbatim — the host has already applied its path policy.
    // The scale attribute matters: URDF-derived models commonly carry
    // millimetre meshes with scale="0.001 …"; dropping it makes MuJoCo
    // load them 1000× too large.
    let mut assets: Vec<(String, String, [f64; 3])> = Vec::new();
    for (li, link) in file.link.iter().enumerate() {
        for (gi, v) in link.visual.iter().enumerate() {
            if let mn::Geom::Mesh { file: f, scale } = &v.geom {
                let name = format!("mesh_{}", assets.len());
                ctx.mesh_name.insert(GeomSlot::Visual(li, gi), name.clone());
                assets.push((name, f.clone(), *scale));
            }
        }
        for (gi, c) in link.collision.iter().enumerate() {
            if let mn::Geom::Mesh { file: f, scale } = &c.geom {
                let name = format!("mesh_{}", assets.len());
                ctx.mesh_name
                    .insert(GeomSlot::Collision(li, gi), name.clone());
                assets.push((name, f.clone(), *scale));
            }
        }
    }
    if !assets.is_empty() {
        s.push_str("  <asset>\n");
        for (name, f, scale) in &assets {
            let unit = (scale[0] - 1.0).abs() < 1e-12
                && (scale[1] - 1.0).abs() < 1e-12
                && (scale[2] - 1.0).abs() < 1e-12;
            if unit {
                s.push_str(&format!("    <mesh name=\"{name}\" file=\"{f}\"/>\n"));
            } else {
                s.push_str(&format!(
                    "    <mesh name=\"{name}\" file=\"{f}\" scale=\"{} {} {}\"/>\n",
                    fmt(scale[0]),
                    fmt(scale[1]),
                    fmt(scale[2]),
                ));
            }
        }
        s.push_str("  </asset>\n\n");
    }

    s.push_str("  <worldbody>\n");

    if let Some(gp) = &opts.ground_plane {
        s.push_str(&format!(
            "    <geom name=\"ground\" type=\"plane\" pos=\"0 0 {z}\" euler=\"{roll} {pitch} 0\" size=\"{hs} {hs} 0.1\" rgba=\"0.5 0.5 0.55 1\"/>\n",
            z = fmt(gp.z),
            roll = fmt(gp.roll),
            pitch = fmt(gp.pitch),
            hs = fmt(gp.half_size),
        ));
    }

    let base_spec = BaseSpec {
        pos: opts.base_pos,
        locked: opts.base_locked_axes,
    };
    write_body(&mut s, &ctx, &file.robot.root, 4, Some(base_spec), opts);

    s.push_str("  </worldbody>\n");

    if opts.add_actuators {
        write_actuators(&mut s, &ctx, opts.bake_actuator_limits);
    }

    write_equalities(&mut s, &ctx);
    write_sensors(&mut s, &ctx);
    write_contact_excludes(&mut s, &ctx);

    s.push_str("</mujoco>\n");
    s
}

/// World-frame placement + per-axis lock state for the root body.
#[derive(Clone, Copy)]
struct BaseSpec {
    pos: [f64; 3],
    /// `[TX, TY, TZ, RX, RY, RZ]`: `true` = locked (no DoF).
    locked: [bool; 6],
}

fn write_body(
    s: &mut String,
    ctx: &ExportCtx,
    link_name: &str,
    indent: usize,
    base_spec: Option<BaseSpec>,
    opts: &MjcfExportOptions,
) {
    let pad: String = " ".repeat(indent);

    let Some(&link_i) = ctx.link_idx.get(link_name) else {
        return;
    };
    let link = &ctx.file.link[link_i];

    // Pose of this body: the connecting joint's origin (the root takes the
    // caller-resolved world position; identity orientation — the floating
    // base joints carry the trunk rotation at runtime).
    let (pos_str, quat_str, joint) = if let Some(spec) = base_spec {
        let [x, y, z] = spec.pos;
        (format!("{} {} {}", fmt(x), fmt(y), fmt(z)), String::new(), None)
    } else if let Some(&ji) = ctx.parent_joint.get(link_name) {
        let joint = &ctx.file.joint[ji];
        let xyz = joint.origin.xyz;
        // MJCF's <body> has no rpy attribute, so the joint-frame rotation
        // (which the child link inherits at q=0) is expressed as a quat
        // (`w x y z`). Dropping it was the root cause of the "forward
        // command produces lateral motion" gait bug on robots that spell
        // their thigh pitch axis as rpy="0 0 π/2" + axis="1 0 0".
        (
            format!("{} {} {}", fmt(xyz[0]), fmt(xyz[1]), fmt(xyz[2])),
            quat_attr_str(&joint.origin),
            Some(joint),
        )
    } else {
        ("0 0 0".into(), String::new(), None)
    };

    s.push_str(&format!(
        "{pad}<body name=\"{link_name}\" pos=\"{pos_str}\"{quat_str}>\n"
    ));

    // Root body: emit floating-base joints based on the per-axis lock state.
    if let Some(spec) = base_spec {
        let any_free = spec.locked.iter().any(|&l| !l);
        let all_free = spec.locked.iter().all(|&l| !l);
        if all_free {
            // Cleanest 6-DoF representation; avoids the gimbal singularity
            // of stacking three hinges for orientation.
            s.push_str(&format!("{pad}  <freejoint/>\n"));
        } else if any_free {
            // Partial constraint: only the unlocked axes, translations
            // first so the chain reads X→Y→Z→roll→pitch→yaw.
            const AXES: [(&str, &str, &str); 6] = [
                ("base_tx", "slide", "1 0 0"),
                ("base_ty", "slide", "0 1 0"),
                ("base_tz", "slide", "0 0 1"),
                ("base_rx", "hinge", "1 0 0"),
                ("base_ry", "hinge", "0 1 0"),
                ("base_rz", "hinge", "0 0 1"),
            ];
            for (i, (jname, jtype, axis)) in AXES.iter().enumerate() {
                if !spec.locked[i] {
                    s.push_str(&format!(
                        "{pad}  <joint name=\"{jname}\" type=\"{jtype}\" axis=\"{axis}\"/>\n",
                    ));
                }
            }
        }
        // else: all 6 locked → no joint; body welds to the world at `pos`.
    }

    // Inertial. Non-zero products of inertia mean the tensor's principal
    // axes are rotated relative to the link frame — `diaginertia` alone
    // would silently drop them, which is enough to make a heavy trunk look
    // unstable on contact. Use `fullinertia` whenever an off-diagonal is
    // non-trivial.
    let inr = &link.inertial;
    if inr.mass > 1e-12 {
        let p = inr.origin.xyz;
        let off_diag_eps = 1e-12;
        let has_off_diag = inr.ixy.abs() > off_diag_eps
            || inr.ixz.abs() > off_diag_eps
            || inr.iyz.abs() > off_diag_eps;
        if has_off_diag {
            s.push_str(&format!(
                "{pad}  <inertial mass=\"{}\" pos=\"{} {} {}\" \
                 fullinertia=\"{} {} {} {} {} {}\"/>\n",
                fmt(inr.mass),
                fmt(p[0]),
                fmt(p[1]),
                fmt(p[2]),
                fmt(inr.ixx),
                fmt(inr.iyy),
                fmt(inr.izz),
                fmt(inr.ixy),
                fmt(inr.ixz),
                fmt(inr.iyz),
            ));
        } else {
            s.push_str(&format!(
                "{pad}  <inertial mass=\"{}\" pos=\"{} {} {}\" diaginertia=\"{} {} {}\"/>\n",
                fmt(inr.mass),
                fmt(p[0]),
                fmt(p[1]),
                fmt(p[2]),
                fmt(inr.ixx),
                fmt(inr.iyy),
                fmt(inr.izz),
            ));
        }
    }

    // Joint. An actuator-mode `Fixed` is the "MJCF-only weld" shortcut:
    // omit the <joint> element so MuJoCo fuses parent and child, while the
    // declared kind stays movable for .misa / URDF round-trips.
    if let Some(joint) = joint {
        let mode_fixed = ctx
            .actuator_mode
            .get(joint.name.as_str())
            .is_some_and(|m| *m == mn::ActuatorMode::Fixed);
        if joint.kind != mn::JointKind::Fixed && !mode_fixed {
            let mjcf_type = match joint.kind {
                mn::JointKind::Revolute | mn::JointKind::Continuous => "hinge",
                mn::JointKind::Prismatic => "slide",
                mn::JointKind::Floating => "free",
                // MuJoCo has no planar joint; emit the name verbatim so the
                // incompatibility is visible instead of silently changing
                // the kinematics.
                mn::JointKind::Planar => "planar",
                mn::JointKind::Fixed => unreachable!(),
            };
            if joint.kind == mn::JointKind::Floating {
                s.push_str(&format!(
                    "{pad}  <joint name=\"{}\" type=\"free\"/>\n",
                    joint.name
                ));
            } else {
                s.push_str(&format!(
                    "{pad}  <joint name=\"{}\" type=\"{mjcf_type}\" axis=\"{} {} {}\"",
                    joint.name,
                    fmt(joint.axis[0]),
                    fmt(joint.axis[1]),
                    fmt(joint.axis[2]),
                ));
                if opts.bake_joint_position_limits && joint.limit.lower < joint.limit.upper {
                    s.push_str(&format!(
                        " range=\"{} {}\"",
                        fmt(joint.limit.lower),
                        fmt(joint.limit.upper),
                    ));
                }
                // Rotor inertia + passive damping map 1:1 to MuJoCo attrs.
                if joint.dynamics.armature > 0.0 {
                    s.push_str(&format!(" armature=\"{}\"", fmt(joint.dynamics.armature)));
                }
                if joint.dynamics.damping > 0.0 {
                    s.push_str(&format!(" damping=\"{}\"", fmt(joint.dynamics.damping)));
                }
                s.push_str("/>\n");
            }
        }
    }

    // Geoms. MuJoCo has no visual/collision split, so use the standard
    // contype/conaffinity/group convention:
    //   visual    → contype=0 conaffinity=0 group=1 (render only)
    //   collision → contype=1 conaffinity=1 group=3 (physics)
    // A link with `collision_enabled = false` gets visual-style bits on its
    // collision geoms — rendered in the group-3 viewer, skipped by physics.
    let visual_extra = " contype=\"0\" conaffinity=\"0\" group=\"1\"";
    let collision_extra = if link.collision_enabled {
        " contype=\"1\" conaffinity=\"1\" group=\"3\""
    } else {
        " contype=\"0\" conaffinity=\"0\" group=\"3\""
    };

    for (gi, vis) in link.visual.iter().enumerate() {
        let rgba = resolve_visual_rgba(vis, &ctx.materials);
        let rgba = format!("{} {} {} {}", rgba[0], rgba[1], rgba[2], rgba[3]);
        write_geom(
            s,
            &pad,
            &vis.geom,
            &vis.origin,
            &rgba,
            visual_extra,
            "",
            ctx.mesh_name.get(&GeomSlot::Visual(link_i, gi)),
        );
    }

    // Collision geoms carry a faint translucent green so they're
    // inspectable in the group-3 viewer; physics ignores rgba.
    let col_rgba = "0.4 0.85 0.4 0.25";
    for (gi, col) in link.collision.iter().enumerate() {
        let mut phys = String::new();
        if let Some(p) = &col.physics {
            if let Some(f) = p.friction {
                phys.push_str(&format!(
                    " friction=\"{} {} {}\"",
                    fmt(f[0]),
                    fmt(f[1]),
                    fmt(f[2])
                ));
            }
            if let Some(c) = p.condim {
                phys.push_str(&format!(" condim=\"{c}\""));
            }
            if let Some(pr) = p.priority {
                phys.push_str(&format!(" priority=\"{pr}\""));
            }
            if let Some(si) = p.solimp {
                phys.push_str(&format!(
                    " solimp=\"{} {} {}\"",
                    fmt(si[0]),
                    fmt(si[1]),
                    fmt(si[2])
                ));
            }
            if let Some(m) = p.margin {
                phys.push_str(&format!(" margin=\"{}\"", fmt(m)));
            }
        }
        write_geom(
            s,
            &pad,
            &col.geom,
            &col.origin,
            col_rgba,
            collision_extra,
            &phys,
            ctx.mesh_name.get(&GeomSlot::Collision(link_i, gi)),
        );
    }

    // Sites for sensors mounted on this link: `<accelerometer>` / `<gyro>`
    // reference a `<site>` for their attachment frame, named `<sensor>_site`
    // so `write_sensors` can refer to it deterministically.
    for sensor in &ctx.file.sensor {
        if sensor.link != link_name {
            continue;
        }
        let p = sensor.origin.xyz;
        let q = origin_rotation(&sensor.origin);
        s.push_str(&format!(
            "{pad}  <site name=\"{}_site\" pos=\"{} {} {}\" quat=\"{} {} {} {}\" size=\"0.005\"/>\n",
            sensor.name,
            fmt(p[0]),
            fmt(p[1]),
            fmt(p[2]),
            fmt(q.w),
            fmt(q.i),
            fmt(q.j),
            fmt(q.k),
        ));
    }

    // Recurse children in joint declaration order.
    if let Some(child_joints) = ctx.children.get(link_name) {
        for &ji in child_joints {
            write_body(s, ctx, &ctx.file.joint[ji].child, indent + 2, None, opts);
        }
    }

    s.push_str(&format!("{pad}</body>\n"));
}

/// Emit one `<geom>` element. Schema dimensions are full sizes; MJCF wants
/// half-extents / half-lengths.
#[allow(clippy::too_many_arguments)]
fn write_geom(
    s: &mut String,
    pad: &str,
    geom: &mn::Geom,
    origin: &mn::Origin,
    rgba: &str,
    extra: &str,
    phys: &str,
    mesh_name: Option<&String>,
) {
    let p = origin.xyz;
    let pos = format!("{} {} {}", fmt(p[0]), fmt(p[1]), fmt(p[2]));
    let quat = quat_attr_str(origin);
    match geom {
        mn::Geom::Box { size } => {
            s.push_str(&format!(
                "{pad}  <geom type=\"box\" pos=\"{pos}\"{quat} size=\"{} {} {}\" rgba=\"{rgba}\"{extra}{phys}/>\n",
                fmt(size[0] / 2.0),
                fmt(size[1] / 2.0),
                fmt(size[2] / 2.0),
            ));
        }
        mn::Geom::Cylinder { radius, length } => {
            s.push_str(&format!(
                "{pad}  <geom type=\"cylinder\" pos=\"{pos}\"{quat} size=\"{} {}\" rgba=\"{rgba}\"{extra}{phys}/>\n",
                fmt(*radius),
                fmt(length / 2.0),
            ));
        }
        mn::Geom::Sphere { radius } => {
            s.push_str(&format!(
                "{pad}  <geom type=\"sphere\" pos=\"{pos}\"{quat} size=\"{}\" rgba=\"{rgba}\"{extra}{phys}/>\n",
                fmt(*radius),
            ));
        }
        mn::Geom::Capsule { radius, length } => {
            s.push_str(&format!(
                "{pad}  <geom type=\"capsule\" pos=\"{pos}\"{quat} size=\"{} {}\" rgba=\"{rgba}\"{extra}{phys}/>\n",
                fmt(*radius),
                fmt(length / 2.0),
            ));
        }
        mn::Geom::Mesh { .. } => {
            if let Some(name) = mesh_name {
                s.push_str(&format!(
                    "{pad}  <geom type=\"mesh\" mesh=\"{name}\" pos=\"{pos}\"{quat} rgba=\"{rgba}\"{extra}{phys}/>\n",
                ));
            }
        }
    }
}

/// Emit one `<motor>` actuator per non-fixed joint (skipping joints whose
/// actuator mode is `Fixed` — the body emitter omitted their `<joint>`, so
/// an actuator would dangle). The MJCF is always plain torque-mode so the
/// same file works for any control strategy; per-joint mode/kp/kv live in
/// the master format for the host to apply externally.
fn write_actuators(s: &mut String, ctx: &ExportCtx, bake_limits: bool) {
    let movable: Vec<&mn::Joint> = ctx
        .file
        .joint
        .iter()
        .filter(|j| {
            j.kind != mn::JointKind::Fixed
                && !ctx
                    .actuator_mode
                    .get(j.name.as_str())
                    .is_some_and(|m| *m == mn::ActuatorMode::Fixed)
        })
        .collect();
    if movable.is_empty() {
        return;
    }
    s.push_str("\n  <actuator>\n");
    for joint in movable {
        let force_attrs = if bake_limits && joint.limit.effort > 0.0 {
            format!(
                " forcelimited=\"true\" forcerange=\"{} {}\"",
                fmt(-joint.limit.effort),
                fmt(joint.limit.effort),
            )
        } else {
            String::new()
        };
        s.push_str(&format!(
            "    <motor name=\"motor_{name}\" joint=\"{name}\" gear=\"1\"{force_attrs}/>\n",
            name = joint.name,
        ));
    }
    s.push_str("  </actuator>\n");
}

/// Emit `<equality>` covering mimics and closed kinematic loops:
///
/// - `<joint joint1=… joint2=… polycoef="off mult 0 0 0">` per mimic.
/// - `<connect body1=… body2=… anchor="x y z">` per 3-DoF loop closure
///   (anchor in body1's local frame; MuJoCo bakes the body2-local point
///   from the rest pose at compile time).
/// - `<weld body1=… body2=… relpose="x y z qw qx qy qz">` per 6-DoF loop.
///
/// Entries that reference unknown bodies / joints are silently dropped so
/// a partial model doesn't make MuJoCo refuse the file.
fn write_equalities(s: &mut String, ctx: &ExportCtx) {
    let active_mimics: Vec<&mn::Mimic> = ctx
        .file
        .mimic
        .iter()
        .filter(|m| {
            ctx.joint_names.contains(m.joint.as_str())
                && ctx.joint_names.contains(m.source.as_str())
        })
        .collect();
    let active_loops: Vec<&mn::LoopClosure> = ctx
        .file
        .loop_closure
        .iter()
        .filter(|lc| {
            ctx.link_idx.contains_key(lc.link_a.as_str())
                && ctx.link_idx.contains_key(lc.link_b.as_str())
        })
        .collect();
    if active_mimics.is_empty() && active_loops.is_empty() {
        return;
    }
    s.push_str("\n  <equality>\n");
    for m in active_mimics {
        s.push_str(&format!(
            "    <joint name=\"mimic_{}\" joint1=\"{}\" joint2=\"{}\" polycoef=\"{} {} 0 0 0\"/>\n",
            m.joint,
            m.joint,
            m.source,
            fmt(m.offset),
            fmt(m.multiplier),
        ));
    }
    for lc in active_loops {
        if lc.pose_6dof {
            // relpose = B seen from A at the constraint instant, from the
            // stored per-side offsets: A · B⁻¹.
            let iso_a = config_iso(lc.offset_a, lc.rot_a);
            let iso_b = config_iso(lc.offset_b, lc.rot_b);
            let rel = iso_a * iso_b.inverse();
            let rt = rel.translation.vector;
            let rq = rel.rotation.quaternion();
            s.push_str(&format!(
                "    <weld name=\"{}\" body1=\"{}\" body2=\"{}\" relpose=\"{} {} {} {} {} {} {}\"/>\n",
                lc.name,
                lc.link_a,
                lc.link_b,
                fmt(rt.x),
                fmt(rt.y),
                fmt(rt.z),
                fmt(rq.w),
                fmt(rq.i),
                fmt(rq.j),
                fmt(rq.k),
            ));
        } else {
            s.push_str(&format!(
                "    <connect name=\"{}\" body1=\"{}\" body2=\"{}\" anchor=\"{} {} {}\"/>\n",
                lc.name,
                lc.link_a,
                lc.link_b,
                fmt(lc.offset_a[0]),
                fmt(lc.offset_a[1]),
                fmt(lc.offset_a[2]),
            ));
        }
    }
    s.push_str("  </equality>\n");
}

/// Emit `<sensor>` entries. Core kinds map to the closest MuJoCo natives;
/// kinds without a direct equivalent become comments so the master record
/// isn't silently lost.
fn write_sensors(s: &mut String, ctx: &ExportCtx) {
    if ctx.file.sensor.is_empty() {
        return;
    }
    // Mount sites are emitted by `write_body` as `<site name="<name>_site"/>`
    // inside the body of `sensor.link`.
    s.push_str("\n  <sensor>\n");
    for sensor in &ctx.file.sensor {
        match &sensor.kind {
            mn::SensorKind::Imu {
                gyro_noise,
                accel_noise,
            } => {
                let accel_noise_attr = if *accel_noise > 0.0 {
                    format!(" noise=\"{}\"", fmt(*accel_noise))
                } else {
                    String::new()
                };
                let gyro_noise_attr = if *gyro_noise > 0.0 {
                    format!(" noise=\"{}\"", fmt(*gyro_noise))
                } else {
                    String::new()
                };
                s.push_str(&format!(
                    "    <accelerometer name=\"{}_accel\" site=\"{}_site\"{accel_noise_attr}/>\n",
                    sensor.name, sensor.name,
                ));
                s.push_str(&format!(
                    "    <gyro name=\"{}_gyro\" site=\"{}_site\"{gyro_noise_attr}/>\n",
                    sensor.name, sensor.name,
                ));
            }
            mn::SensorKind::ForceTorque { joint } => {
                if let Some(j) = joint {
                    s.push_str(&format!(
                        "    <jointactuatorfrc name=\"{}\" joint=\"{}\"/>\n",
                        sensor.name, j,
                    ));
                } else {
                    s.push_str(&format!(
                        "    <!-- force_torque '{}' on link '{}' (no joint specified) -->\n",
                        sensor.name, sensor.link,
                    ));
                }
            }
            mn::SensorKind::Contact { .. } => {
                s.push_str(&format!(
                    "    <touch name=\"{}\" site=\"{}\"/>\n",
                    sensor.name, sensor.link,
                ));
            }
            mn::SensorKind::Camera { .. } => {
                s.push_str(&format!(
                    "    <!-- camera '{}' on link '{}' — use <camera> element manually -->\n",
                    sensor.name, sensor.link,
                ));
            }
            mn::SensorKind::Lidar { .. } => {
                s.push_str(&format!(
                    "    <!-- lidar '{}' on link '{}' — MuJoCo has no native ray sensor -->\n",
                    sensor.name, sensor.link,
                ));
            }
            mn::SensorKind::Generic { kind, .. } => {
                s.push_str(&format!(
                    "    <!-- generic sensor '{}' (kind='{}') on link '{}' -->\n",
                    sensor.name, kind, sensor.link,
                ));
            }
        }
    }
    s.push_str("  </sensor>\n");
}

/// Emit `<contact><exclude>` for every joint's parent-child pair (URDF
/// semantics: links connected by a joint don't collide) plus every pair
/// the user explicitly disabled. MuJoCo's default is "all geoms collide",
/// so `enabled = true` pairs are implicit no-ops.
fn write_contact_excludes(s: &mut String, ctx: &ExportCtx) {
    let mut pairs: Vec<(String, String)> = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let record = |a: &str, b: &str, list: &mut Vec<(String, String)>, seen: &mut HashSet<(String, String)>| {
        // Canonical (sorted) key so (A,B) and (B,A) collapse.
        let (k_a, k_b) = if a <= b {
            (a.to_string(), b.to_string())
        } else {
            (b.to_string(), a.to_string())
        };
        if seen.insert((k_a.clone(), k_b.clone())) {
            list.push((k_a, k_b));
        }
    };
    for j in &ctx.file.joint {
        if ctx.link_idx.contains_key(j.parent.as_str())
            && ctx.link_idx.contains_key(j.child.as_str())
            && j.parent != j.child
        {
            record(&j.parent, &j.child, &mut pairs, &mut seen);
        }
    }
    for p in &ctx.file.collision_pair {
        if p.enabled {
            continue;
        }
        if ctx.link_idx.contains_key(p.link_a.as_str())
            && ctx.link_idx.contains_key(p.link_b.as_str())
        {
            record(&p.link_a, &p.link_b, &mut pairs, &mut seen);
        }
    }
    if pairs.is_empty() {
        return;
    }
    s.push_str("\n  <contact>\n");
    for (a, b) in &pairs {
        s.push_str(&format!("    <exclude body1=\"{a}\" body2=\"{b}\"/>\n"));
    }
    s.push_str("  </contact>\n");
}

// ─── Export helpers ─────────────────────────────────────────────────────

/// ` quat="w x y z"` attribute string, or empty for identity rotation.
fn quat_attr_str(o: &mn::Origin) -> String {
    let q = origin_rotation(o);
    let q = q.quaternion();
    let is_identity =
        (q.w - 1.0).abs() < 1e-9 && q.i.abs() < 1e-9 && q.j.abs() < 1e-9 && q.k.abs() < 1e-9;
    if is_identity {
        String::new()
    } else {
        format!(" quat=\"{} {} {} {}\"", fmt(q.w), fmt(q.i), fmt(q.j), fmt(q.k))
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"<?xml version="1.0"?>
<mujoco model="test_mjcf_robot">
  <compiler angle="radian"/>
  <default>
    <geom rgba="0.5 0.5 0.5 1"/>
  </default>
  <worldbody>
    <body name="base_link" pos="0 0 0">
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.1 0.1 0.05" rgba="0.5 0.5 0.5 1"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="joint1" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <inertial pos="0 0 0.1" mass="0.5" diaginertia="0.001 0.001 0.001"/>
        <geom type="cylinder" size="0.02 0.1" pos="0 0 0.1" rgba="1 0 0 1"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
          <inertial pos="0 0 0.075" mass="0.3" diaginertia="0.0005 0.0005 0.0005"/>
          <geom type="sphere" size="0.03" pos="0 0 0.075" rgba="0.5 0.5 0.5 1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"#;

    fn joint<'a>(f: &'a MisaFile, name: &str) -> &'a mn::Joint {
        f.joint
            .iter()
            .find(|j| j.name == name)
            .unwrap_or_else(|| panic!("joint {name} missing"))
    }

    #[test]
    fn import_basic_structure() {
        let out = import_str(FIXTURE).expect("import");
        let f = &out.file;
        assert!(out.warnings.is_empty(), "{:?}", out.warnings);
        assert_eq!(f.robot.name, "test_mjcf_robot");
        assert_eq!(f.robot.root, "base_link");
        assert_eq!(f.link.len(), 3);
        assert_eq!(f.joint.len(), 2);

        let j1 = joint(f, "joint1");
        assert_eq!(j1.kind, mn::JointKind::Revolute);
        assert_eq!(j1.parent, "base_link");
        assert_eq!(j1.child, "link1");
        assert!((j1.limit.lower - (-1.57)).abs() < 1e-9);
        assert!((j1.limit.upper - 1.57).abs() < 1e-9);
        assert_eq!(j1.axis, [0.0, 1.0, 0.0]);
        assert_eq!(j1.origin.xyz, [0.0, 0.0, 0.05]);

        let base = &f.link[0];
        assert!((base.inertial.mass - 1.0).abs() < 1e-12);
        // MJCF box size is half-extents; the schema stores full dims.
        assert!(matches!(
            base.visual[0].geom,
            mn::Geom::Box { size } if size == [0.2, 0.2, 0.1]
        ));
        // Each geom mirrors into a collision entry too.
        assert_eq!(base.collision.len(), 1);
    }

    #[test]
    fn import_respects_degree_mode_for_ranges_and_euler() {
        let xml = r#"<mujoco model="deg">
  <worldbody>
    <body name="root">
      <body name="child" pos="0 0 1" euler="0 0 90">
        <joint name="j" type="hinge" range="-90 90"/>
      </body>
    </body>
  </worldbody>
</mujoco>"#;
        // No <compiler angle=…> → MJCF default is degree.
        let f = import_str(xml).unwrap().file;
        let j = joint(&f, "j");
        assert!((j.limit.lower - (-std::f64::consts::FRAC_PI_2)).abs() < 1e-9);
        assert!((j.limit.upper - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
        let rpy = j.origin.rpy.expect("euler stored as rpy");
        assert!((rpy[2] - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    }

    #[test]
    fn import_fullinertia_uses_mujoco_order() {
        let xml = r#"<mujoco model="fi">
  <worldbody>
    <body name="root">
      <inertial pos="0 0 0" mass="2" fullinertia="0.1 0.2 0.3 0.01 0.02 0.03"/>
    </body>
  </worldbody>
</mujoco>"#;
        let f = import_str(xml).unwrap().file;
        let i = &f.link[0].inertial;
        assert_eq!(
            (i.ixx, i.iyy, i.izz, i.ixy, i.ixz, i.iyz),
            (0.1, 0.2, 0.3, 0.01, 0.02, 0.03)
        );
    }

    /// Port of articara's class-inheritance regression: a 2-level
    /// `<default>` hierarchy plus an `<actuator>` block with
    /// class-inherited ctrlrange — the same shape as Unitree Go2.
    #[test]
    fn default_class_and_actuator_inheritance() {
        let xml = r#"<mujoco model="cls_test">
  <compiler angle="radian"/>
  <default>
    <default class="robot">
      <joint axis="0 1 0" damping="2" armature="0.01"/>
      <motor ctrlrange="-25 25"/>
      <default class="abduction">
        <joint axis="1 0 0" range="-1.0 1.0"/>
      </default>
      <default class="knee">
        <joint range="-2.7 -0.8"/>
        <motor ctrlrange="-45 45"/>
      </default>
    </default>
  </default>
  <worldbody>
    <body name="base" pos="0 0 0.3" childclass="robot">
      <body name="hip" pos="0.1 0 0">
        <joint name="hip_j" class="abduction"/>
        <body name="thigh" pos="0 0 0">
          <joint name="thigh_j"/>
          <body name="calf" pos="0 0 -0.2">
            <joint name="calf_j" class="knee"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor class="abduction" name="hip_m" joint="hip_j"/>
    <motor class="robot"     name="thigh_m" joint="thigh_j"/>
    <motor class="knee"      name="calf_m" joint="calf_j"/>
  </actuator>
</mujoco>"#;
        let f = import_str(xml).unwrap().file;

        let hip = joint(&f, "hip_j");
        assert_eq!(hip.axis, [1.0, 0.0, 0.0]);
        assert_eq!((hip.limit.lower, hip.limit.upper), (-1.0, 1.0));
        assert_eq!((hip.dynamics.damping, hip.dynamics.armature), (2.0, 0.01));

        let thigh = joint(&f, "thigh_j");
        assert_eq!(thigh.axis, [0.0, 1.0, 0.0]);

        let calf = joint(&f, "calf_j");
        assert_eq!(calf.axis, [0.0, 1.0, 0.0]);
        assert_eq!((calf.limit.lower, calf.limit.upper), (-2.7, -0.8));

        // Effort from class-inherited ctrlrange; mode Torque via <motor>.
        assert_eq!(hip.limit.effort, 25.0);
        assert_eq!(thigh.limit.effort, 25.0);
        assert_eq!(calf.limit.effort, 45.0);
        assert_eq!(f.actuator.len(), 3);
        assert!(f.actuator.iter().all(|a| a.mode == mn::ActuatorMode::Torque));
    }

    #[test]
    fn import_meshdir_and_unnamed_asset_and_class_typed_geom() {
        let xml = r#"<mujoco model="mesh_class_test">
  <compiler angle="radian" meshdir="assets"/>
  <default>
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0"/>
    </default>
  </default>
  <asset>
    <mesh file="tri.obj" scale="0.001 0.001 0.001"/>
  </asset>
  <worldbody>
    <body name="root" pos="0 0 0">
      <geom mesh="tri" class="visual"/>
    </body>
  </worldbody>
</mujoco>"#;
        let f = import_str(xml).unwrap().file;
        let root = &f.link[0];
        assert_eq!(root.visual.len(), 1);
        match &root.visual[0].geom {
            mn::Geom::Mesh { file, scale } => {
                assert_eq!(file, "assets/tri.obj");
                assert_eq!(*scale, [0.001, 0.001, 0.001]);
            }
            other => panic!("expected mesh geom, got {other:?}"),
        }
    }

    #[test]
    fn import_equality_sensors_contact() {
        let xml = r#"<mujoco model="extras">
  <worldbody>
    <body name="a"><body name="b"><joint name="j1"/></body></body>
    <body name="c"><body name="d"><joint name="j2"/></body></body>
  </worldbody>
  <equality>
    <joint joint1="j1" joint2="j2" polycoef="0.5 2 0 0 0"/>
    <connect name="loop1" body1="b" body2="d" anchor="0.1 0 0"/>
    <weld name="weld1" body1="a" body2="c" relpose="1 2 3 1 0 0 0"/>
  </equality>
  <sensor>
    <gyro name="imu" site="a"/>
    <touch name="foot" site="d"/>
  </sensor>
  <contact>
    <exclude body1="a" body2="c"/>
  </contact>
</mujoco>"#;
        let f = import_str(xml).unwrap().file;
        assert_eq!(f.mimic.len(), 1);
        assert_eq!(f.mimic[0].offset, 0.5);
        assert_eq!(f.mimic[0].multiplier, 2.0);
        assert_eq!(f.loop_closure.len(), 2);
        assert!(!f.loop_closure[0].pose_6dof);
        assert_eq!(f.loop_closure[0].offset_a, [0.1, 0.0, 0.0]);
        assert!(f.loop_closure[1].pose_6dof);
        assert_eq!(f.loop_closure[1].offset_a, [1.0, 2.0, 3.0]);
        assert_eq!(f.sensor.len(), 2);
        assert!(matches!(f.sensor[0].kind, mn::SensorKind::Imu { .. }));
        assert!(matches!(f.sensor[1].kind, mn::SensorKind::Contact { .. }));
        assert_eq!(f.collision_pair.len(), 1);
        assert!(!f.collision_pair[0].enabled);
    }

    #[test]
    fn export_basic_and_roundtrip() {
        let out = import_str(FIXTURE).unwrap();
        let xml = export(&out.file, &MjcfExportOptions::default());
        assert!(xml.contains("<mujoco model=\"test_mjcf_robot\">"));
        assert!(xml.contains("joint1"));
        // Parent-child contact excludes are emitted automatically.
        assert!(xml.contains("<exclude body1=\"base_link\" body2=\"link1\"/>"));
        // Half-extent conversion round-trips: full 0.2 → half 0.1.
        assert!(xml.contains("size=\"0.1 0.1 0.05\""), "{xml}");

        let back = import_str(&xml).expect("re-import");
        assert_eq!(back.file.link.len(), out.file.link.len());
        assert_eq!(back.file.joint.len(), out.file.joint.len());
        let j1a = joint(&out.file, "joint1");
        let j1b = joint(&back.file, "joint1");
        assert!((j1a.limit.lower - j1b.limit.lower).abs() < 1e-12);
        assert!((j1a.limit.upper - j1b.limit.upper).abs() < 1e-12);
        assert_eq!(j1a.origin.xyz, j1b.origin.xyz);
    }

    #[test]
    fn export_base_lock_variants() {
        let out = import_str(FIXTURE).unwrap();
        // All free → freejoint.
        let xml = export(&out.file, &MjcfExportOptions::default());
        assert!(xml.contains("<freejoint/>"));
        // All locked → no root joint.
        let xml = export(
            &out.file,
            &MjcfExportOptions {
                base_locked_axes: [true; 6],
                ..Default::default()
            },
        );
        assert!(!xml.contains("<freejoint/>"));
        assert!(!xml.contains("base_tx"));
        // Mixed → individual axes.
        let xml = export(
            &out.file,
            &MjcfExportOptions {
                base_locked_axes: [false, true, true, true, true, false],
                ..Default::default()
            },
        );
        assert!(xml.contains("base_tx"));
        assert!(xml.contains("base_rz"));
        assert!(!xml.contains("base_ty"));
    }

    #[test]
    fn export_actuators_and_ground_plane() {
        let mut out = import_str(FIXTURE).unwrap();
        out.file.joint[0].limit.effort = 20.0;
        let xml = export(
            &out.file,
            &MjcfExportOptions {
                add_actuators: true,
                ground_plane: Some(GroundPlaneCfg {
                    z: -0.1,
                    half_size: 5.0,
                    roll: 0.0,
                    pitch: 0.0,
                }),
                base_pos: [0.0, 0.0, 0.4],
                ..Default::default()
            },
        );
        assert!(xml.contains("<motor name=\"motor_joint1\" joint=\"joint1\" gear=\"1\" forcelimited=\"true\" forcerange=\"-20 20\"/>"));
        assert!(xml.contains("type=\"plane\""));
        assert!(xml.contains("pos=\"0 0 0.4\""));
    }

    #[test]
    fn export_mimic_and_weld_roundtrip() {
        let out = import_str(
            r#"<mujoco model="eq">
  <worldbody>
    <body name="a"><body name="b"><joint name="j1"/></body>
      <body name="c"><joint name="j2"/></body></body>
  </worldbody>
  <equality>
    <joint joint1="j1" joint2="j2" polycoef="0.25 -1.5 0 0 0"/>
    <weld name="w" body1="b" body2="c" relpose="0.5 0 0 1 0 0 0"/>
  </equality>
</mujoco>"#,
        )
        .unwrap();
        let xml = export(&out.file, &MjcfExportOptions::default());
        assert!(xml.contains("polycoef=\"0.25 -1.5 0 0 0\""));
        assert!(xml.contains("relpose=\"0.5 0 0 1 0 0 0\""), "{xml}");

        let back = import_str(&xml).unwrap().file;
        assert_eq!(back.mimic.len(), 1);
        assert_eq!(back.loop_closure.len(), 1);
        assert_eq!(back.loop_closure[0].offset_a, [0.5, 0.0, 0.0]);
        assert!(back.loop_closure[0].pose_6dof);
    }

    #[test]
    fn fixed_actuator_mode_welds_joint() {
        let mut out = import_str(FIXTURE).unwrap();
        out.file.actuator.push(mn::Actuator {
            name: "weld_joint2".into(),
            mode: mn::ActuatorMode::Fixed,
            joints: vec![mn::ActuatorJointRef {
                name: "joint2".into(),
                gear: 1.0,
            }],
            kp: 0.0,
            kv: 0.0,
        });
        let xml = export(
            &out.file,
            &MjcfExportOptions {
                add_actuators: true,
                ..Default::default()
            },
        );
        // joint2 is welded: no <joint> element, no motor.
        assert!(!xml.contains("<joint name=\"joint2\""));
        assert!(!xml.contains("motor_joint2"));
        assert!(xml.contains("<joint name=\"joint1\""));
        assert!(xml.contains("motor_joint1"));
    }
}
