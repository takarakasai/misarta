//! Serde data types for the `.misa` master format.
//!
//! These types describe the on-disk shape of a `.misa` TOML file. They are
//! deliberately decoupled from misarta's runtime types (`Model`,
//! `GeometryModel`, etc.); conversion happens in the parse/save layer so
//! schema evolution is independent from runtime data structures.
//!
//! # Conventions
//!
//! - Units: meters, kilograms, radians, seconds (implicit, no `[units]` table).
//! - Axis: Z-up (implicit). Exporters that target Y-up formats handle the swap.
//! - Identifiers: `^[A-Za-z_][A-Za-z0-9_]*$`. Violations are sanitised on load
//!   and reported via `LoadReport.sanitized_names`.
//! - Field names match TOML table / key names directly (singular nouns for
//!   `[[…]]` arrays — same convention as `crate::config::MisartaConfig`).
//! - Reuse: types that are unchanged from `crate::config` (poses, sequences,
//!   gaits, home, mimics, loop closures, collision pairs) are re-exported
//!   here as type aliases so the schema stays a single point of reference.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ─── Re-exported stable types from crate::config ──────────────────────────
//
// These schemas have not changed in the move from `.misarta.toml` (sidecar)
// to `.misa` (master). Exposing them as aliases keeps the public surface
// consistent and avoids duplicate definitions.

pub use crate::config::ActuatorMode;
pub use crate::config::CollisionPairConfig as CollisionPair;
pub use crate::config::GaitConfigEntry as Gait;
pub use crate::config::GaitTypeConfig;
pub use crate::config::HomeConfig as Home;
pub use crate::config::LoopClosureConfig as LoopClosure;
pub use crate::config::MimicConfig as Mimic;
pub use crate::config::PoseConfig as Pose;
pub use crate::config::SequenceConfig as Sequence;
pub use crate::config::SequenceStepConfig as SequenceStep;

/// Current `.misa` format major version. Bumped on breaking schema changes;
/// additive changes use `#[serde(default)]` and keep the version intact.
pub const CURRENT_VERSION: u32 = 1;

/// The schema string written at the top of every `.misa` file. Used as
/// both the version marker and a content identifier — the bare `.misa`
/// extension alone is ambiguous, this line makes the file self-describing.
pub const SCHEMA_TAG: &str = "misarta/1";

// ─── Top-level file ────────────────────────────────────────────────────────

/// Top-level `.misa` file. Mirrors the on-disk TOML layout one-to-one.
///
/// Field order in this struct is also the canonical write order — the saver
/// emits sections in this sequence so diffs stay stable across edits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MisaFile {
    /// Schema identifier — must equal [`SCHEMA_TAG`] (e.g. `"misarta/1"`).
    /// Loaders reject anything they don't recognise rather than silently
    /// misinterpreting an unknown future format.
    pub schema: String,

    /// Robot-level metadata (name, root link).
    pub robot: RobotMeta,

    /// Optional named materials shared across visuals. Visuals can either
    /// reference one by name (`material = "red_plastic"`) or carry an
    /// inline `color`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub material: Vec<Material>,

    /// Rigid bodies in the kinematic tree. Order is not significant; the
    /// tree is reconstructed via `joint.parent` / `joint.child`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub link: Vec<Link>,

    /// Joints connecting links. Defines the tree topology.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub joint: Vec<Joint>,

    /// Linear coupling between joints (`q_target = mult · q_source + offset`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mimic: Vec<Mimic>,

    /// Closed-loop kinematic constraints — pairs of links pinned to each
    /// other beyond the tree topology.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loop_closure: Vec<LoopClosure>,

    /// Per-pair collision overrides. Pairs not listed here use the host
    /// default (collide); listed pairs with `enabled = false` are excluded.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub collision_pair: Vec<CollisionPair>,

    /// Sensors mounted on links (camera, lidar, IMU, force/torque, contact).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sensor: Vec<Sensor>,

    /// Actuators. Each actuator may drive 1 or N joints with per-joint gear
    /// ratios, supporting differential drives, tendons, and other N:M
    /// transmissions that a per-joint inline form cannot express.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actuator: Vec<Actuator>,

    /// Named joint-space poses for replay.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pose: Vec<Pose>,

    /// Pose sequences (chained replay).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sequence: Vec<Sequence>,

    /// Quadruped gait presets.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gait: Vec<Gait>,

    /// Home pose — joint angles and floating-base transform restored at load.
    #[serde(default)]
    pub home: Home,
}

impl MisaFile {
    /// Construct an empty file with the current schema tag.
    pub fn new(robot_name: impl Into<String>, root_link: impl Into<String>) -> Self {
        Self {
            schema: SCHEMA_TAG.into(),
            robot: RobotMeta {
                name: robot_name.into(),
                root: root_link.into(),
            },
            material: Vec::new(),
            link: Vec::new(),
            joint: Vec::new(),
            mimic: Vec::new(),
            loop_closure: Vec::new(),
            collision_pair: Vec::new(),
            sensor: Vec::new(),
            actuator: Vec::new(),
            pose: Vec::new(),
            sequence: Vec::new(),
            gait: Vec::new(),
            home: Home::default(),
        }
    }

    /// Parse the schema tag into `(vendor, major)`. Returns `None` if the tag
    /// doesn't match the expected `"vendor/major"` shape — callers treat that
    /// as an error.
    pub fn parse_schema(tag: &str) -> Option<(&str, u32)> {
        let (vendor, ver) = tag.split_once('/')?;
        let major = ver.parse().ok()?;
        Some((vendor, major))
    }
}

// ─── Robot metadata ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotMeta {
    pub name: String,
    /// Root link name. Must match exactly one entry in `link`.
    pub root: String,
}

// ─── Material ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Material {
    pub name: String,
    pub color: ColorSpec,
}

/// Color literal — hex string (`"#RRGGBB"` / `"#RRGGBBAA"`) or RGBA float
/// array `[r, g, b, a]` in 0..1. Both round-trip; the loader normalises
/// everything to `[f32; 4]` internally.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ColorSpec {
    Hex(String),
    Rgba([f32; 4]),
}

// ─── Link ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    pub name: String,

    /// Optional human-readable description shown as tooltip in the editor.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,

    /// Inertial properties. Defaults to mass = 0 / zero tensor when omitted
    /// (treated as a frame-only link by the dynamics engine).
    #[serde(default)]
    pub inertial: Inertial,

    /// Visual geometries. Empty vec means "no visual representation".
    /// Field name singular to match TOML `[[link.visual]]` syntax.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub visual: Vec<Visual>,

    /// Collision geometries. Often a coarser approximation of `visual`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub collision: Vec<Collision>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Inertial {
    #[serde(default)]
    pub mass: f64,
    #[serde(default)]
    pub ixx: f64,
    #[serde(default)]
    pub iyy: f64,
    #[serde(default)]
    pub izz: f64,
    #[serde(default)]
    pub ixy: f64,
    #[serde(default)]
    pub ixz: f64,
    #[serde(default)]
    pub iyz: f64,
    /// Inertial frame relative to the link frame. Default identity = inertia
    /// expressed at the link origin.
    #[serde(default, skip_serializing_if = "Origin::is_identity")]
    pub origin: Origin,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Visual {
    #[serde(default, skip_serializing_if = "Origin::is_identity")]
    pub origin: Origin,

    pub geom: Geom,

    /// Inline color. Mutually exclusive with `material`. Loader rejects
    /// entries that specify both.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<ColorSpec>,

    /// Reference to a `[[material]]` entry by name. Mutually exclusive
    /// with `color`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collision {
    #[serde(default, skip_serializing_if = "Origin::is_identity")]
    pub origin: Origin,
    pub geom: Geom,
}

// ─── Geometry tagged union ────────────────────────────────────────────────
//
// Default external tagging: `geom = { box = { size = [w, h, d] } }`.
// Each variant uses URDF-style full dimensions (size / length), not
// half-extents — those are an internal `RobotModel` convention only.

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Geom {
    /// Axis-aligned box. `size = [width, height, depth]` (full dimensions).
    Box { size: [f64; 3] },

    /// Cylinder along the local Z axis. `length` is the full length, not
    /// half-length, matching URDF / SDF / MJCF convention.
    Cylinder { radius: f64, length: f64 },

    Sphere { radius: f64 },

    /// Capsule along the local Z axis. `length` is the **cylindrical
    /// portion** length only — total along-axis span is `length + 2 * radius`.
    Capsule { radius: f64, length: f64 },

    /// Reference to an external mesh file. Path is resolved by the
    /// `AssetSource` (typically relative to the `.misa` file).
    Mesh {
        file: String,
        #[serde(default = "default_one3", skip_serializing_if = "is_one3")]
        scale: [f64; 3],
    },
}

// ─── Joint ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Joint {
    pub name: String,

    /// Joint kind. Renamed to `type` in TOML — `type` is a Rust keyword.
    #[serde(rename = "type")]
    pub kind: JointKind,

    pub parent: String,
    pub child: String,

    /// Axis in the child-link frame. Defaults to `[0, 0, 1]` when omitted;
    /// must be unit-length (loader normalises).
    #[serde(default = "default_z_axis")]
    pub axis: [f64; 3],

    #[serde(default, skip_serializing_if = "Origin::is_identity")]
    pub origin: Origin,

    /// Position / effort / velocity limits. Optional — a missing `limit`
    /// table means "use the dynamics engine's defaults" (typically
    /// unlimited / continuous).
    #[serde(default, skip_serializing_if = "JointLimit::is_default")]
    pub limit: JointLimit,

    /// Passive physical properties (rotor inertia, bearing damping). These
    /// belong on the joint regardless of whether an actuator drives it.
    /// Control gains (kp / kv / mode) live on `[[actuator]]` instead.
    #[serde(default, skip_serializing_if = "JointDynamics::is_default")]
    pub dynamics: JointDynamics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JointKind {
    Revolute,
    Continuous,
    Prismatic,
    Fixed,
    Floating,
    Planar,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JointLimit {
    #[serde(default)]
    pub lower: f64,
    #[serde(default)]
    pub upper: f64,
    /// Maximum |torque| (revolute) or |force| (prismatic). 0 means "no
    /// limit declared".
    #[serde(default)]
    pub effort: f64,
    /// Maximum |velocity| (rad/s or m/s). 0 means "no limit declared".
    #[serde(default)]
    pub velocity: f64,
}

impl JointLimit {
    fn is_default(&self) -> bool {
        self.lower == 0.0 && self.upper == 0.0 && self.effort == 0.0 && self.velocity == 0.0
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JointDynamics {
    /// Reflected rotor inertia (kg·m² for revolute, kg for prismatic).
    /// Real motors / gearboxes have non-zero armature; leaving it at 0
    /// makes the simulator more prone to numerical oscillation than the
    /// physical system would be.
    #[serde(default)]
    pub armature: f64,

    /// Passive linear damping at the joint (N·m·s/rad for revolute,
    /// N·s/m for prismatic). Models bearing friction / lubricant drag.
    #[serde(default)]
    pub damping: f64,

    /// Coulomb (dry) friction. Optional — not all dynamics engines model it.
    #[serde(default)]
    pub friction: f64,
}

impl JointDynamics {
    fn is_default(&self) -> bool {
        self.armature == 0.0 && self.damping == 0.0 && self.friction == 0.0
    }
}

// ─── Actuator (N:M capable) ───────────────────────────────────────────────

/// One actuator, possibly driving multiple joints with per-joint gear
/// ratios. The 1:1 case (one entry in `joints` with `gear = 1.0`) covers
/// most servos; multi-joint forms cover differential drives, tendon
/// networks, and other N:M transmissions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Actuator {
    pub name: String,

    #[serde(default)]
    pub mode: ActuatorMode,

    /// Joints driven by this actuator with their gear ratios. Must be
    /// non-empty.
    pub joints: Vec<ActuatorJointRef>,

    /// Position gain. Used by `Position` and `ComputedTorque` modes;
    /// ignored in `Velocity` / `Torque`.
    #[serde(default = "default_actuator_kp")]
    pub kp: f64,

    /// Damping / velocity gain. Used by `Position`, `Velocity`,
    /// `ComputedTorque`; ignored in `Torque`.
    #[serde(default = "default_actuator_kv")]
    pub kv: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActuatorJointRef {
    pub name: String,

    /// Linear coupling coefficient (MJCF `gear`, tendon `coef`). Defaults
    /// to 1.0 for the common 1:1 case.
    #[serde(default = "default_one")]
    pub gear: f64,
}

// ─── Sensor ────────────────────────────────────────────────────────────────
//
// Distinct from `crate::config::SensorConfig` because that one uses
// flat `[f64; 3]` + `[f64; 4]` for origin/orientation. The `.misa` schema
// uses the unified `Origin` type so the surface is consistent across all
// link-attached entities.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sensor {
    pub name: String,

    /// Link the sensor is rigidly attached to.
    pub link: String,

    #[serde(default, skip_serializing_if = "Origin::is_identity")]
    pub origin: Origin,

    /// Sample rate in Hz. 0 means "let the simulator pick a default".
    #[serde(default)]
    pub update_rate: f64,

    pub kind: SensorKind,
}

/// Sensor type. Default external tagging — `kind = { lidar = { … } }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorKind {
    Camera {
        #[serde(default = "default_fov")]
        fov: f64,
        #[serde(default = "default_width")]
        width: u32,
        #[serde(default = "default_height")]
        height: u32,
        #[serde(default = "default_near")]
        near: f64,
        #[serde(default = "default_far")]
        far: f64,
    },
    Lidar {
        #[serde(default = "default_lidar_range_min")]
        range_min: f64,
        #[serde(default = "default_lidar_range_max")]
        range_max: f64,
        #[serde(default = "default_lidar_h_fov")]
        h_fov: f64,
        #[serde(default = "default_lidar_h_samples")]
        h_samples: u32,
        #[serde(default)]
        v_fov: f64,
        #[serde(default = "default_lidar_v_samples")]
        v_samples: u32,
    },
    Imu {
        #[serde(default)]
        gyro_noise: f64,
        #[serde(default)]
        accel_noise: f64,
    },
    ForceTorque {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        joint: Option<String>,
    },
    Contact {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        partner: Option<String>,
    },
    /// Escape hatch for sensor types not covered above. Round-trips as
    /// raw key/value strings so unknown sensors survive import/export.
    Generic {
        kind: String,
        #[serde(default)]
        params: BTreeMap<String, String>,
    },
}

// ─── Origin (rpy default, quat alternate) ─────────────────────────────────

/// 6-DoF placement. Translation is always `xyz`; rotation may be expressed
/// as either Euler angles (`rpy`, radians, ZYX intrinsic) **or** a
/// quaternion (`quat = [x, y, z, w]`). At most one rotation form may be
/// present in any single instance — the loader rejects entries that
/// specify both. Both omitted = identity rotation.
///
/// `rpy` is the default for human-edited files because angles in radians
/// are immediately interpretable. `quat` is the right choice when exact
/// round-trip with USD or other quaternion-native sources is needed (no
/// gimbal-lock loss of precision near pole orientations).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Origin {
    #[serde(default = "default_zero3", skip_serializing_if = "is_zero3")]
    pub xyz: [f64; 3],

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rpy: Option<[f64; 3]>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quat: Option<[f64; 4]>,
}

impl Origin {
    pub fn is_identity(&self) -> bool {
        is_zero3(&self.xyz)
            && self.rpy.map_or(true, |r| is_zero3(&r))
            && self.quat.map_or(true, |q| is_identity_quat(&q))
    }
}

// ─── Defaults / helpers ────────────────────────────────────────────────────

fn default_zero3() -> [f64; 3] {
    [0.0, 0.0, 0.0]
}
fn default_one3() -> [f64; 3] {
    [1.0, 1.0, 1.0]
}
fn default_z_axis() -> [f64; 3] {
    [0.0, 0.0, 1.0]
}
fn default_one() -> f64 {
    1.0
}
fn default_actuator_kp() -> f64 {
    50.0
}
fn default_actuator_kv() -> f64 {
    5.0
}
fn default_fov() -> f64 {
    std::f64::consts::FRAC_PI_3
}
fn default_width() -> u32 {
    640
}
fn default_height() -> u32 {
    480
}
fn default_near() -> f64 {
    0.05
}
fn default_far() -> f64 {
    100.0
}
fn default_lidar_range_min() -> f64 {
    0.05
}
fn default_lidar_range_max() -> f64 {
    30.0
}
fn default_lidar_h_fov() -> f64 {
    std::f64::consts::TAU
}
fn default_lidar_h_samples() -> u32 {
    360
}
fn default_lidar_v_samples() -> u32 {
    1
}

fn is_zero3(v: &[f64; 3]) -> bool {
    v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0
}
fn is_one3(v: &[f64; 3]) -> bool {
    v[0] == 1.0 && v[1] == 1.0 && v[2] == 1.0
}
fn is_identity_quat(q: &[f64; 4]) -> bool {
    q[0] == 0.0 && q[1] == 0.0 && q[2] == 0.0 && (q[3] == 1.0 || q[3] == -1.0)
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_tag_round_trips() {
        let (vendor, major) = MisaFile::parse_schema(SCHEMA_TAG).unwrap();
        assert_eq!(vendor, "misarta");
        assert_eq!(major, CURRENT_VERSION);
    }

    #[test]
    fn empty_file_serialises() {
        let f = MisaFile::new("test_robot", "base_link");
        let s = toml::to_string_pretty(&f).expect("serialize");
        assert!(s.contains("schema = \"misarta/1\""));
        assert!(s.contains("[robot]"));
        assert!(s.contains("name = \"test_robot\""));
    }

    #[test]
    fn empty_file_round_trips() {
        let f = MisaFile::new("r", "root");
        let s = toml::to_string(&f).unwrap();
        let f2: MisaFile = toml::from_str(&s).unwrap();
        assert_eq!(f2.robot.name, "r");
        assert_eq!(f2.robot.root, "root");
        assert!(f2.link.is_empty());
    }

    #[test]
    fn link_with_visual_and_collision_round_trips() {
        let mut f = MisaFile::new("r", "trunk");
        f.link.push(Link {
            name: "trunk".into(),
            description: String::new(),
            inertial: Inertial {
                mass: 5.0,
                ixx: 0.1,
                iyy: 0.1,
                izz: 0.1,
                ..Default::default()
            },
            visual: vec![Visual {
                origin: Origin::default(),
                geom: Geom::Box {
                    size: [0.30, 0.20, 0.10],
                },
                color: Some(ColorSpec::Hex("#cc6644".into())),
                material: None,
            }],
            collision: vec![Collision {
                origin: Origin::default(),
                geom: Geom::Capsule {
                    radius: 0.04,
                    length: 0.20,
                },
            }],
        });
        let s = toml::to_string(&f).unwrap();
        let f2: MisaFile = toml::from_str(&s).unwrap();
        assert_eq!(f2.link.len(), 1);
        assert_eq!(f2.link[0].visual.len(), 1);
        assert!(matches!(f2.link[0].visual[0].geom, Geom::Box { .. }));
        assert!(matches!(f2.link[0].collision[0].geom, Geom::Capsule { .. }));
    }

    #[test]
    fn joint_with_actuator_n_to_m() {
        let mut f = MisaFile::new("r", "base");
        f.joint.push(Joint {
            name: "wheel_left".into(),
            kind: JointKind::Continuous,
            parent: "base".into(),
            child: "wheel_l".into(),
            axis: [0.0, 1.0, 0.0],
            origin: Origin::default(),
            limit: JointLimit::default(),
            dynamics: JointDynamics {
                armature: 0.001,
                damping: 0.05,
                friction: 0.0,
            },
        });
        f.actuator.push(Actuator {
            name: "diff_drive_a".into(),
            mode: ActuatorMode::Torque,
            joints: vec![
                ActuatorJointRef {
                    name: "wheel_left".into(),
                    gear: 1.0,
                },
                ActuatorJointRef {
                    name: "wheel_right".into(),
                    gear: -1.0,
                },
            ],
            kp: 0.0,
            kv: 0.0,
        });
        let s = toml::to_string(&f).unwrap();
        let f2: MisaFile = toml::from_str(&s).unwrap();
        assert_eq!(f2.actuator.len(), 1);
        assert_eq!(f2.actuator[0].joints.len(), 2);
        assert_eq!(f2.actuator[0].joints[1].gear, -1.0);
    }

    #[test]
    fn origin_with_rpy_or_quat() {
        let o = Origin {
            xyz: [1.0, 2.0, 3.0],
            rpy: Some([0.0, 0.0, 1.5708]),
            quat: None,
        };
        let s = toml::to_string(&o).unwrap();
        assert!(s.contains("rpy"));
        assert!(!s.contains("quat"));
        let o2: Origin = toml::from_str(&s).unwrap();
        assert_eq!(o2.xyz, [1.0, 2.0, 3.0]);
        assert_eq!(o2.rpy, Some([0.0, 0.0, 1.5708]));
    }

    #[test]
    fn identity_origin_serialises_compactly() {
        let v = Visual {
            origin: Origin::default(),
            geom: Geom::Sphere { radius: 0.05 },
            color: None,
            material: Some("red".into()),
        };
        let s = toml::to_string(&v).unwrap();
        // identity origin should be skipped entirely
        assert!(!s.contains("origin"));
    }

    #[test]
    fn geom_variants_use_external_tag() {
        let cases = vec![
            (
                Geom::Box {
                    size: [0.1, 0.2, 0.3],
                },
                "box",
            ),
            (
                Geom::Cylinder {
                    radius: 0.05,
                    length: 0.2,
                },
                "cylinder",
            ),
            (Geom::Sphere { radius: 0.05 }, "sphere"),
            (
                Geom::Capsule {
                    radius: 0.05,
                    length: 0.2,
                },
                "capsule",
            ),
        ];
        for (g, expected_tag) in cases {
            let v = Visual {
                origin: Origin::default(),
                geom: g,
                color: None,
                material: None,
            };
            let s = toml::to_string(&v).unwrap();
            assert!(s.contains(expected_tag), "missing tag {expected_tag} in: {s}");
        }
    }

    #[test]
    fn mesh_scale_skipped_when_unit() {
        let v = Visual {
            origin: Origin::default(),
            geom: Geom::Mesh {
                file: "meshes/trunk.stl".into(),
                scale: [1.0, 1.0, 1.0],
            },
            color: None,
            material: None,
        };
        let s = toml::to_string(&v).unwrap();
        assert!(!s.contains("scale"));
    }

    #[test]
    fn unknown_schema_rejected() {
        assert!(MisaFile::parse_schema("nope").is_none());
        assert!(MisaFile::parse_schema("misarta/1").is_some());
        assert!(MisaFile::parse_schema("misarta/abc").is_none());
    }
}
