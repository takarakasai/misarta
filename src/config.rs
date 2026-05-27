//! Persistent configuration for misarta — `.misarta.toml` files.
//!
//! Stores auxiliary data that cannot be expressed in standard robot
//! description formats (URDF, SDF, MJCF), such as:
//!
//! - Loop-closure (closed kinematic chain) constraints
//!
//! # File format
//!
//! ```toml
//! [misarta]
//! version = 1
//!
//! [[loop_closure]]
//! name = "four_bar_loop"
//! link_a = "coupler"
//! offset_a = [0.3, 0.0, 0.0]
//! link_b = "crank_right"
//! offset_b = [0.0, 0.0, 0.2]
//! pose_6dof = false
//! ```
//!
//! # Usage
//!
//! ```no_run
//! use misarta::config::{MisartaConfig, LoopClosureConfig};
//!
//! // Load
//! let config = MisartaConfig::load("robot.misarta.toml").unwrap();
//!
//! // Create & save
//! let config = MisartaConfig {
//!     misarta: misarta::config::MisartaHeader { version: 1 },
//!     loop_closure: vec![
//!         LoopClosureConfig {
//!             name: "loop".into(),
//!             link_a: "coupler".into(),
//!             offset_a: [0.3, 0.0, 0.0],
//!             link_b: "crank_right".into(),
//!             offset_b: [0.0, 0.0, 0.2],
//!             rot_a: [0.0, 0.0, 0.0, 1.0],
//!             rot_b: [0.0, 0.0, 0.0, 1.0],
//!             pose_6dof: false,
//!         },
//!     ],
//!     pose: vec![],
//!     actuator: vec![],
//!     collision_pair: vec![],
//!     sequence: vec![],
//!     mimic: vec![],
//!     sensor: vec![],
//! };
//! config.save("robot.misarta.toml").unwrap();
//! ```

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Current config file format version.
pub const CURRENT_VERSION: u32 = 1;

/// Top-level misarta configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MisartaConfig {
    /// File header with version.
    pub misarta: MisartaHeader,
    /// Loop-closure constraints (may be empty).
    #[serde(default)]
    pub loop_closure: Vec<LoopClosureConfig>,
    /// Named joint-space poses for replay during simulation (may be empty).
    #[serde(default)]
    pub pose: Vec<PoseConfig>,
    /// Per-joint actuator configuration (mode + gains) used by MJCF export
    /// and the live MuJoCo controller. Only joints with non-default settings
    /// need be present; unmatched joints fall back to the host's defaults.
    #[serde(default)]
    pub actuator: Vec<ActuatorConfig>,
    /// Per-link-pair collision overrides. Pairs not listed here use the
    /// host's default behaviour (which is "all link pairs collide" to match
    /// MuJoCo / SDF defaults). Listed pairs with `enabled = false` are
    /// emitted as `<contact><exclude>` in MJCF and as filtered pairs for
    /// USD/Isaac. Pairs with `enabled = true` are no-ops in formats whose
    /// default is already "collide" — they exist mostly to record the user's
    /// intent and for round-trip preservation.
    #[serde(default)]
    pub collision_pair: Vec<CollisionPairConfig>,
    /// Named pose sequences for chained replay (e.g. crouch → extend → land).
    /// Each step references a pose by name and specifies the transition
    /// duration / curve from the *previous* step (or the model's current
    /// state for the first step).
    #[serde(default)]
    pub sequence: Vec<SequenceConfig>,
    /// Coupled (mimic) joints — `joint = multiplier · source_joint + offset`.
    /// Linear coupling matches the URDF / SDF native form; MJCF can express
    /// the same shape via `<equality><joint polycoef="off mult 0 0 0">`.
    #[serde(default)]
    pub mimic: Vec<MimicConfig>,
    /// Sensors mounted on links. Format-agnostic representation; per-target
    /// formats (SDF/MJCF/USD) translate during export.
    #[serde(default)]
    pub sensor: Vec<SensorConfig>,
    /// Quadruped gait presets. Multiple entries allow saving variants
    /// (e.g. `slow_walk`, `fast_trot`); the host app typically loads the
    /// first as the default but can switch by name.
    #[serde(default)]
    pub gait: Vec<GaitConfigEntry>,
    /// Home pose — initial joint configuration + floating-base transform
    /// applied at load time. Lets a session round-trip the user's last
    /// saved pose without an explicit `[[pose]]` entry. Defaults to
    /// "no information" (URDF defaults remain in effect).
    #[serde(default)]
    pub home: HomeConfig,
}

/// A named joint-space pose stored in the sidecar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseConfig {
    /// Human-readable name shown in the UI.
    pub name: String,
    /// Joint angle / displacement keyed by joint name. Joints not in the map
    /// inherit the model's current value at replay time.
    pub angles: std::collections::BTreeMap<String, f64>,
    /// Default transition time in seconds when the pose is replayed. Acts as
    /// the seed value for the per-play UI control; the user can override it
    /// at playback time without re-saving the pose.
    #[serde(default = "default_duration")]
    pub duration: f64,
    /// Default interpolation curve. Same role as `duration` — a per-pose
    /// default that can be overridden at playback time.
    #[serde(default)]
    pub kind: crate::trajectory::InterpolationKind,
}

fn default_duration() -> f64 {
    1.0
}

impl Default for crate::trajectory::InterpolationKind {
    fn default() -> Self {
        crate::trajectory::InterpolationKind::QuinticSmooth
    }
}

/// File header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MisartaHeader {
    /// Format version (currently 1).
    pub version: u32,
}

/// Control mode for a single actuator. Mirrors the MuJoCo actuator types
/// articara emits in its MJCF export plus the computed-torque mode that
/// only lives in the runtime controller (never serialised into MJCF —
/// MuJoCo's actuator model can't represent inverse-dynamics feedforward).
/// Persisted as a TOML string so the file stays human-editable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActuatorMode {
    Position,
    Velocity,
    Torque,
    /// `τ = M(q)·q̈* + h(q, q̇) + Kp·(q*−q) + Kv·(q̇*−q̇)` (computed-torque).
    /// Round-trips through the host (articara) only; on MJCF export this
    /// mode degrades to `<motor>` since the inverse dynamics live outside
    /// MuJoCo.
    ComputedTorque,
    /// MJCF-export-only flag: the host should emit this joint as a weld
    /// (no DoF, no actuator) in the generated MJCF. The kinematic
    /// `joint_type` in the joint definition itself is preserved, so .misa
    /// and URDF round-trips keep the joint movable for FK and re-export.
    /// Intended for disabling wheel / passive joints in a single MuJoCo
    /// session without rewriting the model.
    Fixed,
}

impl Default for ActuatorMode {
    fn default() -> Self {
        ActuatorMode::Position
    }
}

/// Per-joint actuator settings. Stored once per movable joint so the host
/// can reconstruct identical controller behaviour across sessions / hosts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActuatorConfig {
    /// Joint name (matches the URDF / SDF joint name).
    pub joint_name: String,
    /// Active control mode.
    #[serde(default)]
    pub mode: ActuatorMode,
    /// Position gain (used by `Position`).
    #[serde(default = "default_actuator_kp")]
    pub kp: f64,
    /// Damping / velocity gain (used by `Position` and `Velocity`).
    #[serde(default = "default_actuator_kv")]
    pub kv: f64,
    /// Reflected rotor inertia (kg·m² for revolute, kg for prismatic). Maps to
    /// MuJoCo's `<joint armature="…"/>`.
    #[serde(default)]
    pub armature: f64,
    /// Passive joint damping coefficient. Maps to MuJoCo's `<joint damping="…"/>`.
    #[serde(default)]
    pub joint_damping: f64,
}

fn default_actuator_kp() -> f64 {
    50.0
}

fn default_actuator_kv() -> f64 {
    5.0
}

/// Per-link-pair collision setting.
///
/// Pairs are unordered — `(link_a, link_b)` and `(link_b, link_a)` are
/// equivalent. The host normalises them to alphabetical order on load /
/// save so the TOML stays diff-friendly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionPairConfig {
    pub link_a: String,
    pub link_b: String,
    /// `true` = collide, `false` = explicitly excluded (e.g. self-collision
    /// avoidance for adjacent links whose visual meshes overlap).
    #[serde(default = "default_pair_enabled")]
    pub enabled: bool,
}

fn default_pair_enabled() -> bool {
    true
}

/// A linear coupling between two joints — `joint = multiplier · source + offset`.
///
/// Modelled after URDF's `<mimic>` (the most-common form). The host can
/// translate to MJCF's `<equality><joint polycoef="off mult 0 0 0">` /
/// SDF's `<axis><mimic>`. USD has no native mimic concept; importers may
/// drop or warn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MimicConfig {
    /// Joint that follows.
    pub joint: String,
    /// Joint that drives.
    pub source: String,
    #[serde(default = "default_one")]
    pub multiplier: f64,
    #[serde(default)]
    pub offset: f64,
}

fn default_one() -> f64 {
    1.0
}

/// A sensor mounted on a link, in a format-neutral representation. Each
/// kind carries the parameters most exporters need; format-specific
/// extras can be tunnelled through the optional `params` map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorConfig {
    /// User-visible name (used as the geom / sensor name in exports).
    pub name: String,
    /// Link the sensor is rigidly attached to.
    pub link: String,
    /// Local-frame translation `[x, y, z]`.
    #[serde(default)]
    pub origin: [f64; 3],
    /// Local-frame quaternion rotation `[x, y, z, w]`.
    #[serde(default = "default_quat", skip_serializing_if = "is_identity_quat")]
    pub orientation: [f64; 4],
    /// Sample rate in Hz. `0.0` means "let the simulator pick a default".
    #[serde(default)]
    pub update_rate: f64,
    /// Type-specific parameters.
    #[serde(flatten)]
    pub kind: SensorKind,
}

/// Sensor type discriminator. `#[serde(tag = "type")]` makes the TOML
/// representation use a `type = "..."` field per entry, which round-trips
/// through `SensorConfig.kind`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SensorKind {
    /// Pinhole / depth camera. Renders an image stream in target sims.
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
    /// Ray-cast (LiDAR) sensor — 1D or 2D scan with horizontal/vertical FOVs.
    Lidar {
        #[serde(default = "default_lidar_range_min")]
        range_min: f64,
        #[serde(default = "default_lidar_range_max")]
        range_max: f64,
        /// Horizontal FOV in radians.
        #[serde(default = "default_lidar_h_fov")]
        h_fov: f64,
        #[serde(default = "default_lidar_h_samples")]
        h_samples: u32,
        /// Vertical FOV in radians (0 = single-line scan).
        #[serde(default)]
        v_fov: f64,
        #[serde(default = "default_lidar_v_samples")]
        v_samples: u32,
    },
    /// Accelerometer + gyroscope.
    Imu {
        #[serde(default)]
        gyro_noise: f64,
        #[serde(default)]
        accel_noise: f64,
    },
    /// 6-axis force/torque mounted on a joint or link.
    ForceTorque {
        /// Optional joint to measure (defaults to the parent joint of `link`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        joint: Option<String>,
    },
    /// Boolean / scalar contact sensor on a link.
    Contact {
        /// Optional partner link to filter against. None = any contact.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        partner: Option<String>,
    },
    /// Escape hatch for sensor types we don't yet model. Round-trips raw
    /// key/value strings so importer/exporter round-trips don't lose data.
    Generic {
        kind: String,
        #[serde(default)]
        params: std::collections::BTreeMap<String, String>,
    },
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

/// A chained-pose sequence (e.g. crouch → extend → land).
///
/// Stored as a list of steps, each referencing a [`PoseConfig`] by name and
/// declaring how long to transition into it from the previous step's pose
/// (or the current model state for the first step). Replayed via the host
/// `play_sequence` script API or the Sequences UI section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceConfig {
    pub name: String,
    pub steps: Vec<SequenceStepConfig>,
}

/// One step in a [`SequenceConfig`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceStepConfig {
    /// Name of a registered pose to transition INTO.
    pub pose_name: String,
    /// Time in seconds spent reaching `pose_name` from the previous step's
    /// pose (or the current model state for the first step).
    #[serde(default = "default_step_duration")]
    pub duration: f64,
    /// Interpolation curve for this step.
    #[serde(default)]
    pub kind: crate::trajectory::InterpolationKind,
}

fn default_step_duration() -> f64 {
    1.0
}

/// A single loop-closure constraint definition.
///
/// Stored offsets are local-frame transforms on each link. The constraint
/// solver enforces `link_a · offset_a == link_b · offset_b` (position only
/// when `pose_6dof = false`, full pose otherwise). Rotation fields default
/// to identity and are silently dropped during serialisation when they
/// equal identity, keeping the TOML diff-friendly for the common
/// position-only case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopClosureConfig {
    /// Human-readable name.
    pub name: String,
    /// First link name.
    pub link_a: String,
    /// Translation offset in link_a's local frame `[x, y, z]`.
    #[serde(default)]
    pub offset_a: [f64; 3],
    /// Rotation offset in link_a's local frame as quaternion `[x, y, z, w]`.
    /// Identity (`[0, 0, 0, 1]`) when omitted.
    #[serde(default = "default_quat", skip_serializing_if = "is_identity_quat")]
    pub rot_a: [f64; 4],
    /// Second link name.
    pub link_b: String,
    /// Translation offset in link_b's local frame `[x, y, z]`.
    #[serde(default)]
    pub offset_b: [f64; 3],
    /// Rotation offset in link_b's local frame as quaternion `[x, y, z, w]`.
    /// Identity (`[0, 0, 0, 1]`) when omitted.
    #[serde(default = "default_quat", skip_serializing_if = "is_identity_quat")]
    pub rot_b: [f64; 4],
    /// Whether to constrain full pose (6-DoF) or position only (3-DoF).
    #[serde(default)]
    pub pose_6dof: bool,
}

fn default_quat() -> [f64; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

fn is_identity_quat(q: &[f64; 4]) -> bool {
    (q[0].abs() < 1e-12)
        && (q[1].abs() < 1e-12)
        && (q[2].abs() < 1e-12)
        && ((q[3] - 1.0).abs() < 1e-12 || (q[3] + 1.0).abs() < 1e-12)
}

impl MisartaConfig {
    /// Create an empty config.
    pub fn new() -> Self {
        Self {
            misarta: MisartaHeader {
                version: CURRENT_VERSION,
            },
            loop_closure: Vec::new(),
            pose: Vec::new(),
            actuator: Vec::new(),
            collision_pair: Vec::new(),
            sequence: Vec::new(),
            mimic: Vec::new(),
            sensor: Vec::new(),
            gait: Vec::new(),
            home: HomeConfig::default(),
        }
    }

    /// Whether the config has any meaningful content worth saving.
    pub fn is_empty(&self) -> bool {
        self.loop_closure.is_empty()
            && self.pose.is_empty()
            && self.actuator.is_empty()
            && self.collision_pair.is_empty()
            && self.sequence.is_empty()
            && self.mimic.is_empty()
            && self.sensor.is_empty()
    }

    /// Load from a `.misarta.toml` file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let text = std::fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read {}: {}", path.as_ref().display(), e))?;
        Self::from_toml(&text)
    }

    /// Parse from a TOML string.
    pub fn from_toml(text: &str) -> Result<Self, String> {
        let config: Self =
            toml::from_str(text).map_err(|e| format!("Failed to parse misarta TOML: {e}"))?;
        if config.misarta.version > CURRENT_VERSION {
            return Err(format!(
                "Unsupported misarta config version {} (max supported: {})",
                config.misarta.version, CURRENT_VERSION,
            ));
        }
        Ok(config)
    }

    /// Serialize to a TOML string.
    pub fn to_toml(&self) -> Result<String, String> {
        toml::to_string_pretty(self).map_err(|e| format!("Failed to serialize misarta TOML: {e}"))
    }

    /// Save to a `.misarta.toml` file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let text = self.to_toml()?;
        std::fs::write(path.as_ref(), &text)
            .map_err(|e| format!("Failed to write {}: {}", path.as_ref().display(), e))
    }

    /// Derive the `.misarta.toml` path from a robot model path.
    ///
    /// Given `/path/to/robot.urdf`, returns `/path/to/robot.misarta.toml`.
    pub fn config_path_for(model_path: impl AsRef<Path>) -> std::path::PathBuf {
        let p = model_path.as_ref();
        let stem = p.file_stem().unwrap_or_default().to_string_lossy();
        p.with_file_name(format!("{}.misarta.toml", stem))
    }
}

impl Default for MisartaConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Home pose — joint angles + floating-base transform persisted across
//  sessions, so the model opens up exactly where the user left it.
// ─────────────────────────────────────────────────────────────────────────

/// Initial pose applied right after a sidecar load. All fields default
/// to "no information": empty joint map, origin translation, identity
/// quaternion. The host treats those defaults as no-ops, so a freshly
/// imported URDF / SDF / MJCF / USD with no `[home]` section keeps its
/// own neutral configuration.
///
/// Why this is separate from `[[pose]]`: poses are *named user-saved
/// targets* used for replay (sequences, jumps); home is *the resume
/// state* — the model's current configuration when the user last hit
/// Save. Both can coexist; home is automatic, poses are deliberate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomeConfig {
    /// Joint angles keyed by joint name. Joints not in the map keep
    /// their URDF / SDF default at load time, so partial home entries
    /// don't blow away unrelated state.
    #[serde(default)]
    pub joint_positions: std::collections::BTreeMap<String, f64>,
    /// Floating-base translation in world frame (m).
    #[serde(default = "default_zero3")]
    pub base_position: [f64; 3],
    /// Floating-base orientation as a quaternion `[qx, qy, qz, qw]`.
    /// Defaults to identity `[0, 0, 0, 1]`.
    #[serde(default = "default_identity_quat")]
    pub base_orientation: [f64; 4],
}

fn default_zero3() -> [f64; 3] {
    [0.0, 0.0, 0.0]
}
fn default_identity_quat() -> [f64; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

impl Default for HomeConfig {
    fn default() -> Self {
        Self {
            joint_positions: std::collections::BTreeMap::new(),
            base_position: default_zero3(),
            base_orientation: default_identity_quat(),
        }
    }
}

impl HomeConfig {
    /// True when every field is at its default. The host can use this to
    /// decide whether a `.misarta.toml` round-trip carries any home
    /// information at all (skip writing `[home]` if not).
    pub fn is_empty(&self) -> bool {
        self.joint_positions.is_empty()
            && self.base_position == default_zero3()
            && self.base_orientation == default_identity_quat()
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Quadruped gait — sidecar persistence schema
// ─────────────────────────────────────────────────────────────────────────

/// Gait family. Mirrors `quadruped_gait::GaitType` but lives in the
/// misarta config crate so the persistence layer doesn't pull in the
/// gait library. Stored as a TOML string so the file stays human-readable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GaitTypeConfig {
    Trot,
    Walk,
    Pace,
    Bound,
    Crawl,
}

impl Default for GaitTypeConfig {
    fn default() -> Self {
        GaitTypeConfig::Trot
    }
}

/// One quadruped gait preset stored in the sidecar.
///
/// Holds the **user-tunable** subset of `quadruped_gait::GaitController`
/// state — things the host app can't reconstruct just by looking at the
/// URDF. The leg link lengths / hip offsets are intentionally NOT stored:
/// they're auto-detected from the model on every load via
/// `articara::gait::auto_detect_kinematics_config`. Saving them would let
/// the user's model edit drift out of sync with the cached kinematics
/// silently. Only override the auto-detection by overriding
/// `nominal_foot_body_*` fields explicitly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaitConfigEntry {
    /// Preset name shown in the UI dropdown.
    pub name: String,
    #[serde(default)]
    pub gait_type: GaitTypeConfig,
    /// One full leg cycle (s).
    #[serde(default = "default_cycle_period")]
    pub cycle_period_s: f64,
    /// Fraction of cycle each foot is in stance.
    #[serde(default = "default_duty_factor")]
    pub duty_factor: f64,
    /// Peak swing height above the stance plane (m).
    #[serde(default = "default_swing_height")]
    pub swing_height_m: f64,
    /// Footstep planner clamp (m).
    #[serde(default = "default_max_step")]
    pub max_step_length_m: f64,
    /// Foot link names per leg in canonical FL/FR/RL/RR order. The host
    /// uses these to re-run kinematics auto-detection on load.
    #[serde(default = "default_fl_foot")]
    pub fl_foot: String,
    #[serde(default = "default_fr_foot")]
    pub fr_foot: String,
    #[serde(default = "default_rl_foot")]
    pub rl_foot: String,
    #[serde(default = "default_rr_foot")]
    pub rr_foot: String,
    /// Per-leg knee direction. `[FL, FR, RL, RR]`. `true` = bends forward
    /// (front-leg style), `false` = bends backward. Default all-false
    /// matches the analytical IK's natural sign convention.
    #[serde(default)]
    pub knee_forward: [bool; 4],
    /// LinearCrawl-only: fraction of each per-leg sub-cycle (`T/4`)
    /// spent in 4-support before the leg lifts. Ignored by every
    /// other gait mode. Default `0.5`.
    #[serde(default = "default_four_support_fraction")]
    pub four_support_fraction: f64,
}

fn default_cycle_period() -> f64 { 0.4 }
fn default_duty_factor() -> f64 { 0.5 }
fn default_swing_height() -> f64 { 0.04 }
fn default_max_step() -> f64 { 0.10 }
fn default_four_support_fraction() -> f64 { 0.5 }
fn default_fl_foot() -> String { "FL_foot".into() }
fn default_fr_foot() -> String { "FR_foot".into() }
fn default_rl_foot() -> String { "RL_foot".into() }
fn default_rr_foot() -> String { "RR_foot".into() }

impl Default for GaitConfigEntry {
    fn default() -> Self {
        Self {
            name: "default".into(),
            gait_type: GaitTypeConfig::default(),
            cycle_period_s: default_cycle_period(),
            duty_factor: default_duty_factor(),
            swing_height_m: default_swing_height(),
            max_step_length_m: default_max_step(),
            fl_foot: default_fl_foot(),
            fr_foot: default_fr_foot(),
            rl_foot: default_rl_foot(),
            rr_foot: default_rr_foot(),
            knee_forward: [false; 4],
            four_support_fraction: default_four_support_fraction(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_empty() {
        let config = MisartaConfig::new();
        let toml = config.to_toml().unwrap();
        let parsed = MisartaConfig::from_toml(&toml).unwrap();
        assert!(parsed.loop_closure.is_empty());
        assert_eq!(parsed.misarta.version, CURRENT_VERSION);
    }

    #[test]
    fn roundtrip_with_loop_closures() {
        let config = MisartaConfig {
            misarta: MisartaHeader { version: 1 },
            loop_closure: vec![
                LoopClosureConfig {
                    name: "four_bar".into(),
                    link_a: "coupler".into(),
                    offset_a: [0.3, 0.0, 0.0],
                    rot_a: default_quat(),
                    link_b: "crank_right".into(),
                    offset_b: [0.0, 0.0, 0.2],
                    rot_b: default_quat(),
                    pose_6dof: false,
                },
                LoopClosureConfig {
                    name: "weld".into(),
                    link_a: "hand_l".into(),
                    offset_a: [0.0, 0.0, 0.1],
                    rot_a: default_quat(),
                    link_b: "hand_r".into(),
                    offset_b: [0.0, 0.0, 0.1],
                    rot_b: default_quat(),
                    pose_6dof: true,
                },
            ],
            pose: Vec::new(),
            actuator: Vec::new(),
            collision_pair: Vec::new(),
            sequence: Vec::new(),
            mimic: Vec::new(),
            sensor: Vec::new(),
            gait: Vec::new(),
            home: HomeConfig::default(),
        };
        let toml = config.to_toml().unwrap();
        let parsed = MisartaConfig::from_toml(&toml).unwrap();
        assert_eq!(parsed.loop_closure.len(), 2);
        assert_eq!(parsed.loop_closure[0].name, "four_bar");
        assert_eq!(parsed.loop_closure[0].offset_a, [0.3, 0.0, 0.0]);
        assert!(parsed.loop_closure[1].pose_6dof);
    }

    #[test]
    fn parse_minimal_toml() {
        let text = r#"
[misarta]
version = 1
"#;
        let config = MisartaConfig::from_toml(text).unwrap();
        assert!(config.loop_closure.is_empty());
    }

    #[test]
    fn reject_future_version() {
        let text = r#"
[misarta]
version = 999
"#;
        let err = MisartaConfig::from_toml(text).unwrap_err();
        assert!(err.contains("Unsupported"), "{}", err);
    }

    #[test]
    fn config_path_for_urdf() {
        let p = MisartaConfig::config_path_for("/home/user/robot.urdf");
        assert_eq!(p.to_str().unwrap(), "/home/user/robot.misarta.toml");
    }

    #[test]
    fn config_path_for_sdf() {
        let p = MisartaConfig::config_path_for("models/four_bar.sdf");
        assert_eq!(p.to_str().unwrap(), "models/four_bar.misarta.toml");
    }

    #[test]
    fn roundtrip_with_actuators() {
        let config = MisartaConfig {
            misarta: MisartaHeader { version: 1 },
            loop_closure: Vec::new(),
            pose: Vec::new(),
            actuator: vec![
                ActuatorConfig {
                    joint_name: "hip_l".into(),
                    mode: ActuatorMode::Position,
                    kp: 80.0,
                    kv: 6.5,
                    armature: 0.012,
                    joint_damping: 0.4,
                },
                ActuatorConfig {
                    joint_name: "wheel_l".into(),
                    mode: ActuatorMode::Velocity,
                    kp: 0.0,
                    kv: 12.0,
                    armature: 0.0,
                    joint_damping: 0.0,
                },
            ],
            collision_pair: Vec::new(),
            sequence: Vec::new(),
            mimic: Vec::new(),
            sensor: Vec::new(),
            gait: Vec::new(),
            home: HomeConfig::default(),
        };
        let toml = config.to_toml().unwrap();
        let parsed = MisartaConfig::from_toml(&toml).unwrap();
        assert_eq!(parsed.actuator.len(), 2);
        assert_eq!(parsed.actuator[0].joint_name, "hip_l");
        assert_eq!(parsed.actuator[0].mode, ActuatorMode::Position);
        assert!((parsed.actuator[0].kp - 80.0).abs() < 1e-9);
        assert!((parsed.actuator[0].armature - 0.012).abs() < 1e-9);
        assert!((parsed.actuator[0].joint_damping - 0.4).abs() < 1e-9);
        assert_eq!(parsed.actuator[1].mode, ActuatorMode::Velocity);
        assert!((parsed.actuator[1].armature).abs() < 1e-9);
    }

    #[test]
    fn roundtrip_with_collision_pairs() {
        let config = MisartaConfig {
            misarta: MisartaHeader { version: 1 },
            loop_closure: Vec::new(),
            pose: Vec::new(),
            actuator: Vec::new(),
            collision_pair: vec![
                CollisionPairConfig {
                    link_a: "trunk".into(),
                    link_b: "FL_thigh".into(),
                    enabled: false,
                },
                CollisionPairConfig {
                    link_a: "FR_calf".into(),
                    link_b: "FR_foot".into(),
                    enabled: true,
                },
            ],
            sequence: Vec::new(),
            mimic: Vec::new(),
            sensor: Vec::new(),
            gait: Vec::new(),
            home: HomeConfig::default(),
        };
        let toml = config.to_toml().unwrap();
        let parsed = MisartaConfig::from_toml(&toml).unwrap();
        assert_eq!(parsed.collision_pair.len(), 2);
        assert!(!parsed.collision_pair[0].enabled);
        assert!(parsed.collision_pair[1].enabled);
    }

    #[test]
    fn roundtrip_with_sequences() {
        let config = MisartaConfig {
            misarta: MisartaHeader { version: 1 },
            loop_closure: Vec::new(),
            pose: Vec::new(),
            actuator: Vec::new(),
            collision_pair: Vec::new(),
            sequence: vec![SequenceConfig {
                name: "jump".into(),
                steps: vec![
                    SequenceStepConfig {
                        pose_name: "crouch".into(),
                        duration: 0.5,
                        kind: crate::trajectory::InterpolationKind::QuinticSmooth,
                    },
                    SequenceStepConfig {
                        pose_name: "extended".into(),
                        duration: 0.1,
                        kind: crate::trajectory::InterpolationKind::Linear,
                    },
                ],
            }],
            mimic: Vec::new(),
            sensor: Vec::new(),
            gait: Vec::new(),
            home: HomeConfig::default(),
        };
        let toml = config.to_toml().unwrap();
        let parsed = MisartaConfig::from_toml(&toml).unwrap();
        assert_eq!(parsed.sequence.len(), 1);
        assert_eq!(parsed.sequence[0].name, "jump");
        assert_eq!(parsed.sequence[0].steps.len(), 2);
        assert_eq!(parsed.sequence[0].steps[0].pose_name, "crouch");
        assert!((parsed.sequence[0].steps[1].duration - 0.1).abs() < 1e-9);
    }

    #[test]
    fn roundtrip_with_mimic() {
        let config = MisartaConfig {
            misarta: MisartaHeader { version: 1 },
            loop_closure: Vec::new(),
            pose: Vec::new(),
            actuator: Vec::new(),
            collision_pair: Vec::new(),
            sequence: Vec::new(),
            mimic: vec![MimicConfig {
                joint: "wheel_r".into(),
                source: "wheel_l".into(),
                multiplier: -1.0,
                offset: 0.0,
            }],
            sensor: Vec::new(),
            gait: Vec::new(),
            home: HomeConfig::default(),
        };
        let toml = config.to_toml().unwrap();
        let parsed = MisartaConfig::from_toml(&toml).unwrap();
        assert_eq!(parsed.mimic.len(), 1);
        assert_eq!(parsed.mimic[0].source, "wheel_l");
        assert!((parsed.mimic[0].multiplier - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn roundtrip_with_sensors() {
        let config = MisartaConfig {
            misarta: MisartaHeader { version: 1 },
            loop_closure: Vec::new(),
            pose: Vec::new(),
            actuator: Vec::new(),
            collision_pair: Vec::new(),
            sequence: Vec::new(),
            mimic: Vec::new(),
            sensor: vec![
                SensorConfig {
                    name: "front_camera".into(),
                    link: "head".into(),
                    origin: [0.1, 0.0, 0.05],
                    orientation: default_quat(),
                    update_rate: 30.0,
                    kind: SensorKind::Camera {
                        fov: 1.0,
                        width: 320,
                        height: 240,
                        near: 0.05,
                        far: 50.0,
                    },
                },
                SensorConfig {
                    name: "lidar".into(),
                    link: "lidar_mount".into(),
                    origin: [0.0, 0.0, 0.1],
                    orientation: default_quat(),
                    update_rate: 10.0,
                    kind: SensorKind::Lidar {
                        range_min: 0.05,
                        range_max: 30.0,
                        h_fov: std::f64::consts::TAU,
                        h_samples: 1024,
                        v_fov: 0.5,
                        v_samples: 16,
                    },
                },
                SensorConfig {
                    name: "imu".into(),
                    link: "trunk".into(),
                    origin: [0.0, 0.0, 0.0],
                    orientation: default_quat(),
                    update_rate: 200.0,
                    kind: SensorKind::Imu {
                        gyro_noise: 0.001,
                        accel_noise: 0.01,
                    },
                },
            ],
            gait: Vec::new(),
            home: HomeConfig::default(),
        };
        let toml = config.to_toml().unwrap();
        let parsed = MisartaConfig::from_toml(&toml).unwrap();
        assert_eq!(parsed.sensor.len(), 3);
        assert_eq!(parsed.sensor[0].name, "front_camera");
        match &parsed.sensor[0].kind {
            SensorKind::Camera { width, height, .. } => {
                assert_eq!(*width, 320);
                assert_eq!(*height, 240);
            }
            _ => panic!("expected Camera"),
        }
        match &parsed.sensor[1].kind {
            SensorKind::Lidar { h_samples, .. } => assert_eq!(*h_samples, 1024),
            _ => panic!("expected Lidar"),
        }
    }

    #[test]
    fn roundtrip_with_gait() {
        let mut cfg = MisartaConfig::new();
        cfg.gait.push(GaitConfigEntry {
            name: "fast_trot".into(),
            gait_type: GaitTypeConfig::Trot,
            cycle_period_s: 0.30,
            duty_factor: 0.45,
            swing_height_m: 0.05,
            max_step_length_m: 0.12,
            fl_foot: "FL_paw".into(),
            fr_foot: "FR_paw".into(),
            rl_foot: "RL_paw".into(),
            rr_foot: "RR_paw".into(),
            knee_forward: [true, true, false, false],
            four_support_fraction: 0.5,
        });
        let toml = cfg.to_toml().unwrap();
        let parsed = MisartaConfig::from_toml(&toml).unwrap();
        assert_eq!(parsed.gait.len(), 1);
        let g = &parsed.gait[0];
        assert_eq!(g.name, "fast_trot");
        assert!((g.cycle_period_s - 0.30).abs() < 1e-9);
        assert_eq!(g.fl_foot, "FL_paw");
        assert_eq!(g.knee_forward, [true, true, false, false]);
    }

    #[test]
    fn parse_minimal_toml_has_empty_actuators() {
        let text = r#"
[misarta]
version = 1
"#;
        let config = MisartaConfig::from_toml(text).unwrap();
        assert!(config.actuator.is_empty());
    }

    #[test]
    fn save_and_load_file() {
        let tmp = std::env::temp_dir().join("misarta_test_config.misarta.toml");
        let config = MisartaConfig {
            misarta: MisartaHeader { version: 1 },
            loop_closure: vec![LoopClosureConfig {
                name: "test".into(),
                link_a: "a".into(),
                offset_a: [1.0, 2.0, 3.0],
                rot_a: default_quat(),
                link_b: "b".into(),
                offset_b: [4.0, 5.0, 6.0],
                rot_b: default_quat(),
                pose_6dof: false,
            }],
            pose: Vec::new(),
            actuator: Vec::new(),
            collision_pair: Vec::new(),
            sequence: Vec::new(),
            mimic: Vec::new(),
            sensor: Vec::new(),
            gait: Vec::new(),
            home: HomeConfig::default(),
        };
        config.save(&tmp).unwrap();

        let loaded = MisartaConfig::load(&tmp).unwrap();
        assert_eq!(loaded.loop_closure.len(), 1);
        assert_eq!(loaded.loop_closure[0].link_a, "a");
        assert_eq!(loaded.loop_closure[0].offset_b, [4.0, 5.0, 6.0]);

        std::fs::remove_file(&tmp).ok();
    }
}
