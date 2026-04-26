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
//!             pose_6dof: false,
//!         },
//!     ],
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

/// A single loop-closure constraint definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopClosureConfig {
    /// Human-readable name.
    pub name: String,
    /// First link name.
    pub link_a: String,
    /// Offset in link_a's local frame `[x, y, z]`.
    #[serde(default)]
    pub offset_a: [f64; 3],
    /// Second link name.
    pub link_b: String,
    /// Offset in link_b's local frame `[x, y, z]`.
    #[serde(default)]
    pub offset_b: [f64; 3],
    /// Whether to constrain full pose (6-DoF) or position only (3-DoF).
    #[serde(default)]
    pub pose_6dof: bool,
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
        }
    }

    /// Whether the config has any meaningful content worth saving.
    pub fn is_empty(&self) -> bool {
        self.loop_closure.is_empty() && self.pose.is_empty()
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
                    link_b: "crank_right".into(),
                    offset_b: [0.0, 0.0, 0.2],
                    pose_6dof: false,
                },
                LoopClosureConfig {
                    name: "weld".into(),
                    link_a: "hand_l".into(),
                    offset_a: [0.0, 0.0, 0.1],
                    link_b: "hand_r".into(),
                    offset_b: [0.0, 0.0, 0.1],
                    pose_6dof: true,
                },
            ],
            pose: Vec::new(),
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
    fn save_and_load_file() {
        let tmp = std::env::temp_dir().join("misarta_test_config.misarta.toml");
        let config = MisartaConfig {
            misarta: MisartaHeader { version: 1 },
            loop_closure: vec![LoopClosureConfig {
                name: "test".into(),
                link_a: "a".into(),
                offset_a: [1.0, 2.0, 3.0],
                link_b: "b".into(),
                offset_b: [4.0, 5.0, 6.0],
                pose_6dof: false,
            }],
            pose: Vec::new(),
        };
        config.save(&tmp).unwrap();

        let loaded = MisartaConfig::load(&tmp).unwrap();
        assert_eq!(loaded.loop_closure.len(), 1);
        assert_eq!(loaded.loop_closure[0].link_a, "a");
        assert_eq!(loaded.loop_closure[0].offset_b, [4.0, 5.0, 6.0]);

        std::fs::remove_file(&tmp).ok();
    }
}
