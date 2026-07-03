//! `.misa` — misarta's native master format.
//!
//! `.misa` is a TOML-encoded, articara/misarta-native description of a robot
//! that supersedes the URDF + `.misarta.toml` sidecar split: a single file
//! holds the full kinematic tree, geometry, materials, mimic / loop-closure /
//! collision-pair / sensor / actuator definitions, plus editor metadata
//! (poses, sequences, gaits, home pose).
//!
//! See `doc/refactor_20260502.md` for the design rationale; the on-disk
//! schema lives in [`schema`].
//!
//! # Layered API
//!
//! The module is split into three layers so the file-system dependency
//! can be swapped out for embedded / WASM / test scenarios:
//!
//! - **Layer 1 — [`source`]**: the [`AssetSource`] trait plus four
//!   built-in implementations ([`FileSystemSource`], [`InMemorySource`],
//!   [`StaticBundleSource`], [`NullSource`]).
//! - **Layer 2 — [`parse`] / [`write`] / [`build`]**: pure-Rust
//!   conversion between TOML strings, [`schema::MisaFile`], and runtime
//!   [`crate::model::Model`] / [`crate::geometry::GeometryModel`]. No
//!   `std::fs` access.
//! - **Layer 3 — convenience**: [`load`] / [`save`] wrap layer 2 with
//!   `std::fs` for the common "I have a path on disk" case.
//!
//! # Quick reference
//!
//! ```ignore
//! // Common case: read a .misa from disk.
//! let out = misarta::native::load("robots/namiashi/namiashi.misa")?;
//! if !out.report.is_empty() {
//!     show_dialog(&out.report);
//! }
//! let (model, visual, collision) = misarta::native::build_model(&out.file)?;
//!
//! // Embedded case: parse from a memory buffer with bundled meshes.
//! const ASSETS: &[(&str, &[u8])] = &[ /* ... */ ];
//! let source = StaticBundleSource::new(ASSETS);
//! let bytes = source.read("robot.misa")?;
//! let text = std::str::from_utf8(&bytes)?;
//! let out = misarta::native::parse_str(text, &source)?;
//! ```

pub mod build;
pub mod edit;
pub mod mesh_load;
pub mod parse;
pub mod report;
pub mod schema;
pub mod source;
pub mod write;

pub use build::build_model;
pub use edit::{add_joint, add_link, remove_link, rename_joint, rename_link, EditError};
pub use mesh_load::{load_meshes, load_meshes_into, normalise_mesh_reference, MeshLoadReport};
pub use parse::parse_str;
pub use report::{is_valid_identifier, sanitize_identifier, LoadReport,
                 MaterialCollision, NameSanitization};
pub use schema::{
    Actuator, ActuatorJointRef, ActuatorMode, CollisionPair, ColorSpec, Gait, GaitTypeConfig,
    Geom, Home, Inertial, Joint, JointDynamics, JointKind, JointLimit, Link, LoopClosure,
    Material, MisaFile, Mimic, MjcfPhysics, Origin, Pose, RobotMeta, Sensor, SensorKind, Sequence,
    SequenceStep, Visual, Collision, CURRENT_VERSION, SCHEMA_TAG,
};
pub use source::{
    validate_logical_path, AssetError, AssetSource, FileSystemSource, InMemorySource, NullSource,
    StaticBundleSource,
};
pub use write::write_str;

use std::path::Path;

// ─── Errors ────────────────────────────────────────────────────────────────

/// Top-level error type for `.misa` load / parse / save.
#[derive(Debug, Clone)]
pub enum NativeError {
    /// I/O or asset access failed.
    Io(String),
    /// TOML failed to parse, or the document didn't match the schema.
    Toml(String),
    /// `schema = "..."` header is missing, malformed, or names a version
    /// this build can't read.
    UnsupportedSchema(String),
    /// Structural validation failed (e.g. joint references unknown link,
    /// duplicate name, root link not in `link` list).
    Validation(String),
    /// An [`AssetSource`] reported failure for a required asset.
    Asset(AssetError),
}

impl std::fmt::Display for NativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NativeError::Io(m) => write!(f, "I/O error: {m}"),
            NativeError::Toml(m) => write!(f, "TOML error: {m}"),
            NativeError::UnsupportedSchema(m) => write!(f, "unsupported .misa schema: {m}"),
            NativeError::Validation(m) => write!(f, "validation error: {m}"),
            NativeError::Asset(e) => write!(f, "asset error: {e}"),
        }
    }
}

impl std::error::Error for NativeError {}

impl From<AssetError> for NativeError {
    fn from(e: AssetError) -> Self {
        NativeError::Asset(e)
    }
}

// ─── ParseOutput ───────────────────────────────────────────────────────────

/// The fully-decoded contents of a `.misa` file.
///
/// At the layer-2 boundary we return the raw schema struct rather than
/// converting straight to [`crate::model::Model`]; the conversion lives
/// in a separate step ([`build_model`]) so callers that only want the
/// structural data (e.g. a model linter, a documentation generator)
/// don't pay the cost of building the dynamics model.
#[derive(Debug, Clone)]
pub struct ParseOutput {
    /// The parsed document, post-sanitisation.
    pub file: schema::MisaFile,
    /// Diagnostics — sanitised names, missing meshes, warnings.
    pub report: LoadReport,
}

// ─── Layer 3: load / save (std::fs convenience) ────────────────────────────

/// Read a `.misa` file from disk.
///
/// Wraps [`parse_str`] with a [`FileSystemSource`] rooted at the file's
/// parent directory, so mesh references like `"meshes/trunk.stl"` resolve
/// relative to the `.misa` location.
pub fn load(path: impl AsRef<Path>) -> Result<ParseOutput, NativeError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| {
        NativeError::Io(format!("read {}: {e}", path.display()))
    })?;
    let root = path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let assets = FileSystemSource::new(root);
    parse_str(&text, &assets)
}

/// Write a `.misa` file to disk. The caller is responsible for ensuring
/// `path.parent()` exists.
pub fn save(path: impl AsRef<Path>, file: &schema::MisaFile) -> Result<(), NativeError> {
    let path = path.as_ref();
    let text = write_str(file)?;
    std::fs::write(path, text).map_err(|e| {
        NativeError::Io(format!("write {}: {e}", path.display()))
    })
}

// ─── Integration tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::schema::*;

    /// Build a small but realistic `.misa` document in code, then go
    /// `MisaFile → TOML → MisaFile → Model + GeometryModels` to exercise
    /// the full pipeline end-to-end.
    fn build_sample_file() -> MisaFile {
        let mut f = MisaFile::new("biped_test", "trunk");

        f.material.push(Material {
            name: "red_plastic".into(),
            color: ColorSpec::Hex("#cc4422".into()),
        });

        f.link.push(Link {
            name: "trunk".into(),
            description: String::new(),
            collision_enabled: true,
            inertial: Inertial {
                mass: 5.0,
                ixx: 0.10,
                iyy: 0.10,
                izz: 0.10,
                ..Default::default()
            },
            visual: vec![Visual {
                origin: Origin::default(),
                geom: Geom::Box {
                    size: [0.30, 0.20, 0.10],
                },
                color: None,
                material: Some("red_plastic".into()),
            }],
            collision: vec![Collision {
                origin: Origin::default(),
                geom: Geom::Box {
                    size: [0.30, 0.20, 0.10],
                },
                physics: None,
            }],
        });

        for side in &["left", "right"] {
            f.link.push(Link {
                name: format!("{side}_thigh"),
                description: String::new(),
                collision_enabled: true,
                inertial: Inertial {
                    mass: 0.8,
                    ixx: 0.01,
                    iyy: 0.01,
                    izz: 0.001,
                    ..Default::default()
                },
                visual: vec![Visual {
                    origin: Origin::default(),
                    geom: Geom::Cylinder {
                        radius: 0.03,
                        length: 0.20,
                    },
                    color: Some(ColorSpec::Rgba([0.5, 0.5, 0.5, 1.0])),
                    material: None,
                }],
                collision: Vec::new(),
            });
            f.joint.push(Joint {
                name: format!("{side}_hip_pitch"),
                kind: JointKind::Revolute,
                parent: "trunk".into(),
                child: format!("{side}_thigh"),
                axis: [0.0, 1.0, 0.0],
                origin: Origin {
                    xyz: [0.0, if *side == "left" { 0.10 } else { -0.10 }, -0.05],
                    rpy: Some([0.0, 0.0, 0.0]),
                    quat: None,
                },
                limit: JointLimit {
                    lower: -1.5,
                    upper: 1.5,
                    effort: 30.0,
                    velocity: 10.0,
                },
                dynamics: JointDynamics {
                    armature: 0.001,
                    damping: 0.05,
                    friction: 0.0,
                },
            });
        }

        f.actuator.push(Actuator {
            name: "left_motor".into(),
            mode: ActuatorMode::Position,
            joints: vec![ActuatorJointRef {
                name: "left_hip_pitch".into(),
                gear: 1.0,
            }],
            kp: 100.0,
            kv: 1.2,
        });
        f.actuator.push(Actuator {
            name: "right_motor".into(),
            mode: ActuatorMode::Position,
            joints: vec![ActuatorJointRef {
                name: "right_hip_pitch".into(),
                gear: 1.0,
            }],
            kp: 100.0,
            kv: 1.2,
        });

        f
    }

    #[test]
    fn end_to_end_round_trip() {
        let original = build_sample_file();
        let toml = write_str(&original).expect("write");
        let out = parse_str(&toml, &NullSource).expect("parse");
        assert!(out.report.is_empty(), "unexpected report: {:?}", out.report);
        assert_eq!(out.file.robot.name, "biped_test");
        assert_eq!(out.file.link.len(), 3);
        assert_eq!(out.file.joint.len(), 2);
        assert_eq!(out.file.actuator.len(), 2);

        let (model, visual, collision) = build_model(&out.file).expect("build_model");
        assert_eq!(model.name, "biped_test");
        // 2 movable joints + universe = joints.len() == 3
        assert_eq!(model.joints.len(), 3);
        // 1 visual on trunk + 2 thigh visuals
        assert_eq!(visual.objects.len(), 3);
        // 1 collision on trunk only
        assert_eq!(collision.objects.len(), 1);
    }

    #[test]
    fn missing_root_rejected() {
        let mut f = build_sample_file();
        f.robot.root = "nonexistent".into();
        let toml = write_str(&f).unwrap();
        let err = parse_str(&toml, &NullSource).unwrap_err();
        assert!(matches!(err, NativeError::Validation(_)));
    }

    #[test]
    fn dangling_joint_parent_rejected() {
        let mut f = build_sample_file();
        f.joint[0].parent = "ghost".into();
        let toml = write_str(&f).unwrap();
        let err = parse_str(&toml, &NullSource).unwrap_err();
        match err {
            NativeError::Validation(m) => assert!(m.contains("parent link"), "msg: {m}"),
            _ => panic!("expected Validation, got {err:?}"),
        }
    }

    #[test]
    fn duplicate_link_name_rejected() {
        let mut f = build_sample_file();
        f.link[1].name = "trunk".into();
        // intentionally bypass write_str here; toml::to_string handles it
        let toml = toml::to_string(&f).unwrap();
        let err = parse_str(&toml, &NullSource).unwrap_err();
        match err {
            NativeError::Validation(m) => assert!(m.contains("duplicate")),
            _ => panic!("expected Validation, got {err:?}"),
        }
    }

    #[test]
    fn unknown_actuator_joint_rejected() {
        let mut f = build_sample_file();
        f.actuator[0].joints[0].name = "ghost_joint".into();
        let toml = write_str(&f).unwrap();
        let err = parse_str(&toml, &NullSource).unwrap_err();
        match err {
            NativeError::Validation(m) => assert!(m.contains("unknown joint"), "msg: {m}"),
            _ => panic!("expected Validation, got {err:?}"),
        }
    }

    #[test]
    fn color_and_material_mutual_exclusion() {
        let mut f = build_sample_file();
        f.link[0].visual[0].color = Some(ColorSpec::Hex("#ffffff".into()));
        f.link[0].visual[0].material = Some("red_plastic".into());
        let toml = write_str(&f).unwrap();
        let err = parse_str(&toml, &NullSource).unwrap_err();
        match err {
            NativeError::Validation(m) => assert!(m.contains("both `color` and `material`")),
            _ => panic!("expected Validation, got {err:?}"),
        }
    }

    #[test]
    fn schema_tag_mismatch_rejected() {
        let mut f = build_sample_file();
        f.schema = "future/2".into();
        let err = write_str(&f).unwrap_err();
        assert!(matches!(err, NativeError::UnsupportedSchema(_)));
    }

    #[test]
    fn invalid_schema_at_parse_rejected() {
        let toml = r#"
schema = "not_misarta/1"

[robot]
name = "test"
root = "base"

[[link]]
name = "base"
"#;
        let err = parse_str(toml, &NullSource).unwrap_err();
        assert!(matches!(err, NativeError::UnsupportedSchema(_)));
    }

    #[test]
    fn name_sanitisation_recorded_in_report() {
        // Hand-crafted TOML with an invalid identifier ("front-leg")
        let toml = r#"
schema = "misarta/1"

[robot]
name = "demo"
root = "base"

[[link]]
name = "base"

[[link]]
name = "front-leg"

[[joint]]
name = "hip"
type = "revolute"
parent = "base"
child = "front-leg"
"#;
        let out = parse_str(toml, &NullSource).expect("parse");
        assert_eq!(out.file.link[1].name, "front_leg");
        assert_eq!(out.file.joint[0].child, "front_leg"); // cross-ref patched
        assert_eq!(out.report.sanitized_names.len(), 1);
        assert_eq!(out.report.sanitized_names[0].original, "front-leg");
        assert_eq!(out.report.sanitized_names[0].sanitized, "front_leg");
    }

    #[test]
    fn missing_mesh_recorded_non_fatal() {
        let mut f = build_sample_file();
        f.link[0].visual[0] = Visual {
            origin: Origin::default(),
            geom: Geom::Mesh {
                file: "meshes/missing.stl".into(),
                scale: [1.0, 1.0, 1.0],
            },
            color: None,
            material: Some("red_plastic".into()),
        };
        let toml = write_str(&f).unwrap();
        let out = parse_str(&toml, &NullSource).expect("parse should succeed");
        assert_eq!(out.report.missing_meshes.len(), 1);
        assert_eq!(out.report.missing_meshes[0], "meshes/missing.stl");
    }

    #[test]
    fn mimic_round_trips_through_build() {
        let mut f = build_sample_file();
        f.mimic.push(Mimic {
            joint: "right_hip_pitch".into(),
            source: "left_hip_pitch".into(),
            multiplier: -1.0,
            offset: 0.0,
        });
        let toml = write_str(&f).unwrap();
        let out = parse_str(&toml, &NullSource).expect("parse");
        let (model, _, _) = build_model(&out.file).expect("build_model");
        assert_eq!(model.mimic.len(), 1);
        assert_eq!(model.mimic[0].multiplier, -1.0);
    }

    #[test]
    fn double_child_rejected_use_loop_closure() {
        let mut f = build_sample_file();
        // Make right_hip_pitch share child link with left_hip_pitch
        f.joint[1].child = "left_thigh".into();
        let toml = write_str(&f).unwrap();
        let err = parse_str(&toml, &NullSource).unwrap_err();
        match err {
            NativeError::Validation(m) => assert!(m.contains("loop_closure"), "msg: {m}"),
            _ => panic!("expected Validation, got {err:?}"),
        }
    }
}
