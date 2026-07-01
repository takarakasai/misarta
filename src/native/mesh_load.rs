//! Deferred mesh loading for `.misa`-derived [`GeometryModel`]s.
//!
//! [`crate::native::build_model`] returns `GeometryModel`s with their
//! `mesh_path` populated but `mesh_data` still empty — the structural
//! shape table is enough for analysis, validation, and many headless
//! workflows. Visual rendering and collision queries that need actual
//! triangle data call [`load_meshes`] (or [`load_meshes_into`])
//! afterwards to populate `mesh_data` from an [`AssetSource`].
//!
//! Splitting the pass like this keeps embedded / WASM callers in
//! control of when (and whether) raw mesh bytes hit the linear memory.
//! On a controller that only needs dynamics, you can skip this step
//! entirely and ship without any STL bytes at all.

use crate::geometry::{GeometryModel, GeometryShape};
use crate::mesh::MeshData;

use super::source::{AssetError, AssetSource};
use super::NativeError;

/// Per-call summary of a mesh-loading pass.
///
/// Loading is best-effort: missing or malformed meshes are recorded
/// here rather than aborting the pass. The caller decides whether the
/// result is fatal.
#[derive(Debug, Clone, Default)]
pub struct MeshLoadReport {
    /// Number of `GeometryObject`s whose `mesh_data` was populated by
    /// this call.
    pub loaded: usize,
    /// Number of `GeometryObject`s that were skipped because their
    /// `mesh_data` was already populated.
    pub already_loaded: usize,
    /// Mesh paths the [`AssetSource`] reported as missing.
    pub missing: Vec<String>,
    /// Mesh paths whose bytes loaded but the parser rejected
    /// (invalid STL, unsupported variant, ...). Each entry is
    /// `(path, error message)`.
    pub failed: Vec<(String, String)>,
}

impl MeshLoadReport {
    pub fn is_clean(&self) -> bool {
        self.missing.is_empty() && self.failed.is_empty()
    }
}

/// Walk a [`GeometryModel`] and load mesh data for every
/// `GeometryShape::Mesh` whose `mesh_data` slot is still empty.
///
/// Understands STL, OBJ and DAE (case-insensitive extensions). Other
/// formats log a warning into [`MeshLoadReport::failed`].
///
/// Mesh references are passed through [`normalise_mesh_reference`] before
/// being handed to `assets`, so URDF-style `package://name/sub/foo.stl`
/// and `file://abs/path` references work alongside `.misa`'s
/// already-relative paths. Absolute paths are still rejected by the
/// `AssetSource` sandbox.
///
/// Returns [`Err(NativeError::Asset(_))`] only for *unexpected* asset
/// errors (permission denied, IO error). `NotFound` is non-fatal and
/// surfaces via `report.missing` so a single missing mesh doesn't kill
/// the load.
pub fn load_meshes(
    geom: &mut GeometryModel,
    assets: &dyn AssetSource,
) -> Result<MeshLoadReport, NativeError> {
    let mut report = MeshLoadReport::default();

    for obj in geom.objects.iter_mut() {
        // Already loaded? (e.g. test fixture pre-populated the field)
        if obj.mesh_data.is_some() {
            report.already_loaded += 1;
            continue;
        }
        // Only mesh shapes have something to load.
        let raw_path = match &obj.shape {
            GeometryShape::Mesh { filename, .. } => filename.clone(),
            _ => continue,
        };
        if raw_path.is_empty() {
            // Procedural / placeholder mesh with no source — nothing to do.
            continue;
        }
        let path = normalise_mesh_reference(&raw_path);

        match assets.read(&path) {
            Ok(bytes) => match parse_mesh_bytes(&path, &bytes) {
                Ok(mesh) => {
                    obj.mesh_data = Some(mesh);
                    report.loaded += 1;
                }
                Err(e) => {
                    // Record under the *original* path so users can find
                    // the source reference in their input file.
                    report.failed.push((raw_path, e));
                }
            },
            Err(AssetError::NotFound) => {
                report.missing.push(raw_path);
            }
            Err(other) => return Err(NativeError::Asset(other)),
        }
    }

    Ok(report)
}

/// Normalise a mesh reference into an [`AssetSource`]-compatible logical
/// path. Strips URI prefixes commonly produced by URDF / SDF importers:
///
/// - `package://<pkg>/sub/foo.stl` → `sub/foo.stl`
/// - `file:///absolute/path.stl` → unchanged (caller's `AssetSource`
///   sandbox will reject the absolute form, which is the right outcome)
/// - already-relative paths are returned untouched
///
/// Public so URDF / SDF consumers can pre-normalise references before
/// other passes (validation, asset-existence checks, etc.).
pub fn normalise_mesh_reference(s: &str) -> String {
    if let Some(rest) = s.strip_prefix("package://") {
        // Drop the package name (everything up to the first `/`).
        if let Some(slash) = rest.find('/') {
            return rest[slash + 1..].to_string();
        }
        return rest.to_string();
    }
    if let Some(rest) = s.strip_prefix("file://") {
        return rest.to_string();
    }
    s.to_string()
}

/// Load both visual and collision meshes in a single call.
///
/// Equivalent to calling [`load_meshes`] separately on each model and
/// merging the reports. Convenient when the caller wants the fully
/// realised model for rendering AND collision and treats their
/// missing/failed sets uniformly.
pub fn load_meshes_into(
    visual: &mut GeometryModel,
    collision: &mut GeometryModel,
    assets: &dyn AssetSource,
) -> Result<MeshLoadReport, NativeError> {
    let mut report = load_meshes(visual, assets)?;
    let collision_report = load_meshes(collision, assets)?;
    report.loaded += collision_report.loaded;
    report.already_loaded += collision_report.already_loaded;
    report.missing.extend(collision_report.missing);
    report.failed.extend(collision_report.failed);
    Ok(report)
}

/// Dispatch on extension; new formats slot in here.
fn parse_mesh_bytes(path: &str, bytes: &[u8]) -> Result<MeshData, String> {
    let ext = path
        .rsplit('.')
        .next()
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "stl" => MeshData::from_stl_bytes(bytes),
        "obj" => MeshData::from_obj_bytes(bytes),
        "dae" => {
            // DAE arrives as bytes from the AssetSource, so relative
            // texture paths inside the file cannot be resolved against a
            // real directory — geometry loads fine, texture references
            // stay as written.
            let xml = std::str::from_utf8(bytes)
                .map_err(|e| format!("DAE is not valid UTF-8: {e}"))?;
            crate::collada::load_dae_string(xml, std::path::Path::new("."))
        }
        other => Err(format!(
            "unsupported mesh extension '.{other}' (supported: stl, obj, dae)"
        )),
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{GeometryObject, GeometryShape};
    use crate::native::source::InMemorySource;
    use crate::se3;
    use nalgebra::Vector3;

    /// Minimal valid binary-STL bytes for a single triangle.
    ///
    /// 80-byte header + 4-byte triangle count + 1 × 50-byte triangle
    /// (12 floats + 2-byte attribute) = 134 bytes total.
    fn one_triangle_stl_bytes() -> Vec<u8> {
        let mut buf = vec![0u8; 80]; // header
        buf.extend_from_slice(&1u32.to_le_bytes()); // triangle count
        // normal
        for f in [0.0_f32, 0.0, 1.0] {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        // 3 vertices
        for v in [[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for f in v {
                buf.extend_from_slice(&f.to_le_bytes());
            }
        }
        // attribute byte count
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf
    }

    fn mesh_object(name: &str, file: &str) -> GeometryObject {
        GeometryObject {
            name: name.into(),
            parent_joint: 0,
            placement: se3::identity(),
            shape: GeometryShape::Mesh {
                filename: file.into(),
                scale: Vector3::new(1.0, 1.0, 1.0),
            },
            mesh_path: Some(file.into()),
            mesh_scale: Some(Vector3::new(1.0, 1.0, 1.0)),
            mesh_data: None,
            material: None,
        }
    }

    #[test]
    fn loads_stl_via_in_memory_source() {
        let mut src = InMemorySource::new();
        src.insert("meshes/cube.stl", one_triangle_stl_bytes());

        let mut geom = GeometryModel::new();
        geom.add(mesh_object("a", "meshes/cube.stl"));
        geom.add(mesh_object("b", "meshes/cube.stl"));

        let report = load_meshes(&mut geom, &src).unwrap();
        assert_eq!(report.loaded, 2);
        assert_eq!(report.already_loaded, 0);
        assert!(report.is_clean());
        for obj in &geom.objects {
            let m = obj.mesh_data.as_ref().expect("mesh_data populated");
            assert_eq!(m.vertices.len(), 3);
            assert_eq!(m.indices.len(), 1);
        }
    }

    #[test]
    fn missing_mesh_is_non_fatal_and_reported() {
        let src = InMemorySource::new(); // empty
        let mut geom = GeometryModel::new();
        geom.add(mesh_object("a", "meshes/missing.stl"));

        let report = load_meshes(&mut geom, &src).unwrap();
        assert_eq!(report.loaded, 0);
        assert_eq!(report.missing.len(), 1);
        assert_eq!(report.missing[0], "meshes/missing.stl");
        assert!(geom.objects[0].mesh_data.is_none());
    }

    #[test]
    fn malformed_stl_is_recorded_as_failed() {
        let mut src = InMemorySource::new();
        src.insert("meshes/garbage.stl", b"not an stl".to_vec());
        let mut geom = GeometryModel::new();
        geom.add(mesh_object("g", "meshes/garbage.stl"));

        let report = load_meshes(&mut geom, &src).unwrap();
        assert_eq!(report.loaded, 0);
        assert_eq!(report.failed.len(), 1);
        assert_eq!(report.failed[0].0, "meshes/garbage.stl");
    }

    #[test]
    fn already_loaded_mesh_is_skipped() {
        let mut src = InMemorySource::new();
        src.insert("meshes/cube.stl", one_triangle_stl_bytes());
        let mut geom = GeometryModel::new();
        geom.add(mesh_object("pre", "meshes/cube.stl"));
        // Pre-populate to simulate already-loaded state
        geom.objects[0].mesh_data = Some(MeshData::from_stl_bytes(
            &one_triangle_stl_bytes(),
        ).unwrap());

        let report = load_meshes(&mut geom, &src).unwrap();
        assert_eq!(report.loaded, 0);
        assert_eq!(report.already_loaded, 1);
    }

    #[test]
    fn unsupported_extension_recorded_as_failed() {
        // `.xyzzy` is not in the dispatcher; load_meshes should record it
        // as failed (not crash). `.stl` and `.obj` are the supported set.
        let mut src = InMemorySource::new();
        src.insert("meshes/foo.xyzzy", b"garbage".to_vec());
        let mut geom = GeometryModel::new();
        geom.add(mesh_object("o", "meshes/foo.xyzzy"));

        let report = load_meshes(&mut geom, &src).unwrap();
        assert_eq!(report.loaded, 0);
        assert_eq!(report.failed.len(), 1);
        assert!(report.failed[0].1.contains("unsupported mesh extension"));
    }

    /// A 2-triangle OBJ — enough to exercise tobj's triangulation and
    /// vertex deduplication paths in `MeshData::from_obj_bytes`.
    const TINY_OBJ_BYTES: &[u8] = b"\
o tri
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
v 0.0 0.0 1.0
f 1 2 3
f 1 3 4
";

    #[test]
    fn loads_obj_via_in_memory_source() {
        // Regression: pre-fix this fell into the `unsupported extension`
        // arm and the OBJ data was discarded.
        let mut src = InMemorySource::new();
        src.insert("meshes/tri.obj", TINY_OBJ_BYTES.to_vec());
        let mut geom = GeometryModel::new();
        geom.add(mesh_object("o", "meshes/tri.obj"));

        let report = load_meshes(&mut geom, &src).unwrap();
        assert_eq!(report.loaded, 1);
        assert!(
            report.is_clean(),
            "OBJ load reported issues: missing={:?} failed={:?}",
            report.missing, report.failed
        );
        let m = geom.objects[0]
            .mesh_data
            .as_ref()
            .expect("mesh_data populated");
        assert_eq!(m.indices.len(), 2, "expected 2 triangles");
        assert_eq!(m.vertices.len(), 4, "expected 4 deduped verts");
    }

    #[test]
    fn non_mesh_shapes_are_skipped() {
        let mut geom = GeometryModel::new();
        geom.add(GeometryObject {
            name: "box".into(),
            parent_joint: 0,
            placement: se3::identity(),
            shape: GeometryShape::Box { x: 1.0, y: 1.0, z: 1.0 },
            mesh_path: None,
            mesh_scale: None,
            mesh_data: None,
            material: None,
        });
        let report = load_meshes(&mut geom, &InMemorySource::new()).unwrap();
        assert_eq!(report.loaded, 0);
        assert_eq!(report.missing.len(), 0);
        assert_eq!(report.failed.len(), 0);
    }

    #[test]
    fn empty_filename_is_skipped() {
        let mut geom = GeometryModel::new();
        geom.add(mesh_object("p", ""));
        let report = load_meshes(&mut geom, &InMemorySource::new()).unwrap();
        assert_eq!(report.loaded, 0);
        assert!(report.missing.is_empty());
    }

    #[test]
    fn normalise_mesh_reference_strips_package_uri() {
        assert_eq!(
            normalise_mesh_reference("package://my_robot/meshes/trunk.stl"),
            "meshes/trunk.stl"
        );
        assert_eq!(normalise_mesh_reference("package://only_pkg"), "only_pkg");
        assert_eq!(normalise_mesh_reference("file:///abs/path.stl"), "/abs/path.stl");
        assert_eq!(normalise_mesh_reference("meshes/foo.stl"), "meshes/foo.stl");
        assert_eq!(normalise_mesh_reference("foo.stl"), "foo.stl");
    }

    #[test]
    fn load_meshes_handles_urdf_package_uri() {
        // Simulate a URDF-style mesh reference: the GeometryObject was
        // populated by misarta::urdf::load_urdf_geometry which leaves
        // package:// intact. load_meshes should normalise it before
        // calling AssetSource.
        let mut src = InMemorySource::new();
        src.insert("meshes/cube.stl", one_triangle_stl_bytes());
        let mut geom = GeometryModel::new();
        geom.add(mesh_object("u", "package://my_robot/meshes/cube.stl"));

        let report = load_meshes(&mut geom, &src).unwrap();
        assert_eq!(report.loaded, 1);
        assert!(report.is_clean());
        assert!(geom.objects[0].mesh_data.is_some());
    }

    #[test]
    fn load_meshes_into_combines_visual_and_collision() {
        let mut src = InMemorySource::new();
        src.insert("meshes/cube.stl", one_triangle_stl_bytes());

        let mut visual = GeometryModel::new();
        visual.add(mesh_object("v", "meshes/cube.stl"));
        let mut collision = GeometryModel::new();
        collision.add(mesh_object("c", "meshes/cube.stl"));

        let report = load_meshes_into(&mut visual, &mut collision, &src).unwrap();
        assert_eq!(report.loaded, 2);
        assert!(visual.objects[0].mesh_data.is_some());
        assert!(collision.objects[0].mesh_data.is_some());
    }
}
