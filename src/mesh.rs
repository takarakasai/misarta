//! Triangle mesh loading and conversion.
//!
//! Loads external mesh files (**STL**, **Collada / DAE**) into [`MeshData`] — an
//! indexed triangle mesh with per-face normals, optional per-vertex normals,
//! texture coordinates, materials and sub-meshes — and provides conversion to
//! parry3d's [`TriMesh`](parry3d::shape::TriMesh) for collision detection.
//!
//! # Example
//!
//! ```no_run
//! use misarta::mesh::MeshData;
//! use nalgebra::Vector3;
//! use std::path::Path;
//!
//! let mesh = MeshData::from_stl(Path::new("robot.stl")).unwrap();
//! let scaled = mesh.scaled(&Vector3::new(0.001, 0.001, 0.001));
//! println!("{} vertices, {} triangles", scaled.num_vertices(), scaled.num_triangles());
//!
//! // Convert to parry3d TriMesh for collision queries.
//! let trimesh = scaled.to_trimesh();
//! ```

use nalgebra::{Point2, Point3, Vector3};
use std::path::Path;

// ─── Material ───────────────────────────────────────────────────────────────

/// Surface material properties (Phong / Lambert).
///
/// Colours are linear RGBA in `[0, 1]`.  Only `diffuse` is required; the
/// others default to sensible values for a matte surface.
#[derive(Debug, Clone, PartialEq)]
pub struct Material {
    /// Human-readable name (may be empty).
    pub name: String,
    /// Diffuse colour — the primary surface colour.
    pub diffuse: [f64; 4],
    /// Specular colour (default `[0, 0, 0, 1]`).
    pub specular: [f64; 4],
    /// Ambient colour (default = same as diffuse).
    pub ambient: [f64; 4],
    /// Emissive colour (default `[0, 0, 0, 1]`).
    pub emission: [f64; 4],
    /// Phong shininess exponent (default `0`).
    pub shininess: f64,
    /// File path of the diffuse texture image (empty = none).
    pub texture_diffuse: Option<String>,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: String::new(),
            diffuse: [0.8, 0.8, 0.8, 1.0],
            specular: [0.0, 0.0, 0.0, 1.0],
            ambient: [0.8, 0.8, 0.8, 1.0],
            emission: [0.0, 0.0, 0.0, 1.0],
            shininess: 0.0,
            texture_diffuse: None,
        }
    }
}

impl Material {
    /// Create a simple material with only a diffuse colour.
    pub fn from_color(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self {
            diffuse: [r, g, b, a],
            ambient: [r, g, b, a],
            ..Default::default()
        }
    }
}

// ─── SubMesh ────────────────────────────────────────────────────────────────

/// A contiguous range of triangles that share a single [`Material`].
///
/// `tri_start..tri_start + tri_count` indexes into [`MeshData::indices`].
#[derive(Debug, Clone, PartialEq)]
pub struct SubMesh {
    /// Human-readable name (e.g. the Collada `<geometry>` id).
    pub name: String,
    /// Index of the first triangle in [`MeshData::indices`].
    pub tri_start: usize,
    /// Number of triangles in this sub-mesh.
    pub tri_count: usize,
    /// Index into [`MeshData::materials`].
    pub material_index: Option<usize>,
}

// ─── MeshData ───────────────────────────────────────────────────────────────

/// Indexed triangle mesh with per-face normals.
///
/// Vertices are deduplicated so that shared vertices between triangles have a
/// single entry in [`vertices`](MeshData::vertices). Triangle connectivity is
/// stored in [`indices`](MeshData::indices).
///
/// Optionally stores:
/// - per-vertex normals ([`vertex_normals`](MeshData::vertex_normals))
/// - texture coordinates ([`texcoords`](MeshData::texcoords))
/// - [`materials`](MeshData::materials) and [`submeshes`](MeshData::submeshes)
#[derive(Debug, Clone)]
pub struct MeshData {
    /// Unique vertex positions.
    pub vertices: Vec<Point3<f64>>,
    /// Triangle indices — each `[u32; 3]` references three entries in `vertices`.
    pub indices: Vec<[u32; 3]>,
    /// Per-face normals (one per triangle, same length as `indices`).
    pub face_normals: Vec<Vector3<f64>>,

    // ── Optional attributes ─────────────────────────────────────────────

    /// Per-vertex normals (same length as `vertices`, or empty).
    pub vertex_normals: Vec<Vector3<f64>>,
    /// Per-vertex texture coordinates (same length as `vertices`, or empty).
    pub texcoords: Vec<Point2<f64>>,

    // ── Materials & sub-meshes ───────────────────────────────────────────

    /// Material table.  Indexed by [`SubMesh::material_index`].
    pub materials: Vec<Material>,
    /// Sub-meshes — each refers to a contiguous triangle range and an optional
    /// material.  If empty the whole mesh has no material assignment.
    pub submeshes: Vec<SubMesh>,
}

impl MeshData {
    // ── Constructors ────────────────────────────────────────────────────

    /// Load mesh data from an **STL** file (binary or ASCII).
    ///
    /// `stl_io` returns an [`IndexedMesh`](stl_io::IndexedMesh) with already-
    /// deduplicated vertices, so no additional merging is performed.
    pub fn from_stl(path: &Path) -> Result<Self, String> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| format!("cannot open STL file {}: {e}", path.display()))?;
        let stl = stl_io::read_stl(&mut file)
            .map_err(|e| format!("STL parse error for {}: {e}", path.display()))?;

        Self::from_indexed_mesh(&stl)
    }

    /// Parse mesh data from an in-memory STL byte buffer (binary or ASCII).
    ///
    /// Same parser as [`MeshData::from_stl`] but works without `std::fs` —
    /// useful for callers that obtain bytes via an asset abstraction
    /// (e.g. [`crate::native::AssetSource`]) or have the data
    /// embedded in a binary via `include_bytes!`.
    pub fn from_stl_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut cursor = std::io::Cursor::new(bytes);
        let stl = stl_io::read_stl(&mut cursor)
            .map_err(|e| format!("STL parse error: {e}"))?;
        Self::from_indexed_mesh(&stl)
    }

    /// Build `MeshData` from an `stl_io::IndexedMesh`.
    pub fn from_indexed_mesh(mesh: &stl_io::IndexedMesh) -> Result<Self, String> {
        let vertices: Vec<Point3<f64>> = mesh
            .vertices
            .iter()
            .map(|v| Point3::new(v[0] as f64, v[1] as f64, v[2] as f64))
            .collect();

        let indices: Vec<[u32; 3]> = mesh
            .faces
            .iter()
            .map(|f| [f.vertices[0] as u32, f.vertices[1] as u32, f.vertices[2] as u32])
            .collect();

        let face_normals: Vec<Vector3<f64>> = indices
            .iter()
            .enumerate()
            .map(|(fi, tri)| {
                let v0 = &vertices[tri[0] as usize];
                let v1 = &vertices[tri[1] as usize];
                let v2 = &vertices[tri[2] as usize];
                let n = (v1 - v0).cross(&(v2 - v0));
                let len = n.norm();
                if len > 1e-30 {
                    n / len
                } else {
                    // Degenerate — fall back to STL stored normal.
                    let sn = &mesh.faces[fi].normal;
                    Vector3::new(sn[0] as f64, sn[1] as f64, sn[2] as f64)
                }
            })
            .collect();

        Ok(Self {
            vertices,
            indices,
            face_normals,
            vertex_normals: Vec::new(),
            texcoords: Vec::new(),
            materials: Vec::new(),
            submeshes: Vec::new(),
        })
    }

    /// Build `MeshData` from raw `stl_io::Triangle` slices (non-indexed,
    /// useful for testing / procedural generation without a file).
    ///
    /// Vertices are deduplicated with an epsilon tolerance of `1e-10`.
    pub fn from_stl_triangles(faces: &[stl_io::Triangle]) -> Result<Self, String> {
        if faces.is_empty() {
            return Ok(Self {
                vertices: Vec::new(),
                indices: Vec::new(),
                face_normals: Vec::new(),
                vertex_normals: Vec::new(),
                texcoords: Vec::new(),
                materials: Vec::new(),
                submeshes: Vec::new(),
            });
        }

        // Vertex deduplication with spatial hashing.
        let eps = 1e-10_f64;
        let inv_cell = 1.0 / eps.max(1e-12);
        let mut vertex_map: std::collections::HashMap<(i64, i64, i64), u32> =
            std::collections::HashMap::new();
        let mut vertices: Vec<Point3<f64>> = Vec::new();
        let mut indices: Vec<[u32; 3]> = Vec::with_capacity(faces.len());
        let mut face_normals: Vec<Vector3<f64>> = Vec::with_capacity(faces.len());

        let quantise = |v: f64| -> i64 { (v * inv_cell).round() as i64 };

        for face in faces {
            let mut tri_idx = [0u32; 3];
            for (k, vert) in face.vertices.iter().enumerate() {
                let x = vert[0] as f64;
                let y = vert[1] as f64;
                let z = vert[2] as f64;
                let key = (quantise(x), quantise(y), quantise(z));
                let idx = vertex_map.entry(key).or_insert_with(|| {
                    let i = vertices.len() as u32;
                    vertices.push(Point3::new(x, y, z));
                    i
                });
                tri_idx[k] = *idx;
            }
            indices.push(tri_idx);

            // Compute face normal from the triangle vertices.
            let v0 = &vertices[tri_idx[0] as usize];
            let v1 = &vertices[tri_idx[1] as usize];
            let v2 = &vertices[tri_idx[2] as usize];
            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let n = edge1.cross(&edge2);
            let len = n.norm();
            if len > 1e-30 {
                face_normals.push(n / len);
            } else {
                face_normals.push(Vector3::new(
                    face.normal[0] as f64,
                    face.normal[1] as f64,
                    face.normal[2] as f64,
                ));
            }
        }

        Ok(Self {
            vertices,
            indices,
            face_normals,
            vertex_normals: Vec::new(),
            texcoords: Vec::new(),
            materials: Vec::new(),
            submeshes: Vec::new(),
        })
    }

    /// Build `MeshData` from a flat interleaved `[x, y, z, nx, ny, nz, ...]`
    /// buffer (stride 6, as used by the OpenGL renderer).
    ///
    /// Vertices are spatially deduplicated with an epsilon of `1e-10`.
    pub fn from_flat_vertices_f32(flat: &[f32]) -> Self {
        if flat.len() < 18 {
            return Self {
                vertices: Vec::new(),
                indices: Vec::new(),
                face_normals: Vec::new(),
                vertex_normals: Vec::new(),
                texcoords: Vec::new(),
                materials: Vec::new(),
                submeshes: Vec::new(),
            };
        }

        let eps = 1e-10_f64;
        let inv_cell = 1.0 / eps.max(1e-12);
        let quantise = |v: f64| -> i64 { (v * inv_cell).round() as i64 };

        let mut vertex_map: std::collections::HashMap<(i64, i64, i64), u32> =
            std::collections::HashMap::new();
        let mut vertices: Vec<Point3<f64>> = Vec::new();
        let mut indices: Vec<[u32; 3]> = Vec::new();
        let mut face_normals: Vec<Vector3<f64>> = Vec::new();

        let n_tris = flat.len() / 18;
        for ti in 0..n_tris {
            let base = ti * 18;
            let mut tri_idx = [0u32; 3];

            for k in 0..3 {
                let vbase = base + k * 6;
                let x = flat[vbase] as f64;
                let y = flat[vbase + 1] as f64;
                let z = flat[vbase + 2] as f64;
                let key = (quantise(x), quantise(y), quantise(z));
                let idx = vertex_map.entry(key).or_insert_with(|| {
                    let i = vertices.len() as u32;
                    vertices.push(Point3::new(x, y, z));
                    i
                });
                tri_idx[k] = *idx;
            }
            indices.push(tri_idx);

            // Face normal from the first vertex's stored normal
            let nx = flat[base + 3] as f64;
            let ny = flat[base + 4] as f64;
            let nz = flat[base + 5] as f64;
            let n = Vector3::new(nx, ny, nz);
            let len = n.norm();
            face_normals.push(if len > 1e-30 { n / len } else { Vector3::zeros() });
        }

        Self {
            vertices,
            indices,
            face_normals,
            vertex_normals: Vec::new(),
            texcoords: Vec::new(),
            materials: Vec::new(),
            submeshes: Vec::new(),
        }
    }

    // ── Queries ─────────────────────────────────────────────────────────

    /// Number of unique vertices.
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Number of triangles.
    pub fn num_triangles(&self) -> usize {
        self.indices.len()
    }

    /// `true` if per-vertex normals are populated (same count as `vertices`).
    pub fn has_vertex_normals(&self) -> bool {
        self.vertex_normals.len() == self.vertices.len()
    }

    /// `true` if texture coordinates are populated (same count as `vertices`).
    pub fn has_texcoords(&self) -> bool {
        self.texcoords.len() == self.vertices.len()
    }

    /// Number of materials.
    pub fn num_materials(&self) -> usize {
        self.materials.len()
    }

    /// Number of sub-meshes.
    pub fn num_submeshes(&self) -> usize {
        self.submeshes.len()
    }

    /// Return the material for triangle `tri_idx`, if any.
    pub fn material_for_triangle(&self, tri_idx: usize) -> Option<&Material> {
        for sm in &self.submeshes {
            if tri_idx >= sm.tri_start && tri_idx < sm.tri_start + sm.tri_count {
                return sm.material_index.and_then(|mi| self.materials.get(mi));
            }
        }
        None
    }

    /// Axis-aligned bounding box `(min_corner, max_corner)`.
    ///
    /// Returns `None` for an empty mesh.
    pub fn aabb(&self) -> Option<(Point3<f64>, Point3<f64>)> {
        if self.vertices.is_empty() {
            return None;
        }
        let mut min = self.vertices[0];
        let mut max = self.vertices[0];
        for v in &self.vertices[1..] {
            for k in 0..3 {
                if v[k] < min[k] {
                    min[k] = v[k];
                }
                if v[k] > max[k] {
                    max[k] = v[k];
                }
            }
        }
        Some((min, max))
    }

    // ── Transformations ─────────────────────────────────────────────────

    /// Return a new `MeshData` with each vertex multiplied component-wise by
    /// `scale`.
    pub fn scaled(&self, scale: &Vector3<f64>) -> Self {
        let vertices = self
            .vertices
            .iter()
            .map(|v| Point3::new(v.x * scale.x, v.y * scale.y, v.z * scale.z))
            .collect::<Vec<_>>();

        // Recompute face normals after (possibly non-uniform) scaling.
        let face_normals = self
            .indices
            .iter()
            .map(|tri| {
                let v0 = &vertices[tri[0] as usize];
                let v1 = &vertices[tri[1] as usize];
                let v2 = &vertices[tri[2] as usize];
                let n = (v1 - v0).cross(&(v2 - v0));
                let len = n.norm();
                if len > 1e-30 {
                    n / len
                } else {
                    Vector3::zeros()
                }
            })
            .collect();

        // Recompute vertex normals after scaling.
        let vertex_normals = if self.vertex_normals.is_empty() {
            Vec::new()
        } else {
            self.vertex_normals.iter().enumerate().map(|(i, _)| {
                // Approximate: use inverse-transpose of scale diagonal.
                // For axis-aligned scale S, n' = normalize(S^{-T} n).
                let n = &self.vertex_normals[i];
                let sn = Vector3::new(n.x / scale.x, n.y / scale.y, n.z / scale.z);
                let len = sn.norm();
                if len > 1e-30 { sn / len } else { Vector3::zeros() }
            }).collect()
        };

        Self {
            vertices,
            indices: self.indices.clone(),
            face_normals,
            vertex_normals,
            texcoords: self.texcoords.clone(),
            materials: self.materials.clone(),
            submeshes: self.submeshes.clone(),
        }
    }

    // ── Conversion ──────────────────────────────────────────────────────

    /// Convert to a parry3d `TriMesh` for collision detection.
    ///
    /// Returns `Err` if the mesh is degenerate (e.g. zero triangles).
    pub fn to_trimesh(&self) -> Result<parry3d::shape::TriMesh, String> {
        parry3d::shape::TriMesh::new(self.vertices.clone(), self.indices.clone())
            .map_err(|e| format!("TriMesh build error: {e:?}"))
    }

    /// Convert to a parry3d `SharedShape` wrapping a `TriMesh`.
    ///
    /// Returns `Err` if the mesh is degenerate.
    pub fn to_shared_shape(&self) -> Result<parry3d::shape::SharedShape, String> {
        parry3d::shape::SharedShape::trimesh(self.vertices.clone(), self.indices.clone())
            .map_err(|e| format!("TriMesh build error: {e:?}"))
    }

    /// Flatten vertex + per-face-normal data into an interleaved buffer
    /// `[x, y, z, nx, ny, nz, …]` suitable for OpenGL rendering.
    ///
    /// Each triangle produces 3 vertices × 6 floats = 18 floats. The returned
    /// buffer has `num_triangles() * 18` elements.
    pub fn to_flat_vertices_f32(&self) -> Vec<f32> {
        let mut buf = Vec::with_capacity(self.indices.len() * 18);
        for (tri, n) in self.indices.iter().zip(self.face_normals.iter()) {
            let nx = n.x as f32;
            let ny = n.y as f32;
            let nz = n.z as f32;
            for &vi in tri {
                let v = &self.vertices[vi as usize];
                buf.push(v.x as f32);
                buf.push(v.y as f32);
                buf.push(v.z as f32);
                buf.push(nx);
                buf.push(ny);
                buf.push(nz);
            }
        }
        buf
    }

    /// Flatten vertex + per-face-normal data into `f64`.
    pub fn to_flat_vertices_f64(&self) -> Vec<f64> {
        let mut buf = Vec::with_capacity(self.indices.len() * 18);
        for (tri, n) in self.indices.iter().zip(self.face_normals.iter()) {
            for &vi in tri {
                let v = &self.vertices[vi as usize];
                buf.push(v.x);
                buf.push(v.y);
                buf.push(v.z);
                buf.push(n.x);
                buf.push(n.y);
                buf.push(n.z);
            }
        }
        buf
    }
}

// ─── Convenience loader ─────────────────────────────────────────────────────

/// Load an STL file and return a parry3d `TriMesh` directly.
///
/// This is a shorthand for `MeshData::from_stl(path)?.scaled(scale).to_trimesh()`.
pub fn load_stl_as_trimesh(path: &Path, scale: &Vector3<f64>) -> Result<parry3d::shape::TriMesh, String> {
    let mesh = MeshData::from_stl(path)?;
    mesh.scaled(scale).to_trimesh()
}

// ─── GeometryModel mesh loading ─────────────────────────────────────────────

use crate::geometry::{GeometryModel, GeometryShape};

/// Resolve a URDF/SDF mesh URI to an absolute filesystem path.
///
/// Supported URI schemes:
/// - `package://PKG_NAME/rest/of/path` → `<package_dir>/rest/of/path`
/// - `model://…` → same as `package://`
/// - `file:///absolute/…` → `/absolute/…`
/// - bare relative path → resolved relative to `base_dir`
///
/// `package_dir` is typically the **parent** of the directory that contains the
/// URDF/SDF file (i.e. the ROS package root).
pub fn resolve_mesh_uri(
    uri: &str,
    base_dir: &Path,
    package_dir: Option<&Path>,
) -> std::path::PathBuf {
    if let Some(rest) = uri.strip_prefix("package://") {
        // Strip the package name segment.
        let after_pkg = rest.find('/').map(|i| &rest[i + 1..]).unwrap_or(rest);
        package_dir.unwrap_or(base_dir).join(after_pkg)
    } else if let Some(rest) = uri.strip_prefix("model://") {
        let after_pkg = rest.find('/').map(|i| &rest[i + 1..]).unwrap_or(rest);
        package_dir.unwrap_or(base_dir).join(after_pkg)
    } else if let Some(rest) = uri.strip_prefix("file://") {
        std::path::PathBuf::from(rest)
    } else {
        base_dir.join(uri)
    }
}

/// Attempt to load mesh data for every `GeometryShape::Mesh` object in `gmodel`.
///
/// For each object whose `shape` is `Mesh` and whose `mesh_data` is `None`,
/// this function resolves the URI, reads the file (currently STL only), and
/// stores the resulting [`MeshData`] in `mesh_data`.
///
/// Objects that already have `mesh_data` populated are skipped.  Files that
/// cannot be read (missing, unsupported format, …) are silently skipped — the
/// object's `mesh_data` remains `None`.
///
/// # Arguments
///
/// * `gmodel` — geometry model to mutate.
/// * `base_dir` — directory of the URDF/SDF file (for relative paths).
/// * `package_dir` — optional ROS package root (for `package://` URIs).
pub fn load_meshes_for_geometry_model(
    gmodel: &mut GeometryModel,
    base_dir: &Path,
    package_dir: Option<&Path>,
) {
    for obj in &mut gmodel.objects {
        if obj.mesh_data.is_some() {
            continue;
        }
        if let GeometryShape::Mesh { ref filename, .. } = obj.shape {
            let resolved = resolve_mesh_uri(filename, base_dir, package_dir);
            if let Some(ext) = resolved.extension().and_then(|e| e.to_str()) {
                match ext.to_ascii_lowercase().as_str() {
                    "stl" => {
                        if let Ok(md) = MeshData::from_stl(&resolved) {
                            obj.mesh_data = Some(md);
                        }
                    }
                    "dae" => {
                        if let Ok(md) = crate::collada::load_dae(&resolved) {
                            obj.mesh_data = Some(md);
                        }
                    }
                    _ => {
                        // Unsupported format — skip silently.
                    }
                }
            }
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Helper: create a single-triangle STL face.
    fn make_triangle(
        v0: [f32; 3],
        v1: [f32; 3],
        v2: [f32; 3],
    ) -> stl_io::Triangle {
        stl_io::Triangle {
            normal: stl_io::Normal::new([0.0, 0.0, 1.0]),
            vertices: [
                stl_io::Vertex::new(v0),
                stl_io::Vertex::new(v1),
                stl_io::Vertex::new(v2),
            ],
        }
    }

    #[test]
    fn empty_mesh() {
        let mesh = MeshData::from_stl_triangles(&[]).unwrap();
        assert_eq!(mesh.num_vertices(), 0);
        assert_eq!(mesh.num_triangles(), 0);
        assert!(mesh.aabb().is_none());
    }

    #[test]
    fn single_triangle() {
        let faces = vec![make_triangle(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        )];
        let mesh = MeshData::from_stl_triangles(&faces).unwrap();

        assert_eq!(mesh.num_vertices(), 3);
        assert_eq!(mesh.num_triangles(), 1);
        assert_eq!(mesh.indices[0], [0, 1, 2]);

        // Normal should point in +Z.
        assert_relative_eq!(mesh.face_normals[0].z, 1.0, epsilon = 1e-10);
        assert_relative_eq!(mesh.face_normals[0].x, 0.0, epsilon = 1e-10);

        // AABB
        let (min, max) = mesh.aabb().unwrap();
        assert_relative_eq!(min.x, 0.0);
        assert_relative_eq!(max.x, 1.0);
        assert_relative_eq!(max.y, 1.0);
    }

    #[test]
    fn vertex_deduplication() {
        // Two triangles sharing an edge (v0–v1).
        let faces = vec![
            make_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            make_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        ];
        let mesh = MeshData::from_stl_triangles(&faces).unwrap();

        // 4 unique vertices (not 6).
        assert_eq!(mesh.num_vertices(), 4);
        assert_eq!(mesh.num_triangles(), 2);
    }

    #[test]
    fn scaling() {
        let faces = vec![make_triangle(
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        )];
        let mesh = MeshData::from_stl_triangles(&faces).unwrap();
        let scaled = mesh.scaled(&Vector3::new(0.001, 0.001, 0.001));

        assert_eq!(scaled.num_vertices(), 3);
        assert_relative_eq!(scaled.vertices[0].x, 0.001, epsilon = 1e-12);
        assert_relative_eq!(scaled.vertices[0].y, 0.002, epsilon = 1e-12);
        assert_relative_eq!(scaled.vertices[0].z, 0.003, epsilon = 1e-12);
    }

    #[test]
    fn flat_vertices_f32_layout() {
        let faces = vec![make_triangle(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        )];
        let mesh = MeshData::from_stl_triangles(&faces).unwrap();
        let flat = mesh.to_flat_vertices_f32();

        // 1 triangle × 3 verts × 6 floats = 18
        assert_eq!(flat.len(), 18);
        // First vertex position
        assert_relative_eq!(flat[0], 1.0_f32);
        assert_relative_eq!(flat[1], 0.0_f32);
        assert_relative_eq!(flat[2], 0.0_f32);
        // First vertex normal (should be -Z for this winding).
        // edge1 = (-1,1,0), edge2 = (-1,0,0), cross = (0,0,1)
        // Actually: v0=(1,0,0), v1=(0,1,0), v2=(0,0,0)
        // edge1 = v1-v0 = (-1,1,0), edge2 = v2-v0 = (-1,0,0)
        // cross = (1*0 - 0*0, 0*(-1) - (-1)*0, (-1)*0 - 1*(-1)) = (0, 0, 1)
        assert_relative_eq!(flat[5], 1.0_f32, epsilon = 1e-6);
    }

    #[test]
    fn to_trimesh_conversion() {
        let faces = vec![
            make_triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            make_triangle([1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
        ];
        let mesh = MeshData::from_stl_triangles(&faces).unwrap();
        let nv = mesh.num_vertices();
        let nt = mesh.num_triangles();
        let trimesh = mesh.to_trimesh().unwrap();

        assert_eq!(trimesh.vertices().len(), nv);
        assert_eq!(trimesh.indices().len(), nt);
    }

    #[test]
    fn to_shared_shape_conversion() {
        let faces = vec![make_triangle(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        )];
        let mesh = MeshData::from_stl_triangles(&faces).unwrap();
        let shape = mesh.to_shared_shape().unwrap();

        // SharedShape wrapping a TriMesh should be usable for parry3d queries.
        assert!(shape.as_trimesh().is_some());
    }

    #[test]
    fn load_stl_as_trimesh_convenience() {
        // We cannot test actual file loading without a test fixture, but we
        // can verify the function signature compiles and round-trips through
        // from_stl_triangles + scaled + to_trimesh.
        let faces = vec![make_triangle(
            [0.0, 0.0, 0.0],
            [1000.0, 0.0, 0.0],
            [0.0, 1000.0, 0.0],
        )];
        let mesh = MeshData::from_stl_triangles(&faces).unwrap();
        let scaled = mesh.scaled(&Vector3::new(0.001, 0.001, 0.001));
        let trimesh = scaled.to_trimesh().unwrap();

        assert_relative_eq!(trimesh.vertices()[1].x, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn non_uniform_scaling_recomputes_normals() {
        // Triangle in XY plane, normal = +Z.
        let faces = vec![make_triangle(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        )];
        let mesh = MeshData::from_stl_triangles(&faces).unwrap();
        assert_relative_eq!(mesh.face_normals[0].z, 1.0, epsilon = 1e-10);

        // Non-uniform scale that stretches Z — normal should still point +Z.
        let scaled = mesh.scaled(&Vector3::new(1.0, 1.0, 10.0));
        assert_relative_eq!(scaled.face_normals[0].z, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn load_stl_file() {
        let stl_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/meshes/test_box.stl");
        let mesh = MeshData::from_stl(&stl_path).unwrap();
        assert!(mesh.num_triangles() > 0, "STL should have at least 1 triangle");
        assert!(mesh.num_vertices() > 0, "STL should have at least 1 vertex");
        // Should be convertible to TriMesh.
        let trimesh = mesh.to_trimesh().unwrap();
        assert_eq!(trimesh.indices().len(), mesh.num_triangles());
    }

    #[test]
    fn resolve_mesh_uri_package() {
        let base = Path::new("/robot/urdf");
        let pkg = Path::new("/robot");
        let resolved = super::resolve_mesh_uri(
            "package://my_robot/meshes/arm.stl",
            base,
            Some(pkg),
        );
        assert_eq!(resolved, std::path::PathBuf::from("/robot/meshes/arm.stl"));
    }

    #[test]
    fn resolve_mesh_uri_relative() {
        let base = Path::new("/robot/urdf");
        let resolved = super::resolve_mesh_uri("../meshes/arm.stl", base, None);
        assert_eq!(
            resolved,
            std::path::PathBuf::from("/robot/urdf/../meshes/arm.stl")
        );
    }
}
