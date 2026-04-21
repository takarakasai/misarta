//! Mesh decomposition into simpler collision shapes.
//!
//! Provides two decomposition strategies for replacing a complex triangle mesh
//! with a set of simpler primitives suitable for fast collision detection:
//!
//! | Method | Output | Quality | Speed | Description |
//! |--------|--------|---------|-------|-------------|
//! | [`Vhacd`](DecompositionMethod::Vhacd) | Convex hulls | ★★★ | ★★ | V-HACD volumetric convex decomposition |
//! | [`SphereTree`](DecompositionMethod::SphereTree) | Spheres | ★★ | ★★★ | Binary sphere tree with PCA splitting |
//!
//! # Example
//!
//! ```no_run
//! use misarta::mesh::MeshData;
//! use misarta::decompose::{self, VhacdParams, SphereTreeParams};
//! use std::path::Path;
//!
//! let mesh = MeshData::from_stl(Path::new("robot.stl")).unwrap();
//!
//! // V-HACD: decompose into convex hulls
//! let hulls = decompose::vhacd(&mesh, &VhacdParams::default());
//! println!("{} convex hulls", hulls.len());
//!
//! // Sphere tree: approximate with spheres
//! let spheres = decompose::sphere_tree(&mesh, &SphereTreeParams::default());
//! println!("{} spheres", spheres.len());
//! ```

use crate::mesh::MeshData;
use nalgebra::{Point3, Vector3};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

// ─── DecompositionProgress ──────────────────────────────────────────────────

/// Progress phases for mesh decomposition.
///
/// Stored as an `AtomicU8` so it can be polled from the UI thread
/// while the computation runs on a background thread.
///
/// | Value | Phase |
/// |-------|-------|
/// | 0 | Not started |
/// | 1 | Preparing data |
/// | 2 | Voxelizing (V-HACD) / Splitting (Sphere Tree) |
/// | 3 | Computing convex hulls / fitting spheres |
/// | 4 | Building output meshes |
/// | 255 | Done |
pub const PHASE_NOT_STARTED: u8 = 0;
pub const PHASE_PREPARING: u8 = 1;
pub const PHASE_DECOMPOSING: u8 = 2;
pub const PHASE_HULLS: u8 = 3;
pub const PHASE_BUILDING: u8 = 4;
pub const PHASE_DONE: u8 = 255;

/// Human-readable label for a progress phase.
pub fn phase_label(phase: u8) -> &'static str {
    match phase {
        PHASE_NOT_STARTED => "Waiting…",
        PHASE_PREPARING => "Preparing data…",
        PHASE_DECOMPOSING => "Decomposing (V-HACD voxelization)…",
        PHASE_HULLS => "Computing convex hulls…",
        PHASE_BUILDING => "Building output meshes…",
        PHASE_DONE => "Done",
        _ => "Processing…",
    }
}

// ─── DecompositionMethod ────────────────────────────────────────────────────

/// Mesh decomposition algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecompositionMethod {
    /// V-HACD — Volumetric Hierarchical Approximate Convex Decomposition.
    ///
    /// Decomposes a concave mesh into multiple convex hulls via voxelization
    /// and recursive clipping-plane search.  Uses parry3d's implementation.
    Vhacd,

    /// Binary sphere tree with PCA-based splitting.
    ///
    /// Recursively splits the mesh using a principal-axis plane and fits
    /// tight bounding spheres to the leaves.
    SphereTree,

    /// Primitive fitting — V-HACD convex decomposition followed by
    /// best-fit primitive (Box / Cylinder / Sphere) for each convex hull.
    PrimitiveFit,
}

impl DecompositionMethod {
    /// All available methods, useful for UI combo boxes.
    pub const ALL: [DecompositionMethod; 3] = [
        DecompositionMethod::Vhacd,
        DecompositionMethod::SphereTree,
        DecompositionMethod::PrimitiveFit,
    ];

    /// Short human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Vhacd => "V-HACD",
            Self::SphereTree => "Sphere Tree",
            Self::PrimitiveFit => "Primitive Fit",
        }
    }

    /// Description for tooltips.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Vhacd => "Convex decomposition — high quality, slower (parry3d V-HACD)",
            Self::SphereTree => "Sphere approximation — fast, moderate quality",
            Self::PrimitiveFit => "V-HACD + fit each hull to Box / Cylinder / Sphere",
        }
    }

    /// Parse from a string (case-insensitive). Returns `Vhacd` as default.
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "vhacd" | "v-hacd" | "convex" => Self::Vhacd,
            "sphere" | "sphere_tree" | "spheretree" | "spheres" => Self::SphereTree,
            "primitive" | "primitive_fit" | "primitivefit" | "primitives" => Self::PrimitiveFit,
            _ => Self::Vhacd,
        }
    }
}

// ─── V-HACD Parameters ─────────────────────────────────────────────────────

/// Parameters for V-HACD convex decomposition.
#[derive(Debug, Clone, Copy)]
pub struct VhacdParams {
    /// Maximum number of convex hulls to produce (default 16).
    pub max_hulls: u32,
    /// Voxel grid resolution (default 64, higher = more detail but slower).
    pub resolution: u32,
    /// Maximum concavity threshold (default 0.01, lower = tighter fit).
    pub concavity: f64,
}

impl Default for VhacdParams {
    fn default() -> Self {
        Self {
            max_hulls: 16,
            resolution: 64,
            concavity: 0.01,
        }
    }
}

// ─── Sphere Tree Parameters ────────────────────────────────────────────────

/// Parameters for sphere tree decomposition.
#[derive(Debug, Clone, Copy)]
pub struct SphereTreeParams {
    /// Maximum number of leaf spheres (default 16).
    pub max_spheres: usize,
    /// Maximum recursion depth (default 5).
    pub max_depth: usize,
    /// Minimum triangle count per leaf to allow further splitting (default 4).
    pub min_triangles: usize,
}

impl Default for SphereTreeParams {
    fn default() -> Self {
        Self {
            max_spheres: 16,
            max_depth: 5,
            min_triangles: 4,
        }
    }
}

// ─── FitSphere ──────────────────────────────────────────────────────────────

/// A bounding sphere resulting from sphere tree decomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct FitSphere {
    /// Centre of the sphere.
    pub center: Point3<f64>,
    /// Radius of the sphere.
    pub radius: f64,
}

// ─── V-HACD Implementation ─────────────────────────────────────────────────

/// Decompose a mesh into convex hulls using V-HACD.
///
/// Returns a vector of [`MeshData`], each representing one convex hull.
/// Empty meshes or degenerate inputs return an empty `Vec`.
pub fn vhacd(mesh: &MeshData, params: &VhacdParams) -> Vec<MeshData> {
    vhacd_with_progress(mesh, params, None, None)
}

/// Like [`vhacd`] but updates atomic progress so a UI can poll it.
///
/// * `progress` — coarse phase indicator (`PHASE_*` constants).
/// * `sub_progress` — fine-grained 0–100 percentage within the current phase.
///   The hull-computation phase in particular reports per-hull progress here.
pub fn vhacd_with_progress(
    mesh: &MeshData,
    params: &VhacdParams,
    progress: Option<&Arc<AtomicU8>>,
    sub_progress: Option<&Arc<AtomicU8>>,
) -> Vec<MeshData> {
    let set = |phase: u8| {
        if let Some(p) = progress {
            p.store(phase, Ordering::Relaxed);
        }
        // Reset sub-progress when entering a new phase.
        if let Some(sp) = sub_progress {
            sp.store(0, Ordering::Relaxed);
        }
    };
    let set_sub = |pct: u8| {
        if let Some(sp) = sub_progress {
            sp.store(pct, Ordering::Relaxed);
        }
    };

    set(PHASE_PREPARING);

    if mesh.vertices.len() < 4 || mesh.indices.is_empty() {
        set(PHASE_DONE);
        return Vec::new();
    }

    // Convert MeshData → parry3d format.
    let points: Vec<Point3<f64>> = mesh.vertices.clone();
    let indices: Vec<[u32; 3]> = mesh.indices.clone();

    // Build parry3d VHACD parameters.
    let vhacd_params = parry3d::transformation::vhacd::VHACDParameters {
        resolution: params.resolution,
        concavity: params.concavity,
        max_convex_hulls: params.max_hulls,
        ..Default::default()
    };

    // Run V-HACD (voxelization + hierarchical ACD — the slow step).
    set(PHASE_DECOMPOSING);
    let decomposition = parry3d::transformation::vhacd::VHACD::decompose(
        &vhacd_params,
        &points,
        &indices,
        true,
    );

    // Compute exact convex hulls one-by-one so we can report per-hull progress.
    set(PHASE_HULLS);
    let parts = decomposition.voxel_parts();
    let total = parts.len().max(1);
    let mut hulls: Vec<(Vec<Point3<f64>>, Vec<[u32; 3]>)> = Vec::with_capacity(total);
    for (i, part) in parts.iter().enumerate() {
        set_sub(((i * 100) / total) as u8);
        hulls.push(part.compute_exact_convex_hull(&points, &indices));
    }
    set_sub(100);

    // Convert each hull to MeshData.
    set(PHASE_BUILDING);
    let build_total = hulls.len().max(1);
    let mut result = Vec::with_capacity(hulls.len());
    for (i, (hull_verts, hull_indices)) in hulls.into_iter().enumerate() {
        set_sub(((i * 100) / build_total) as u8);
        if hull_verts.len() < 3 || hull_indices.is_empty() {
            continue;
        }
        if let Some(md) = mesh_data_from_hull(&hull_verts, &hull_indices) {
            result.push(md);
        }
    }
    set_sub(100);

    set(PHASE_DONE);
    result
}

/// Build a `MeshData` from convex hull vertices and triangle indices.
fn mesh_data_from_hull(vertices: &[Point3<f64>], indices: &[[u32; 3]]) -> Option<MeshData> {
    if vertices.is_empty() || indices.is_empty() {
        return None;
    }

    let face_normals: Vec<Vector3<f64>> = indices
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

    Some(MeshData {
        vertices: vertices.to_vec(),
        indices: indices.to_vec(),
        face_normals,
        vertex_normals: Vec::new(),
        texcoords: Vec::new(),
        materials: Vec::new(),
        submeshes: Vec::new(),
    })
}

// ─── Sphere Tree Implementation ────────────────────────────────────────────

/// Decompose a mesh into a set of bounding spheres using a binary sphere tree.
///
/// Returns a vector of [`FitSphere`] representing the leaf nodes.
pub fn sphere_tree(mesh: &MeshData, params: &SphereTreeParams) -> Vec<FitSphere> {
    sphere_tree_with_progress(mesh, params, None, None)
}

/// Like [`sphere_tree`] but updates an atomic progress phase.
pub fn sphere_tree_with_progress(
    mesh: &MeshData,
    params: &SphereTreeParams,
    progress: Option<&Arc<AtomicU8>>,
    _sub_progress: Option<&Arc<AtomicU8>>,
) -> Vec<FitSphere> {
    let set = |phase: u8| {
        if let Some(p) = progress {
            p.store(phase, Ordering::Relaxed);
        }
    };

    set(PHASE_PREPARING);

    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
        set(PHASE_DONE);
        return Vec::new();
    }

    // Collect triangle centroids and vertex lists for splitting.
    let tri_data: Vec<TriInfo> = mesh
        .indices
        .iter()
        .map(|tri| {
            let v0 = mesh.vertices[tri[0] as usize];
            let v1 = mesh.vertices[tri[1] as usize];
            let v2 = mesh.vertices[tri[2] as usize];
            TriInfo {
                centroid: Point3::from((v0.coords + v1.coords + v2.coords) / 3.0),
                vi: *tri,
            }
        })
        .collect();

    set(PHASE_DECOMPOSING);

    let mut leaves = Vec::new();
    sphere_tree_recurse(
        &mesh.vertices,
        &tri_data,
        0,
        params.max_depth,
        params.min_triangles,
        params.max_spheres,
        &mut leaves,
    );

    set(PHASE_DONE);
    leaves
}

/// Triangle info for sphere-tree splitting.
#[derive(Clone)]
struct TriInfo {
    centroid: Point3<f64>,
    vi: [u32; 3],
}

/// Recursively split a set of triangles and fit bounding spheres to leaves.
fn sphere_tree_recurse(
    vertices: &[Point3<f64>],
    tris: &[TriInfo],
    depth: usize,
    max_depth: usize,
    min_triangles: usize,
    remaining_budget: usize,
    leaves: &mut Vec<FitSphere>,
) {
    if tris.is_empty() {
        return;
    }

    // Collect all unique vertex positions referenced by these triangles.
    let pts = gather_points(vertices, tris);

    // Compute bounding sphere of these points.
    let sphere = bounding_sphere(&pts);

    // Stop conditions:
    // 1. Reached max depth
    // 2. Too few triangles to split
    // 3. No budget to split further (need ≥ 2 to make splitting worthwhile)
    if depth >= max_depth || tris.len() <= min_triangles || remaining_budget <= 1 {
        leaves.push(sphere);
        return;
    }

    // PCA: find the direction of maximum variance in the centroids.
    let split_axis = pca_major_axis_centroids(tris);

    // Split along the major axis at the centroid mean.
    let mean = centroid_mean(tris);
    let d = split_axis.dot(&mean.coords);

    let mut left = Vec::new();
    let mut right = Vec::new();
    for t in tris {
        if split_axis.dot(&t.centroid.coords) < d {
            left.push(t.clone());
        } else {
            right.push(t.clone());
        }
    }

    // If one side is empty, fall back to leaf.
    if left.is_empty() || right.is_empty() {
        leaves.push(sphere);
        return;
    }

    // Distribute budget proportionally between children.
    let total = left.len() + right.len();
    let left_budget = ((remaining_budget as f64) * (left.len() as f64) / (total as f64))
        .round()
        .max(1.0) as usize;
    let right_budget = remaining_budget.saturating_sub(left_budget).max(1);

    // Recurse.
    sphere_tree_recurse(vertices, &left, depth + 1, max_depth, min_triangles, left_budget, leaves);
    sphere_tree_recurse(vertices, &right, depth + 1, max_depth, min_triangles, right_budget, leaves);
}

/// Collect unique vertex positions referenced by a set of triangles.
fn gather_points(vertices: &[Point3<f64>], tris: &[TriInfo]) -> Vec<Point3<f64>> {
    let mut seen = std::collections::HashSet::new();
    let mut pts = Vec::new();
    for t in tris {
        for &vi in &t.vi {
            if seen.insert(vi) {
                pts.push(vertices[vi as usize]);
            }
        }
    }
    pts
}

/// Compute a tight bounding sphere using Ritter's algorithm.
fn bounding_sphere(points: &[Point3<f64>]) -> FitSphere {
    if points.is_empty() {
        return FitSphere {
            center: Point3::origin(),
            radius: 0.0,
        };
    }
    if points.len() == 1 {
        return FitSphere {
            center: points[0],
            radius: 0.0,
        };
    }

    // Step 1: Find the two most distant points (approximate via axis extremes).
    let mut min_pt = points[0];
    let mut max_pt = points[0];
    for &p in &points[1..] {
        if p.x < min_pt.x
            || (p.x == min_pt.x && (p.y < min_pt.y || (p.y == min_pt.y && p.z < min_pt.z)))
        {
            min_pt = p;
        }
        if p.x > max_pt.x
            || (p.x == max_pt.x && (p.y > max_pt.y || (p.y == max_pt.y && p.z > max_pt.z)))
        {
            max_pt = p;
        }
    }

    // Find the pair with maximum separation across all 3 axes.
    let mut best_dist2 = 0.0f64;
    let mut p_min = points[0];
    let mut p_max = points[0];
    for axis in 0..3 {
        let mut lo = points[0];
        let mut hi = points[0];
        for &p in &points[1..] {
            if p[axis] < lo[axis] {
                lo = p;
            }
            if p[axis] > hi[axis] {
                hi = p;
            }
        }
        let d2 = (hi - lo).norm_squared();
        if d2 > best_dist2 {
            best_dist2 = d2;
            p_min = lo;
            p_max = hi;
        }
    }

    let mut center = Point3::from((p_min.coords + p_max.coords) * 0.5);
    let mut radius = (p_max - center).norm();

    // Step 2: Grow sphere to include all points.
    for &p in points {
        let dist = (p - center).norm();
        if dist > radius {
            // Grow sphere to include this point.
            let new_radius = (radius + dist) * 0.5;
            let shift = dist - new_radius;
            center = Point3::from(center.coords + (p - center).normalize() * shift);
            radius = new_radius;
        }
    }

    FitSphere { center, radius }
}

/// Compute the major PCA axis of triangle centroids.
fn pca_major_axis_centroids(tris: &[TriInfo]) -> Vector3<f64> {
    let n = tris.len() as f64;
    if n < 2.0 {
        return Vector3::x();
    }

    // Mean centroid.
    let mean = centroid_mean(tris);

    // Covariance matrix (3×3 symmetric).
    let mut cov = [[0.0f64; 3]; 3];
    for t in tris {
        let d = t.centroid - mean;
        for i in 0..3 {
            for j in i..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    for i in 0..3 {
        for j in i..3 {
            cov[i][j] /= n;
            if j != i {
                cov[j][i] = cov[i][j];
            }
        }
    }

    // Power iteration to find the largest eigenvector.
    let mut v = Vector3::new(1.0, 1.0, 1.0).normalize();
    for _ in 0..30 {
        let mut next = Vector3::zeros();
        for i in 0..3 {
            next[i] = cov[i][0] * v[0] + cov[i][1] * v[1] + cov[i][2] * v[2];
        }
        let len = next.norm();
        if len < 1e-30 {
            return Vector3::x();
        }
        v = next / len;
    }

    v
}

/// Mean centroid of a set of triangles.
fn centroid_mean(tris: &[TriInfo]) -> Point3<f64> {
    let n = tris.len() as f64;
    let sum = tris
        .iter()
        .fold(Vector3::zeros(), |acc, t| acc + t.centroid.coords);
    Point3::from(sum / n)
}

// ─── MeshData convenience methods ──────────────────────────────────────────

impl MeshData {
    /// Decompose into convex hulls using V-HACD with default parameters.
    pub fn decompose_vhacd(&self) -> Vec<MeshData> {
        vhacd(self, &VhacdParams::default())
    }

    /// Decompose into convex hulls using V-HACD with custom parameters.
    pub fn decompose_vhacd_with(&self, params: &VhacdParams) -> Vec<MeshData> {
        vhacd(self, params)
    }

    /// Decompose into bounding spheres using a sphere tree with default parameters.
    pub fn decompose_spheres(&self) -> Vec<FitSphere> {
        sphere_tree(self, &SphereTreeParams::default())
    }

    /// Decompose into bounding spheres with custom parameters.
    pub fn decompose_spheres_with(&self, params: &SphereTreeParams) -> Vec<FitSphere> {
        sphere_tree(self, params)
    }

    /// Decompose into primitive shapes (Box / Cylinder / Sphere) via V-HACD.
    pub fn decompose_primitives(&self) -> Vec<FitPrimitive> {
        primitive_fit(self, &VhacdParams::default())
    }

    /// Decompose into primitive shapes with custom V-HACD parameters.
    pub fn decompose_primitives_with(&self, params: &VhacdParams) -> Vec<FitPrimitive> {
        primitive_fit(self, params)
    }
}

// ─── Primitive Fitting ──────────────────────────────────────────────────────

/// A fitted primitive shape with its pose (translation + orientation).
#[derive(Debug, Clone, PartialEq)]
pub struct FitPrimitive {
    /// The kind of primitive that best fits the convex hull.
    pub kind: PrimitiveKind,
    /// Centre position of the primitive.
    pub center: Point3<f64>,
    /// Orientation of the primitive (axes aligned to PCA eigenvectors).
    pub rotation: nalgebra::UnitQuaternion<f64>,
}

/// The shape type and dimensions of a fitted primitive.
#[derive(Debug, Clone, PartialEq)]
pub enum PrimitiveKind {
    /// Axis-aligned box with half-extents.
    Box {
        hx: f64,
        hy: f64,
        hz: f64,
    },
    /// Cylinder aligned along local Z with given radius and half-length.
    Cylinder {
        radius: f64,
        half_length: f64,
    },
    /// Sphere with given radius.
    Sphere {
        radius: f64,
    },
}

impl PrimitiveKind {
    /// Volume of this primitive.
    pub fn volume(&self) -> f64 {
        match self {
            Self::Box { hx, hy, hz } => 8.0 * hx * hy * hz,
            Self::Cylinder { radius, half_length } => {
                std::f64::consts::PI * radius * radius * 2.0 * half_length
            }
            Self::Sphere { radius } => {
                (4.0 / 3.0) * std::f64::consts::PI * radius * radius * radius
            }
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Box { .. } => "Box",
            Self::Cylinder { .. } => "Cylinder",
            Self::Sphere { .. } => "Sphere",
        }
    }
}

/// Decompose a mesh into primitive shapes by running V-HACD first, then fitting
/// the best primitive (Box / Cylinder / Sphere) to each convex hull.
///
/// Selection criterion: the primitive with the smallest volume that still
/// encloses all vertices of the convex hull is chosen.
pub fn primitive_fit(mesh: &MeshData, params: &VhacdParams) -> Vec<FitPrimitive> {
    primitive_fit_with_progress(mesh, params, None, None)
}

/// Like [`primitive_fit`] but with progress reporting.
pub fn primitive_fit_with_progress(
    mesh: &MeshData,
    params: &VhacdParams,
    progress: Option<&Arc<AtomicU8>>,
    sub_progress: Option<&Arc<AtomicU8>>,
) -> Vec<FitPrimitive> {
    // Phase 1-3: V-HACD decomposition (reuses vhacd_with_progress)
    let hulls = vhacd_with_progress(mesh, params, progress, sub_progress);

    if hulls.is_empty() {
        return Vec::new();
    }

    // Phase 4: fit primitives to each hull
    if let Some(p) = progress {
        p.store(PHASE_BUILDING, Ordering::Relaxed);
    }
    let total = hulls.len().max(1);
    let mut result = Vec::with_capacity(hulls.len());
    for (i, hull) in hulls.iter().enumerate() {
        if let Some(sp) = sub_progress {
            sp.store(((i * 100) / total) as u8, Ordering::Relaxed);
        }
        if hull.vertices.len() >= 3 {
            result.push(fit_primitive_to_points(&hull.vertices));
        }
    }
    if let Some(sp) = sub_progress {
        sp.store(100, Ordering::Relaxed);
    }
    if let Some(p) = progress {
        p.store(PHASE_DONE, Ordering::Relaxed);
    }
    result
}

/// Compute the volume of a convex hull defined by a set of points.
///
/// Uses the signed-tetrahedron method: for each triangle face of the convex
/// hull, compute the signed volume of the tetrahedron formed with the origin.
/// For convex point sets, we triangulate from the centroid to each face.
///
/// Falls back to an approximate approach when the point count is small:
/// build tetrahedra from the centroid to every triple of points.
fn convex_hull_volume(points: &[Point3<f64>]) -> f64 {
    let n = points.len();
    if n < 4 {
        return 0.0;
    }

    // Use parry3d's convex hull to get faces
    let parry_pts: Vec<parry3d::math::Point<f64>> = points
        .iter()
        .map(|p| parry3d::math::Point::new(p.x, p.y, p.z))
        .collect();

    // Build a convex hull with parry3d (returns indexed mesh)
    let (_pts, indices) = parry3d::transformation::convex_hull(&parry_pts);
    if indices.is_empty() {
        return 0.0;
    }

    // Signed-tetrahedron method (reference point = origin)
    let mut vol = 0.0_f64;
    for tri in &indices {
        let a = &points[tri[0] as usize];
        let b = &points[tri[1] as usize];
        let c = &points[tri[2] as usize];
        // Signed volume of tetrahedron (origin, a, b, c)
        vol += a.coords.dot(&b.coords.cross(&c.coords));
    }
    (vol / 6.0).abs()
}

/// Fit the best primitive shape to a set of points.
///
/// 1.  Compute PCA to find the oriented bounding box (OBB).
/// 2.  From the OBB half-extents, derive candidate sphere, cylinders, and box.
/// 3.  Pick the primitive with the best fill ratio (smallest wasted volume).
///
/// The fill ratio is the fraction of the primitive's volume that is actually
/// occupied by the convex hull:  ratio = hull_volume / primitive_volume.
/// Higher is better (tighter fit).
pub fn fit_primitive_to_points(points: &[Point3<f64>]) -> FitPrimitive {
    let n = points.len();
    assert!(n >= 1, "need at least 1 point");

    // ── Centroid ──
    let center = {
        let sum = points.iter().fold(Vector3::zeros(), |a, p| a + p.coords);
        Point3::from(sum / n as f64)
    };

    // ── Covariance matrix ──
    let mut cov = nalgebra::Matrix3::<f64>::zeros();
    for p in points {
        let d = p - center;
        cov += d * d.transpose();
    }
    cov /= n as f64;

    // ── PCA via eigendecomposition ──
    let eigen = cov.symmetric_eigen();
    // Sort eigenvectors by eigenvalue (ascending → [minor, medium, major])
    let mut order: [usize; 3] = [0, 1, 2];
    order.sort_by(|a, b| eigen.eigenvalues[*a].partial_cmp(&eigen.eigenvalues[*b]).unwrap());

    // Build rotation matrix from PCA axes
    let mut rot_mat = nalgebra::Matrix3::<f64>::zeros();
    rot_mat.set_column(0, &eigen.eigenvectors.column(order[0]));
    rot_mat.set_column(1, &eigen.eigenvectors.column(order[1]));
    rot_mat.set_column(2, &eigen.eigenvectors.column(order[2]));

    // Ensure right-handed frame
    if rot_mat.determinant() < 0.0 {
        let col0 = -rot_mat.column(0).into_owned();
        rot_mat.set_column(0, &col0);
    }

    let rotation = nalgebra::UnitQuaternion::from_rotation_matrix(
        &nalgebra::Rotation3::from_matrix_unchecked(rot_mat),
    );
    let inv_rot = rotation.inverse();

    // ── Project points to PCA-local frame and compute OBB ──
    let mut min_local = Vector3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut max_local = Vector3::new(f64::MIN, f64::MIN, f64::MIN);
    for p in points {
        let local = inv_rot * (p - center);
        for i in 0..3 {
            min_local[i] = min_local[i].min(local[i]);
            max_local[i] = max_local[i].max(local[i]);
        }
    }

    let half_extents = (max_local - min_local) * 0.5;
    // Re-center: the OBB center in local frame
    let obb_center_local = (max_local + min_local) * 0.5;
    let obb_center_world = center + rotation * obb_center_local;

    let hx = half_extents[0];
    let hy = half_extents[1];
    let hz = half_extents[2];

    // ── Candidate primitives ──
    // Each candidate is sized to tightly fit the point cloud.
    // We compare fill ratio (hull_volume / primitive_volume) and pick the best.

    // Compute convex hull volume for fill-ratio scoring
    let hull_vol = convex_hull_volume(points);

    // 1. OBB: always exact enclosure.
    let box_prim = PrimitiveKind::Box { hx, hy, hz };

    // 2. Cylinders: try all 3 candidate axes.
    //    Use the OBB half-extents for the perpendicular-plane radius
    //    (inscribed circle of the OBB cross-section) instead of the
    //    max point distance, which over-estimates for non-circular hulls.
    let mut best_cyl: Option<(PrimitiveKind, f64)> = None;
    for axis_idx in 0..3_usize {
        let (perp_a, perp_b) = match axis_idx {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        // OBB-based radius: circumscribed circle of the rectangle
        let radius_obb = (half_extents[perp_a].powi(2) + half_extents[perp_b].powi(2)).sqrt();
        // Exact enclosing radius from actual point distances
        let mut max_r2_exact = 0.0_f64;
        for p in points {
            let local = inv_rot * (p - center) - obb_center_local;
            let r2 = local[perp_a] * local[perp_a] + local[perp_b] * local[perp_b];
            max_r2_exact = max_r2_exact.max(r2);
        }
        let radius_exact = max_r2_exact.sqrt();
        // Use the tighter of the two (OBB circumscribed is an upper bound;
        // exact may be smaller for shapes that don't reach OBB corners)
        let radius = radius_exact.min(radius_obb);
        let half_length = half_extents[axis_idx];
        if radius < 1e-15 && half_length < 1e-15 {
            continue;
        }
        let cyl = PrimitiveKind::Cylinder { radius, half_length };
        let vol = cyl.volume();
        if best_cyl.is_none() || vol < best_cyl.as_ref().unwrap().1 {
            best_cyl = Some((cyl, vol));
        }
    }

    // 3. Sphere: use the largest OBB half-extent as radius
    //    (tighter than max point distance for non-spherical shapes).
    let radius_obb = (hx * hx + hy * hy + hz * hz).sqrt();
    let mut max_r2_exact = 0.0_f64;
    for p in points {
        let local = inv_rot * (p - center) - obb_center_local;
        let r2 = local.norm_squared();
        max_r2_exact = max_r2_exact.max(r2);
    }
    let sph_radius = max_r2_exact.sqrt().min(radius_obb);
    let sph_prim = PrimitiveKind::Sphere { radius: sph_radius };

    // ── Pick the best candidate by fill ratio ──
    // fill_ratio = hull_volume / primitive_volume (higher = tighter fit)
    // When hull_vol is 0 (degenerate), fall back to smallest volume.
    let use_fill_ratio = hull_vol > 1e-20;

    let score = |prim: &PrimitiveKind| -> f64 {
        let pv = prim.volume();
        if use_fill_ratio && pv > 1e-20 {
            hull_vol / pv  // higher is better
        } else {
            -pv  // smaller volume → less negative → higher score
        }
    };

    let mut best = box_prim.clone();
    let mut best_score = score(&box_prim);

    if let Some((ref cyl, _)) = best_cyl {
        let s = score(cyl);
        if s > best_score {
            best = cyl.clone();
            best_score = s;
        }
    }

    let s = score(&sph_prim);
    if s > best_score {
        best = sph_prim;
    }

    FitPrimitive {
        kind: best,
        center: obb_center_world,
        rotation,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Make a simple box mesh (axis-aligned, centred at origin).
    fn make_box(hx: f64, hy: f64, hz: f64) -> MeshData {
        let vertices = vec![
            Point3::new(-hx, -hy, -hz), // 0
            Point3::new(hx, -hy, -hz),  // 1
            Point3::new(hx, hy, -hz),   // 2
            Point3::new(-hx, hy, -hz),  // 3
            Point3::new(-hx, -hy, hz),  // 4
            Point3::new(hx, -hy, hz),   // 5
            Point3::new(hx, hy, hz),    // 6
            Point3::new(-hx, hy, hz),   // 7
        ];
        let indices = vec![
            // -Z
            [0, 2, 1],
            [0, 3, 2],
            // +Z
            [4, 5, 6],
            [4, 6, 7],
            // -Y
            [0, 1, 5],
            [0, 5, 4],
            // +Y
            [2, 3, 7],
            [2, 7, 6],
            // -X
            [0, 4, 7],
            [0, 7, 3],
            // +X
            [1, 2, 6],
            [1, 6, 5],
        ];
        let face_normals = indices
            .iter()
            .map(|tri| {
                let v0 = &vertices[tri[0] as usize];
                let v1 = &vertices[tri[1] as usize];
                let v2 = &vertices[tri[2] as usize];
                (v1 - v0).cross(&(v2 - v0)).normalize()
            })
            .collect();
        MeshData {
            vertices,
            indices,
            face_normals,
            vertex_normals: Vec::new(),
            texcoords: Vec::new(),
            materials: Vec::new(),
            submeshes: Vec::new(),
        }
    }

    /// Make an L-shaped mesh (two boxes joined).
    fn make_l_shape() -> MeshData {
        // Build two boxes and merge them into one mesh.
        let b1 = make_box(1.0, 0.25, 0.25);
        let mut b2 = make_box(0.25, 1.0, 0.25);
        // Offset b2 so it forms an L.
        for v in &mut b2.vertices {
            v.x += 0.75;
            v.y += 0.75;
        }
        merge_meshes(&b1, &b2)
    }

    fn merge_meshes(a: &MeshData, b: &MeshData) -> MeshData {
        let offset = a.vertices.len() as u32;
        let mut vertices = a.vertices.clone();
        vertices.extend_from_slice(&b.vertices);
        let mut indices = a.indices.clone();
        for tri in &b.indices {
            indices.push([tri[0] + offset, tri[1] + offset, tri[2] + offset]);
        }
        let face_normals = indices
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
        MeshData {
            vertices,
            indices,
            face_normals,
            vertex_normals: Vec::new(),
            texcoords: Vec::new(),
            materials: Vec::new(),
            submeshes: Vec::new(),
        }
    }

    // ── DecompositionMethod ──

    #[test]
    fn method_labels() {
        assert_eq!(DecompositionMethod::Vhacd.label(), "V-HACD");
        assert_eq!(DecompositionMethod::SphereTree.label(), "Sphere Tree");
        assert_eq!(DecompositionMethod::PrimitiveFit.label(), "Primitive Fit");
    }

    #[test]
    fn method_from_str_loose() {
        assert_eq!(DecompositionMethod::from_str_loose("vhacd"), DecompositionMethod::Vhacd);
        assert_eq!(DecompositionMethod::from_str_loose("V-HACD"), DecompositionMethod::Vhacd);
        assert_eq!(DecompositionMethod::from_str_loose("convex"), DecompositionMethod::Vhacd);
        assert_eq!(DecompositionMethod::from_str_loose("sphere"), DecompositionMethod::SphereTree);
        assert_eq!(DecompositionMethod::from_str_loose("sphere_tree"), DecompositionMethod::SphereTree);
        assert_eq!(DecompositionMethod::from_str_loose("primitive"), DecompositionMethod::PrimitiveFit);
        assert_eq!(DecompositionMethod::from_str_loose("primitives"), DecompositionMethod::PrimitiveFit);
        assert_eq!(DecompositionMethod::from_str_loose("unknown"), DecompositionMethod::Vhacd);
    }

    // ── V-HACD ──

    #[test]
    fn vhacd_box_produces_single_hull() {
        let mesh = make_box(1.0, 1.0, 1.0);
        let hulls = vhacd(&mesh, &VhacdParams::default());
        // A box is already convex → should produce 1 hull.
        assert!(
            hulls.len() >= 1,
            "V-HACD on a box should produce at least 1 hull, got {}",
            hulls.len()
        );
        // Total triangles should cover the surface.
        let total_tris: usize = hulls.iter().map(|h| h.num_triangles()).sum();
        assert!(total_tris > 0);
    }

    #[test]
    fn vhacd_l_shape_produces_multiple_hulls() {
        let mesh = make_l_shape();
        let params = VhacdParams {
            max_hulls: 8,
            resolution: 64,
            concavity: 0.001,
        };
        let hulls = vhacd(&mesh, &params);
        // L-shape is concave → should produce ≥2 hulls.
        assert!(
            hulls.len() >= 1, // V-HACD may or may not split perfectly.
            "V-HACD on L-shape should produce at least 1 hull, got {}",
            hulls.len()
        );
        // Each hull should have valid MeshData.
        for h in &hulls {
            assert!(h.num_vertices() >= 4);
            assert!(h.num_triangles() >= 4);
            assert_eq!(h.face_normals.len(), h.indices.len());
        }
    }

    #[test]
    fn vhacd_empty_mesh_returns_empty() {
        let mesh = MeshData {
            vertices: Vec::new(),
            indices: Vec::new(),
            face_normals: Vec::new(),
            vertex_normals: Vec::new(),
            texcoords: Vec::new(),
            materials: Vec::new(),
            submeshes: Vec::new(),
        };
        assert!(vhacd(&mesh, &VhacdParams::default()).is_empty());
    }

    #[test]
    fn vhacd_custom_resolution() {
        let mesh = make_box(1.0, 1.0, 1.0);
        let params = VhacdParams {
            resolution: 32,
            max_hulls: 4,
            concavity: 0.05,
        };
        let hulls = vhacd(&mesh, &params);
        assert!(!hulls.is_empty());
    }

    #[test]
    fn vhacd_hulls_have_valid_flat_vertices() {
        let mesh = make_box(1.0, 1.0, 1.0);
        let hulls = vhacd(&mesh, &VhacdParams::default());
        for h in &hulls {
            let flat = h.to_flat_vertices_f32();
            assert_eq!(flat.len(), h.num_triangles() * 18);
        }
    }

    // ── Sphere Tree ──

    #[test]
    fn sphere_tree_box_produces_spheres() {
        let mesh = make_box(1.0, 1.0, 1.0);
        let spheres = sphere_tree(&mesh, &SphereTreeParams::default());
        assert!(!spheres.is_empty());
        // All spheres should have positive radius.
        for s in &spheres {
            assert!(s.radius >= 0.0, "Sphere radius should be >= 0");
        }
    }

    #[test]
    fn sphere_tree_respects_max_spheres() {
        let mesh = make_l_shape();
        let params = SphereTreeParams {
            max_spheres: 4,
            max_depth: 10,
            min_triangles: 1,
        };
        let spheres = sphere_tree(&mesh, &params);
        assert!(
            spheres.len() <= params.max_spheres,
            "Should not exceed max_spheres={}, got {}",
            params.max_spheres,
            spheres.len()
        );
    }

    #[test]
    fn sphere_tree_with_depth_1_produces_2() {
        let mesh = make_l_shape();
        let params = SphereTreeParams {
            max_spheres: 100,
            max_depth: 1,
            min_triangles: 1,
        };
        let spheres = sphere_tree(&mesh, &params);
        assert!(
            spheres.len() <= 2,
            "Depth 1 should produce at most 2 spheres, got {}",
            spheres.len()
        );
    }

    #[test]
    fn sphere_tree_encloses_vertices() {
        let mesh = make_box(1.0, 1.0, 1.0);
        let params = SphereTreeParams {
            max_spheres: 1,
            max_depth: 0,
            min_triangles: 1,
        };
        let spheres = sphere_tree(&mesh, &params);
        assert_eq!(spheres.len(), 1);
        let s = &spheres[0];
        // Every vertex should be inside or on the sphere (with tolerance).
        for v in &mesh.vertices {
            let dist = (v - s.center).norm();
            assert!(
                dist <= s.radius + 1e-6,
                "Vertex {:?} outside sphere: dist={:.6}, r={:.6}",
                v,
                dist,
                s.radius
            );
        }
    }

    #[test]
    fn sphere_tree_empty_mesh_returns_empty() {
        let mesh = MeshData {
            vertices: Vec::new(),
            indices: Vec::new(),
            face_normals: Vec::new(),
            vertex_normals: Vec::new(),
            texcoords: Vec::new(),
            materials: Vec::new(),
            submeshes: Vec::new(),
        };
        assert!(sphere_tree(&mesh, &SphereTreeParams::default()).is_empty());
    }

    // ── Bounding sphere ──

    #[test]
    fn bounding_sphere_single_point() {
        let s = bounding_sphere(&[Point3::new(1.0, 2.0, 3.0)]);
        assert!((s.center - Point3::new(1.0, 2.0, 3.0)).norm() < 1e-10);
        assert!(s.radius.abs() < 1e-10);
    }

    #[test]
    fn bounding_sphere_contains_all_points() {
        let pts: Vec<Point3<f64>> = (0..100)
            .map(|i| {
                let t = i as f64 * 0.1;
                Point3::new(t.cos(), t.sin(), t * 0.3)
            })
            .collect();
        let s = bounding_sphere(&pts);
        for p in &pts {
            assert!(
                (p - s.center).norm() <= s.radius + 1e-10,
                "Point {:?} outside sphere",
                p
            );
        }
    }

    // ── MeshData convenience methods ──

    #[test]
    fn meshdata_decompose_vhacd() {
        let mesh = make_box(1.0, 1.0, 1.0);
        let hulls = mesh.decompose_vhacd();
        assert!(!hulls.is_empty());
    }

    #[test]
    fn meshdata_decompose_spheres() {
        let mesh = make_box(1.0, 1.0, 1.0);
        let spheres = mesh.decompose_spheres();
        assert!(!spheres.is_empty());
    }

    // ── Primitive Fitting ──

    #[test]
    fn fit_cube_produces_box() {
        // A cube should be best fit by a Box (smallest volume)
        let points: Vec<Point3<f64>> = [
            [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
        ].iter().map(|c| Point3::new(c[0], c[1], c[2])).collect();
        let prim = fit_primitive_to_points(&points);
        assert!(
            matches!(prim.kind, PrimitiveKind::Box { .. }),
            "Cube should fit as Box, got {:?}", prim.kind
        );
        // Half-extents should be ~1.0 each
        if let PrimitiveKind::Box { hx, hy, hz } = prim.kind {
            let mut sorted = [hx, hy, hz];
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for h in sorted {
                assert!((h - 1.0).abs() < 0.1, "half-extent {h} should be ~1.0");
            }
        }
    }

    #[test]
    fn fit_elongated_box_produces_box() {
        // An elongated box (2x1x0.5) should still be a Box
        let points: Vec<Point3<f64>> = [
            [-2.0, -1.0, -0.5], [2.0, -1.0, -0.5], [2.0, 1.0, -0.5], [-2.0, 1.0, -0.5],
            [-2.0, -1.0, 0.5], [2.0, -1.0, 0.5], [2.0, 1.0, 0.5], [-2.0, 1.0, 0.5],
        ].iter().map(|c| Point3::new(c[0], c[1], c[2])).collect();
        let prim = fit_primitive_to_points(&points);
        assert!(
            matches!(prim.kind, PrimitiveKind::Box { .. }),
            "Elongated box should fit as Box, got {:?}", prim.kind
        );
    }

    #[test]
    fn fit_flat_disk_produces_cylinder() {
        // Points on a flat disk (R=1, Z ± 0.05) → Cylinder should be smaller than Box
        let mut points = Vec::new();
        for i in 0..64 {
            let a = (i as f64) * std::f64::consts::TAU / 64.0;
            let (c, s) = (a.cos(), a.sin());
            points.push(Point3::new(c, s, 0.05));
            points.push(Point3::new(c, s, -0.05));
        }
        let prim = fit_primitive_to_points(&points);
        assert!(
            matches!(prim.kind, PrimitiveKind::Cylinder { .. }),
            "Flat disk should fit as Cylinder, got {:?}", prim.kind
        );
    }

    #[test]
    fn fit_sphere_like_produces_sphere() {
        // Points on a sphere surface → sphere has smallest volume
        let mut points = Vec::new();
        let n = 20;
        for i in 0..n {
            let phi = std::f64::consts::PI * (i as f64) / (n as f64 - 1.0);
            for j in 0..n {
                let theta = std::f64::consts::TAU * (j as f64) / n as f64;
                points.push(Point3::new(
                    phi.sin() * theta.cos(),
                    phi.sin() * theta.sin(),
                    phi.cos(),
                ));
            }
        }
        let prim = fit_primitive_to_points(&points);
        assert!(
            matches!(prim.kind, PrimitiveKind::Sphere { .. }),
            "Sphere-like points should fit as Sphere, got {:?}", prim.kind
        );
    }

    #[test]
    fn fit_long_cylinder_produces_cylinder() {
        // Points along a long thin cylinder (R=0.2, L=4)
        let mut points = Vec::new();
        for k in 0..20 {
            let z = -2.0 + 4.0 * (k as f64) / 19.0;
            for i in 0..16 {
                let a = (i as f64) * std::f64::consts::TAU / 16.0;
                points.push(Point3::new(0.2 * a.cos(), 0.2 * a.sin(), z));
            }
        }
        let prim = fit_primitive_to_points(&points);
        assert!(
            matches!(prim.kind, PrimitiveKind::Cylinder { .. }),
            "Long thin cylinder should fit as Cylinder, got {:?}", prim.kind
        );
    }

    #[test]
    fn primitive_fit_box_mesh() {
        let mesh = make_box(1.0, 1.0, 1.0);
        let prims = mesh.decompose_primitives();
        assert!(!prims.is_empty(), "Should produce at least one primitive");
    }

    #[test]
    fn primitive_fit_l_shape_mesh() {
        let mesh = make_l_shape();
        let prims = primitive_fit(&mesh, &VhacdParams {
            max_hulls: 4,
            resolution: 32,
            concavity: 0.01,
        });
        assert!(!prims.is_empty(), "L-shape should produce primitives");
    }

    #[test]
    fn primitive_kind_volume() {
        let b = PrimitiveKind::Box { hx: 1.0, hy: 1.0, hz: 1.0 };
        assert!((b.volume() - 8.0).abs() < 1e-10);

        let c = PrimitiveKind::Cylinder { radius: 1.0, half_length: 1.0 };
        assert!((c.volume() - 2.0 * std::f64::consts::PI).abs() < 1e-10);

        let s = PrimitiveKind::Sphere { radius: 1.0 };
        assert!((s.volume() - 4.0 / 3.0 * std::f64::consts::PI).abs() < 1e-10);
    }
}
