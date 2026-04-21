//! Mesh decimation algorithms.
//!
//! Provides multiple edge-collapse strategies for reducing triangle count:
//!
//! | Method | Quality | Speed | Description |
//! |--------|---------|-------|-------------|
//! | [`Qem`](DecimationMethod::Qem) | ★★★ | ★★ | Garland & Heckbert Quadric Error Metrics |
//! | [`EdgeLength`](DecimationMethod::EdgeLength) | ★★ | ★★★ | Collapse shortest edges first |
//! | [`VertexClustering`](DecimationMethod::VertexClustering) | ★ | ★★★★ | Spatial grid vertex merging |
//!
//! # Example
//!
//! ```no_run
//! use misarta::mesh::MeshData;
//! use misarta::decimate::DecimationMethod;
//! use std::path::Path;
//!
//! let mesh = MeshData::from_stl(Path::new("robot.stl")).unwrap();
//! // QEM (default, best quality)
//! let reduced = mesh.decimate(0.5);
//! // Edge-length (faster)
//! let reduced_fast = mesh.decimate_with(0.5, DecimationMethod::EdgeLength);
//! // Vertex clustering (fastest)
//! let reduced_vcluster = mesh.decimate_with(0.5, DecimationMethod::VertexClustering);
//! ```

use crate::mesh::MeshData;
use nalgebra::{Matrix4, Point3, Vector3, Vector4};
use std::collections::{BinaryHeap, HashSet};

// ─── DecimationMethod ───────────────────────────────────────────────────────

/// Mesh decimation algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecimationMethod {
    /// Quadric Error Metrics — best quality, moderate speed.
    ///
    /// Minimises geometric error by tracking per-vertex error quadrics.
    /// The gold standard for offline mesh simplification.
    Qem,

    /// Shortest-edge collapse — good quality, fast.
    ///
    /// Always collapses the shortest edge, placing the merged vertex at the
    /// midpoint. Produces reasonable results especially for uniformly
    /// tessellated meshes.
    EdgeLength,

    /// Vertex clustering — lowest quality, fastest.
    ///
    /// Partitions space into a uniform grid and merges all vertices within
    /// each cell. Very fast but can distort thin features.
    VertexClustering,
}

impl DecimationMethod {
    /// All available methods, useful for UI combo boxes.
    pub const ALL: [DecimationMethod; 3] = [
        DecimationMethod::Qem,
        DecimationMethod::EdgeLength,
        DecimationMethod::VertexClustering,
    ];

    /// Short human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Qem => "QEM",
            Self::EdgeLength => "Edge Length",
            Self::VertexClustering => "Vertex Clustering",
        }
    }

    /// Description for tooltips.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Qem => "Best quality — minimises geometric error (Garland & Heckbert)",
            Self::EdgeLength => "Good quality — collapses shortest edges first (fast)",
            Self::VertexClustering => "Fastest — spatial grid vertex merging (lower quality)",
        }
    }

    /// Parse from a string (case-insensitive). Returns `Qem` as default.
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "qem" | "quadric" => Self::Qem,
            "edge" | "edge_length" | "edgelength" => Self::EdgeLength,
            "cluster" | "vertex_clustering" | "vertexclustering" | "vcluster" => {
                Self::VertexClustering
            }
            _ => Self::Qem,
        }
    }
}

// ─── Quadric Error Matrix ───────────────────────────────────────────────────

/// A 4×4 symmetric matrix representing the quadric error for a vertex.
/// Stored as a flat upper-triangular array for efficiency (10 elements).
#[derive(Debug, Clone, Copy)]
struct Quadric {
    /// Upper-triangular entries: a00, a01, a02, a03, a11, a12, a13, a22, a23, a33
    q: [f64; 10],
}

impl Quadric {
    fn zero() -> Self {
        Self { q: [0.0; 10] }
    }

    /// Build quadric from a plane `ax + by + cz + d = 0` (normal must be unit).
    fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            q: [
                a * a, a * b, a * c, a * d, // row 0
                b * b, b * c, b * d, // row 1 (upper tri)
                c * c, c * d, // row 2
                d * d, // row 3
            ],
        }
    }

    fn add(&self, other: &Quadric) -> Self {
        let mut r = Quadric::zero();
        for i in 0..10 {
            r.q[i] = self.q[i] + other.q[i];
        }
        r
    }

    /// Evaluate the error `v^T Q v` for a point `(x, y, z)`.
    fn error(&self, x: f64, y: f64, z: f64) -> f64 {
        let q = &self.q;
        // v = [x, y, z, 1]
        // v^T Q v expanded from upper-triangular storage
        q[0] * x * x
            + 2.0 * q[1] * x * y
            + 2.0 * q[2] * x * z
            + 2.0 * q[3] * x
            + q[4] * y * y
            + 2.0 * q[5] * y * z
            + 2.0 * q[6] * y
            + q[7] * z * z
            + 2.0 * q[8] * z
            + q[9]
    }

    /// Try to find the optimal vertex position that minimizes the error.
    /// Returns `None` if the system is singular (falls back to midpoint).
    fn optimal_point(&self, v1: &Point3<f64>, v2: &Point3<f64>) -> Point3<f64> {
        let q = &self.q;
        // Build the 4×4 matrix with last row = [0, 0, 0, 1]
        #[rustfmt::skip]
        let mat = Matrix4::new(
            q[0], q[1], q[2], q[3],
            q[1], q[4], q[5], q[6],
            q[2], q[5], q[7], q[8],
            0.0,  0.0,  0.0,  1.0,
        );

        if let Some(inv) = mat.try_inverse() {
            let opt = inv * Vector4::new(0.0, 0.0, 0.0, 1.0);
            if opt.w.abs() > 1e-12 {
                let p = Point3::new(opt.x / opt.w, opt.y / opt.w, opt.z / opt.w);
                // Sanity: don't allow the optimal point to be too far from the edge
                let mid = Point3::from((v1.coords + v2.coords) * 0.5);
                let edge_len = (v1 - v2).norm();
                if (p - mid).norm() < edge_len * 3.0 {
                    return p;
                }
            }
        }

        // Fallback: pick the endpoint or midpoint with smallest error
        let mid = Point3::from((v1.coords + v2.coords) * 0.5);
        let e1 = self.error(v1.x, v1.y, v1.z);
        let e2 = self.error(v2.x, v2.y, v2.z);
        let em = self.error(mid.x, mid.y, mid.z);
        if e1 <= e2 && e1 <= em {
            *v1
        } else if e2 <= em {
            *v2
        } else {
            mid
        }
    }
}

// ─── Edge collapse entry ────────────────────────────────────────────────────

/// An edge candidate for collapse, stored in a priority queue.
#[derive(Debug, Clone)]
struct CollapseCandidate {
    /// QEM error cost for this collapse.
    cost: f64,
    /// Vertex indices forming the edge (v_a < v_b).
    v_a: u32,
    v_b: u32,
    /// Optimal position for the merged vertex.
    target: Point3<f64>,
    /// Generation counter — used to invalidate stale entries.
    generation: u32,
}

impl PartialEq for CollapseCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for CollapseCandidate {}

impl PartialOrd for CollapseCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CollapseCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: reverse ordering so smallest cost is popped first
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ─── Helper: edge key ──────────────────────────────────────────────────────

fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Decimate a mesh using the specified [`DecimationMethod`].
///
/// - `target_ratio` is clamped to `[0.0, 1.0]`.
/// - `method` selects the simplification algorithm.
///
/// See [`DecimationMethod`] for trade-offs between quality and speed.
pub fn decimate_with(mesh: &MeshData, target_ratio: f64, method: DecimationMethod) -> MeshData {
    match method {
        DecimationMethod::Qem => decimate_qem(mesh, target_ratio),
        DecimationMethod::EdgeLength => decimate_edge_length(mesh, target_ratio),
        DecimationMethod::VertexClustering => decimate_vertex_clustering(mesh, target_ratio),
    }
}

/// Decimate using QEM (default, backward-compatible).
pub fn decimate(mesh: &MeshData, target_ratio: f64) -> MeshData {
    decimate_qem(mesh, target_ratio)
}

// ─── QEM (Quadric Error Metrics) ────────────────────────────────────────────

/// Decimate a mesh to approximately `target_ratio` of its original triangle
/// count using the Quadric Error Metrics (QEM) edge-collapse algorithm.
///
/// - `target_ratio` is clamped to `[0.0, 1.0]`.
///   - `0.5` means half the triangles.
///   - `1.0` returns a clone.
///   - `0.0` reduces as much as possible.
///
/// The returned mesh has updated vertices, indices and face normals.
/// Vertex normals, texcoords, materials, and submeshes are **not** preserved
/// (they are cleared) since the topology changes significantly.
fn decimate_qem(mesh: &MeshData, target_ratio: f64) -> MeshData {
    let ratio = target_ratio.clamp(0.0, 1.0);
    let target_tris = ((mesh.num_triangles() as f64) * ratio).ceil() as usize;

    if mesh.num_triangles() <= 4 || target_tris >= mesh.num_triangles() {
        return mesh.clone();
    }

    let n_verts = mesh.vertices.len();
    let n_tris = mesh.indices.len();

    // ── Working copies ──

    let mut positions: Vec<Point3<f64>> = mesh.vertices.clone();

    // alive[tri_idx] = true if the triangle still exists
    let mut alive = vec![true; n_tris];

    // Triangle indices (mutable — vertex references change during collapse)
    let mut tris: Vec<[u32; 3]> = mesh.indices.clone();

    // Map: vertex → set of triangle indices referencing it
    let mut vert_tris: Vec<HashSet<usize>> = vec![HashSet::new(); n_verts];
    for (ti, tri) in tris.iter().enumerate() {
        for &vi in tri {
            vert_tris[vi as usize].insert(ti);
        }
    }

    // Vertex liveness — when collapsed, a vertex is "redirected" to another
    // redirect[v] = v means v is live; redirect[v] = u means v was merged into u
    let mut redirect: Vec<u32> = (0..n_verts as u32).collect();

    // Generation counter per vertex — incremented after each collapse involving it
    let mut generation: Vec<u32> = vec![0; n_verts];

    // ── Compute initial per-vertex quadrics ──

    let mut quadrics: Vec<Quadric> = vec![Quadric::zero(); n_verts];

    for tri in tris.iter() {
        let v0 = &positions[tri[0] as usize];
        let v1 = &positions[tri[1] as usize];
        let v2 = &positions[tri[2] as usize];
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let n = edge1.cross(&edge2);
        let len = n.norm();
        if len < 1e-30 {
            continue;
        }
        let n = n / len;
        let d = -n.dot(&v0.coords);
        let q = Quadric::from_plane(n.x, n.y, n.z, d);

        // Weight by triangle area (optional but improves quality)
        let area = len * 0.5;
        let weighted = Quadric {
            q: q.q.map(|v| v * area),
        };

        for &vi in tri {
            quadrics[vi as usize] = quadrics[vi as usize].add(&weighted);
        }
    }

    // ── Build initial edge->candidate heap ──

    let mut heap: BinaryHeap<CollapseCandidate> = BinaryHeap::new();
    let mut seen_edges: HashSet<(u32, u32)> = HashSet::new();

    for tri in &tris {
        let edges = [
            edge_key(tri[0], tri[1]),
            edge_key(tri[1], tri[2]),
            edge_key(tri[0], tri[2]),
        ];
        for (a, b) in edges {
            if seen_edges.insert((a, b)) {
                let combined = quadrics[a as usize].add(&quadrics[b as usize]);
                let target = combined.optimal_point(
                    &positions[a as usize],
                    &positions[b as usize],
                );
                let cost = combined.error(target.x, target.y, target.z).max(0.0);
                heap.push(CollapseCandidate {
                    cost,
                    v_a: a,
                    v_b: b,
                    target,
                    generation: generation[a as usize] + generation[b as usize],
                });
            }
        }
    }

    // ── Main collapse loop ──

    let mut current_tris = n_tris;

    while current_tris > target_tris {
        let candidate = match heap.pop() {
            Some(c) => c,
            None => break,
        };

        // Resolve redirections
        let mut va = candidate.v_a;
        while redirect[va as usize] != va {
            va = redirect[va as usize];
        }
        let mut vb = candidate.v_b;
        while redirect[vb as usize] != vb {
            vb = redirect[vb as usize];
        }

        // Skip if both endpoints collapsed to the same vertex
        if va == vb {
            continue;
        }

        // Check generation — skip stale entries
        let expected_gen = generation[va as usize] + generation[vb as usize];
        if candidate.generation != expected_gen {
            continue;
        }

        // ── Perform the collapse: merge vb → va ──

        // Move va to optimal position
        positions[va as usize] = candidate.target;

        // Merge quadrics
        quadrics[va as usize] = quadrics[va as usize].add(&quadrics[vb as usize]);

        // Redirect vb → va
        redirect[vb as usize] = va;
        generation[va as usize] += 1;
        generation[vb as usize] += 1;

        // Update all triangles referencing vb
        let vb_tris: Vec<usize> = vert_tris[vb as usize].iter().copied().collect();
        for ti in vb_tris {
            if !alive[ti] {
                vert_tris[vb as usize].remove(&ti);
                continue;
            }
            // Replace vb with va in this triangle
            for slot in &mut tris[ti] {
                if *slot == vb {
                    *slot = va;
                }
            }
            vert_tris[vb as usize].remove(&ti);
            vert_tris[va as usize].insert(ti);

            // Check if triangle became degenerate (two or more identical vertices)
            let t = &tris[ti];
            if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
                alive[ti] = false;
                current_tris -= 1;
                // Remove from all vertex adjacency lists
                for &vi in &tris[ti] {
                    vert_tris[vi as usize].remove(&ti);
                }
            }
        }

        // Re-insert edges involving va into the heap
        let va_tris: Vec<usize> = vert_tris[va as usize].iter().copied().collect();
        for ti in va_tris {
            if !alive[ti] {
                continue;
            }
            for &vi in &tris[ti] {
                if vi != va {
                    let (ea, eb) = edge_key(va, vi);
                    let combined = quadrics[ea as usize].add(&quadrics[eb as usize]);
                    let target = combined.optimal_point(
                        &positions[ea as usize],
                        &positions[eb as usize],
                    );
                    let cost = combined.error(target.x, target.y, target.z).max(0.0);
                    heap.push(CollapseCandidate {
                        cost,
                        v_a: ea,
                        v_b: eb,
                        target,
                        generation: generation[ea as usize] + generation[eb as usize],
                    });
                }
            }
        }
    }

    compact_mesh(&positions, &tris, &alive, mesh)
}

// ─── Edge Length decimation ──────────────────────────────────────────────────

/// Simple edge-collapse heuristic: always collapse the shortest edge first,
/// placing the merged vertex at the midpoint.
///
/// Faster than QEM but less shape-preserving on curved surfaces.
fn decimate_edge_length(mesh: &MeshData, target_ratio: f64) -> MeshData {
    let ratio = target_ratio.clamp(0.0, 1.0);
    let target_tris = ((mesh.num_triangles() as f64) * ratio).ceil() as usize;

    if mesh.num_triangles() <= 4 || target_tris >= mesh.num_triangles() {
        return mesh.clone();
    }

    let n_verts = mesh.vertices.len();
    let n_tris = mesh.indices.len();

    let mut positions: Vec<Point3<f64>> = mesh.vertices.clone();
    let mut alive = vec![true; n_tris];
    let mut tris: Vec<[u32; 3]> = mesh.indices.clone();

    let mut vert_tris: Vec<HashSet<usize>> = vec![HashSet::new(); n_verts];
    for (ti, tri) in tris.iter().enumerate() {
        for &vi in tri {
            vert_tris[vi as usize].insert(ti);
        }
    }

    let mut redirect: Vec<u32> = (0..n_verts as u32).collect();
    let mut generation: Vec<u32> = vec![0; n_verts];

    // Build heap: cost = edge length
    let mut heap: BinaryHeap<CollapseCandidate> = BinaryHeap::new();
    let mut seen_edges: HashSet<(u32, u32)> = HashSet::new();

    for tri in &tris {
        let edges = [
            edge_key(tri[0], tri[1]),
            edge_key(tri[1], tri[2]),
            edge_key(tri[0], tri[2]),
        ];
        for (a, b) in edges {
            if seen_edges.insert((a, b)) {
                let len = (positions[a as usize] - positions[b as usize]).norm();
                let mid = Point3::from((positions[a as usize].coords + positions[b as usize].coords) * 0.5);
                heap.push(CollapseCandidate {
                    cost: len,
                    v_a: a,
                    v_b: b,
                    target: mid,
                    generation: generation[a as usize] + generation[b as usize],
                });
            }
        }
    }

    let mut current_tris = n_tris;

    while current_tris > target_tris {
        let candidate = match heap.pop() {
            Some(c) => c,
            None => break,
        };

        let mut va = candidate.v_a;
        while redirect[va as usize] != va {
            va = redirect[va as usize];
        }
        let mut vb = candidate.v_b;
        while redirect[vb as usize] != vb {
            vb = redirect[vb as usize];
        }

        if va == vb {
            continue;
        }

        let expected_gen = generation[va as usize] + generation[vb as usize];
        if candidate.generation != expected_gen {
            continue;
        }

        // Merge vb → va at midpoint
        positions[va as usize] = Point3::from(
            (positions[va as usize].coords + positions[vb as usize].coords) * 0.5,
        );

        redirect[vb as usize] = va;
        generation[va as usize] += 1;
        generation[vb as usize] += 1;

        let vb_tris: Vec<usize> = vert_tris[vb as usize].iter().copied().collect();
        for ti in vb_tris {
            if !alive[ti] {
                vert_tris[vb as usize].remove(&ti);
                continue;
            }
            for slot in &mut tris[ti] {
                if *slot == vb {
                    *slot = va;
                }
            }
            vert_tris[vb as usize].remove(&ti);
            vert_tris[va as usize].insert(ti);

            let t = &tris[ti];
            if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
                alive[ti] = false;
                current_tris -= 1;
                for &vi in &tris[ti] {
                    vert_tris[vi as usize].remove(&ti);
                }
            }
        }

        // Re-insert edges from va
        let va_tris: Vec<usize> = vert_tris[va as usize].iter().copied().collect();
        for ti in va_tris {
            if !alive[ti] {
                continue;
            }
            for &vi in &tris[ti] {
                if vi != va {
                    let len = (positions[va as usize] - positions[vi as usize]).norm();
                    let mid = Point3::from(
                        (positions[va as usize].coords + positions[vi as usize].coords) * 0.5,
                    );
                    let (ea, eb) = edge_key(va, vi);
                    heap.push(CollapseCandidate {
                        cost: len,
                        v_a: ea,
                        v_b: eb,
                        target: mid,
                        generation: generation[ea as usize] + generation[eb as usize],
                    });
                }
            }
        }
    }

    compact_mesh(&positions, &tris, &alive, mesh)
}

// ─── Vertex Clustering decimation ───────────────────────────────────────────

/// Vertex clustering: partition space into a uniform grid and merge all
/// vertices within each cell to the cell centroid.
///
/// The grid resolution is chosen so that the resulting triangle count
/// approximates `target_ratio * original_count`. This is the fastest method
/// but produces the lowest quality.
fn decimate_vertex_clustering(mesh: &MeshData, target_ratio: f64) -> MeshData {
    let ratio = target_ratio.clamp(0.0, 1.0);

    if mesh.num_triangles() <= 4 || ratio >= 1.0 {
        return mesh.clone();
    }

    // Compute AABB
    let (bb_min, bb_max) = match mesh.aabb() {
        Some(bb) => bb,
        None => return mesh.clone(),
    };

    let extent = bb_max - bb_min;
    let max_extent = extent.x.max(extent.y).max(extent.z);
    if max_extent < 1e-30 {
        return mesh.clone();
    }

    // Choose grid resolution: higher ratio ≈ more cells ≈ more detail
    // Heuristic: n_cells_per_axis ∝ ratio^(1/3) * some_base
    // We want ratio=1.0 → very fine grid, ratio→0 → very coarse
    let base_cells = (mesh.num_vertices() as f64).cbrt().ceil().max(2.0);
    let cells_per_axis = (base_cells * ratio.powf(0.33)).ceil().max(2.0) as usize;
    let cell_size = max_extent / cells_per_axis as f64;
    let inv_cell = 1.0 / cell_size;

    // Assign each vertex to a grid cell and compute centroid per cell
    let cell_key = |p: &Point3<f64>| -> (i64, i64, i64) {
        let dx = p.x - bb_min.x;
        let dy = p.y - bb_min.y;
        let dz = p.z - bb_min.z;
        (
            (dx * inv_cell).floor() as i64,
            (dy * inv_cell).floor() as i64,
            (dz * inv_cell).floor() as i64,
        )
    };

    // Map: cell key → (accumulated position sum, count, new vertex index)
    let mut cell_map: std::collections::HashMap<(i64, i64, i64), (Point3<f64>, usize, u32)> =
        std::collections::HashMap::new();

    // First pass: assign vertex → cell, accumulate
    let mut vert_to_new: Vec<u32> = Vec::with_capacity(mesh.vertices.len());

    for v in &mesh.vertices {
        let key = cell_key(v);
        let entry = cell_map.entry(key).or_insert_with(|| {
            (Point3::origin(), 0, 0)
        });
        entry.0.coords += v.coords;
        entry.1 += 1;
        vert_to_new.push(0); // placeholder, filled in second pass
    }

    // Second pass: assign final indices and compute centroids
    let mut new_vertices: Vec<Point3<f64>> = Vec::with_capacity(cell_map.len());
    for (_, (sum, count, idx)) in cell_map.iter_mut() {
        *idx = new_vertices.len() as u32;
        new_vertices.push(Point3::from(sum.coords / (*count as f64)));
    }

    // Map old vertices to new indices
    for (i, v) in mesh.vertices.iter().enumerate() {
        let key = cell_key(v);
        let (_, _, new_idx) = &cell_map[&key];
        vert_to_new[i] = *new_idx;
    }

    // Rebuild triangles, skipping degenerates
    let mut new_indices: Vec<[u32; 3]> = Vec::new();
    let mut new_normals: Vec<Vector3<f64>> = Vec::new();

    for tri in &mesh.indices {
        let i0 = vert_to_new[tri[0] as usize];
        let i1 = vert_to_new[tri[1] as usize];
        let i2 = vert_to_new[tri[2] as usize];

        if i0 == i1 || i1 == i2 || i0 == i2 {
            continue;
        }

        // Recompute face normal
        let v0 = &new_vertices[i0 as usize];
        let v1 = &new_vertices[i1 as usize];
        let v2 = &new_vertices[i2 as usize];
        let n = (v1 - v0).cross(&(v2 - v0));
        let len = n.norm();
        if len < 1e-30 {
            continue; // degenerate
        }

        new_indices.push([i0, i1, i2]);
        new_normals.push(n / len);
    }

    // Remove duplicate triangles (same 3 indices in any order)
    let mut seen_tris: HashSet<[u32; 3]> = HashSet::new();
    let mut deduped_indices: Vec<[u32; 3]> = Vec::new();
    let mut deduped_normals: Vec<Vector3<f64>> = Vec::new();

    for (ti, tri) in new_indices.iter().enumerate() {
        let mut sorted = *tri;
        sorted.sort();
        if seen_tris.insert(sorted) {
            deduped_indices.push(*tri);
            deduped_normals.push(new_normals[ti]);
        }
    }

    // Remove unused vertices
    let mut used = vec![false; new_vertices.len()];
    for tri in &deduped_indices {
        for &vi in tri {
            used[vi as usize] = true;
        }
    }
    let mut remap = vec![0u32; new_vertices.len()];
    let mut final_vertices: Vec<Point3<f64>> = Vec::new();
    for (i, &is_used) in used.iter().enumerate() {
        if is_used {
            remap[i] = final_vertices.len() as u32;
            final_vertices.push(new_vertices[i]);
        }
    }
    let final_indices: Vec<[u32; 3]> = deduped_indices
        .iter()
        .map(|tri| [remap[tri[0] as usize], remap[tri[1] as usize], remap[tri[2] as usize]])
        .collect();

    MeshData {
        vertices: final_vertices,
        indices: final_indices,
        face_normals: deduped_normals,
        vertex_normals: Vec::new(),
        texcoords: Vec::new(),
        materials: mesh.materials.clone(),
        submeshes: Vec::new(),
    }
}

// ─── Shared compacting helper ───────────────────────────────────────────────

/// Compact edge-collapse results into a clean `MeshData`.
fn compact_mesh(
    positions: &[Point3<f64>],
    tris: &[[u32; 3]],
    alive: &[bool],
    original: &MeshData,
) -> MeshData {
    let n_verts = positions.len();

    let mut used = vec![false; n_verts];
    for (ti, tri) in tris.iter().enumerate() {
        if alive[ti] {
            for &vi in tri {
                used[vi as usize] = true;
            }
        }
    }

    let mut new_idx = vec![0u32; n_verts];
    let mut new_vertices: Vec<Point3<f64>> = Vec::new();
    for (i, &is_used) in used.iter().enumerate() {
        if is_used {
            new_idx[i] = new_vertices.len() as u32;
            new_vertices.push(positions[i]);
        }
    }

    let mut new_indices: Vec<[u32; 3]> = Vec::new();
    let mut new_normals: Vec<Vector3<f64>> = Vec::new();

    for (ti, tri) in tris.iter().enumerate() {
        if !alive[ti] {
            continue;
        }
        let i0 = new_idx[tri[0] as usize];
        let i1 = new_idx[tri[1] as usize];
        let i2 = new_idx[tri[2] as usize];
        if i0 == i1 || i1 == i2 || i0 == i2 {
            continue;
        }
        new_indices.push([i0, i1, i2]);

        let v0 = &new_vertices[i0 as usize];
        let v1 = &new_vertices[i1 as usize];
        let v2 = &new_vertices[i2 as usize];
        let n = (v1 - v0).cross(&(v2 - v0));
        let len = n.norm();
        new_normals.push(if len > 1e-30 { n / len } else { Vector3::zeros() });
    }

    MeshData {
        vertices: new_vertices,
        indices: new_indices,
        face_normals: new_normals,
        vertex_normals: Vec::new(),
        texcoords: Vec::new(),
        materials: original.materials.clone(),
        submeshes: Vec::new(),
    }
}

// ─── Convenience methods on MeshData ────────────────────────────────────────

impl MeshData {
    /// Decimate this mesh to approximately `target_ratio` of the original
    /// triangle count using Quadric Error Metrics (QEM, default).
    ///
    /// See [`decimate()`] for full documentation.
    pub fn decimate(&self, target_ratio: f64) -> MeshData {
        decimate(self, target_ratio)
    }

    /// Decimate with a specific algorithm.
    ///
    /// See [`DecimationMethod`] for available options and trade-offs.
    pub fn decimate_with(&self, target_ratio: f64, method: DecimationMethod) -> MeshData {
        decimate_with(self, target_ratio, method)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Helper: create a simple quad (2 triangles, 4 vertices).
    fn make_quad() -> MeshData {
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let indices = vec![[0, 1, 2], [0, 2, 3]];
        let face_normals = vec![Vector3::z(), Vector3::z()];
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

    /// Helper: create a subdivided grid mesh with known triangle count.
    fn make_grid(n: usize) -> MeshData {
        let mut vertices = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                vertices.push(Point3::new(i as f64, j as f64, 0.0));
            }
        }
        let mut indices = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let v00 = (j * (n + 1) + i) as u32;
                let v10 = v00 + 1;
                let v01 = v00 + (n + 1) as u32;
                let v11 = v01 + 1;
                indices.push([v00, v10, v11]);
                indices.push([v00, v11, v01]);
            }
        }
        let face_normals = vec![Vector3::z(); indices.len()];
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

    #[test]
    fn decimate_ratio_1_returns_clone() {
        let mesh = make_quad();
        let result = mesh.decimate(1.0);
        assert_eq!(result.num_triangles(), 2);
        assert_eq!(result.num_vertices(), 4);
    }

    #[test]
    fn decimate_too_few_triangles_returns_clone() {
        let mesh = make_quad();
        let result = mesh.decimate(0.1);
        // With only 2 triangles (≤ 4), should return clone
        assert_eq!(result.num_triangles(), 2);
    }

    #[test]
    fn decimate_grid_reduces_triangle_count() {
        let mesh = make_grid(10); // 200 triangles
        assert_eq!(mesh.num_triangles(), 200);

        let reduced = mesh.decimate(0.5); // target ~100
        assert!(
            reduced.num_triangles() < 200,
            "Should reduce: got {} tris",
            reduced.num_triangles()
        );
        assert!(
            reduced.num_triangles() > 0,
            "Should have at least some triangles"
        );
    }

    #[test]
    fn decimate_grid_aggressive() {
        let mesh = make_grid(20); // 800 triangles
        assert_eq!(mesh.num_triangles(), 800);

        let reduced = mesh.decimate(0.1); // target ~80
        assert!(
            reduced.num_triangles() < 200,
            "Aggressive reduction should give < 200 tris, got {}",
            reduced.num_triangles()
        );
    }

    #[test]
    fn decimate_preserves_manifold() {
        let mesh = make_grid(10);
        let reduced = mesh.decimate(0.3);

        // All indices should reference valid vertices
        for tri in &reduced.indices {
            for &vi in tri {
                assert!(
                    (vi as usize) < reduced.vertices.len(),
                    "Invalid vertex index {} (max {})",
                    vi,
                    reduced.vertices.len()
                );
            }
        }
        // No degenerate triangles
        for tri in &reduced.indices {
            assert_ne!(tri[0], tri[1]);
            assert_ne!(tri[1], tri[2]);
            assert_ne!(tri[0], tri[2]);
        }
    }

    #[test]
    fn decimate_face_normals_count() {
        let mesh = make_grid(10);
        let reduced = mesh.decimate(0.5);
        assert_eq!(
            reduced.face_normals.len(),
            reduced.indices.len(),
            "Face normals count must equal triangle count"
        );
    }

    #[test]
    fn decimate_flat_vertices_roundtrip() {
        let mesh = make_grid(10);
        let reduced = mesh.decimate(0.5);
        let flat = reduced.to_flat_vertices_f32();
        assert_eq!(flat.len(), reduced.num_triangles() * 18);
    }

    #[test]
    fn quadric_plane_error() {
        // Plane z = 0 → normal = (0,0,1), d = 0
        let q = Quadric::from_plane(0.0, 0.0, 1.0, 0.0);
        // Point on plane: error should be 0
        assert_relative_eq!(q.error(1.0, 2.0, 0.0), 0.0, epsilon = 1e-10);
        // Point off plane by 1 unit: error should be 1
        assert_relative_eq!(q.error(0.0, 0.0, 1.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn decimate_stl_file() {
        let stl_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/meshes/test_box.stl");
        if !stl_path.exists() {
            return; // skip if fixture not available
        }
        let mesh = MeshData::from_stl(&stl_path).unwrap();
        let original_tris = mesh.num_triangles();
        if original_tris <= 4 {
            return; // too small to decimate
        }
        let reduced = mesh.decimate(0.5);
        assert!(
            reduced.num_triangles() <= original_tris,
            "Reduced mesh should have fewer or equal triangles"
        );
    }

    // ── Edge Length tests ──

    #[test]
    fn edge_length_reduces_grid() {
        let mesh = make_grid(10); // 200 triangles
        let reduced = mesh.decimate_with(0.5, DecimationMethod::EdgeLength);
        assert!(
            reduced.num_triangles() < 200,
            "Edge length should reduce: got {} tris",
            reduced.num_triangles()
        );
        assert!(reduced.num_triangles() > 0);
    }

    #[test]
    fn edge_length_preserves_manifold() {
        let mesh = make_grid(10);
        let reduced = mesh.decimate_with(0.3, DecimationMethod::EdgeLength);
        for tri in &reduced.indices {
            for &vi in tri {
                assert!((vi as usize) < reduced.vertices.len());
            }
            assert_ne!(tri[0], tri[1]);
            assert_ne!(tri[1], tri[2]);
            assert_ne!(tri[0], tri[2]);
        }
        assert_eq!(reduced.face_normals.len(), reduced.indices.len());
    }

    #[test]
    fn edge_length_ratio_1_returns_clone() {
        let mesh = make_grid(10);
        let result = mesh.decimate_with(1.0, DecimationMethod::EdgeLength);
        assert_eq!(result.num_triangles(), 200);
    }

    // ── Vertex Clustering tests ──

    #[test]
    fn vertex_clustering_reduces_grid() {
        let mesh = make_grid(10); // 200 triangles
        let reduced = mesh.decimate_with(0.5, DecimationMethod::VertexClustering);
        assert!(
            reduced.num_triangles() < 200,
            "Vertex clustering should reduce: got {} tris",
            reduced.num_triangles()
        );
        assert!(reduced.num_triangles() > 0);
    }

    #[test]
    fn vertex_clustering_preserves_valid_indices() {
        let mesh = make_grid(10);
        let reduced = mesh.decimate_with(0.3, DecimationMethod::VertexClustering);
        for tri in &reduced.indices {
            for &vi in tri {
                assert!((vi as usize) < reduced.vertices.len());
            }
            assert_ne!(tri[0], tri[1]);
            assert_ne!(tri[1], tri[2]);
            assert_ne!(tri[0], tri[2]);
        }
        assert_eq!(reduced.face_normals.len(), reduced.indices.len());
    }

    #[test]
    fn vertex_clustering_ratio_1_returns_clone() {
        let mesh = make_grid(10);
        let result = mesh.decimate_with(1.0, DecimationMethod::VertexClustering);
        assert_eq!(result.num_triangles(), 200);
    }

    #[test]
    fn vertex_clustering_aggressive() {
        let mesh = make_grid(20); // 800 triangles
        let reduced = mesh.decimate_with(0.1, DecimationMethod::VertexClustering);
        assert!(
            reduced.num_triangles() < 400,
            "Aggressive clustering should significantly reduce, got {}",
            reduced.num_triangles()
        );
    }

    // ── Cross-method comparison ──

    #[test]
    fn all_methods_produce_valid_output() {
        let mesh = make_grid(10);
        for method in DecimationMethod::ALL {
            let reduced = mesh.decimate_with(0.5, method);
            assert!(reduced.num_triangles() > 0, "{:?} produced 0 tris", method);
            assert!(reduced.num_triangles() <= 200, "{:?} didn't reduce", method);
            assert_eq!(reduced.face_normals.len(), reduced.indices.len());
            let flat = reduced.to_flat_vertices_f32();
            assert_eq!(flat.len(), reduced.num_triangles() * 18);
        }
    }

    #[test]
    fn method_from_str_loose() {
        assert_eq!(DecimationMethod::from_str_loose("qem"), DecimationMethod::Qem);
        assert_eq!(DecimationMethod::from_str_loose("QEM"), DecimationMethod::Qem);
        assert_eq!(DecimationMethod::from_str_loose("edge"), DecimationMethod::EdgeLength);
        assert_eq!(DecimationMethod::from_str_loose("edge_length"), DecimationMethod::EdgeLength);
        assert_eq!(DecimationMethod::from_str_loose("cluster"), DecimationMethod::VertexClustering);
        assert_eq!(DecimationMethod::from_str_loose("vcluster"), DecimationMethod::VertexClustering);
        assert_eq!(DecimationMethod::from_str_loose("unknown"), DecimationMethod::Qem); // default
    }
}
