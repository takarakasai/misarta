//! Geometry model — visual and collision shape descriptions.
//!
//! Follows Pinocchio's design: a `GeometryModel` stores a collection of
//! `GeometryObject`s, each attached to a parent joint/frame with a local
//! placement offset. Two separate `GeometryModel` instances are typically
//! maintained — one for **visual** geometry, one for **collision** geometry.
//!
//! # Example
//!
//! ```
//! use misarta::geometry::*;
//! use misarta::se3;
//!
//! let mut gmodel = GeometryModel::new();
//! gmodel.add(GeometryObject {
//!     name: "base_visual".into(),
//!     parent_joint: 0,
//!     placement: se3::identity(),
//!     shape: GeometryShape::Box { x: 0.2, y: 0.2, z: 0.1 },
//!     mesh_path: None,
//!     mesh_scale: None,
//!     mesh_data: None,
//! });
//! assert_eq!(gmodel.num_objects(), 1);
//! ```

use crate::mesh::MeshData;
use crate::se3::SE3;
use nalgebra::Vector3;

// ─── Primitive shapes ───────────────────────────────────────────────────────

/// Geometric shape primitive or mesh reference.
///
/// Mirrors the shape types supported by Pinocchio / HPP-FCL / coal.
#[derive(Debug, Clone, PartialEq)]
pub enum GeometryShape {
    /// Axis-aligned box with half-extents along X, Y, Z.
    /// `x`, `y`, `z` are the **full** side lengths (URDF/SDF convention).
    Box { x: f64, y: f64, z: f64 },
    /// Sphere with given radius.
    Sphere { radius: f64 },
    /// Cylinder along Z with given radius and length.
    Cylinder { radius: f64, length: f64 },
    /// Capsule (cylinder with hemispherical caps) along Z.
    Capsule { radius: f64, length: f64 },
    /// Cone along Z.
    Cone { radius: f64, length: f64 },
    /// External mesh file (STL, OBJ, DAE, …).
    ///
    /// The path is always stored.  Call [`MeshData::from_stl`] and attach
    /// the result to [`GeometryObject::mesh_data`] to make the mesh
    /// available for collision and rendering.
    Mesh {
        filename: String,
        scale: Vector3<f64>,
    },
}

// ─── Geometry object ────────────────────────────────────────────────────────

/// A single geometry attached to a frame in the kinematic tree.
///
/// Corresponds to Pinocchio's `GeometryObject`.
#[derive(Debug, Clone)]
pub struct GeometryObject {
    /// Human-readable name (e.g. `"base_link_visual_0"`).
    pub name: String,
    /// Index of the parent joint (0 = universe / root link).
    pub parent_joint: usize,
    /// Placement of this geometry relative to the parent joint frame.
    pub placement: SE3<f64>,
    /// Shape description (primitive or mesh reference).
    pub shape: GeometryShape,
    /// Original mesh file path, if loaded from URDF/SDF `<mesh>`.
    pub mesh_path: Option<String>,
    /// Mesh scale factor, if specified.
    pub mesh_scale: Option<Vector3<f64>>,
    /// Loaded triangle-mesh data (populated when `shape` is `Mesh` and the
    /// file was successfully read).  Used for collision detection and
    /// rendering.
    pub mesh_data: Option<MeshData>,
}

// ─── Geometry model ─────────────────────────────────────────────────────────

/// Collection of geometry objects (either visual **or** collision).
///
/// Pinocchio maintains two instances: one for visuals and one for collisions.
/// misarta follows the same convention.
///
/// # Example
///
/// ```
/// use misarta::geometry::*;
/// use misarta::se3;
///
/// let mut visual = GeometryModel::new();
/// visual.add(GeometryObject {
///     name: "box_visual".into(),
///     parent_joint: 0,
///     placement: se3::identity(),
///     shape: GeometryShape::Box { x: 1.0, y: 1.0, z: 1.0 },
///     mesh_path: None,
///     mesh_scale: None,
///     mesh_data: None,
/// });
/// assert_eq!(visual.num_objects(), 1);
/// assert_eq!(visual.objects[0].name, "box_visual");
/// ```
#[derive(Debug, Clone, Default)]
pub struct GeometryModel {
    /// Ordered list of geometry objects.
    pub objects: Vec<GeometryObject>,
}

impl GeometryModel {
    /// Create an empty geometry model.
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
        }
    }

    /// Add a geometry object. Returns the index of the newly added object.
    pub fn add(&mut self, obj: GeometryObject) -> usize {
        let idx = self.objects.len();
        self.objects.push(obj);
        idx
    }

    /// Number of geometry objects.
    pub fn num_objects(&self) -> usize {
        self.objects.len()
    }

    /// Find a geometry object by name.
    pub fn find_by_name(&self, name: &str) -> Option<usize> {
        self.objects.iter().position(|o| o.name == name)
    }

    /// Get all objects attached to a given joint index.
    pub fn objects_for_joint(&self, joint_idx: usize) -> Vec<usize> {
        self.objects
            .iter()
            .enumerate()
            .filter(|(_, o)| o.parent_joint == joint_idx)
            .map(|(i, _)| i)
            .collect()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::se3;

    #[test]
    fn empty_geometry_model() {
        let gm = GeometryModel::new();
        assert_eq!(gm.num_objects(), 0);
    }

    #[test]
    fn add_and_find_objects() {
        let mut gm = GeometryModel::new();
        let idx = gm.add(GeometryObject {
            name: "base_box".into(),
            parent_joint: 0,
            placement: se3::identity(),
            shape: GeometryShape::Box {
                x: 0.2,
                y: 0.2,
                z: 0.1,
            },
            mesh_path: None,
            mesh_scale: None,
            mesh_data: None,
        });
        assert_eq!(idx, 0);
        assert_eq!(gm.num_objects(), 1);
        assert_eq!(gm.find_by_name("base_box"), Some(0));
        assert_eq!(gm.find_by_name("nonexistent"), None);
    }

    #[test]
    fn objects_for_joint_filter() {
        let mut gm = GeometryModel::new();
        gm.add(GeometryObject {
            name: "a".into(),
            parent_joint: 0,
            placement: se3::identity(),
            shape: GeometryShape::Sphere { radius: 0.1 },
            mesh_path: None,
            mesh_scale: None,
            mesh_data: None,
        });
        gm.add(GeometryObject {
            name: "b".into(),
            parent_joint: 1,
            placement: se3::identity(),
            shape: GeometryShape::Cylinder {
                radius: 0.02,
                length: 0.2,
            },
            mesh_path: None,
            mesh_scale: None,
            mesh_data: None,
        });
        gm.add(GeometryObject {
            name: "c".into(),
            parent_joint: 0,
            placement: se3::identity(),
            shape: GeometryShape::Capsule {
                radius: 0.05,
                length: 0.3,
            },
            mesh_path: None,
            mesh_scale: None,
            mesh_data: None,
        });

        assert_eq!(gm.objects_for_joint(0), vec![0, 2]);
        assert_eq!(gm.objects_for_joint(1), vec![1]);
        assert!(gm.objects_for_joint(99).is_empty());
    }

    #[test]
    fn all_shape_variants() {
        // Verify all shape variants can be constructed
        let shapes = vec![
            GeometryShape::Box {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            GeometryShape::Sphere { radius: 0.5 },
            GeometryShape::Cylinder {
                radius: 0.1,
                length: 1.0,
            },
            GeometryShape::Capsule {
                radius: 0.1,
                length: 1.0,
            },
            GeometryShape::Cone {
                radius: 0.1,
                length: 0.5,
            },
            GeometryShape::Mesh {
                filename: "robot.stl".into(),
                scale: nalgebra::Vector3::new(1.0, 1.0, 1.0),
            },
        ];
        assert_eq!(shapes.len(), 6);
    }
}
