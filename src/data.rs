//! Computation data — mutable workspace filled by algorithms.
//!
//! Pinocchio separates immutable Model from mutable Data. We follow the same
//! pattern, but our algorithm functions are *pure*: they take `&Model<T>` + `&[T]`
//! and return a fresh `Data<T>`, rather than mutating one in place.
//! All types are generic over `T: RealField`.

use crate::model::Model;
use crate::se3::{self, SE3};
use nalgebra::{DMatrix, RealField};

/// Stores the results of kinematic/dynamic computations.
///
/// Indexed in parallel with `Model::joints` (index 0 = universe).
#[allow(non_snake_case)]
#[derive(Debug, Clone)]
pub struct Data<T: RealField> {
    /// Joint placement relative to parent: parent_M_joint(q).
    pub joint_placements: Vec<SE3<T>>,
    /// Absolute placement of each joint frame in the world: world_M_joint.
    pub oMi: Vec<SE3<T>>,
    /// Body-frame Jacobians (6×nv), one per joint (only populated by `jacobian`).
    pub J: DMatrix<T>,
}

impl<T: RealField> Data<T> {
    /// Allocate data structures sized for the given model.
    pub fn new(model: &Model<T>) -> Self {
        let n = model.joints.len();
        Self {
            joint_placements: vec![se3::identity(); n],
            oMi: vec![se3::identity(); n],
            J: DMatrix::zeros(6, model.nv),
        }
    }
}
