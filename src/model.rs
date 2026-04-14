//! Robot model — the kinematic tree structure.
//!
//! Follows Pinocchio's design philosophy:
//! - **Model** is immutable and describes the robot topology + constant parameters.
//! - **Data** (separate module) holds mutable computation results.
//!
//! The model is built via a builder pattern, then frozen as an immutable value.
//! All types are generic over `T: RealField`.

use crate::joint::JointType;
use crate::se3::{self, SE3};
use nalgebra::{RealField, Vector3};

// ─── Single joint frame ─────────────────────────────────────────────────────

/// One joint in the kinematic tree.
#[derive(Debug, Clone)]
pub struct JointModel<T: RealField> {
    /// Human-readable name.
    pub name: String,
    /// Joint type (revolute / prismatic / fixed / free-flyer).
    pub joint_type: JointType<T>,
    /// Parent joint index (0 = universe / root).
    pub parent: usize,
    /// Fixed placement from parent joint frame to this joint's reference frame.
    /// In Pinocchio notation: ¹M_J (joint placement in parent frame).
    pub placement: SE3<T>,
}

// ─── Link (body) ────────────────────────────────────────────────────────────

/// Inertial properties of a rigid body (link).
#[derive(Debug, Clone)]
pub struct LinkInertia<T: RealField> {
    pub mass: T,
    pub center_of_mass: Vector3<T>,
    // Rotational inertia could be added later (Matrix3<T>).
}

impl<T: RealField> LinkInertia<T> {
    pub fn zero() -> Self {
        Self {
            mass: T::zero(),
            center_of_mass: Vector3::zeros(),
        }
    }
}

// ─── Model ──────────────────────────────────────────────────────────────────

/// Immutable robot model describing the kinematic tree.
///
/// Joint indices are 1-based; index 0 represents the universe (fixed root).
/// This matches Pinocchio's convention.
#[derive(Debug, Clone)]
pub struct Model<T: RealField> {
    /// Joint models, index 0 is a dummy "universe" joint.
    pub joints: Vec<JointModel<T>>,
    /// Link inertias, indexed in parallel with `joints`.
    pub inertias: Vec<LinkInertia<T>>,
    /// Starting index of each joint's configuration in the q vector.
    pub q_idx: Vec<usize>,
    /// Starting index of each joint's velocity in the v vector.
    pub v_idx: Vec<usize>,
    /// Total configuration dimension.
    pub nq: usize,
    /// Total velocity dimension.
    pub nv: usize,
    /// Gravity vector in the world frame
    pub gravity: Vector3<T>,
}

impl<T: RealField> Model<T> {
    /// Number of joints (excluding the universe).
    pub fn num_joints(&self) -> usize {
        self.joints.len() - 1
    }

    /// Zero configuration vector.
    pub fn neutral_q(&self) -> Vec<T> {
        let mut q = vec![T::zero(); self.nq];
        // For free-flyer joints, set quaternion w to 1.
        for (i, joint) in self.joints.iter().enumerate() {
            if let JointType::FreeFlyer = &joint.joint_type {
                let idx = self.q_idx[i];
                q[idx + 6] = T::one(); // qw = 1
            }
        }
        q
    }
}

// ─── Builder ────────────────────────────────────────────────────────────────

/// Builder for constructing a `Model` incrementally.
pub struct ModelBuilder<T: RealField> {
    joints: Vec<JointModel<T>>,
    inertias: Vec<LinkInertia<T>>,
    nq: usize,
    nv: usize,
    q_idx: Vec<usize>,
    v_idx: Vec<usize>,
    gravity: Vector3<T>,
}

impl<T: RealField> ModelBuilder<T> {
    /// Create a new builder with the universe joint at index 0.
    pub fn new() -> Self {
        let universe = JointModel {
            name: "universe".into(),
            joint_type: JointType::Fixed,
            parent: 0,
            placement: se3::identity(),
        };
        Self {
            joints: vec![universe],
            inertias: vec![LinkInertia::zero()],
            nq: 0,
            nv: 0,
            q_idx: vec![0],
            v_idx: vec![0],
            gravity: Vector3::new(T::zero(), T::zero(), nalgebra::convert(-9.81)),
        }
    }

    /// Set the gravity vector.
    pub fn gravity(mut self, g: Vector3<T>) -> Self {
        self.gravity = g;
        self
    }

    /// Add a joint (and its associated link) to the model.
    ///
    /// - `name`: human-readable joint name
    /// - `parent`: index of the parent joint (0 = universe)
    /// - `joint_type`: the joint type
    /// - `placement`: fixed placement from the parent joint frame to this joint's reference frame
    /// - `inertia`: link inertia attached to this joint
    ///
    /// Returns the index of the newly added joint.
    pub fn add_joint(
        mut self,
        name: impl Into<String>,
        parent: usize,
        joint_type: JointType<T>,
        placement: SE3<T>,
        inertia: LinkInertia<T>,
    ) -> Self {
        let qi = self.nq;
        let vi = self.nv;
        self.nq += joint_type.nq();
        self.nv += joint_type.nv();
        self.q_idx.push(qi);
        self.v_idx.push(vi);
        self.joints.push(JointModel {
            name: name.into(),
            joint_type,
            parent,
            placement,
        });
        self.inertias.push(inertia);
        self
    }

    /// Consume the builder and produce an immutable `Model`.
    pub fn build(self) -> Model<T> {
        Model {
            joints: self.joints,
            inertias: self.inertias,
            q_idx: self.q_idx,
            v_idx: self.v_idx,
            nq: self.nq,
            nv: self.nv,
            gravity: self.gravity,
        }
    }
}

impl<T: RealField> Default for ModelBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint;

    #[test]
    fn build_simple_chain() {
        let model = ModelBuilder::<f64>::new()
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .add_joint("j2", 1, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .build();

        assert_eq!(model.num_joints(), 2);
        assert_eq!(model.nq, 2);
        assert_eq!(model.nv, 2);
        assert_eq!(model.joints[1].parent, 0);
        assert_eq!(model.joints[2].parent, 1);
    }
}
