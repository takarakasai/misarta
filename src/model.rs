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
use nalgebra::{Matrix3, RealField, Vector3};

// ─── Mimic (coupled) joint ──────────────────────────────────────────────────

/// A mimic (coupled) joint relation: `q_slave = multiplier * q_master + offset`.
///
/// Corresponds to URDF's `<mimic joint="master" multiplier="m" offset="o"/>`.
/// The slave joint retains its original `JointType` (e.g. `Revolute`) in the
/// kinematic tree — its configuration is simply overwritten by this affine
/// mapping before FK / dynamics algorithms are called.
///
/// # Constraints
///
/// - Both `master` and `slave` must be 1-DOF joints (`Revolute` or `Prismatic`).
/// - `master` and `slave` are 1-based joint indices (0 = universe is invalid).
/// - Chained mimic joints (slave of a slave) are **not** checked at build
///   time but will work correctly with [`crate::mimic::enforce_mimic`] which
///   iterates in declaration order.
#[derive(Debug, Clone)]
pub struct MimicJoint<T: RealField> {
    /// Index of the slave joint (1-based).
    pub slave: usize,
    /// Index of the master joint (1-based).
    pub master: usize,
    /// Gear ratio: `q_slave = multiplier * q_master + offset`.
    pub multiplier: T,
    /// Offset: `q_slave = multiplier * q_master + offset`.
    pub offset: T,
}

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
    /// Rotational inertia about the center of mass, expressed in the body frame.
    pub rotational_inertia: Matrix3<T>,
}

impl<T: RealField> LinkInertia<T> {
    pub fn zero() -> Self {
        Self {
            mass: T::zero(),
            center_of_mass: Vector3::zeros(),
            rotational_inertia: Matrix3::zeros(),
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
    /// Human-readable robot name (from URDF `<robot name>` or SDF `<model name>`).
    pub name: String,
    /// Joint models, index 0 is a dummy "universe" joint.
    pub joints: Vec<JointModel<T>>,
    /// Link inertias, indexed in parallel with `joints`.
    pub inertias: Vec<LinkInertia<T>>,
    /// Link names, indexed in parallel with `joints`.
    /// `link_names[0]` is the root link; `link_names[i]` is the child link of `joints[i]`.
    pub link_names: Vec<String>,
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
    /// Mimic (coupled) joint constraints.
    ///
    /// Each entry maps a slave joint to its master via an affine relation.
    /// See [`MimicJoint`] for details.
    pub mimic: Vec<MimicJoint<T>>,
}

impl<T: RealField> Model<T> {
    /// Number of joints (excluding the universe).
    pub fn num_joints(&self) -> usize {
        self.joints.len() - 1
    }

    /// Check whether two models describe the same robot within a tolerance.
    ///
    /// Compares **by joint index** (not by name-matching): two models are
    /// considered equal when they have the same number of joints and, for
    /// every joint index, the name, type, parent index, placement, and
    /// link inertia all agree within `epsilon`.
    ///
    /// This is the structural / numerical analogue of `PartialEq`, but with
    /// a user-chosen tolerance for floating-point quantities.
    ///
    /// # Example
    ///
    /// ```
    /// # use misarta::model::*;
    /// # use misarta::{se3, joint};
    /// let a = ModelBuilder::<f64>::new()
    ///     .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
    ///     .build();
    /// let b = a.clone();
    /// assert!(a.approx_eq(&b, 1e-12));
    /// ```
    pub fn approx_eq(&self, other: &Model<T>, epsilon: T) -> bool {
        if self.name != other.name {
            return false;
        }
        if self.joints.len() != other.joints.len() {
            return false;
        }
        if self.nq != other.nq || self.nv != other.nv {
            return false;
        }
        if self.q_idx != other.q_idx || self.v_idx != other.v_idx {
            return false;
        }
        if (self.gravity.clone() - other.gravity.clone()).norm() > epsilon.clone() {
            return false;
        }
        for (a, b) in self.joints.iter().zip(other.joints.iter()) {
            if a.name != b.name {
                return false;
            }
            if a.parent != b.parent {
                return false;
            }
            if !a.joint_type.approx_eq(&b.joint_type, epsilon.clone()) {
                return false;
            }
            // Compare placements via homogeneous matrices
            let diff = (se3::to_homogeneous(&a.placement)
                - se3::to_homogeneous(&b.placement))
            .norm();
            if diff > epsilon.clone() {
                return false;
            }
        }
        for (a, b) in self.inertias.iter().zip(other.inertias.iter()) {
            if (a.mass.clone() - b.mass.clone()).abs() > epsilon.clone() {
                return false;
            }
            if (a.center_of_mass.clone() - b.center_of_mass.clone()).norm() > epsilon.clone() {
                return false;
            }
            if (a.rotational_inertia.clone() - b.rotational_inertia.clone()).norm() > epsilon.clone() {
                return false;
            }
        }
        // Compare mimic constraints
        if self.mimic.len() != other.mimic.len() {
            return false;
        }
        for (a, b) in self.mimic.iter().zip(other.mimic.iter()) {
            if a.slave != b.slave || a.master != b.master {
                return false;
            }
            if (a.multiplier.clone() - b.multiplier.clone()).abs() > epsilon.clone() {
                return false;
            }
            if (a.offset.clone() - b.offset.clone()).abs() > epsilon.clone() {
                return false;
            }
        }
        true
    }

    /// Like [`approx_eq`](Self::approx_eq) but matches joints **by name**
    /// instead of by index. This is useful when comparing models loaded from
    /// different formats (e.g. URDF vs SDF) that may order joints differently
    /// or include extra joints (e.g. fixed joints).
    ///
    /// Only joints whose names appear in **both** models are compared.
    /// Returns `(matching, mismatches)` where:
    /// - `matching` = number of joints that match within `epsilon`
    /// - `mismatches` = list of `(name, reason)` for joints that differ
    pub fn approx_eq_by_name(
        &self,
        other: &Model<T>,
        epsilon: T,
    ) -> (usize, Vec<(String, String)>) {
        let mut matching = 0usize;
        let mut mismatches: Vec<(String, String)> = Vec::new();

        for joint_a in &self.joints {
            if joint_a.name == "universe" {
                continue;
            }
            if let Some(joint_b) = other.joints.iter().find(|j| j.name == joint_a.name) {
                let mut ok = true;
                let mut reason = String::new();

                if !joint_a.joint_type.approx_eq(&joint_b.joint_type, epsilon.clone()) {
                    ok = false;
                    reason.push_str("joint_type ");
                }

                let diff = (se3::to_homogeneous(&joint_a.placement)
                    - se3::to_homogeneous(&joint_b.placement))
                .norm();
                if diff > epsilon.clone() {
                    ok = false;
                    reason.push_str("placement ");
                }

                // Compare corresponding inertias by position in self / other
                let idx_a = self.joints.iter().position(|j| j.name == joint_a.name).unwrap();
                let idx_b = other.joints.iter().position(|j| j.name == joint_a.name).unwrap();
                let ia = &self.inertias[idx_a];
                let ib = &other.inertias[idx_b];
                if (ia.mass.clone() - ib.mass.clone()).abs() > epsilon.clone() {
                    ok = false;
                    reason.push_str("mass ");
                }
                if (ia.center_of_mass.clone() - ib.center_of_mass.clone()).norm() > epsilon.clone()
                {
                    ok = false;
                    reason.push_str("center_of_mass ");
                }
                if (ia.rotational_inertia.clone() - ib.rotational_inertia.clone()).norm()
                    > epsilon.clone()
                {
                    ok = false;
                    reason.push_str("rotational_inertia ");
                }

                if ok {
                    matching += 1;
                } else {
                    mismatches.push((joint_a.name.clone(), reason.trim().to_string()));
                }
            }
        }

        (matching, mismatches)
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

    /// Collect the chain of ancestor joint indices from `joint_idx` up to
    /// (but not including) the universe (0).
    ///
    /// Returns the path in **child → root** order, e.g. `[joint_idx, parent, grandparent, …]`.
    pub fn ancestors_of(&self, joint_idx: usize) -> Vec<usize> {
        let mut chain = Vec::new();
        let mut cur = joint_idx;
        while cur > 0 {
            chain.push(cur);
            cur = self.joints[cur].parent;
        }
        chain
    }

    /// Check whether `ancestor` is on the path from `descendant` to the root.
    pub fn is_ancestor(&self, ancestor: usize, descendant: usize) -> bool {
        let mut cur = descendant;
        while cur > 0 {
            if cur == ancestor {
                return true;
            }
            cur = self.joints[cur].parent;
        }
        ancestor == 0
    }
}

// ─── Builder ────────────────────────────────────────────────────────────────

/// Builder for constructing a `Model` incrementally.
pub struct ModelBuilder<T: RealField> {
    name: String,
    joints: Vec<JointModel<T>>,
    inertias: Vec<LinkInertia<T>>,
    link_names: Vec<String>,
    nq: usize,
    nv: usize,
    q_idx: Vec<usize>,
    v_idx: Vec<usize>,
    gravity: Vector3<T>,
    mimic: Vec<MimicJoint<T>>,
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
            name: String::new(),
            joints: vec![universe],
            inertias: vec![LinkInertia::zero()],
            link_names: vec!["base_link".to_string()],
            nq: 0,
            nv: 0,
            q_idx: vec![0],
            v_idx: vec![0],
            gravity: Vector3::new(T::zero(), T::zero(), nalgebra::convert(-9.81)),
            mimic: Vec::new(),
        }
    }

    /// Set the robot name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the root link name (default: `"base_link"`).
    pub fn root_link_name(mut self, name: impl Into<String>) -> Self {
        self.link_names[0] = name.into();
        self
    }

    /// Set the gravity vector.
    pub fn gravity(mut self, g: Vector3<T>) -> Self {
        self.gravity = g;
        self
    }

    /// Reconstruct a builder from an existing `Model`, preserving all joints,
    /// inertias, link names, and gravity — but **not** mimic constraints.
    ///
    /// This is useful when you need to clone a model's tree structure and
    /// selectively re-add mimic constraints.
    pub fn from_model(model: &Model<T>) -> Self {
        Self {
            name: model.name.clone(),
            joints: model.joints.clone(),
            inertias: model.inertias.clone(),
            link_names: model.link_names.clone(),
            nq: model.nq,
            nv: model.nv,
            q_idx: model.q_idx.clone(),
            v_idx: model.v_idx.clone(),
            gravity: model.gravity.clone(),
            mimic: Vec::new(),
        }
    }

    /// Add a joint (and its associated link) to the model.
    ///
    /// The child link name is auto-generated as `"link_{n}"`. To specify
    /// an explicit link name, use [`add_joint_with_link`](Self::add_joint_with_link).
    ///
    /// - `name`: human-readable joint name
    /// - `parent`: index of the parent joint (0 = universe)
    /// - `joint_type`: the joint type
    /// - `placement`: fixed placement from the parent joint frame to this joint's reference frame
    /// - `inertia`: link inertia attached to this joint
    pub fn add_joint(
        self,
        name: impl Into<String>,
        parent: usize,
        joint_type: JointType<T>,
        placement: SE3<T>,
        inertia: LinkInertia<T>,
    ) -> Self {
        let link_name = format!("link_{}", self.joints.len());
        self.add_joint_with_link(name, parent, joint_type, placement, inertia, link_name)
    }

    /// Add a joint with an explicit child link name.
    pub fn add_joint_with_link(
        mut self,
        name: impl Into<String>,
        parent: usize,
        joint_type: JointType<T>,
        placement: SE3<T>,
        inertia: LinkInertia<T>,
        link_name: impl Into<String>,
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
        self.link_names.push(link_name.into());
        self
    }

    /// Add a mimic (coupled) joint constraint.
    ///
    /// - `slave`: index of the slave joint (1-based)
    /// - `master`: index of the master joint (1-based)
    /// - `multiplier`: gear ratio (`q_slave = multiplier * q_master + offset`)
    /// - `offset`: constant offset
    ///
    /// # Panics
    ///
    /// Panics if `slave` or `master` are out of range or refer to joints
    /// that are not 1-DOF (revolute or prismatic).
    pub fn add_mimic(
        mut self,
        slave: usize,
        master: usize,
        multiplier: T,
        offset: T,
    ) -> Self {
        assert!(
            slave > 0 && slave < self.joints.len(),
            "mimic slave index {} out of range (model has {} joints)",
            slave,
            self.joints.len() - 1,
        );
        assert!(
            master > 0 && master < self.joints.len(),
            "mimic master index {} out of range (model has {} joints)",
            master,
            self.joints.len() - 1,
        );
        assert!(
            self.joints[slave].joint_type.nq() == 1,
            "mimic slave joint '{}' must be 1-DOF (revolute or prismatic)",
            self.joints[slave].name,
        );
        assert!(
            self.joints[master].joint_type.nq() == 1,
            "mimic master joint '{}' must be 1-DOF (revolute or prismatic)",
            self.joints[master].name,
        );
        self.mimic.push(MimicJoint {
            slave,
            master,
            multiplier,
            offset,
        });
        self
    }

    /// Consume the builder and produce an immutable `Model`.
    pub fn build(self) -> Model<T> {
        Model {
            name: self.name,
            joints: self.joints,
            inertias: self.inertias,
            link_names: self.link_names,
            q_idx: self.q_idx,
            v_idx: self.v_idx,
            nq: self.nq,
            nv: self.nv,
            gravity: self.gravity,
            mimic: self.mimic,
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

    #[test]
    fn approx_eq_identical_models() {
        let model = ModelBuilder::<f64>::new()
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .add_joint("j2", 1, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .build();
        let clone = model.clone();
        assert!(model.approx_eq(&clone, 1e-14));
    }

    #[test]
    fn approx_eq_detects_different_joint_count() {
        let a = ModelBuilder::<f64>::new()
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .build();
        let b = ModelBuilder::<f64>::new()
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .add_joint("j2", 1, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .build();
        assert!(!a.approx_eq(&b, 1e-12));
    }

    #[test]
    fn approx_eq_detects_different_joint_name() {
        let a = ModelBuilder::<f64>::new()
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .build();
        let b = ModelBuilder::<f64>::new()
            .add_joint("j_other", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .build();
        assert!(!a.approx_eq(&b, 1e-12));
    }

    #[test]
    fn approx_eq_detects_different_joint_type() {
        let a = ModelBuilder::<f64>::new()
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .build();
        let b = ModelBuilder::<f64>::new()
            .add_joint("j1", 0, joint::prismatic_z(), se3::identity(), LinkInertia::zero())
            .build();
        assert!(!a.approx_eq(&b, 1e-12));
    }

    #[test]
    fn approx_eq_detects_different_axis() {
        let a = ModelBuilder::<f64>::new()
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .build();
        let b = ModelBuilder::<f64>::new()
            .add_joint("j1", 0, joint::revolute_x(), se3::identity(), LinkInertia::zero())
            .build();
        assert!(!a.approx_eq(&b, 1e-12));
    }

    #[test]
    fn approx_eq_detects_different_placement() {
        let offset = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &nalgebra::Vector3::new(1.0, 0.0, 0.0),
        );
        let a = ModelBuilder::<f64>::new()
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .build();
        let b = ModelBuilder::<f64>::new()
            .add_joint("j1", 0, joint::revolute_z(), offset, LinkInertia::zero())
            .build();
        assert!(!a.approx_eq(&b, 1e-12));
    }

    #[test]
    fn approx_eq_detects_different_mass() {
        let a = ModelBuilder::<f64>::new()
            .add_joint(
                "j1", 0, joint::revolute_z(), se3::identity(),
                LinkInertia { mass: 1.0, center_of_mass: nalgebra::Vector3::zeros(), rotational_inertia: nalgebra::Matrix3::zeros() },
            )
            .build();
        let b = ModelBuilder::<f64>::new()
            .add_joint(
                "j1", 0, joint::revolute_z(), se3::identity(),
                LinkInertia { mass: 2.0, center_of_mass: nalgebra::Vector3::zeros(), rotational_inertia: nalgebra::Matrix3::zeros() },
            )
            .build();
        assert!(!a.approx_eq(&b, 1e-12));
    }

    #[test]
    fn approx_eq_by_name_subset_match() {
        // Model A has j1 + j_extra; Model B has j1 only.
        // approx_eq_by_name should report j1 matches.
        let a = ModelBuilder::<f64>::new()
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .add_joint("j_extra", 1, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .build();
        let b = ModelBuilder::<f64>::new()
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .build();
        let (matching, mismatches) = a.approx_eq_by_name(&b, 1e-12);
        assert_eq!(matching, 1);
        assert!(mismatches.is_empty());
    }

    #[test]
    fn approx_eq_by_name_reports_mismatch() {
        let a = ModelBuilder::<f64>::new()
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .build();
        let b = ModelBuilder::<f64>::new()
            .add_joint("j1", 0, joint::prismatic_z(), se3::identity(), LinkInertia::zero())
            .build();
        let (matching, mismatches) = a.approx_eq_by_name(&b, 1e-12);
        assert_eq!(matching, 0);
        assert_eq!(mismatches.len(), 1);
        assert!(mismatches[0].1.contains("joint_type"));
    }
}
