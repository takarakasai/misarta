//! Constraint type definitions — [`ConstraintType`], [`ReferenceFrame`],
//! [`RigidConstraint`], and [`ConstraintModel`].

use crate::frames::Frame;
use crate::se3::{self, SE3};
use nalgebra::RealField;

// ─── Constraint type ────────────────────────────────────────────────────────

/// Dimensionality / type of a rigid constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintType {
    /// Full 6-D constraint (position + orientation).
    Contact6D,
    /// Position-only 3-D constraint.
    Contact3D,
}

impl ConstraintType {
    /// Number of rows this constraint contributes.
    pub fn dim(&self) -> usize {
        match self {
            ConstraintType::Contact6D => 6,
            ConstraintType::Contact3D => 3,
        }
    }
}

// ─── Reference frame for expressing the constraint ─────────────────────────

/// In which coordinate frame the constraint error and Jacobian are expressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceFrame {
    /// World (spatial / fixed) frame.
    World,
    /// Frame 1 (the first frame of the constraint pair).
    Local,
}

// ─── Single rigid constraint ────────────────────────────────────────────────

/// A rigid constraint between two operational frames.
///
/// The constraint enforces that the relative placement of `frame2` w.r.t.
/// `frame1` equals `desired_relative_placement` (identity by default,
/// meaning the two frames should coincide).
///
/// Either frame can have `parent_joint == 0` to anchor it to the world.
#[derive(Debug, Clone)]
pub struct RigidConstraint<T: RealField> {
    /// Human-readable name.
    pub name: String,
    /// First frame (reference).
    pub frame1: Frame<T>,
    /// Second frame (target).
    pub frame2: Frame<T>,
    /// Desired relative placement: $M_1^{-1} M_2 = M_{\text{des}}$.
    /// Default is identity (frames should coincide).
    pub desired_relative_placement: SE3<T>,
    /// Constraint type (6D or 3D).
    pub constraint_type: ConstraintType,
    /// Reference frame for the error/Jacobian.
    pub reference_frame: ReferenceFrame,
}

impl<T: RealField> RigidConstraint<T> {
    /// Create a 6-D (pose) constraint between two frames.
    ///
    /// The desired relative placement defaults to identity (frames coincide).
    pub fn pose(frame1: Frame<T>, frame2: Frame<T>) -> Self {
        Self {
            name: format!("{}-{}", frame1.name, frame2.name),
            frame1,
            frame2,
            desired_relative_placement: se3::identity(),
            constraint_type: ConstraintType::Contact6D,
            reference_frame: ReferenceFrame::World,
        }
    }

    /// Create a 3-D (position-only) constraint between two frames.
    pub fn position(frame1: Frame<T>, frame2: Frame<T>) -> Self {
        Self {
            name: format!("{}-{}", frame1.name, frame2.name),
            frame1,
            frame2,
            desired_relative_placement: se3::identity(),
            constraint_type: ConstraintType::Contact3D,
            reference_frame: ReferenceFrame::World,
        }
    }

    /// Builder: set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Builder: set the desired relative placement.
    pub fn with_desired_placement(mut self, m: SE3<T>) -> Self {
        self.desired_relative_placement = m;
        self
    }

    /// Builder: set the reference frame.
    pub fn with_reference_frame(mut self, rf: ReferenceFrame) -> Self {
        self.reference_frame = rf;
        self
    }

    /// Number of scalar constraint rows.
    pub fn dim(&self) -> usize {
        self.constraint_type.dim()
    }
}

// ─── Constraint model (collection) ─────────────────────────────────────────

/// Collection of rigid constraints.
#[derive(Debug, Clone)]
pub struct ConstraintModel<T: RealField> {
    pub constraints: Vec<RigidConstraint<T>>,
}

impl<T: RealField> ConstraintModel<T> {
    /// Create an empty constraint model.
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    /// Create from a vec of constraints.
    pub fn from_constraints(constraints: Vec<RigidConstraint<T>>) -> Self {
        Self { constraints }
    }

    /// Add a constraint.
    pub fn add(&mut self, c: RigidConstraint<T>) {
        self.constraints.push(c);
    }

    /// Total number of constraint rows.
    pub fn total_dim(&self) -> usize {
        self.constraints.iter().map(|c| c.dim()).sum()
    }

    /// Number of constraints.
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// Whether there are no constraints.
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }
}

impl<T: RealField> Default for ConstraintModel<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::Frame;

    fn frame_at_joint(name: &str, joint_idx: usize) -> Frame<f64> {
        Frame {
            name: name.to_string(),
            parent_joint: joint_idx,
            placement: se3::identity(),
        }
    }

    #[test]
    fn empty_constraint_model() {
        let cm = ConstraintModel::<f64>::new();
        assert_eq!(cm.total_dim(), 0);
        assert!(cm.is_empty());
    }

    #[test]
    fn constraint_model_dimensions() {
        let f1 = frame_at_joint("a", 1);
        let f2 = frame_at_joint("b", 2);
        let f3 = frame_at_joint("c", 3);

        let mut cm = ConstraintModel::new();
        cm.add(RigidConstraint::pose(f1.clone(), f2.clone()));
        cm.add(RigidConstraint::position(f2, f3));

        assert_eq!(cm.len(), 2);
        assert_eq!(cm.total_dim(), 9); // 6 + 3
    }
}
