//! Joint types — Pinocchio-compatible revolute / prismatic / fixed / free-flyer.
//!
//! Each joint knows:
//! - how to compute its placement from a configuration vector slice (exp map)
//! - how to compute its motion subspace matrix S (for Jacobian)
//!
//! All functions are pure — no mutation of self.
//! All types are generic over `T: RealField`.

use crate::se3::{self, SE3};
use nalgebra::{Matrix6xX, RealField, Translation3, UnitQuaternion, Vector3};

// ─── Joint enum (tagged union — Rust idiom for Visitor / State patterns) ────

/// Supported joint types, matching Pinocchio's taxonomy.
#[derive(Debug, Clone)]
pub enum JointType<T: RealField> {
    /// 1-DOF rotation about a fixed axis.
    Revolute { axis: Vector3<T> },
    /// 1-DOF translation along a fixed axis.
    Prismatic { axis: Vector3<T> },
    /// 0-DOF rigid attachment.
    Fixed,
    /// 6-DOF free-flyer (e.g. floating base).
    FreeFlyer,
}

impl<T: RealField> JointType<T> {
    /// Number of degrees of freedom for this joint type.
    pub fn nq(&self) -> usize {
        match self {
            JointType::Revolute { .. } => 1,
            JointType::Prismatic { .. } => 1,
            JointType::Fixed => 0,
            JointType::FreeFlyer => 7, // quaternion (4) + translation (3)
        }
    }

    /// Number of velocity degrees of freedom (tangent space dimension).
    pub fn nv(&self) -> usize {
        match self {
            JointType::Revolute { .. } => 1,
            JointType::Prismatic { .. } => 1,
            JointType::Fixed => 0,
            JointType::FreeFlyer => 6,
        }
    }

    /// Joint placement from configuration: M_J(q).
    ///
    /// Pure function: `q` is a slice into the full configuration vector.
    /// Returns the SE(3) placement of the joint frame relative to its reference.
    pub fn forward(&self, q: &[T]) -> SE3<T> {
        match self {
            JointType::Revolute { axis } => {
                let angle = q[0].clone();
                let rot = UnitQuaternion::from_axis_angle(
                    &nalgebra::Unit::new_normalize(axis.clone()),
                    angle,
                );
                SE3::from_parts(Translation3::identity(), rot)
            }
            JointType::Prismatic { axis } => {
                let d = q[0].clone();
                SE3::from_parts(
                    Translation3::from(axis * d),
                    UnitQuaternion::identity(),
                )
            }
            JointType::Fixed => se3::identity(),
            JointType::FreeFlyer => {
                // q = [x, y, z, qx, qy, qz, qw]
                let t = Translation3::new(q[0].clone(), q[1].clone(), q[2].clone());
                let quat = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                    q[6].clone(),
                    q[3].clone(),
                    q[4].clone(),
                    q[5].clone(),
                ));
                SE3::from_parts(t, quat)
            }
        }
    }

    /// Motion subspace matrix S ∈ ℝ^{6×nv}.
    ///
    /// Maps joint velocity q̇ to spatial motion: v_J = S(q) q̇.
    /// For revolute/prismatic this is constant; for free-flyer it depends on q.
    pub fn motion_subspace(&self, _q: &[T]) -> Matrix6xX<T> {
        match self {
            JointType::Revolute { axis } => {
                let mut s = Matrix6xX::zeros(1);
                s[(0, 0)] = axis[0].clone();
                s[(1, 0)] = axis[1].clone();
                s[(2, 0)] = axis[2].clone();
                s
            }
            JointType::Prismatic { axis } => {
                let mut s = Matrix6xX::zeros(1);
                s[(3, 0)] = axis[0].clone();
                s[(4, 0)] = axis[1].clone();
                s[(5, 0)] = axis[2].clone();
                s
            }
            JointType::Fixed => Matrix6xX::zeros(0),
            JointType::FreeFlyer => {
                // S = I₆ (identity) for free-flyer in body frame
                let mut s = Matrix6xX::zeros(6);
                for i in 0..6 {
                    s[(i, i)] = T::one();
                }
                s
            }
        }
    }
}

// ─── Convenience constructors ───────────────────────────────────────────────

/// Revolute joint about the X axis.
pub fn revolute_x<T: RealField>() -> JointType<T> {
    JointType::Revolute {
        axis: Vector3::x(),
    }
}

/// Revolute joint about the Y axis.
pub fn revolute_y<T: RealField>() -> JointType<T> {
    JointType::Revolute {
        axis: Vector3::y(),
    }
}

/// Revolute joint about the Z axis.
pub fn revolute_z<T: RealField>() -> JointType<T> {
    JointType::Revolute {
        axis: Vector3::z(),
    }
}

/// Prismatic joint along the X axis.
pub fn prismatic_x<T: RealField>() -> JointType<T> {
    JointType::Prismatic {
        axis: Vector3::x(),
    }
}

/// Prismatic joint along the Y axis.
pub fn prismatic_y<T: RealField>() -> JointType<T> {
    JointType::Prismatic {
        axis: Vector3::y(),
    }
}

/// Prismatic joint along the Z axis.
pub fn prismatic_z<T: RealField>() -> JointType<T> {
    JointType::Prismatic {
        axis: Vector3::z(),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn revolute_z_quarter_turn() {
        let joint = revolute_z::<f64>();
        let placement = joint.forward(&[FRAC_PI_2]);
        let p = se3::act_on_point(&placement, &Vector3::new(1.0, 0.0, 0.0));
        assert_relative_eq!(p, Vector3::new(0.0, 1.0, 0.0), epsilon = 1e-12);
    }

    #[test]
    fn prismatic_x_displacement() {
        let joint = prismatic_x::<f64>();
        let placement = joint.forward(&[2.5]);
        assert_relative_eq!(
            se3::translation(&placement),
            Vector3::new(2.5, 0.0, 0.0),
            epsilon = 1e-14
        );
    }

    #[test]
    fn fixed_is_identity() {
        let joint: JointType<f64> = JointType::Fixed;
        let placement = joint.forward(&[]);
        assert_relative_eq!(
            se3::to_homogeneous(&placement),
            nalgebra::Matrix4::identity(),
            epsilon = 1e-14
        );
    }

    #[test]
    fn revolute_subspace_is_axis() {
        let joint = revolute_z::<f64>();
        let s = joint.motion_subspace(&[0.0]);
        assert_eq!(s.nrows(), 6);
        assert_eq!(s.ncols(), 1);
        assert_relative_eq!(s[(2, 0)], 1.0, epsilon = 1e-14);
    }
}
