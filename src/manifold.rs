//! Manifold-aware low-level configuration operations.
//!
//! These helpers are foundational for IK, trajectory optimization, and
//! simulation loops:
//! - `integrate`   : q ⊕ v·dt
//! - `difference`  : q1 ⊖ q0
//! - `interpolate` : q(α) between q0 and q1 on the configuration manifold
//! - `normalize_configuration` : enforce valid quaternion state

use crate::joint::JointType;
use crate::model::Model;
use crate::se3;
use nalgebra::RealField;

fn wrap_to_pi<T: RealField>(angle: T) -> T {
    let pi: T = nalgebra::convert(std::f64::consts::PI);
    let two_pi = pi.clone() + pi.clone();
    let mut a = angle;
    while a > pi {
        a = a - two_pi.clone();
    }
    while a <= -pi.clone() {
        a = a + two_pi.clone();
    }
    a
}

pub fn normalize_configuration<T: RealField>(model: &Model<T>, q: &[T]) -> Vec<T> {
    assert_eq!(q.len(), model.nq);
    let mut out = q.to_vec();

    for (i, joint) in model.joints.iter().enumerate().skip(1) {
        if let JointType::FreeFlyer = joint.joint_type {
            let qi = model.q_idx[i];
            let qx = out[qi + 3].clone();
            let qy = out[qi + 4].clone();
            let qz = out[qi + 5].clone();
            let qw = out[qi + 6].clone();
            let norm = (qx.clone() * qx + qy.clone() * qy + qz.clone() * qz + qw.clone() * qw)
                .sqrt();

            if norm <= nalgebra::convert(1e-15) {
                out[qi + 3] = T::zero();
                out[qi + 4] = T::zero();
                out[qi + 5] = T::zero();
                out[qi + 6] = T::one();
            } else {
                let inv = T::one() / norm;
                out[qi + 3] = out[qi + 3].clone() * inv.clone();
                out[qi + 4] = out[qi + 4].clone() * inv.clone();
                out[qi + 5] = out[qi + 5].clone() * inv.clone();
                out[qi + 6] = out[qi + 6].clone() * inv;
            }
        }
    }

    out
}

pub fn integrate<T: RealField>(model: &Model<T>, q: &[T], v: &[T], dt: T) -> Vec<T> {
    assert_eq!(q.len(), model.nq);
    assert_eq!(v.len(), model.nv);

    let mut q_new = q.to_vec();

    for (i, joint) in model.joints.iter().enumerate().skip(1) {
        let qi = model.q_idx[i];
        let vi = model.v_idx[i];

        match &joint.joint_type {
            JointType::Revolute { .. } => {
                let angle = q[qi].clone() + v[vi].clone() * dt.clone();
                q_new[qi] = wrap_to_pi(angle);
            }
            JointType::Prismatic { .. } => {
                q_new[qi] = q[qi].clone() + v[vi].clone() * dt.clone();
            }
            JointType::Fixed => {}
            JointType::FreeFlyer => {
                // q = [x,y,z,qx,qy,qz,qw], v = [wx,wy,wz,vx,vy,vz]
                let q_slice = &q[qi..qi + 7];
                let pose = joint.joint_type.forward(q_slice);

                let mut twist = se3::Motion::zeros();
                for k in 0..6 {
                    twist[k] = v[vi + k].clone() * dt.clone();
                }
                let delta = se3::exp(&twist);
                let pose_new = se3::compose(&pose, &delta);

                let t = se3::translation(&pose_new);
                let quat = pose_new.rotation.quaternion();
                q_new[qi] = t[0].clone();
                q_new[qi + 1] = t[1].clone();
                q_new[qi + 2] = t[2].clone();
                q_new[qi + 3] = quat.i.clone();
                q_new[qi + 4] = quat.j.clone();
                q_new[qi + 5] = quat.k.clone();
                q_new[qi + 6] = quat.w.clone();
            }
        }
    }

    normalize_configuration(model, &q_new)
}

pub fn difference<T: RealField>(model: &Model<T>, q0: &[T], q1: &[T]) -> Vec<T> {
    assert_eq!(q0.len(), model.nq);
    assert_eq!(q1.len(), model.nq);

    let mut dq = vec![T::zero(); model.nv];

    for (i, joint) in model.joints.iter().enumerate().skip(1) {
        let qi = model.q_idx[i];
        let vi = model.v_idx[i];

        match &joint.joint_type {
            JointType::Revolute { .. } => {
                dq[vi] = wrap_to_pi(q1[qi].clone() - q0[qi].clone());
            }
            JointType::Prismatic { .. } => {
                dq[vi] = q1[qi].clone() - q0[qi].clone();
            }
            JointType::Fixed => {}
            JointType::FreeFlyer => {
                let p0 = joint.joint_type.forward(&q0[qi..qi + 7]);
                let p1 = joint.joint_type.forward(&q1[qi..qi + 7]);
                let rel = se3::compose(&se3::inverse(&p0), &p1);
                let twist = se3::log(&rel);
                for k in 0..6 {
                    dq[vi + k] = twist[k].clone();
                }
            }
        }
    }

    dq
}

pub fn interpolate<T: RealField>(
    model: &Model<T>,
    q0: &[T],
    q1: &[T],
    alpha: T,
) -> Vec<T> {
    let dq = difference(model, q0, q1);
    let scaled: Vec<T> = dq.into_iter().map(|x| x * alpha.clone()).collect();
    integrate(model, q0, &scaled, T::one())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint;
    use crate::model::{LinkInertia, ModelBuilder};
    use crate::se3;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;

    fn revolute_model() -> Model<f64> {
        ModelBuilder::new()
            .add_joint(
                "j1",
                0,
                joint::revolute_z(),
                se3::identity(),
                LinkInertia::zero(),
            )
            .build()
    }

    fn freeflyer_model() -> Model<f64> {
        ModelBuilder::new()
            .add_joint(
                "base",
                0,
                JointType::FreeFlyer,
                se3::identity(),
                LinkInertia::zero(),
            )
            .build()
    }

    #[test]
    fn difference_revolute_uses_shortest_arc() {
        let m = revolute_model();
        let q0 = vec![std::f64::consts::PI - 0.01];
        let q1 = vec![-std::f64::consts::PI + 0.01];
        let dq = difference(&m, &q0, &q1);
        assert_relative_eq!(dq[0], 0.02, epsilon = 1e-10);
    }

    #[test]
    fn integrate_revolute_wraps_angle() {
        let m = revolute_model();
        let q = vec![std::f64::consts::PI - 0.01];
        let v = vec![0.02];
        let qn = integrate(&m, &q, &v, 1.0);
        assert!(qn[0] < 0.0);
        assert_relative_eq!(qn[0], -std::f64::consts::PI + 0.01, epsilon = 1e-10);
    }

    #[test]
    fn freeflyer_difference_integrate_roundtrip() {
        let m = freeflyer_model();
        let q0 = m.neutral_q();
        let v = vec![0.1, -0.2, 0.05, 0.3, 0.0, -0.1];

        let q1 = integrate(&m, &q0, &v, 0.2);
        let dq = difference(&m, &q0, &q1);
        let q1_recovered = integrate(&m, &q0, &dq, 1.0);

        let p1 = JointType::FreeFlyer.forward(&q1);
        let p2 = JointType::FreeFlyer.forward(&q1_recovered);

        assert_relative_eq!(se3::to_homogeneous(&p1), se3::to_homogeneous(&p2), epsilon = 1e-8);
    }

    #[test]
    fn interpolate_endpoints_match() {
        let m = revolute_model();
        let q0 = vec![0.2];
        let q1 = vec![1.1];
        let qa = interpolate(&m, &q0, &q1, 0.0);
        let qb = interpolate(&m, &q0, &q1, 1.0);
        assert_relative_eq!(qa[0], q0[0], epsilon = 1e-12);
        assert_relative_eq!(qb[0], q1[0], epsilon = 1e-12);
    }

    #[test]
    fn normalize_freeflyer_quaternion() {
        let m = freeflyer_model();
        let mut q = m.neutral_q();
        q[3] = 1.0;
        q[4] = 2.0;
        q[5] = 3.0;
        q[6] = 4.0;

        let qn = normalize_configuration(&m, &q);
        let n = (qn[3] * qn[3] + qn[4] * qn[4] + qn[5] * qn[5] + qn[6] * qn[6]).sqrt();
        assert_relative_eq!(n, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn normalize_zero_quaternion_fallback_identity() {
        let m = freeflyer_model();
        let mut q = m.neutral_q();
        q[3] = 0.0;
        q[4] = 0.0;
        q[5] = 0.0;
        q[6] = 0.0;
        let qn = normalize_configuration(&m, &q);
        assert_relative_eq!(qn[3], 0.0, epsilon = 1e-12);
        assert_relative_eq!(qn[4], 0.0, epsilon = 1e-12);
        assert_relative_eq!(qn[5], 0.0, epsilon = 1e-12);
        assert_relative_eq!(qn[6], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn prismatic_difference_linear() {
        let m = ModelBuilder::new()
            .add_joint(
                "j1",
                0,
                joint::prismatic_x(),
                se3::identity(),
                LinkInertia::zero(),
            )
            .build();
        let q0 = vec![0.3];
        let q1 = vec![1.8];
        let dq = difference(&m, &q0, &q1);
        assert_relative_eq!(dq[0], 1.5, epsilon = 1e-12);
    }

    #[test]
    fn freeflyer_translation_respects_body_frame_update() {
        let m = freeflyer_model();
        let mut q = m.neutral_q();

        // 90 deg yaw so local +x points to world +y
        let rot = nalgebra::UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::FRAC_PI_2);
        let quat = rot.quaternion();
        q[3] = quat.i;
        q[4] = quat.j;
        q[5] = quat.k;
        q[6] = quat.w;

        // local linear velocity +x, dt=1 -> world translation +y
        let v = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let qn = integrate(&m, &q, &v, 1.0);

        assert_relative_eq!(qn[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(qn[1], 1.0, epsilon = 1e-10);
    }
}
