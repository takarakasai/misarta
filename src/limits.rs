//! Joint limit utilities for configuration, velocity, and torque.

use crate::joint::JointType;
use crate::manifold;
use crate::model::Model;

#[derive(Debug, Clone)]
pub struct JointLimits {
    pub q_min: Vec<f64>,
    pub q_max: Vec<f64>,
    pub v_max: Vec<f64>,
    pub tau_max: Vec<f64>,
}

impl JointLimits {
    pub fn unbounded(model: &Model<f64>) -> Self {
        Self {
            q_min: vec![f64::NEG_INFINITY; model.nq],
            q_max: vec![f64::INFINITY; model.nq],
            v_max: vec![f64::INFINITY; model.nv],
            tau_max: vec![f64::INFINITY; model.nv],
        }
    }

    pub fn validate(&self, model: &Model<f64>) {
        assert_eq!(self.q_min.len(), model.nq, "q_min length mismatch");
        assert_eq!(self.q_max.len(), model.nq, "q_max length mismatch");
        assert_eq!(self.v_max.len(), model.nv, "v_max length mismatch");
        assert_eq!(self.tau_max.len(), model.nv, "tau_max length mismatch");
    }
}

pub fn clamp_configuration(model: &Model<f64>, q: &[f64], limits: &JointLimits) -> Vec<f64> {
    assert_eq!(q.len(), model.nq);
    limits.validate(model);

    let mut out = q.to_vec();
    for (i, joint) in model.joints.iter().enumerate().skip(1) {
        let qi = model.q_idx[i];
        match &joint.joint_type {
            JointType::Fixed => {}
            JointType::Revolute { .. } | JointType::Prismatic { .. } => {
                out[qi] = out[qi].clamp(limits.q_min[qi], limits.q_max[qi]);
            }
            JointType::FreeFlyer => {
                // Clamp translation only; quaternion is normalized afterward.
                for k in 0..3 {
                    out[qi + k] = out[qi + k].clamp(limits.q_min[qi + k], limits.q_max[qi + k]);
                }
            }
        }
    }

    manifold::normalize_configuration(model, &out)
}

pub fn saturate_velocity(model: &Model<f64>, v: &[f64], limits: &JointLimits) -> Vec<f64> {
    assert_eq!(v.len(), model.nv);
    limits.validate(model);

    v.iter()
        .enumerate()
        .map(|(i, x)| {
            let vmax = limits.v_max[i].abs();
            if vmax.is_infinite() {
                *x
            } else {
                x.clamp(-vmax, vmax)
            }
        })
        .collect()
}

pub fn project_torques(model: &Model<f64>, tau: &[f64], limits: &JointLimits) -> Vec<f64> {
    assert_eq!(tau.len(), model.nv);
    limits.validate(model);

    tau.iter()
        .enumerate()
        .map(|(i, x)| {
            let tmax = limits.tau_max[i].abs();
            if tmax.is_infinite() {
                *x
            } else {
                x.clamp(-tmax, tmax)
            }
        })
        .collect()
}

pub fn is_within_configuration_limits(model: &Model<f64>, q: &[f64], limits: &JointLimits) -> bool {
    assert_eq!(q.len(), model.nq);
    limits.validate(model);

    q.iter()
        .enumerate()
        .all(|(i, x)| *x >= limits.q_min[i] - 1e-12 && *x <= limits.q_max[i] + 1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint;
    use crate::model::{LinkInertia, ModelBuilder};
    use crate::se3;
    use approx::assert_relative_eq;

    fn one_revolute_model() -> Model<f64> {
        ModelBuilder::new()
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .build()
    }

    #[test]
    fn clamp_configuration_revolute() {
        let model = one_revolute_model();
        let mut limits = JointLimits::unbounded(&model);
        limits.q_min[0] = -1.0;
        limits.q_max[0] = 0.5;

        let q = vec![1.2];
        let qc = clamp_configuration(&model, &q, &limits);
        assert_relative_eq!(qc[0], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn saturate_velocity_and_torque() {
        let model = one_revolute_model();
        let mut limits = JointLimits::unbounded(&model);
        limits.v_max[0] = 2.0;
        limits.tau_max[0] = 3.0;

        let v = saturate_velocity(&model, &[5.0], &limits);
        let tau = project_torques(&model, &[-10.0], &limits);
        assert_relative_eq!(v[0], 2.0, epsilon = 1e-12);
        assert_relative_eq!(tau[0], -3.0, epsilon = 1e-12);
    }

    #[test]
    fn freeflyer_quaternion_normalized_after_clamp() {
        let model = ModelBuilder::new()
            .add_joint(
                "base",
                0,
                crate::joint::JointType::FreeFlyer,
                se3::identity(),
                LinkInertia::zero(),
            )
            .build();
        let mut limits = JointLimits::unbounded(&model);
        limits.q_min[0] = -0.5;
        limits.q_max[0] = 0.5;

        let mut q = model.neutral_q();
        q[0] = 3.0;
        q[3] = 1.0;
        q[4] = 2.0;
        q[5] = 3.0;
        q[6] = 4.0;

        let qc = clamp_configuration(&model, &q, &limits);
        assert_relative_eq!(qc[0], 0.5, epsilon = 1e-12);
        let n = (qc[3] * qc[3] + qc[4] * qc[4] + qc[5] * qc[5] + qc[6] * qc[6]).sqrt();
        assert_relative_eq!(n, 1.0, epsilon = 1e-12);
    }
}
