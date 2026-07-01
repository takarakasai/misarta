//! Static load analysis: gravity torque margins, end-effector payload
//! capacity, and torque utilisation.
//!
//! Everything here is expressed on [`Model`] + [`JointLimits`] only — no
//! editor / GUI concepts. Callers that track joints under their own
//! indexing (e.g. articara's `RobotModel`) map results back via the
//! returned misarta joint indices.
//!
//! Conventions:
//! - Gravity direction and magnitude come from [`Model::gravity`], so a
//!   payload of `m` kg exerts the force `m * model.gravity` at the
//!   end-effector.
//! - Only single-DoF joints (revolute / prismatic) produce entries;
//!   free-flyer DoF have no actuator to overload.

use crate::jacobian;
use crate::limits::JointLimits;
use crate::model::Model;
use crate::rnea;
use nalgebra::DVector;

/// Static gravity load on one actuated DoF.
#[derive(Debug, Clone)]
pub struct GravityLoad {
    /// misarta joint index (1-based; 0 = universe never appears).
    pub joint_idx: usize,
    /// Index into the velocity vector / `JointLimits::tau_max`.
    pub v_idx: usize,
    /// Static torque (N·m) or force (N) required to hold the pose.
    pub gravity_torque: f64,
    /// Effort limit from [`JointLimits::tau_max`] (`INFINITY` if unbounded).
    pub tau_max: f64,
    /// `tau_max - |gravity_torque|`. Negative means the joint is already
    /// overloaded by gravity alone.
    pub torque_margin: f64,
}

/// Compute the static gravity load of every single-DoF joint at
/// configuration `q`.
///
/// Delegates to [`rnea::compute_gravity`] and pairs each DoF with its
/// effort limit.
pub fn gravity_loads(model: &Model<f64>, q: &[f64], limits: &JointLimits) -> Vec<GravityLoad> {
    limits.validate(model);
    let g_full = rnea::compute_gravity(model, q);

    let mut out = Vec::new();
    for (ji, joint) in model.joints.iter().enumerate().skip(1) {
        if joint.joint_type.nv() != 1 {
            continue;
        }
        let vi = model.v_idx[ji];
        let tau = g_full[vi];
        let tau_max = limits.tau_max[vi];
        out.push(GravityLoad {
            joint_idx: ji,
            v_idx: vi,
            gravity_torque: tau,
            tau_max,
            torque_margin: tau_max - tau.abs(),
        });
    }
    out
}

/// Result of a payload-capacity analysis at one end-effector joint.
#[derive(Debug, Clone)]
pub struct PayloadCapacity {
    /// Maximum payload mass (kg) the robot can hold statically before any
    /// actuated joint exceeds its effort limit. `0.0` when gravity alone
    /// already saturates a joint.
    pub max_mass_kg: f64,
    /// misarta joint index of the bottleneck joint.
    pub limiting_joint: usize,
    /// Additional actuator torque per kg of payload for every velocity
    /// DoF (`-J_lin^T · gravity`, length `nv`; same sign convention as
    /// `rnea::compute_gravity`). Entries of joints outside the
    /// root → end-effector chain are zero.
    pub tau_per_kg: DVector<f64>,
}

/// Additional **actuator** torque per kg of payload hung at `ee_joint`,
/// for every velocity DoF: `-J_lin(ee)^T · model.gravity`.
///
/// Sign convention matches [`rnea::compute_gravity`]: the value is the
/// torque the actuator must produce, not the generalized force of the
/// payload (they differ by sign — τ_actuator = −Jᵀ·F_external). The two
/// therefore add directly: `τ_total = g(q) + m · tau_per_kg`.
pub fn payload_tau_per_kg(model: &Model<f64>, q: &[f64], ee_joint: usize) -> DVector<f64> {
    // 6×nv world-frame Jacobian; rows 0–2 angular, rows 3–5 linear.
    let jac = jacobian::compute_joint_jacobian(model, q, ee_joint);
    let j_lin = jac.rows(3, 3);
    -(j_lin.transpose() * model.gravity)
}

/// Compute the maximum static payload at `ee_joint`.
///
/// Solves, per actuated DoF, for the largest `m ≥ 0` with
/// `|g_tau + m · tau_per_kg| ≤ tau_max` and takes the tightest bound.
/// Returns `None` when no DoF that the payload affects has a finite
/// effort limit (capacity would be unbounded / meaningless).
pub fn payload_capacity(
    model: &Model<f64>,
    q: &[f64],
    ee_joint: usize,
    limits: &JointLimits,
) -> Option<PayloadCapacity> {
    let loads = gravity_loads(model, q, limits);
    let tau_per_kg = payload_tau_per_kg(model, q, ee_joint);

    let mut max_mass = f64::INFINITY;
    let mut limiting = 0usize;

    for load in &loads {
        let tau_p = tau_per_kg[load.v_idx];
        if tau_p.abs() < 1e-12 {
            continue; // payload does not load this joint
        }
        if !load.tau_max.is_finite() || load.tau_max <= 0.0 {
            continue; // no meaningful effort limit
        }
        // Largest m with  -tau_max ≤ g_tau + m·tau_p ≤ tau_max :
        let hi = if tau_p > 0.0 {
            (load.tau_max - load.gravity_torque) / tau_p
        } else {
            (-load.tau_max - load.gravity_torque) / tau_p
        };
        if hi < max_mass {
            max_mass = hi;
            limiting = load.joint_idx;
        }
    }

    if max_mass.is_infinite() {
        return None;
    }
    Some(PayloadCapacity {
        max_mass_kg: max_mass.max(0.0),
        limiting_joint: limiting,
        tau_per_kg,
    })
}

/// Per-joint torque utilisation with a payload of `mass_kg` at `ee_joint`.
///
/// Returns `(joint_idx, |g_tau + m·tau_p| / tau_max)` for every actuated
/// DoF with a finite positive effort limit. Values above `1.0` mean the
/// joint is overloaded.
pub fn payload_utilisation(
    model: &Model<f64>,
    q: &[f64],
    ee_joint: usize,
    mass_kg: f64,
    limits: &JointLimits,
) -> Vec<(usize, f64)> {
    let loads = gravity_loads(model, q, limits);
    let tau_per_kg = payload_tau_per_kg(model, q, ee_joint);

    loads
        .iter()
        .filter(|l| l.tau_max.is_finite() && l.tau_max > 0.0)
        .map(|l| {
            let total = (l.gravity_torque + mass_kg * tau_per_kg[l.v_idx]).abs();
            (l.joint_idx, total / l.tau_max)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint;
    use crate::model::{LinkInertia, ModelBuilder};
    use crate::se3;
    use approx::assert_relative_eq;
    use nalgebra::{Matrix3, Vector3};

    const G: f64 = 9.81;

    /// Horizontal 1-link pendulum: revolute-Y joint at the origin, point
    /// mass `m` at distance `l` along +X. Holding torque = m·g·l.
    fn pendulum(m: f64, l: f64) -> Model<f64> {
        let inertia = LinkInertia {
            mass: m,
            center_of_mass: Vector3::new(l, 0.0, 0.0),
            rotational_inertia: Matrix3::zeros(),
        };
        ModelBuilder::new()
            .gravity(Vector3::new(0.0, 0.0, -G))
            .add_joint("shoulder", 0, joint::revolute_y(), se3::identity(), inertia)
            .build()
    }

    #[test]
    fn gravity_load_matches_analytic_pendulum() {
        let (m, l) = (2.0, 0.5);
        let model = pendulum(m, l);
        let limits = JointLimits::unbounded(&model);

        let loads = gravity_loads(&model, &[0.0], &limits);
        assert_eq!(loads.len(), 1);
        // Revolute-Y with mass at +X: gravity pulls -Z, holding torque m·g·l.
        assert_relative_eq!(loads[0].gravity_torque.abs(), m * G * l, epsilon = 1e-9);
        assert!(loads[0].tau_max.is_infinite());
    }

    #[test]
    fn payload_capacity_matches_analytic_pendulum() {
        let (m, l) = (2.0, 0.5);
        let model = pendulum(m, l);
        let mut limits = JointLimits::unbounded(&model);
        let tau_max = 30.0;
        limits.tau_max[0] = tau_max;

        // Payload hangs at the joint's own frame end: use the joint itself
        // as the end-effector; lever arm equals the joint origin → attach a
        // second fixed frame? Simplest: the payload acts at the joint frame
        // of `shoulder`'s child link origin (x = 0), so instead test with a
        // child prismatic-free chain: hang at distance l via a fixed-offset
        // revolute chain.
        let inertia_tip = LinkInertia {
            mass: 0.0,
            center_of_mass: Vector3::zeros(),
            rotational_inertia: Matrix3::zeros(),
        };
        let model2 = ModelBuilder::from_model(&model)
            .add_joint(
                "wrist",
                1,
                joint::revolute_y(),
                se3::from_rotation_and_translation(
                    &nalgebra::Rotation3::identity(),
                    &Vector3::new(l, 0.0, 0.0),
                ),
                inertia_tip,
            )
            .build();
        let mut limits2 = JointLimits::unbounded(&model2);
        limits2.tau_max[0] = tau_max;

        let cap = payload_capacity(&model2, &[0.0, 0.0], 2, &limits2).expect("finite capacity");
        // Shoulder already holds m·g·l; payload adds mass_kg·g·l at x = l.
        // m_max = (tau_max - m·g·l) / (g·l)
        let expected = (tau_max - m * G * l) / (G * l);
        assert_relative_eq!(cap.max_mass_kg, expected, epsilon = 1e-6);
        assert_eq!(cap.limiting_joint, 1);
    }

    #[test]
    fn utilisation_reaches_one_at_capacity() {
        let (m, l) = (1.0, 0.4);
        let model = pendulum(m, l);
        let inertia_tip = LinkInertia {
            mass: 0.0,
            center_of_mass: Vector3::zeros(),
            rotational_inertia: Matrix3::zeros(),
        };
        let model2 = ModelBuilder::from_model(&model)
            .add_joint(
                "wrist",
                1,
                joint::revolute_y(),
                se3::from_rotation_and_translation(
                    &nalgebra::Rotation3::identity(),
                    &Vector3::new(l, 0.0, 0.0),
                ),
                inertia_tip,
            )
            .build();
        let mut limits = JointLimits::unbounded(&model2);
        limits.tau_max[0] = 20.0;

        let cap = payload_capacity(&model2, &[0.0, 0.0], 2, &limits).unwrap();
        let util = payload_utilisation(&model2, &[0.0, 0.0], 2, cap.max_mass_kg, &limits);
        let shoulder = util.iter().find(|(ji, _)| *ji == 1).unwrap();
        assert_relative_eq!(shoulder.1, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn overloaded_pose_gives_zero_capacity() {
        let (m, l) = (10.0, 1.0);
        let model = pendulum(m, l);
        let inertia_tip = LinkInertia {
            mass: 0.0,
            center_of_mass: Vector3::zeros(),
            rotational_inertia: Matrix3::zeros(),
        };
        let model2 = ModelBuilder::from_model(&model)
            .add_joint(
                "wrist",
                1,
                joint::revolute_y(),
                se3::from_rotation_and_translation(
                    &nalgebra::Rotation3::identity(),
                    &Vector3::new(l, 0.0, 0.0),
                ),
                inertia_tip,
            )
            .build();
        let mut limits = JointLimits::unbounded(&model2);
        // Gravity alone needs 10·9.81·1 ≈ 98 N·m; allow only 50.
        limits.tau_max[0] = 50.0;

        let cap = payload_capacity(&model2, &[0.0, 0.0], 2, &limits).unwrap();
        assert_eq!(cap.max_mass_kg, 0.0);
    }

    /// The sign convention of `payload_tau_per_kg` must match RNEA:
    /// attaching a real point mass at the end-effector and re-running
    /// `compute_gravity` must reproduce `m · tau_per_kg` exactly.
    #[test]
    fn tau_per_kg_matches_rnea_finite_difference() {
        let (m, l) = (2.0, 0.5);
        let base = pendulum(m, l);
        let arm = |tip_mass: f64| {
            let tip = LinkInertia {
                mass: tip_mass,
                center_of_mass: Vector3::zeros(),
                rotational_inertia: Matrix3::zeros(),
            };
            ModelBuilder::from_model(&base)
                .add_joint(
                    "wrist",
                    1,
                    joint::revolute_y(),
                    se3::from_rotation_and_translation(
                        &nalgebra::Rotation3::identity(),
                        &Vector3::new(l, 0.0, 0.0),
                    ),
                    tip,
                )
                .build()
        };

        let q = [0.3, -0.2];
        let mu = 1.7;
        let g0 = crate::rnea::compute_gravity(&arm(0.0), &q);
        let g1 = crate::rnea::compute_gravity(&arm(mu), &q);
        let tpk = payload_tau_per_kg(&arm(0.0), &q, 2);

        for v in 0..2 {
            let fd = (g1[v] - g0[v]) / mu;
            assert_relative_eq!(tpk[v], fd, epsilon = 1e-9);
        }
    }
}
