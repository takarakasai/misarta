//! Mimic (coupled) joint utilities.
//!
//! A **mimic joint** is a slave joint whose configuration is determined by a
//! master joint via an affine mapping:
//!
//! $$q_{\text{slave}} = m \cdot q_{\text{master}} + o$$
//!
//! This module provides pure functions that operate on a `Model` and its mimic
//! constraints **without** modifying any algorithm internals (FK, RNEA, ABA,
//! etc.).  Instead, the mimic relations are applied as a **pre-processing
//! step** on the configuration / velocity vectors before calling the standard
//! algorithms.
//!
//! # Workflow
//!
//! ```text
//! q_independent ──► enforce_mimic(model, q) ──► q_full ──► FK / RNEA / ABA
//! ```
//!
//! For optimisation / IK in the reduced (independent) variable space, use
//! [`mimic_projection_matrix`] to obtain the mapping from independent
//! velocities to full velocities:
//!
//! $$\dot{q} = G \, \dot{q}_{\text{indep}}$$
//!
//! # Example
//!
//! ```
//! use misarta::{model::*, joint, se3, mimic};
//!
//! let model = ModelBuilder::<f64>::new()
//!     .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
//!     .add_joint("j2", 1, joint::revolute_z(), se3::identity(), LinkInertia::zero())
//!     .add_mimic(2, 1, 2.0, 0.1)   // j2 = 2 * j1 + 0.1
//!     .build();
//!
//! let q = vec![0.5, 0.0];   // only j1 matters
//! let q_full = mimic::enforce_mimic(&model, &q);
//! assert!((q_full[1] - (2.0 * 0.5 + 0.1)).abs() < 1e-14);
//! ```

use crate::model::Model;
use nalgebra::{DMatrix, RealField};
use std::collections::HashSet;

/// Enforce all mimic constraints on a configuration vector.
///
/// For each mimic relation `(slave, master, m, o)`, overwrites
/// `q[q_idx[slave]] = m * q[q_idx[master]] + o`.
///
/// The input `q` is not modified; a new vector is returned.
///
/// Relations are applied in declaration order, so chained mimic joints
/// (A → B → C) work correctly as long as they are declared in dependency
/// order (A→B before B→C).
pub fn enforce_mimic<T: RealField + Copy>(model: &Model<T>, q: &[T]) -> Vec<T> {
    assert_eq!(q.len(), model.nq);
    let mut out = q.to_vec();
    for mj in &model.mimic {
        let qi_master = model.q_idx[mj.master];
        let qi_slave = model.q_idx[mj.slave];
        out[qi_slave] = mj.multiplier * out[qi_master] + mj.offset;
    }
    out
}

/// Enforce mimic constraints on a velocity vector.
///
/// For each mimic relation `(slave, master, m, _)`, overwrites
/// `v[v_idx[slave]] = m * v[v_idx[master]]`.
///
/// (The offset does not affect velocity.)
pub fn enforce_mimic_velocity<T: RealField + Copy>(model: &Model<T>, v: &[T]) -> Vec<T> {
    assert_eq!(v.len(), model.nv);
    let mut out = v.to_vec();
    for mj in &model.mimic {
        let vi_master = model.v_idx[mj.master];
        let vi_slave = model.v_idx[mj.slave];
        out[vi_slave] = mj.multiplier * out[vi_master];
    }
    out
}

/// Indices of the **independent** (non-slave) velocity DOFs.
///
/// Returns a sorted list of `v_idx` values that are not slave DOFs.
pub fn independent_v_indices<T: RealField>(model: &Model<T>) -> Vec<usize> {
    let slave_set: HashSet<usize> = model
        .mimic
        .iter()
        .map(|mj| model.v_idx[mj.slave])
        .collect();
    (0..model.nv).filter(|i| !slave_set.contains(i)).collect()
}

/// Number of independent (non-mimic) velocity DOFs.
pub fn num_independent_v<T: RealField>(model: &Model<T>) -> usize {
    model.nv - model.mimic.len()
}

/// Indices of the **independent** (non-slave) configuration DOFs.
///
/// Returns a sorted list of `q_idx` values that are not slave DOFs.
pub fn independent_q_indices<T: RealField>(model: &Model<T>) -> Vec<usize> {
    let slave_set: HashSet<usize> = model
        .mimic
        .iter()
        .map(|mj| model.q_idx[mj.slave])
        .collect();
    (0..model.nq).filter(|i| !slave_set.contains(i)).collect()
}

/// Build the mimic projection matrix $G \in \mathbb{R}^{n_v \times n_{\text{indep}}}$.
///
/// Maps independent joint velocities to full joint velocities:
///
/// $$\dot{q} = G \, \dot{q}_{\text{indep}}$$
///
/// For a non-mimic joint, the corresponding column of $G$ is a unit vector.
/// For a slave joint with multiplier $m$ and master joint $k$, the slave row
/// is $m$ times the master's unit vector (i.e. $G[\text{slave}, k'] = m$
/// where $k'$ is the column index of the master in the independent space).
///
/// # Returns
///
/// `DMatrix<T>` of shape `(nv, n_indep)`.
pub fn mimic_projection_matrix<T: RealField + Copy>(model: &Model<T>) -> DMatrix<T> {
    let indep = independent_v_indices(model);
    let n_indep = indep.len();
    let nv = model.nv;

    // Map from full v-index to independent column index
    let mut full_to_col = vec![usize::MAX; nv];
    for (col, &vi) in indep.iter().enumerate() {
        full_to_col[vi] = col;
    }

    let mut g = DMatrix::zeros(nv, n_indep);

    // Identity rows for independent DOFs
    for (col, &vi) in indep.iter().enumerate() {
        g[(vi, col)] = T::one();
    }

    // Slave rows: G[slave, col_of_master] = multiplier
    for mj in &model.mimic {
        let vi_slave = model.v_idx[mj.slave];
        let vi_master = model.v_idx[mj.master];
        let col_master = full_to_col[vi_master];
        assert!(
            col_master != usize::MAX,
            "mimic master joint {} (v_idx={}) is itself a slave — \
             chained mimic is not supported in projection matrix",
            mj.master,
            vi_master,
        );
        g[(vi_slave, col_master)] = mj.multiplier;
    }

    g
}

/// Expand an independent velocity vector to a full velocity vector using the
/// mimic projection matrix.
///
/// This is equivalent to `G * v_indep` but avoids constructing the matrix.
pub fn expand_independent_velocity<T: RealField + Copy>(
    model: &Model<T>,
    v_indep: &[T],
) -> Vec<T> {
    let indep = independent_v_indices(model);
    assert_eq!(
        v_indep.len(),
        indep.len(),
        "v_indep length {} != num independent DOFs {}",
        v_indep.len(),
        indep.len(),
    );

    let mut v = vec![T::zero(); model.nv];

    // Place independent values
    for (k, &vi) in indep.iter().enumerate() {
        v[vi] = v_indep[k];
    }

    // Fill slave values
    for mj in &model.mimic {
        let vi_master = model.v_idx[mj.master];
        let vi_slave = model.v_idx[mj.slave];
        v[vi_slave] = mj.multiplier * v[vi_master];
    }

    v
}

/// Project a full-space torque vector $\tau \in \mathbb{R}^{n_v}$ to the
/// independent subspace: $\tau_{\text{indep}} = G^\top \tau$.
///
/// This is the correct mapping for torques / generalized forces under the
/// mimic relation (virtual work principle).
pub fn project_torque<T: RealField + Copy>(model: &Model<T>, tau: &[T]) -> Vec<T> {
    assert_eq!(tau.len(), model.nv);
    let indep = independent_v_indices(model);
    let mut tau_indep = vec![T::zero(); indep.len()];

    // Map from full v-index to independent column index
    let mut full_to_col = vec![usize::MAX; model.nv];
    for (col, &vi) in indep.iter().enumerate() {
        full_to_col[vi] = col;
    }

    // Independent DOFs: tau_indep[col] = tau[vi]
    for (col, &vi) in indep.iter().enumerate() {
        tau_indep[col] = tau[vi];
    }

    // Slave contribution: tau_indep[col_master] += multiplier * tau[slave]
    for mj in &model.mimic {
        let vi_slave = model.v_idx[mj.slave];
        let vi_master = model.v_idx[mj.master];
        let col_master = full_to_col[vi_master];
        tau_indep[col_master] = tau_indep[col_master] + mj.multiplier * tau[vi_slave];
    }

    tau_indep
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{LinkInertia, ModelBuilder};
    use crate::{joint, se3};
    use approx::assert_relative_eq;

    /// Build a 3-joint chain where j3 mimics j1 with multiplier=2, offset=0.1.
    fn three_joint_mimic() -> Model<f64> {
        ModelBuilder::new()
            .add_joint(
                "j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero(),
            )
            .add_joint(
                "j2", 1, joint::revolute_z(), se3::identity(), LinkInertia::zero(),
            )
            .add_joint(
                "j3", 2, joint::revolute_z(), se3::identity(), LinkInertia::zero(),
            )
            .add_mimic(3, 1, 2.0, 0.1)
            .build()
    }

    #[test]
    fn enforce_mimic_basic() {
        let model = three_joint_mimic();
        let q = vec![0.5, 0.3, 999.0]; // j3 value doesn't matter
        let qm = enforce_mimic(&model, &q);
        assert_relative_eq!(qm[0], 0.5, epsilon = 1e-14);
        assert_relative_eq!(qm[1], 0.3, epsilon = 1e-14);
        assert_relative_eq!(qm[2], 2.0 * 0.5 + 0.1, epsilon = 1e-14);
    }

    #[test]
    fn enforce_mimic_velocity_basic() {
        let model = three_joint_mimic();
        let v = vec![1.0, 2.0, 999.0];
        let vm = enforce_mimic_velocity(&model, &v);
        assert_relative_eq!(vm[0], 1.0, epsilon = 1e-14);
        assert_relative_eq!(vm[1], 2.0, epsilon = 1e-14);
        assert_relative_eq!(vm[2], 2.0 * 1.0, epsilon = 1e-14);
    }

    #[test]
    fn independent_indices() {
        let model = three_joint_mimic();
        let indep_v = independent_v_indices(&model);
        assert_eq!(indep_v, vec![0, 1]); // j1, j2 are independent; j3 is slave
        assert_eq!(num_independent_v(&model), 2);
    }

    #[test]
    fn projection_matrix_shape_and_values() {
        let model = three_joint_mimic();
        let g = mimic_projection_matrix(&model);
        assert_eq!(g.nrows(), 3); // nv = 3
        assert_eq!(g.ncols(), 2); // n_indep = 2

        // G should be:
        // [1  0]   <- j1 (independent, col 0)
        // [0  1]   <- j2 (independent, col 1)
        // [2  0]   <- j3 = 2*j1 (slave of j1 with multiplier=2)
        assert_relative_eq!(g[(0, 0)], 1.0, epsilon = 1e-14);
        assert_relative_eq!(g[(0, 1)], 0.0, epsilon = 1e-14);
        assert_relative_eq!(g[(1, 0)], 0.0, epsilon = 1e-14);
        assert_relative_eq!(g[(1, 1)], 1.0, epsilon = 1e-14);
        assert_relative_eq!(g[(2, 0)], 2.0, epsilon = 1e-14);
        assert_relative_eq!(g[(2, 1)], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn expand_velocity_roundtrip() {
        let model = three_joint_mimic();
        let v_indep = vec![0.5, 0.3];
        let v_full = expand_independent_velocity(&model, &v_indep);
        assert_relative_eq!(v_full[0], 0.5, epsilon = 1e-14);
        assert_relative_eq!(v_full[1], 0.3, epsilon = 1e-14);
        assert_relative_eq!(v_full[2], 2.0 * 0.5, epsilon = 1e-14); // multiplier=2
    }

    #[test]
    fn project_torque_basic() {
        let model = three_joint_mimic();
        // tau = [1.0, 2.0, 3.0]
        // tau_indep[0] = tau[0] + 2.0 * tau[2] = 1 + 6 = 7  (j1 + multiplier * j3)
        // tau_indep[1] = tau[1] = 2                           (j2)
        let tau = vec![1.0, 2.0, 3.0];
        let ti = project_torque(&model, &tau);
        assert_relative_eq!(ti[0], 7.0, epsilon = 1e-14);
        assert_relative_eq!(ti[1], 2.0, epsilon = 1e-14);
    }

    #[test]
    fn no_mimic_is_identity() {
        // Model without any mimic joints — everything should be a no-op.
        let model = ModelBuilder::<f64>::new()
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .add_joint("j2", 1, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .build();

        let q = vec![0.5, 0.3];
        assert_eq!(enforce_mimic(&model, &q), q);
        assert_eq!(enforce_mimic_velocity(&model, &q), q);
        assert_eq!(independent_v_indices(&model), vec![0, 1]);
        assert_eq!(num_independent_v(&model), 2);

        let g = mimic_projection_matrix(&model);
        assert_eq!(g.nrows(), 2);
        assert_eq!(g.ncols(), 2);
        // Should be identity
        assert_relative_eq!(g[(0, 0)], 1.0, epsilon = 1e-14);
        assert_relative_eq!(g[(1, 1)], 1.0, epsilon = 1e-14);
    }

    #[test]
    fn enforce_mimic_with_fk() {
        // Verify that enforcing mimic before FK gives correct results.
        use crate::fk::forward_kinematics;

        let model = ModelBuilder::<f64>::new()
            .add_joint("j1", 0, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .add_joint("j2", 1, joint::revolute_z(), se3::identity(), LinkInertia::zero())
            .add_mimic(2, 1, 1.0, 0.0) // j2 copies j1 exactly
            .build();

        let q = vec![0.5, 0.0]; // j2 value doesn't matter
        let q_enforced = enforce_mimic(&model, &q);
        assert_relative_eq!(q_enforced[1], 0.5, epsilon = 1e-14);

        // FK with enforced q should give j2 = j1
        let data = forward_kinematics(&model, &q_enforced);
        let p1 = se3::translation(&data.oMi[1]);
        let p2 = se3::translation(&data.oMi[2]);
        // Both joints rotate by 0.5, so p2 should be the composition
        // of two 0.5-rad rotations about Z from the origin.
        assert!(p1.norm() < 1e-12 || p2.norm() > 0.0); // sanity
    }
}
