//! Analytical derivatives of RNEA (inverse dynamics).
//!
//! Computes ∂τ/∂q, ∂τ/∂v, ∂τ/∂a for the Recursive Newton-Euler Algorithm.
//!
//! Reference: Carpentier & Mansard, "Analytical Derivatives of Rigid Body
//! Dynamics Algorithms", RSS 2018.
//!
//! The algorithm is structured as:
//! 1. **Forward pass** — standard RNEA forward pass, storing intermediate quantities.
//! 2. **Backward pass** — accumulate composite rigid-body inertias.
//! 3. **Derivative pass** — for each joint k, forward-propagate perturbations through
//!    the subtree, backward-accumulate force perturbations, then propagate wrenches
//!    to ancestors.
//!
//! **Pure function**: `(model, q, v, a) → RneaDerivatives`.

use crate::model::Model;
use crate::se3;
use nalgebra::{DMatrix, Matrix6, RealField, Vector3, Vector6};

/// Output of RNEA analytical derivatives.
#[derive(Debug, Clone)]
pub struct RneaDerivatives<T: RealField> {
    /// ∂τ/∂q — nv × nv matrix.
    pub dtau_dq: DMatrix<T>,
    /// ∂τ/∂v — nv × nv matrix.
    pub dtau_dv: DMatrix<T>,
    /// ∂τ/∂a — nv × nv matrix (equals the mass matrix M(q)).
    pub dtau_da: DMatrix<T>,
}

/// Compute the analytical derivatives of RNEA.
///
/// Returns ∂τ/∂q, ∂τ/∂v, ∂τ/∂a where τ = RNEA(q, v, a).
///
/// Note: ∂τ/∂a = M(q) (the mass matrix).
///
/// # Arguments
///
/// * `model` — robot model
/// * `q`     — configuration vector (length `nq`)
/// * `v`     — velocity vector (length `nv`)
/// * `a`     — acceleration vector (length `nv`)
pub fn compute_rnea_derivatives<T: RealField>(
    model: &Model<T>,
    q: &[T],
    v: &[T],
    a: &[T],
) -> RneaDerivatives<T> {
    assert_eq!(q.len(), model.nq);
    assert_eq!(v.len(), model.nv);
    assert_eq!(a.len(), model.nv);

    let n = model.joints.len(); // includes universe at 0
    let nv = model.nv;

    // ── Per-body storage ────────────────────────────────────────────────
    let mut x_j: Vec<se3::SE3<T>> = vec![se3::identity(); n];
    let mut vel: Vec<Vector6<T>> = vec![Vector6::zeros(); n];
    let mut acc: Vec<Vector6<T>> = vec![Vector6::zeros(); n];
    let mut s_store: Vec<nalgebra::Matrix6xX<T>> = Vec::with_capacity(n);
    for _ in 0..n {
        s_store.push(nalgebra::Matrix6xX::zeros(0));
    }

    // Body spatial inertia
    let mut y_body: Vec<Matrix6<T>> = vec![Matrix6::zeros(); n];

    // Quantities needed for derivatives
    let mut v_pa_in_body: Vec<Vector6<T>> = vec![Vector6::zeros(); n];
    let mut a_pa_in_body: Vec<Vector6<T>> = vec![Vector6::zeros(); n];
    let mut v_joint: Vec<Vector6<T>> = vec![Vector6::zeros(); n];

    // Gravity as spatial acceleration of the universe (Featherstone trick)
    let mut a0 = Vector6::<T>::zeros();
    a0[3] = -model.gravity[0].clone();
    a0[4] = -model.gravity[1].clone();
    a0[5] = -model.gravity[2].clone();

    // ════════════════════════════════════════════════════════════════════
    // Pass 1: Forward — compute velocities, accelerations, forces
    // ════════════════════════════════════════════════════════════════════

    for i in 1..n {
        let joint = &model.joints[i];
        let qi = model.q_idx[i];
        let vi = model.v_idx[i];
        let nq_j = joint.joint_type.nq();
        let nv_j = joint.joint_type.nv();
        let q_slice = &q[qi..qi + nq_j];

        let m_j = joint.joint_type.forward(q_slice);
        x_j[i] = se3::compose(&joint.placement, &m_j);

        let r = se3::rotation_matrix(&x_j[i]);
        let p = se3::translation(&x_j[i]);
        let rt = r.transpose();

        let parent = joint.parent;
        let v_parent = &vel[parent];
        let omega_parent = Vector3::new(
            v_parent[0].clone(), v_parent[1].clone(), v_parent[2].clone(),
        );
        let vlin_parent = Vector3::new(
            v_parent[3].clone(), v_parent[4].clone(), v_parent[5].clone(),
        );
        let omega_body = &rt * &omega_parent;
        let vlin_body = &rt * (vlin_parent - p.cross(&omega_parent));
        v_pa_in_body[i] = Vector6::new(
            omega_body[0].clone(), omega_body[1].clone(), omega_body[2].clone(),
            vlin_body[0].clone(), vlin_body[1].clone(), vlin_body[2].clone(),
        );

        let s = joint.joint_type.motion_subspace(q_slice);

        let mut v_j = Vector6::<T>::zeros();
        for c in 0..nv_j {
            let qd_c = v[vi + c].clone();
            for row in 0..6 {
                v_j[row] += s[(row, c)].clone() * qd_c.clone();
            }
        }
        v_joint[i] = v_j.clone();
        vel[i] = &v_pa_in_body[i] + &v_j;

        let a_parent = if parent == 0 { &a0 } else { &acc[parent] };
        let omega_a = Vector3::new(
            a_parent[0].clone(), a_parent[1].clone(), a_parent[2].clone(),
        );
        let vlin_a = Vector3::new(
            a_parent[3].clone(), a_parent[4].clone(), a_parent[5].clone(),
        );
        let alpha_body = &rt * &omega_a;
        let alin_body = &rt * (vlin_a - p.cross(&omega_a));
        a_pa_in_body[i] = Vector6::new(
            alpha_body[0].clone(), alpha_body[1].clone(), alpha_body[2].clone(),
            alin_body[0].clone(), alin_body[1].clone(), alin_body[2].clone(),
        );

        let mut a_j = Vector6::<T>::zeros();
        for c in 0..nv_j {
            let qdd_c = a[vi + c].clone();
            for row in 0..6 {
                a_j[row] += s[(row, c)].clone() * qdd_c.clone();
            }
        }

        let vx = se3::motion_cross(&vel[i]);
        let cross_term = vx * &v_j;
        acc[i] = &a_pa_in_body[i] + &a_j + &cross_term;

        let inertia = &model.inertias[i];
        y_body[i] = se3::spatial_inertia(
            inertia.mass.clone(),
            &inertia.center_of_mass,
            &inertia.rotational_inertia,
        );

        s_store[i] = s;
    }

    // Compute body-level forces
    let mut f: Vec<Vector6<T>> = vec![Vector6::zeros(); n];
    for i in 1..n {
        let y_a = &y_body[i] * &acc[i];
        let y_v = &y_body[i] * &vel[i];
        let vxstar = se3::force_cross(&vel[i]);
        f[i] = y_a + vxstar * y_v;
    }

    // ════════════════════════════════════════════════════════════════════
    // Pass 2: Backward — composite inertias + force accumulation
    // ════════════════════════════════════════════════════════════════════

    let mut y_c: Vec<Matrix6<T>> = y_body.clone();

    for i in (1..n).rev() {
        let parent = model.joints[i].parent;
        if parent > 0 {
            let r = se3::rotation_matrix(&x_j[i]);
            let p = se3::translation(&x_j[i]);

            // Composite inertia
            let y_c_transformed = transform_inertia_to_parent(&y_c[i], &r, &p);
            y_c[parent] = &y_c[parent] + &y_c_transformed;

            // Backward force accumulation: f_parent += X_i^* f_i
            let f_to_parent = transform_wrench_to_parent(&r, &p, &f[i]);
            f[parent] = &f[parent] + &f_to_parent;
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // Pass 3: Derivative computation (column by column)
    // ════════════════════════════════════════════════════════════════════
    //
    // For each joint k, for each DoF c of joint k:
    //   1. Compute initial perturbation (δv, δa) at body k
    //   2. Forward-propagate through the subtree of k (δv, δa at each descendant)
    //   3. Compute δf at each body in the subtree using body inertia Y_body
    //   4. Backward-accumulate δf within subtree
    //   5. Project onto S for each joint in subtree → dtau entries
    //   6. Walk up ancestors: just transform wrench (no body-level terms)

    let mut dtau_dq = DMatrix::zeros(nv, nv);
    let mut dtau_dv = DMatrix::zeros(nv, nv);
    let mut dtau_da = DMatrix::zeros(nv, nv);

    // ∂τ/∂a = M(q) via CRBA pattern
    compute_dtau_da(model, &x_j, &y_c, &s_store, &mut dtau_da);

    // For each joint k
    for k in 1..n {
        let joint_k = &model.joints[k];
        let vk = model.v_idx[k];
        let nv_k = joint_k.joint_type.nv();
        if nv_k == 0 {
            continue;
        }

        // Identify subtree of k (k and all descendants)
        let mut in_subtree = vec![false; n];
        in_subtree[k] = true;
        for j in (k + 1)..n {
            if in_subtree[model.joints[j].parent] {
                in_subtree[j] = true;
            }
        }

        for c in 0..nv_k {
            let s_col = s_store[k].column(c).into_owned();

            // ── Initial perturbation at body k ──────────────────────
            // ∂/∂qₖ:  δvₖ = −[Sₖ]× v_{pa→k},  δaₖ = −[Sₖ]× a_{pa→k} + [δvₖ]× v_{J,k}
            let s_cross = se3::motion_cross(&s_col);
            let dv_dq_init = -(s_cross.clone() * &v_pa_in_body[k]);
            let da_dq_init = -(s_cross * &a_pa_in_body[k])
                + se3::motion_cross(&dv_dq_init) * &v_joint[k];

            // ∂/∂q̇ₖ:  δvₖ = Sₖ,  δaₖ = [δvₖ]× v_{J,k} + [vₖ]× δvₖ
            let dv_dv_init = s_col.clone();
            let da_dv_init = se3::motion_cross(&dv_dv_init) * &v_joint[k]
                + se3::motion_cross(&vel[k]) * &dv_dv_init;

            // Per-body perturbation arrays
            let mut dv_dq: Vec<Vector6<T>> = vec![Vector6::zeros(); n];
            let mut da_dq: Vec<Vector6<T>> = vec![Vector6::zeros(); n];
            let mut df_dq: Vec<Vector6<T>> = vec![Vector6::zeros(); n];
            let mut dv_dv: Vec<Vector6<T>> = vec![Vector6::zeros(); n];
            let mut da_dv: Vec<Vector6<T>> = vec![Vector6::zeros(); n];
            let mut df_dv: Vec<Vector6<T>> = vec![Vector6::zeros(); n];

            dv_dq[k] = dv_dq_init;
            da_dq[k] = da_dq_init;
            dv_dv[k] = dv_dv_init;
            da_dv[k] = da_dv_init;

            // ── Forward propagate + compute δf ──────────────────────
            for j in k..n {
                if !in_subtree[j] {
                    continue;
                }

                // Propagate from parent (for j > k)
                if j > k {
                    let r = se3::rotation_matrix(&x_j[j]);
                    let p = se3::translation(&x_j[j]);
                    let rt = r.transpose();
                    let pa = model.joints[j].parent;

                    dv_dq[j] = transform_motion_to_child(&rt, &p, &dv_dq[pa]);
                    dv_dv[j] = transform_motion_to_child(&rt, &p, &dv_dv[pa]);

                    da_dq[j] = transform_motion_to_child(&rt, &p, &da_dq[pa])
                        + se3::motion_cross(&dv_dq[j]) * &v_joint[j];
                    da_dv[j] = transform_motion_to_child(&rt, &p, &da_dv[pa])
                        + se3::motion_cross(&dv_dv[j]) * &v_joint[j];
                }

                // δf_j = Y_j δa_j + [δv_j]×* (Y_j v_j) + [v_j]×* (Y_j δv_j)
                let y_j_vj = &y_body[j] * &vel[j];
                df_dq[j] = &y_body[j] * &da_dq[j]
                    + se3::force_cross(&dv_dq[j]) * &y_j_vj
                    + se3::force_cross(&vel[j]) * (&y_body[j] * &dv_dq[j]);
                df_dv[j] = &y_body[j] * &da_dv[j]
                    + se3::force_cross(&dv_dv[j]) * &y_j_vj
                    + se3::force_cross(&vel[j]) * (&y_body[j] * &dv_dv[j]);
            }

            // ── Backward accumulate within subtree (leaves → k) ─────
            for j in ((k + 1)..n).rev() {
                if !in_subtree[j] {
                    continue;
                }
                let pa = model.joints[j].parent;
                let r = se3::rotation_matrix(&x_j[j]);
                let p = se3::translation(&x_j[j]);

                let df_dq_pa = transform_wrench_to_parent(&r, &p, &df_dq[j]);
                let df_dv_pa = transform_wrench_to_parent(&r, &p, &df_dv[j]);
                df_dq[pa] = &df_dq[pa] + &df_dq_pa;
                df_dv[pa] = &df_dv[pa] + &df_dv_pa;
            }

            // ── Project dtau for joints in subtree ──────────────────
            for j in k..n {
                if !in_subtree[j] {
                    continue;
                }
                let vj = model.v_idx[j];
                let nv_j = model.joints[j].joint_type.nv();
                for rr in 0..nv_j {
                    dtau_dq[(vj + rr, vk + c)] = s_store[j].column(rr).dot(&df_dq[j]);
                    dtau_dv[(vj + rr, vk + c)] = s_store[j].column(rr).dot(&df_dv[j]);
                }
            }

            // ── Walk up ancestors ────────────────────────────────
            // For ∂/∂qₖ: the transform X_k^* depends on q_k, giving an
            // extra term: ∂(X_k^* f_k)/∂q_k = X_k^* ([S_k]^{×*} f_total_k)
            // This must be added before transforming to the parent frame.
            let s_col_cross_star = se3::force_cross(&s_col);
            let mut df_dq_up = &df_dq[k] + &(s_col_cross_star * &f[k]);
            let mut df_dv_up = df_dv[k].clone();
            let mut j = k;
            while model.joints[j].parent > 0 {
                let pa = model.joints[j].parent;
                let r = se3::rotation_matrix(&x_j[j]);
                let p = se3::translation(&x_j[j]);

                df_dq_up = transform_wrench_to_parent(&r, &p, &df_dq_up);
                df_dv_up = transform_wrench_to_parent(&r, &p, &df_dv_up);

                let vpa = model.v_idx[pa];
                let nv_pa = model.joints[pa].joint_type.nv();
                for rr in 0..nv_pa {
                    dtau_dq[(vpa + rr, vk + c)] = s_store[pa].column(rr).dot(&df_dq_up);
                    dtau_dv[(vpa + rr, vk + c)] = s_store[pa].column(rr).dot(&df_dv_up);
                }

                j = pa;
            }
        }
    }

    RneaDerivatives {
        dtau_dq,
        dtau_dv,
        dtau_da,
    }
}

// ─── ∂τ/∂a = M(q) via CRBA pattern ─────────────────────────────────────────

fn compute_dtau_da<T: RealField>(
    model: &Model<T>,
    x_j: &[se3::SE3<T>],
    y_c: &[Matrix6<T>],
    s_store: &[nalgebra::Matrix6xX<T>],
    m: &mut DMatrix<T>,
) {
    let n = model.joints.len();

    for i in 1..n {
        let joint_i = &model.joints[i];
        let vi = model.v_idx[i];
        let nv_i = joint_i.joint_type.nv();
        if nv_i == 0 {
            continue;
        }

        let mut f_cols: Vec<Vector6<T>> = Vec::with_capacity(nv_i);
        for c_idx in 0..nv_i {
            f_cols.push(&y_c[i] * s_store[i].column(c_idx).into_owned());
        }

        for ci in 0..nv_i {
            for ri in 0..nv_i {
                m[(vi + ri, vi + ci)] = s_store[i].column(ri).dot(&f_cols[ci]);
            }
        }

        let mut f_parent: Vec<Vector6<T>> = f_cols
            .iter()
            .map(|fc| {
                let r = se3::rotation_matrix(&x_j[i]);
                let p = se3::translation(&x_j[i]);
                transform_wrench_to_parent(&r, &p, fc)
            })
            .collect();

        let mut j = joint_i.parent;
        while j > 0 {
            let joint_j = &model.joints[j];
            let vj = model.v_idx[j];
            let nv_j = joint_j.joint_type.nv();

            if nv_j > 0 {
                for ci in 0..nv_i {
                    for rj in 0..nv_j {
                        let dot = s_store[j].column(rj).dot(&f_parent[ci]);
                        m[(vj + rj, vi + ci)] = dot.clone();
                        m[(vi + ci, vj + rj)] = dot;
                    }
                }
            }

            f_parent = f_parent
                .iter()
                .map(|fc| {
                    let r = se3::rotation_matrix(&x_j[j]);
                    let p = se3::translation(&x_j[j]);
                    transform_wrench_to_parent(&r, &p, fc)
                })
                .collect();

            j = joint_j.parent;
        }
    }
}

// ─── Spatial algebra helpers ────────────────────────────────────────────────

/// Transform a wrench from child frame to parent frame: f_pa = X^{*} f_ch.
fn transform_wrench_to_parent<T: RealField>(
    r: &nalgebra::Matrix3<T>,
    p: &Vector3<T>,
    f: &Vector6<T>,
) -> Vector6<T> {
    let f_ang = Vector3::new(f[0].clone(), f[1].clone(), f[2].clone());
    let f_lin = Vector3::new(f[3].clone(), f[4].clone(), f[5].clone());
    let r_f_lin = r * &f_lin;
    let r_f_ang = r * &f_ang;
    let f_p_ang = &r_f_ang + p.cross(&r_f_lin);
    Vector6::new(
        f_p_ang[0].clone(), f_p_ang[1].clone(), f_p_ang[2].clone(),
        r_f_lin[0].clone(), r_f_lin[1].clone(), r_f_lin[2].clone(),
    )
}

/// Transform a motion from parent frame to child frame: m_ch = X⁻¹ m_pa.
fn transform_motion_to_child<T: RealField>(
    rt: &nalgebra::Matrix3<T>,
    p: &Vector3<T>,
    m: &Vector6<T>,
) -> Vector6<T> {
    let omega = Vector3::new(m[0].clone(), m[1].clone(), m[2].clone());
    let vlin = Vector3::new(m[3].clone(), m[4].clone(), m[5].clone());
    let omega_ch = rt * &omega;
    let vlin_ch = rt * (vlin - p.cross(&omega));
    Vector6::new(
        omega_ch[0].clone(), omega_ch[1].clone(), omega_ch[2].clone(),
        vlin_ch[0].clone(), vlin_ch[1].clone(), vlin_ch[2].clone(),
    )
}

/// Transform a 6×6 spatial inertia from child to parent: I_pa = X^{-T} I_ch X^{-1}.
fn transform_inertia_to_parent<T: RealField>(
    ic: &Matrix6<T>,
    r: &nalgebra::Matrix3<T>,
    p: &Vector3<T>,
) -> Matrix6<T> {
    let px = se3::skew(p);
    let rt = r.transpose();
    let mut x_inv = Matrix6::<T>::zeros();
    x_inv.fixed_view_mut::<3, 3>(0, 0).copy_from(&rt);
    let neg_rt_px = -&rt * &px;
    x_inv.fixed_view_mut::<3, 3>(3, 0).copy_from(&neg_rt_px);
    x_inv.fixed_view_mut::<3, 3>(3, 3).copy_from(&rt);
    let x_inv_t = x_inv.transpose();
    &x_inv_t * ic * &x_inv
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crba::crba;
    use crate::joint;
    use crate::model::{LinkInertia, ModelBuilder};
    use crate::rnea::rnea;
    use approx::assert_relative_eq;
    use nalgebra::{Matrix3, Vector3};

    fn simple_pendulum() -> Model<f64> {
        let offset = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.5),
        );
        ModelBuilder::new()
            .add_joint(
                "joint1", 0, joint::revolute_y(), offset,
                LinkInertia {
                    mass: 1.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                    rotational_inertia: Matrix3::new(
                        0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.01,
                    ),
                },
            )
            .build()
    }

    fn two_link_arm() -> Model<f64> {
        let offset1 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.5),
        );
        let offset2 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -1.0),
        );
        ModelBuilder::new()
            .add_joint(
                "j1", 0, joint::revolute_y(), offset1,
                LinkInertia {
                    mass: 2.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                    rotational_inertia: Matrix3::new(
                        0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.02,
                    ),
                },
            )
            .add_joint(
                "j2", 1, joint::revolute_y(), offset2,
                LinkInertia {
                    mass: 1.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                    rotational_inertia: Matrix3::new(
                        0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.01,
                    ),
                },
            )
            .build()
    }

    fn three_link_arm() -> Model<f64> {
        let offset1 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.3),
        );
        let offset2 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.5),
        );
        let offset3 = se3::from_rotation_and_translation(
            &nalgebra::Rotation3::identity(),
            &Vector3::new(0.0, 0.0, -0.4),
        );
        ModelBuilder::new()
            .add_joint(
                "j1", 0, joint::revolute_y(), offset1,
                LinkInertia {
                    mass: 3.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.25),
                    rotational_inertia: Matrix3::new(
                        0.3, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.03,
                    ),
                },
            )
            .add_joint(
                "j2", 1, joint::revolute_z(), offset2,
                LinkInertia {
                    mass: 2.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.2),
                    rotational_inertia: Matrix3::new(
                        0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.02,
                    ),
                },
            )
            .add_joint(
                "j3", 2, joint::revolute_x(), offset3,
                LinkInertia {
                    mass: 1.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.15),
                    rotational_inertia: Matrix3::new(
                        0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.01,
                    ),
                },
            )
            .build()
    }

    fn fd_dtau_dq(model: &Model<f64>, q: &[f64], v: &[f64], a: &[f64], eps: f64) -> DMatrix<f64> {
        let nv = model.nv;
        let mut result = DMatrix::zeros(nv, nv);
        for k in 0..nv {
            let mut q_p = q.to_vec();
            let mut q_m = q.to_vec();
            q_p[k] += eps;
            q_m[k] -= eps;
            let tau_p = rnea(model, &q_p, v, a);
            let tau_m = rnea(model, &q_m, v, a);
            let col = (&tau_p - &tau_m) / (2.0 * eps);
            for r in 0..nv { result[(r, k)] = col[r]; }
        }
        result
    }

    fn fd_dtau_dv(model: &Model<f64>, q: &[f64], v: &[f64], a: &[f64], eps: f64) -> DMatrix<f64> {
        let nv = model.nv;
        let mut result = DMatrix::zeros(nv, nv);
        for k in 0..nv {
            let mut v_p = v.to_vec();
            let mut v_m = v.to_vec();
            v_p[k] += eps;
            v_m[k] -= eps;
            let tau_p = rnea(model, q, &v_p, a);
            let tau_m = rnea(model, q, &v_m, a);
            let col = (&tau_p - &tau_m) / (2.0 * eps);
            for r in 0..nv { result[(r, k)] = col[r]; }
        }
        result
    }

    #[test]
    fn dtau_da_equals_mass_matrix_pendulum() {
        let model = simple_pendulum();
        let derivs = compute_rnea_derivatives(&model, &[0.3], &[1.0], &[0.5]);
        let m = crba(&model, &[0.3]);
        assert_relative_eq!(derivs.dtau_da, m, epsilon = 1e-10);
    }

    #[test]
    fn dtau_da_equals_mass_matrix_two_link() {
        let model = two_link_arm();
        let derivs = compute_rnea_derivatives(&model, &[0.3, -0.5], &[1.0, -0.5], &[2.0, -1.0]);
        let m = crba(&model, &[0.3, -0.5]);
        assert_relative_eq!(derivs.dtau_da, m, epsilon = 1e-10);
    }

    #[test]
    fn dtau_dq_pendulum_vs_fd() {
        let model = simple_pendulum();
        let q = [0.3]; let v = [1.0]; let a = [0.5];
        let derivs = compute_rnea_derivatives(&model, &q, &v, &a);
        let fd = fd_dtau_dq(&model, &q, &v, &a, 1e-7);
        assert_relative_eq!(derivs.dtau_dq, fd, epsilon = 1e-5);
    }

    #[test]
    fn dtau_dq_two_link_vs_fd() {
        let model = two_link_arm();
        let q = [0.3, -0.5]; let v = [1.0, -0.5]; let a = [2.0, -1.0];
        let derivs = compute_rnea_derivatives(&model, &q, &v, &a);
        let fd = fd_dtau_dq(&model, &q, &v, &a, 1e-7);
        assert_relative_eq!(derivs.dtau_dq, fd, epsilon = 1e-4);
    }

    #[test]
    fn dtau_dq_three_link_vs_fd() {
        let model = three_link_arm();
        let q = [0.4, -0.3, 0.2]; let v = [1.0, -0.5, 0.8]; let a = [2.0, -1.0, 0.3];
        let derivs = compute_rnea_derivatives(&model, &q, &v, &a);
        let fd = fd_dtau_dq(&model, &q, &v, &a, 1e-7);
        assert_relative_eq!(derivs.dtau_dq, fd, epsilon = 1e-4);
    }

    #[test]
    fn dtau_dv_pendulum_vs_fd() {
        let model = simple_pendulum();
        let q = [0.3]; let v = [1.0]; let a = [0.5];
        let derivs = compute_rnea_derivatives(&model, &q, &v, &a);
        let fd = fd_dtau_dv(&model, &q, &v, &a, 1e-7);
        assert_relative_eq!(derivs.dtau_dv, fd, epsilon = 1e-5);
    }

    #[test]
    fn dtau_dv_two_link_vs_fd() {
        let model = two_link_arm();
        let q = [0.3, -0.5]; let v = [1.0, -0.5]; let a = [2.0, -1.0];
        let derivs = compute_rnea_derivatives(&model, &q, &v, &a);
        let fd = fd_dtau_dv(&model, &q, &v, &a, 1e-7);
        assert_relative_eq!(derivs.dtau_dv, fd, epsilon = 1e-4);
    }

    #[test]
    fn dtau_dv_three_link_vs_fd() {
        let model = three_link_arm();
        let q = [0.4, -0.3, 0.2]; let v = [1.0, -0.5, 0.8]; let a = [2.0, -1.0, 0.3];
        let derivs = compute_rnea_derivatives(&model, &q, &v, &a);
        let fd = fd_dtau_dv(&model, &q, &v, &a, 1e-7);
        assert_relative_eq!(derivs.dtau_dv, fd, epsilon = 1e-4);
    }

    #[test]
    fn dtau_dq_gravity_only() {
        let model = simple_pendulum();
        let derivs = compute_rnea_derivatives(&model, &[0.3], &[0.0], &[0.0]);
        let fd = fd_dtau_dq(&model, &[0.3], &[0.0], &[0.0], 1e-7);
        assert_relative_eq!(derivs.dtau_dq, fd, epsilon = 1e-5);
    }

    #[test]
    fn dtau_dv_zero_velocity() {
        let model = two_link_arm();
        let q = [0.3, -0.5]; let z = [0.0, 0.0];
        let derivs = compute_rnea_derivatives(&model, &q, &z, &z);
        let fd = fd_dtau_dv(&model, &q, &z, &z, 1e-7);
        assert_relative_eq!(derivs.dtau_dv, fd, epsilon = 1e-5);
    }
}
