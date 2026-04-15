//! Composite Rigid-Body Algorithm (CRBA) — joint-space inertia matrix.
//!
//! Computes the symmetric positive-definite mass matrix M(q) such that
//! the kinetic energy is T = ½ q̇ᵀ M(q) q̇.
//!
//! Algorithm (Featherstone 2008, §6.2 / Pinocchio `crba`):
//!
//! 1. **Forward pass**: compute body-frame transformations.
//! 2. **Backward pass**: accumulate composite rigid-body inertias from leaves
//!    to root.
//! 3. **Matrix fill**: project composite inertia onto motion subspaces.
//!
//! **Pure function**: `(model, q) → DMatrix`.  Generic over `T: RealField`.

use crate::model::Model;
use crate::se3;
use nalgebra::{DMatrix, Matrix6, RealField, Vector3, Vector6};

/// Compute the joint-space inertia (mass) matrix M(q).
///
/// Returns a symmetric `nv × nv` matrix.
///
/// # Panics
///
/// Panics if `q.len() != model.nq`.
pub fn crba<T: RealField>(model: &Model<T>, q: &[T]) -> DMatrix<T> {
    assert_eq!(q.len(), model.nq);

    let n = model.joints.len();

    // ── Forward pass: compute body-frame transformations ────────────────
    let mut x_j: Vec<se3::SE3<T>> = vec![se3::identity(); n];
    for i in 1..n {
        let joint = &model.joints[i];
        let qi = model.q_idx[i];
        let nq_j = joint.joint_type.nq();
        let q_slice = &q[qi..qi + nq_j];
        let m_j = joint.joint_type.forward(q_slice);
        x_j[i] = se3::compose(&joint.placement, &m_j);
    }

    // ── Build body spatial inertias ─────────────────────────────────────
    let mut ic: Vec<Matrix6<T>> = Vec::with_capacity(n);
    ic.push(Matrix6::zeros()); // universe
    for i in 1..n {
        let inertia = &model.inertias[i];
        ic.push(se3::spatial_inertia(
            inertia.mass.clone(),
            &inertia.center_of_mass,
            &inertia.rotational_inertia,
        ));
    }

    // ── Backward pass: composite inertias ───────────────────────────────
    for i in (1..n).rev() {
        let parent = model.joints[i].parent;
        if parent > 0 {
            let r = se3::rotation_matrix(&x_j[i]);
            let p = se3::translation(&x_j[i]);

            let ic_child = ic[i].clone();
            let transformed = transform_spatial_inertia(&ic_child, &r, &p);
            let ic_parent = &ic[parent] + &transformed;
            ic[parent] = ic_parent;
        }
    }

    // ── Fill mass matrix ────────────────────────────────────────────────
    let mut m = DMatrix::zeros(model.nv, model.nv);
    compute_crba_full(model, q, &x_j, &ic, &mut m);

    // Symmetrize: copy upper triangle to lower
    for r in 0..model.nv {
        for c in (r + 1)..model.nv {
            m[(c, r)] = m[(r, c)].clone();
        }
    }

    m
}

/// Internal: compute mass matrix using the standard CRBA pattern.
fn compute_crba_full<T: RealField>(
    model: &Model<T>,
    q: &[T],
    x_j: &[se3::SE3<T>],
    ic: &[Matrix6<T>],
    m: &mut DMatrix<T>,
) {
    let n = model.joints.len();

    // Clear the matrix first
    m.fill(T::zero());

    for i in 1..n {
        let joint_i = &model.joints[i];
        let vi = model.v_idx[i];
        let nv_i = joint_i.joint_type.nv();
        if nv_i == 0 {
            continue;
        }

        let qi = model.q_idx[i];
        let s_i = joint_i.joint_type.motion_subspace(&q[qi..qi + joint_i.joint_type.nq()]);

        // F = Ic_i * S_i  (each column is a 6-vector force)
        let mut f_cols: Vec<Vector6<T>> = Vec::with_capacity(nv_i);
        for c in 0..nv_i {
            f_cols.push(&ic[i] * s_i.column(c).into_owned());
        }

        // Diagonal block: M[i,i] = S_i^T * Ic_i * S_i
        for ci in 0..nv_i {
            for ri in ci..nv_i {
                let dot = s_i.column(ri).dot(&f_cols[ci]);
                m[(vi + ri, vi + ci)] = dot;
            }
        }

        // Off-diagonal: walk j from parent(i) to root, transforming F each step
        let mut j = joint_i.parent;
        // Transform F from frame i to frame parent(i) using x_j[i]
        let mut f_parent: Vec<Vector6<T>> = f_cols
            .iter()
            .map(|fc| transform_wrench(&x_j[i], fc))
            .collect();

        while j > 0 {
            let joint_j = &model.joints[j];
            let vj = model.v_idx[j];
            let nv_j = joint_j.joint_type.nv();

            if nv_j > 0 {
                let qj = model.q_idx[j];
                let s_j = joint_j.joint_type.motion_subspace(
                    &q[qj..qj + joint_j.joint_type.nq()],
                );

                // M[j, i] = S_j^T * F_parent
                for ci in 0..nv_i {
                    for rj in 0..nv_j {
                        let dot = s_j.column(rj).dot(&f_parent[ci]);
                        m[(vj + rj, vi + ci)] = dot;
                    }
                }
            }

            // Transform F from j's frame to j's parent frame
            f_parent = f_parent
                .iter()
                .map(|fc| transform_wrench(&x_j[j], fc))
                .collect();

            j = joint_j.parent;
        }
    }
}

/// Transform a wrench from child frame to parent frame.
///
/// Given the placement `parent_X_child` (child frame expressed in parent frame),
/// the force transform is:
///
/// ```text
///   f_parent = X^{*} f_child
///   where X^* = [ R    [p]×R ]
///               [ 0       R  ]
/// ```
fn transform_wrench<T: RealField>(parent_x_child: &se3::SE3<T>, f: &Vector6<T>) -> Vector6<T> {
    let r = se3::rotation_matrix(parent_x_child);
    let p = se3::translation(parent_x_child);

    let f_ang = Vector3::new(f[0].clone(), f[1].clone(), f[2].clone());
    let f_lin = Vector3::new(f[3].clone(), f[4].clone(), f[5].clone());

    let r_f_lin = &r * &f_lin;
    let r_f_ang = &r * &f_ang;
    let f_p_ang = &r_f_ang + p.cross(&r_f_lin);

    Vector6::new(
        f_p_ang[0].clone(),
        f_p_ang[1].clone(),
        f_p_ang[2].clone(),
        r_f_lin[0].clone(),
        r_f_lin[1].clone(),
        r_f_lin[2].clone(),
    )
}

/// Transform a 6×6 spatial inertia from child frame to parent frame.
///
/// Given placement `parent_X_child`, compute:
///   Ic_parent = X^{-T} Ic_child X^{-1}
fn transform_spatial_inertia<T: RealField>(
    ic: &Matrix6<T>,
    r: &nalgebra::Matrix3<T>,
    p: &Vector3<T>,
) -> Matrix6<T> {
    // Build the 6×6 motion transform X (parent_X_child):
    //   X = [ R    0   ]
    //       [ [p]×R  R ]
    // X^{-1} = [ R^T         0    ]
    //          [ -R^T [p]×   R^T  ]
    //
    // For inertia: I_parent = X^{*T} I_child X^{-1}
    //            = X^{-T} I_child X^{-1}
    // But it's simpler to compute using the full 6×6 transform.
    let px = se3::skew(p);
    let rt = r.transpose();

    // Build X^{-1}
    let mut x_inv = Matrix6::<T>::zeros();
    x_inv.fixed_view_mut::<3, 3>(0, 0).copy_from(&rt);
    let neg_rt_px = -&rt * &px;
    x_inv.fixed_view_mut::<3, 3>(3, 0).copy_from(&neg_rt_px);
    x_inv.fixed_view_mut::<3, 3>(3, 3).copy_from(&rt);

    // X^{-T}
    let x_inv_t = x_inv.transpose();

    // I_parent = X^{-T} * I_child * X^{-1}
    &x_inv_t * ic * &x_inv
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint;
    use crate::model::{LinkInertia, ModelBuilder};
    use crate::rnea;
    use approx::assert_relative_eq;
    use nalgebra::{DVector, Matrix3, Vector3};

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
                "j1",
                0,
                joint::revolute_y(),
                offset1,
                LinkInertia {
                    mass: 2.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                    rotational_inertia: Matrix3::new(
                        0.2, 0.0, 0.0,
                        0.0, 0.2, 0.0,
                        0.0, 0.0, 0.02,
                    ),
                },
            )
            .add_joint(
                "j2",
                1,
                joint::revolute_y(),
                offset2,
                LinkInertia {
                    mass: 1.0,
                    center_of_mass: Vector3::new(0.0, 0.0, -0.5),
                    rotational_inertia: Matrix3::new(
                        0.1, 0.0, 0.0,
                        0.0, 0.1, 0.0,
                        0.0, 0.0, 0.01,
                    ),
                },
            )
            .build()
    }

    #[test]
    fn crba_is_symmetric() {
        let model = two_link_arm();
        let q = vec![0.3, -0.5];
        let m = crba(&model, &q);
        assert_relative_eq!(m[(0, 1)], m[(1, 0)], epsilon = 1e-12);
    }

    #[test]
    fn crba_is_positive_definite() {
        let model = two_link_arm();
        let q = vec![0.3, -0.5];
        let m = crba(&model, &q);
        // Check eigenvalues are positive
        let eig = nalgebra::SymmetricEigen::new(m);
        for ev in eig.eigenvalues.iter() {
            assert!(*ev > 0.0, "eigenvalue {} is not positive", ev);
        }
    }

    #[test]
    fn crba_consistent_with_rnea() {
        // M(q) * a = rnea(q, 0, a) - g(q)
        let model = two_link_arm();
        let q = vec![0.3, -0.5];
        let zero = vec![0.0, 0.0];
        let a = vec![1.0, 0.0]; // unit acceleration on joint 1

        let m = crba(&model, &q);
        let g = rnea::compute_gravity(&model, &q);
        let tau = rnea::rnea(&model, &q, &zero, &a);

        let ma: DVector<f64> = (&m * DVector::from_column_slice(&a)).column(0).into_owned();
        let expected = &tau - &g;

        assert_relative_eq!(ma, expected, epsilon = 1e-10);
    }

    #[test]
    fn crba_consistent_with_rnea_joint2() {
        let model = two_link_arm();
        let q = vec![0.3, -0.5];
        let zero = vec![0.0, 0.0];
        let a = vec![0.0, 1.0]; // unit acceleration on joint 2

        let m = crba(&model, &q);
        let g = rnea::compute_gravity(&model, &q);
        let tau = rnea::rnea(&model, &q, &zero, &a);

        let ma: DVector<f64> = (&m * DVector::from_column_slice(&a)).column(0).into_owned();
        let expected = &tau - &g;

        assert_relative_eq!(ma, expected, epsilon = 1e-10);
    }

    #[test]
    fn crba_is_pure() {
        let model = two_link_arm();
        let q = vec![0.3, -0.5];
        let m1 = crba(&model, &q);
        let m2 = crba(&model, &q);
        assert_relative_eq!(m1, m2, epsilon = 1e-14);
    }
}
