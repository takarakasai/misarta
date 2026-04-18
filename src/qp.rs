//! Dense quadratic programming solver — primal active-set method.
//!
//! Solves problems of the form:
//!
//! $$\min_x \frac{1}{2} x^T H x + c^T x$$
//!
//! subject to:
//! - $A_{eq}\, x = b_{eq}$  (equality constraints)
//! - $A_{iq}\, x \le b_{iq}$  (inequality constraints)
//!
//! The Hessian $H$ must be positive definite (a tiny diagonal regularisation
//! is added automatically when the Cholesky factorisation fails).
//! The solver uses a **primal active-set algorithm** that is efficient for the
//! small, dense QPs (n ≤ 50) that arise in constrained inverse kinematics.
//!
//! # Example
//!
//! ```
//! use nalgebra::{DMatrix, DVector};
//! use misarta::qp::{solve_qp, QpConfig, QpStatus};
//!
//! // min 0.5*((x₁−2)² + (x₂−2)²)  s.t.  x₁ ≤ 1, x₂ ≤ 1
//! let h = DMatrix::identity(2, 2);
//! let c = DVector::from_vec(vec![-2.0, -2.0]); // c = -[2, 2]
//! let a_iq = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
//! let b_iq = DVector::from_vec(vec![1.0, 1.0]);
//!
//! let sol = solve_qp(&h, &c, None, None,
//!                    Some(&a_iq), Some(&b_iq), None, &QpConfig::default());
//! assert_eq!(sol.status, QpStatus::Optimal);
//! assert!((sol.x[0] - 1.0).abs() < 1e-6);
//! assert!((sol.x[1] - 1.0).abs() < 1e-6);
//! ```

use nalgebra::{DMatrix, DVector};

// ─── Public types ───────────────────────────────────────────────────────────

/// Status of the QP solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QpStatus {
    /// KKT conditions satisfied within tolerance.
    Optimal,
    /// Active-set iteration limit exceeded.
    MaxIterations,
    /// No feasible point could be found.
    Infeasible,
    /// A singular matrix or other numerical issue was encountered.
    NumericalFailure,
}

/// Solution returned by [`solve_qp`].
#[derive(Debug, Clone)]
pub struct QpSolution {
    /// Optimal (or best-found) decision variable.
    pub x: DVector<f64>,
    /// Objective value $\frac{1}{2} x^T H x + c^T x$.
    pub objective: f64,
    /// Lagrange multipliers for equality constraints (length `m_eq`).
    pub lambda_eq: DVector<f64>,
    /// Lagrange multipliers for inequality constraints (length `m_iq`).
    /// Non-zero only for active inequalities at the solution.
    pub lambda_iq: DVector<f64>,
    /// Solver status.
    pub status: QpStatus,
    /// Number of active-set iterations performed.
    pub iterations: usize,
}

/// Configuration parameters for [`solve_qp`].
#[derive(Debug, Clone)]
pub struct QpConfig {
    /// Maximum active-set iterations.
    pub max_iters: usize,
    /// Tolerance for constraint feasibility checks.
    pub feasibility_tol: f64,
    /// Tolerance for step-norm and multiplier optimality checks.
    pub optimality_tol: f64,
}

impl Default for QpConfig {
    fn default() -> Self {
        Self {
            max_iters: 500,
            feasibility_tol: 1e-10,
            optimality_tol: 1e-8,
        }
    }
}

// ─── Solver ─────────────────────────────────────────────────────────────────

/// Solve a dense QP with the primal active-set method.
///
/// # Arguments
///
/// * `h` — Hessian (n × n), must be positive (semi-)definite.
/// * `c` — Linear cost (n).
/// * `a_eq`, `b_eq` — Equality constraints $A_{eq} x = b_{eq}$.
///   Pass `None` for both when there are no equalities.
/// * `a_iq`, `b_iq` — Inequality constraints $A_{iq} x \le b_{iq}$.
///   Pass `None` for both when there are no inequalities.
/// * `x0` — Optional initial feasible point.  When `None` the solver tries
///   $x = 0$ (no equalities) or the least-norm equality-feasible point.
/// * `config` — Solver parameters.
pub fn solve_qp(
    h: &DMatrix<f64>,
    c: &DVector<f64>,
    a_eq: Option<&DMatrix<f64>>,
    b_eq: Option<&DVector<f64>>,
    a_iq: Option<&DMatrix<f64>>,
    b_iq: Option<&DVector<f64>>,
    x0: Option<&DVector<f64>>,
    config: &QpConfig,
) -> QpSolution {
    let n = h.nrows();
    assert_eq!(h.ncols(), n, "H must be square");
    assert_eq!(c.nrows(), n, "c length must match H dimension");

    // ── Unpack / default equality & inequality matrices ──────────────
    let (ae, be) = unpack_pair(a_eq, b_eq, n, "a_eq / b_eq");
    let (ai, bi) = unpack_pair(a_iq, b_iq, n, "a_iq / b_iq");
    let m_eq = ae.nrows();
    let m_iq = ai.nrows();

    // ── Cholesky of H (with regularisation fallback) ─────────────────
    let chol = match h.clone().cholesky() {
        Some(c) => c,
        None => {
            let h_reg = h + &DMatrix::identity(n, n) * 1e-12;
            match h_reg.cholesky() {
                Some(c) => c,
                None => return fail(n, m_eq, m_iq, QpStatus::NumericalFailure),
            }
        }
    };

    // ── Initial feasible point ───────────────────────────────────────
    let mut x = match x0 {
        Some(v) => {
            assert_eq!(v.nrows(), n, "x0 length must match H dimension");
            v.clone()
        }
        None => initial_feasible(n, &ae, &be, config),
    };

    // Verify feasibility
    if m_eq > 0 {
        let residual = (&ae * &x - &be).norm();
        if residual > config.feasibility_tol * (1.0 + be.norm().max(1.0)) {
            return fail(n, m_eq, m_iq, QpStatus::Infeasible);
        }
    }
    if m_iq > 0 {
        let vals = &ai * &x;
        for i in 0..m_iq {
            if vals[i] > bi[i] + config.feasibility_tol {
                return fail(n, m_eq, m_iq, QpStatus::Infeasible);
            }
        }
    }

    // ── Working set: active inequalities at x ────────────────────────
    let mut ws: Vec<usize> = Vec::new();
    if m_iq > 0 {
        let vals = &ai * &x;
        for i in 0..m_iq {
            if vals[i] >= bi[i] - config.feasibility_tol {
                ws.push(i);
            }
        }
    }

    // ── Active-set iterations ────────────────────────────────────────
    let mut lam_eq = DVector::zeros(m_eq);
    let mut lam_iq = DVector::zeros(m_iq);

    for iter in 0..config.max_iters {
        let grad = h * &x + c;
        let m_w = ws.len();
        let m_act = m_eq + m_w;

        if m_act == 0 {
            // Unconstrained step
            let p = chol.solve(&(-&grad));
            if p.norm() < config.optimality_tol {
                return optimal(x, h, c, lam_eq, lam_iq, iter);
            }
            let (alpha, blocking) = step_length(&x, &p, &ai, &bi, &ws, config);
            x += alpha * &p;
            if let Some(idx) = blocking {
                ws.push(idx);
            }
        } else {
            // Build active constraint matrix  Â = [A_eq; A_W]
            let a_act = build_active_matrix(&ae, &ai, &ws, m_eq, n);

            // Schur complement: S = Â H⁻¹ Âᵀ
            let h_inv_at = chol.solve(&a_act.transpose());
            let s = &a_act * &h_inv_at;

            let r = -&grad;
            let h_inv_r = chol.solve(&r);
            let rhs = &a_act * &h_inv_r;

            let nu = match s.clone().lu().solve(&rhs) {
                Some(v) => v,
                None => {
                    // Singular Schur complement → dependent constraints
                    if m_w > 0 {
                        ws.pop();
                        continue;
                    }
                    return make_sol(x, h, c, lam_eq, lam_iq, QpStatus::NumericalFailure, iter);
                }
            };

            let p = &h_inv_r - &h_inv_at * &nu;

            if p.norm() < config.optimality_tol {
                // ── Check multipliers for active inequalities ────
                let mut all_ok = true;
                let mut worst_val = 0.0;
                let mut worst_k = 0usize;

                for k in 0..m_w {
                    let mu = nu[m_eq + k];
                    if mu < -config.optimality_tol && mu < worst_val {
                        all_ok = false;
                        worst_val = mu;
                        worst_k = k;
                    }
                }

                if all_ok {
                    for i in 0..m_eq {
                        lam_eq[i] = nu[i];
                    }
                    for (k, &wi) in ws.iter().enumerate() {
                        lam_iq[wi] = nu[m_eq + k];
                    }
                    return optimal(x, h, c, lam_eq, lam_iq, iter);
                } else {
                    ws.remove(worst_k);
                }
            } else {
                let (alpha, blocking) = step_length(&x, &p, &ai, &bi, &ws, config);
                x += alpha * &p;
                if let Some(idx) = blocking {
                    ws.push(idx);
                }
            }
        }
    }

    make_sol(x, h, c, lam_eq, lam_iq, QpStatus::MaxIterations, config.max_iters)
}

// ─── Internals ──────────────────────────────────────────────────────────────

fn unpack_pair(
    a: Option<&DMatrix<f64>>,
    b: Option<&DVector<f64>>,
    n: usize,
    name: &str,
) -> (DMatrix<f64>, DVector<f64>) {
    match (a, b) {
        (Some(a), Some(b)) => {
            assert_eq!(a.ncols(), n, "{name}: column count must match n");
            assert_eq!(a.nrows(), b.nrows(), "{name}: row counts must match");
            (a.clone(), b.clone())
        }
        (None, None) => (DMatrix::zeros(0, n), DVector::zeros(0)),
        _ => panic!("{name}: must both be Some or both be None"),
    }
}

fn initial_feasible(
    n: usize,
    ae: &DMatrix<f64>,
    be: &DVector<f64>,
    _config: &QpConfig,
) -> DVector<f64> {
    let m_eq = ae.nrows();
    if m_eq == 0 {
        return DVector::zeros(n);
    }
    // Least-norm: x = Aᵀ (A Aᵀ)⁻¹ b
    let aat = ae * ae.transpose();
    match aat.lu().solve(be) {
        Some(y) => ae.transpose() * y,
        None => DVector::zeros(n),
    }
}

fn build_active_matrix(
    ae: &DMatrix<f64>,
    ai: &DMatrix<f64>,
    ws: &[usize],
    m_eq: usize,
    n: usize,
) -> DMatrix<f64> {
    let m_act = m_eq + ws.len();
    let mut a = DMatrix::zeros(m_act, n);
    for i in 0..m_eq {
        a.row_mut(i).copy_from(&ae.row(i));
    }
    for (k, &wi) in ws.iter().enumerate() {
        a.row_mut(m_eq + k).copy_from(&ai.row(wi));
    }
    a
}

fn step_length(
    x: &DVector<f64>,
    p: &DVector<f64>,
    ai: &DMatrix<f64>,
    bi: &DVector<f64>,
    ws: &[usize],
    config: &QpConfig,
) -> (f64, Option<usize>) {
    let m_iq = ai.nrows();
    if m_iq == 0 {
        return (1.0, None);
    }
    let mut alpha = 1.0;
    let mut blocking = None;

    for i in 0..m_iq {
        if ws.contains(&i) {
            continue;
        }
        let ai_p = row_dot(ai, i, p);
        if ai_p > config.feasibility_tol {
            let slack = bi[i] - row_dot(ai, i, x);
            let alpha_i = (slack / ai_p).max(0.0);
            if alpha_i < alpha {
                alpha = alpha_i;
                blocking = Some(i);
            }
        }
    }
    (alpha, blocking)
}

/// Row-vector · column-vector dot product (avoids nalgebra shape mismatch).
#[inline]
fn row_dot(mat: &DMatrix<f64>, row: usize, v: &DVector<f64>) -> f64 {
    let n = v.nrows();
    let mut s = 0.0;
    for k in 0..n {
        s += mat[(row, k)] * v[k];
    }
    s
}

fn fail(n: usize, m_eq: usize, m_iq: usize, status: QpStatus) -> QpSolution {
    QpSolution {
        x: DVector::zeros(n),
        objective: 0.0,
        lambda_eq: DVector::zeros(m_eq),
        lambda_iq: DVector::zeros(m_iq),
        status,
        iterations: 0,
    }
}

fn optimal(
    x: DVector<f64>,
    h: &DMatrix<f64>,
    c: &DVector<f64>,
    lam_eq: DVector<f64>,
    lam_iq: DVector<f64>,
    iters: usize,
) -> QpSolution {
    make_sol(x, h, c, lam_eq, lam_iq, QpStatus::Optimal, iters)
}

fn make_sol(
    x: DVector<f64>,
    h: &DMatrix<f64>,
    c: &DVector<f64>,
    lam_eq: DVector<f64>,
    lam_iq: DVector<f64>,
    status: QpStatus,
    iters: usize,
) -> QpSolution {
    let obj = 0.5 * x.dot(&(h * &x)) + c.dot(&x);
    QpSolution {
        x,
        objective: obj,
        lambda_eq: lam_eq,
        lambda_iq: lam_iq,
        status,
        iterations: iters,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn unconstrained_minimum() {
        // min 0.5*(x1² + 2*x2²) - 3*x1 - x2
        // H = diag(1, 2), c = [-3, -1]
        // Solution: x = H^{-1} (-c) = [3, 0.5]
        let h = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        let c = DVector::from_vec(vec![-3.0, -1.0]);

        let sol = solve_qp(&h, &c, None, None, None, None, None, &QpConfig::default());
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(sol.x[1], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn inequality_active_at_optimum() {
        // min 0.5*(x1-2)² + 0.5*(x2-2)²  s.t.  x1 ≤ 1, x2 ≤ 1
        // H = I, c = [-2, -2]
        // Unconstrained: (2,2).  Constrained: (1,1).
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-2.0, -2.0]);
        let a_iq = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b_iq = DVector::from_vec(vec![1.0, 1.0]);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), None, &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 1.0, epsilon = 1e-8);
        assert_relative_eq!(sol.x[1], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn inequality_not_active() {
        // min 0.5*(x1² + x2²)  s.t.  x1 ≤ 5, x2 ≤ 5
        // Unconstrained min at (0,0) is already feasible.
        let h = DMatrix::identity(2, 2);
        let c = DVector::zeros(2);
        let a_iq = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b_iq = DVector::from_vec(vec![5.0, 5.0]);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), None, &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 0.0, epsilon = 1e-8);
        assert_relative_eq!(sol.x[1], 0.0, epsilon = 1e-8);
    }

    #[test]
    fn one_active_one_inactive() {
        // min 0.5*(x1-3)² + 0.5*(x2-0.5)²  s.t.  x1 ≤ 1, x2 ≤ 2
        // Unconstrained: (3, 0.5).  x1 ≤ 1 is active, x2 ≤ 2 is not.
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-3.0, -0.5]);
        let a_iq = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b_iq = DVector::from_vec(vec![1.0, 2.0]);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), None, &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 1.0, epsilon = 1e-8);
        assert_relative_eq!(sol.x[1], 0.5, epsilon = 1e-8);
    }

    #[test]
    fn box_constraints() {
        // min 0.5*(x1-5)² + 0.5*(x2+3)²  s.t.  -1 ≤ x ≤ 2
        // Unconstrained: (5, -3).  Box: (2, -1).
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-5.0, 3.0]);
        // x1 ≤ 2, x2 ≤ 2, -x1 ≤ 1, -x2 ≤ 1
        let a_iq = DMatrix::from_row_slice(4, 2, &[
            1.0,  0.0,
            0.0,  1.0,
            -1.0, 0.0,
            0.0, -1.0,
        ]);
        let b_iq = DVector::from_vec(vec![2.0, 2.0, 1.0, 1.0]);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), None, &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 2.0, epsilon = 1e-8);
        assert_relative_eq!(sol.x[1], -1.0, epsilon = 1e-8);
    }

    #[test]
    fn equality_only() {
        // min 0.5*(x1² + x2²)  s.t.  x1 + x2 = 1
        // Solution on the line x1+x2=1: closest to origin is (0.5, 0.5).
        let h = DMatrix::identity(2, 2);
        let c = DVector::zeros(2);
        let a_eq = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let b_eq = DVector::from_element(1, 1.0);

        let sol = solve_qp(
            &h, &c,
            Some(&a_eq), Some(&b_eq),
            None, None, None, &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 0.5, epsilon = 1e-8);
        assert_relative_eq!(sol.x[1], 0.5, epsilon = 1e-8);
    }

    #[test]
    fn equality_and_inequality() {
        // min 0.5*(x1-3)² + 0.5*(x2-3)²  s.t.  x1 + x2 = 2, x1 ≥ 0, x2 ≥ 0
        // On the line x1+x2=2, closest to (3,3) is (1,1).
        // But with x1 ≥ 0, x2 ≥ 0 (not active), solution is still (1,1).
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-3.0, -3.0]);
        let a_eq = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let b_eq = DVector::from_element(1, 2.0);
        // -x1 ≤ 0, -x2 ≤ 0
        let a_iq = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -1.0]);
        let b_iq = DVector::zeros(2);

        let sol = solve_qp(
            &h, &c,
            Some(&a_eq), Some(&b_eq),
            Some(&a_iq), Some(&b_iq),
            None,
            &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(sol.x[1], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn equality_and_active_inequality() {
        // min 0.5*(x1² + x2²)  s.t.  x1 + x2 = 2, x1 ≤ 0.5
        // On x1+x2=2 the unconstrained closest-to-origin is (1,1). But x1 ≤ 0.5
        // forces (0.5, 1.5).
        let h = DMatrix::identity(2, 2);
        let c = DVector::zeros(2);
        let a_eq = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let b_eq = DVector::from_element(1, 2.0);
        let a_iq = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let b_iq = DVector::from_element(1, 0.5);

        let sol = solve_qp(
            &h, &c,
            Some(&a_eq), Some(&b_eq),
            Some(&a_iq), Some(&b_iq),
            None,
            &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(sol.x[1], 1.5, epsilon = 1e-6);
    }

    #[test]
    fn user_provided_x0() {
        // Same as box_constraints but with user-provided x0 = (0,0).
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-5.0, 3.0]);
        let a_iq = DMatrix::from_row_slice(4, 2, &[
            1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0,
        ]);
        let b_iq = DVector::from_vec(vec![2.0, 2.0, 1.0, 1.0]);
        let x0 = DVector::from_vec(vec![0.0, 0.0]);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), Some(&x0), &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 2.0, epsilon = 1e-8);
        assert_relative_eq!(sol.x[1], -1.0, epsilon = 1e-8);
    }

    #[test]
    fn larger_problem() {
        // min 0.5 ||x||²  s.t.  Σx_i ≥ 5 and 0 ≤ x_i ≤ 3 for i=0..4
        // Closest to origin on Σx≥5 with box: each x_i = 1 (sum = 5).
        let n = 5;
        let h = DMatrix::identity(n, n);
        let c = DVector::zeros(n);

        // -Σx_i ≤ -5, x_i ≤ 3, -x_i ≤ 0
        let mut rows = Vec::new();
        // sum >= 5
        let mut sum_row = vec![0.0; n];
        for v in &mut sum_row {
            *v = -1.0;
        }
        rows.push((sum_row, -5.0));
        for i in 0..n {
            let mut row_upper = vec![0.0; n];
            row_upper[i] = 1.0;
            rows.push((row_upper, 3.0));
            let mut row_lower = vec![0.0; n];
            row_lower[i] = -1.0;
            rows.push((row_lower, 0.0));
        }

        let m = rows.len();
        let mut a_data = Vec::with_capacity(m * n);
        let mut b_data = Vec::with_capacity(m);
        for (r, b_val) in &rows {
            a_data.extend_from_slice(r);
            b_data.push(*b_val);
        }
        let a_iq = DMatrix::from_row_slice(m, n, &a_data);
        let b_iq = DVector::from_vec(b_data);

        let x0 = DVector::from_element(n, 1.0); // feasible start: sum=5
        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), Some(&x0), &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        for i in 0..n {
            assert_relative_eq!(sol.x[i], 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn objective_value_correct() {
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-2.0, -2.0]);
        let a_iq = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b_iq = DVector::from_vec(vec![1.0, 1.0]);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), None, &QpConfig::default(),
        );
        // x = (1,1), obj = 0.5*(1+1) + (-2-2) = 1 - 4 = -3
        assert_relative_eq!(sol.objective, -3.0, epsilon = 1e-8);
    }

    #[test]
    fn multipliers_positive_for_active_inequality() {
        // min 0.5*(x-2)²  s.t.  x ≤ 1
        // Active at x=1, multiplier = ∂f/∂b = 2-1 = 1
        let h = DMatrix::from_element(1, 1, 1.0);
        let c = DVector::from_element(1, -2.0);
        let a_iq = DMatrix::from_element(1, 1, 1.0);
        let b_iq = DVector::from_element(1, 1.0);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), None, &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 1.0, epsilon = 1e-10);
        assert!(sol.lambda_iq[0] > 0.0, "active inequality multiplier should be positive");
        assert_relative_eq!(sol.lambda_iq[0], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn coupled_inequality_constraints() {
        // min 0.5*(x1² + x2²)  s.t.  x1 + x2 ≤ 1, x1 - x2 ≤ 1
        // Unconstrained: (0,0), which satisfies both → solution is (0,0).
        let h = DMatrix::identity(2, 2);
        let c = DVector::zeros(2);
        let a_iq = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, -1.0]);
        let b_iq = DVector::from_vec(vec![1.0, 1.0]);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), None, &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x.norm(), 0.0, epsilon = 1e-8);
    }

    #[test]
    fn coupled_inequality_active() {
        // min 0.5*((x1-2)² + (x2-2)²)  s.t.  x1 + x2 ≤ 1
        // Unconstrained: (2,2), violates x1+x2≤1.
        // Constrained min on x1+x2=1 closest to (2,2): (0.5, 0.5).
        let h = DMatrix::identity(2, 2);
        let c = DVector::from_vec(vec![-2.0, -2.0]);
        let a_iq = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let b_iq = DVector::from_element(1, 1.0);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), None, &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 0.5, epsilon = 1e-8);
        assert_relative_eq!(sol.x[1], 0.5, epsilon = 1e-8);
    }

    #[test]
    fn non_identity_hessian() {
        // min 0.5*(3*x1² + x2² + 2*x1*x2) - x1  s.t.  -1 ≤ x ≤ 1
        // H = [[3, 1], [1, 1]], c = [-1, 0]
        // Unconstrained: H^{-1}(-c) = [[1,-1],[-1,3]]/2 * [1,0] = [0.5, -0.5]
        let h = DMatrix::from_row_slice(2, 2, &[3.0, 1.0, 1.0, 1.0]);
        let c = DVector::from_vec(vec![-1.0, 0.0]);
        // box: x1 ≤ 1, x2 ≤ 1, -x1 ≤ 1, -x2 ≤ 1
        let a_iq = DMatrix::from_row_slice(4, 2, &[
            1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0,
        ]);
        let b_iq = DVector::from_vec(vec![1.0, 1.0, 1.0, 1.0]);

        let sol = solve_qp(
            &h, &c, None, None,
            Some(&a_iq), Some(&b_iq), None, &QpConfig::default(),
        );
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_relative_eq!(sol.x[0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(sol.x[1], -0.5, epsilon = 1e-6);
    }
}
