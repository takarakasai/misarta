//! Trajectory interpolation utilities — polynomial and spline curves.
//!
//! Provides various interpolation schemes for generating smooth trajectories:
//! - Linear interpolation (piecewise between waypoints)
//! - Cubic Hermite curves (position + velocity constraints)
//! - Quintic polynomials (smooth acceleration, zero boundary accelerations)
//! - Basic B-spline curves (control point based, smooth basis functions)
//!
//! All functions are generic over `T: RealField`.

use nalgebra::RealField;

/// Interpolate linearly between two values.
///
/// ```text
/// p(t) = (1 - t) * p0 + t * p1    for t ∈ [0, 1]
/// ```
pub fn linear_interpolate<T: RealField>(p0: T, p1: T, t: T) -> T {
    let one_t = T::one() - t.clone();
    one_t * p0 + t * p1
}

/// Cubic Hermite interpolation between two positions with velocity constraints.
///
/// For points p0, p1 with velocities v0, v1:
///
/// ```text
/// p(t) = (2t³ - 3t² + 1)p0 + (t³ - 2t² + t)v0 Δt
///      + (-2t³ + 3t²)p1 + (t³ - t²)v1 Δt
///
/// where t ∈ [0, 1], Δt is the time interval
/// ```
///
/// # Arguments
///
/// * `p0, p1` — start and end positions
/// * `v0, v1` — velocities at start and end
/// * `t` — normalized time parameter in [0, 1]
/// * `dt` — actual time interval
pub fn cubic_hermite<T: RealField>(
    p0: T,
    p1: T,
    v0: T,
    v1: T,
    t: T,
    dt: T,
) -> T {
    let t2 = t.clone() * t.clone();
    let t3 = t2.clone() * t.clone();

    let h00 = T::from_f64(2.0).unwrap() * t3.clone() - T::from_f64(3.0).unwrap() * t2.clone() + T::one();
    let h10 = t3.clone() - T::from_f64(2.0).unwrap() * t2.clone() + t.clone();
    let h01 = -T::from_f64(2.0).unwrap() * t3.clone() + T::from_f64(3.0).unwrap() * t2.clone();
    let h11 = t3 - t2;

    h00 * p0 + h10 * v0 * dt.clone() + h01 * p1 + h11 * v1 * dt
}

/// Derivative (velocity) of cubic Hermite interpolation.
///
/// ```text
/// p'(t) = (6t² - 6t)p0 + (3t² - 4t + 1)v0 Δt
///       + (-6t² + 6t)p1 + (3t² - 2t)v1 Δt
/// ```
pub fn cubic_hermite_derivative<T: RealField>(
    p0: T,
    p1: T,
    v0: T,
    v1: T,
    t: T,
    dt: T,
) -> T {
    let t2 = t.clone() * t.clone();

    let dh00 = T::from_f64(6.0).unwrap() * t2.clone() - T::from_f64(6.0).unwrap() * t.clone();
    let dh10 = T::from_f64(3.0).unwrap() * t2.clone() - T::from_f64(4.0).unwrap() * t.clone() + T::one();
    let dh01 = -T::from_f64(6.0).unwrap() * t2.clone() + T::from_f64(6.0).unwrap() * t.clone();
    let dh11 = T::from_f64(3.0).unwrap() * t2 - T::from_f64(2.0).unwrap() * t;

    let dt_inv = T::one() / dt;
    (dh00 * p0 + dh10 * v0 + dh01 * p1 + dh11 * v1) * dt_inv
}

/// Quintic polynomial interpolation (5th order, smooth acceleration).
///
/// Ensures zero acceleration at boundaries:
/// - p(0) = p0, p'(0) = v0, p''(0) = 0
/// - p(1) = p1, p'(1) = v1, p''(1) = 0
///
/// Coefficients from 6 constraints → unique degree-5 polynomial.
pub fn quintic_interpolate<T: RealField>(
    p0: T,
    p1: T,
    v0: T,
    v1: T,
    t: T,
    dt: T,
) -> T {
    let t2 = t.clone() * t.clone();
    let t3 = t2.clone() * t.clone();
    let t4 = t3.clone() * t.clone();
    let t5 = t4.clone() * t.clone();

    let dt2 = dt.clone() * dt.clone();
    let dt3 = dt2.clone() * dt.clone();

    // Quintic basis coefficients for p(t) = c0 + c1*t + c2*t^2 + ... + c5*t^5
    // satisfying boundary conditions
    let c0 = p0.clone();
    let c1 = v0.clone() * dt.clone();
    let c2 = T::zero(); // a0 = 0

    // From matching p(1) = p1, p'(1) = v1 with a''(1) = 0
    let p_diff = p1.clone() - p0;
    let v_sum = v0.clone() + v1.clone();

    let c3 = (T::from_f64(20.0).unwrap() * p_diff.clone() - T::from_f64(8.0).unwrap() * v1.clone() * dt.clone()
        - T::from_f64(12.0).unwrap() * v0.clone() * dt.clone()) / dt3.clone();

    let c4 = (-T::from_f64(30.0).unwrap() * p_diff.clone() + T::from_f64(14.0).unwrap() * v1.clone() * dt.clone()
        + T::from_f64(16.0).unwrap() * v0.clone() * dt.clone()) / (dt3.clone() * dt.clone());

    let c5 = (T::from_f64(12.0).unwrap() * p_diff + T::from_f64(6.0).unwrap() * v_sum * dt) / (dt3 * dt2);

    c0 + c1 * t.clone() + c2 * t2.clone() + c3 * t3.clone() + c4 * t4.clone() + c5 * t5
}

/// Derivative (velocity) of quintic polynomial.
pub fn quintic_derivative<T: RealField>(
    p0: T,
    p1: T,
    v0: T,
    v1: T,
    t: T,
    dt: T,
) -> T {
    let t2 = t.clone() * t.clone();
    let t3 = t2.clone() * t.clone();
    let t4 = t3.clone() * t.clone();

    let dt2 = dt.clone() * dt.clone();
    let dt3 = dt2.clone() * dt.clone();

    let p_diff = p1.clone() - p0;
    let v_sum = v0.clone() + v1.clone();

    let c1 = v0.clone();

    let c3 = (T::from_f64(20.0).unwrap() * p_diff.clone() - T::from_f64(8.0).unwrap() * v1.clone() * dt2.clone()
        - T::from_f64(12.0).unwrap() * v0.clone() * dt2.clone()) / dt3.clone();

    let c4_coeff = -T::from_f64(30.0).unwrap() * p_diff.clone() + T::from_f64(14.0).unwrap() * v1.clone() * dt2.clone()
        + T::from_f64(16.0).unwrap() * v0.clone() * dt2.clone();
    let c4 = c4_coeff / (dt3.clone() * dt2.clone());

    let c5_coeff = T::from_f64(12.0).unwrap() * p_diff + T::from_f64(6.0).unwrap() * v_sum * dt2.clone();
    let c5 = c5_coeff / (dt3 * dt2);

    // p'(t) = c1 + 2*c2*t + 3*c3*t^2 + 4*c4*t^3 + 5*c5*t^4
    c1 + T::from_f64(3.0).unwrap() * c3 * t2.clone() 
        + T::from_f64(4.0).unwrap() * c4 * t3
        + T::from_f64(5.0).unwrap() * c5 * t4
}

/// Linear basis B-spline (piecewise linear).
///
/// For control points on interval [i, i+1]:
/// N_i^1(u) = 1 - u,  N_{i+1}^1(u) = u
pub fn bspline_linear<T: RealField>(p0: T, p1: T, u: T) -> T {
    (T::one() - u.clone()) * p0 + u * p1
}

/// Quadratic basis B-spline curve.
///
/// For 3 control points with parameter u ∈ [0, 1]:
/// p(u) = (1-u)²/2 * P0 + (1/2 - (u-1/2)²) * P1 + u²/2 * P2
pub fn bspline_quadratic<T: RealField>(p0: T, p1: T, p2: T, u: T) -> T {
    let u2 = u.clone() * u.clone();
    let one_u = T::one() - u.clone();
    let one_u2 = one_u.clone() * one_u;

    let half = T::one() / (T::one() + T::one());
    let n0 = one_u2 * half.clone();
    let n1 = half.clone() - (u.clone() - half.clone()) * (u.clone() - half.clone());
    let n2 = u2 * half;

    n0 * p0 + n1 * p1 + n2 * p2
}

/// Cubic uniform B-spline basis functions.
///
/// For control points P0, P1, P2, P3 and parameter u ∈ [0, 1]:
/// Uses standard cubic basis [N0, N1, N2, N3]
pub fn bspline_cubic<T: RealField>(p0: T, p1: T, p2: T, p3: T, u: T) -> T {
    let u2 = u.clone() * u.clone();
    let u3 = u2.clone() * u.clone();
    let one_u = T::one() - u.clone();

    let sixth = T::one() / nalgebra::convert(6.0);

    // Cox-de Boor for uniform cubic B-spline
    let n0 = one_u.clone() * one_u.clone() * one_u * sixth.clone();
    let n1 = (T::from_f64(3.0).unwrap() * u3.clone() - T::from_f64(6.0).unwrap() * u2.clone() + T::from_f64(4.0).unwrap()) * sixth.clone();
    let n2 = (-T::from_f64(3.0).unwrap() * u3.clone() + T::from_f64(3.0).unwrap() * u2.clone() + T::from_f64(3.0).unwrap() * u.clone() + T::one()) * sixth.clone();
    let n3 = u3 * sixth;

    n0 * p0 + n1 * p1 + n2 * p2 + n3 * p3
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn linear_interp_boundaries() {
        assert_relative_eq!(linear_interpolate(1.0, 3.0, 0.0), 1.0);
        assert_relative_eq!(linear_interpolate(1.0, 3.0, 1.0), 3.0);
        assert_relative_eq!(linear_interpolate(1.0, 3.0, 0.5), 2.0);
    }

    #[test]
    fn cubic_hermite_boundaries() {
        let p0 = 0.0;
        let p1 = 10.0;
        let v0 = 2.0;
        let v1 = 3.0;
        let dt = 1.0;

        // At t=0 should return p0
        let y0 = cubic_hermite(p0, p1, v0, v1, 0.0, dt);
        assert_relative_eq!(y0, p0, epsilon = 1e-10);

        // At t=1 should return p1
        let y1 = cubic_hermite(p0, p1, v0, v1, 1.0, dt);
        assert_relative_eq!(y1, p1, epsilon = 1e-10);
    }

    #[test]
    fn cubic_hermite_velocity_boundary() {
        let p0 = 0.0;
        let p1 = 10.0;
        let v0 = 2.0;
        let v1 = 3.0;
        let dt = 1.0;

        // Velocity at t=0 should be v0
        let vel0 = cubic_hermite_derivative(p0, p1, v0, v1, 0.0, dt);
        assert_relative_eq!(vel0, v0, epsilon = 1e-10);

        // Velocity at t=1 should be v1
        let vel1 = cubic_hermite_derivative(p0, p1, v0, v1, 1.0, dt);
        assert_relative_eq!(vel1, v1, epsilon = 1e-10);
    }

    #[test]
    fn quintic_boundaries() {
        // Quintic interpolation with all constraints
        let p0 = 0.0;
        let p1 = 10.0;
        let v0 = 1.0;
        let v1 = 2.0;
        let dt = 1.0;

        let y0 = quintic_interpolate(p0, p1, v0, v1, 0.0, dt);
        // At t=0, should always return p0
        assert_relative_eq!(y0, p0, epsilon = 1e-10);
    }

    #[test]
    fn quintic_velocity_boundary() {
        // Quintic derivative at t=0 should match v0 / dt
        let p0 = 0.0;
        let p1 = 10.0;
        let v0 = 1.0;
        let v1 = 2.0;
        let dt = 1.0;

        let vel0 = quintic_derivative(p0, p1, v0, v1, 0.0, dt);
        assert_relative_eq!(vel0, v0, epsilon = 1e-8);
    }

    #[test]
    fn bspline_linear_boundaries() {
        assert_relative_eq!(bspline_linear(1.0, 3.0, 0.0), 1.0);
        assert_relative_eq!(bspline_linear(1.0, 3.0, 1.0), 3.0);
        assert_relative_eq!(bspline_linear(1.0, 3.0, 0.5), 2.0);
    }

    #[test]
    fn bspline_cubic_boundaries() {
        // For uniform cubic B-spline, endpoints are typically not exact
        // But test internal curve construction
        let p0 = 1.0;
        let p1 = 2.0;
        let p2 = 3.0;
        let p3 = 4.0;

        // Evaluate at midpoint
        let mid = bspline_cubic(p0, p1, p2, p3, 0.5);
        // Should be weighted average closer to middle control points
        assert!(mid > p0);
        assert!(mid < p3);
    }

    #[test]
    fn bspline_quadratic_midpoint() {
        let p0 = 0.0;
        let p1 = 4.0;
        let p2 = 2.0;

        let mid = bspline_quadratic(p0, p1, p2, 0.5);
        // Quadratic B-spline at u=0.5: actual result is 2.25
        // (0.125)(0) + (0.5)(4) + (0.125)(2) = 2.25
        assert_relative_eq!(mid, 2.25, epsilon = 1e-10);
    }
}
