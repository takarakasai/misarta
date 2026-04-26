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

// =============================================================================
//  Pose-to-pose joint-space transition
// =============================================================================

/// Choice of interpolation curve for a [`PoseTransition`].
///
/// All variants enter and leave at zero velocity (and zero acceleration where
/// the polynomial supports it), so the robot can latch onto the new target
/// without a jerk discontinuity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum InterpolationKind {
    /// Constant-velocity ramp `p(s) = (1-s)·p0 + s·p1`.
    /// Discontinuous velocity at the endpoints; cheap and predictable.
    Linear,
    /// Cubic Hermite with `v(0) = v(1) = 0` — smooth start and stop.
    CubicSmooth,
    /// 5-th order polynomial with `v(0) = v(1) = 0` and `a(0) = a(1) = 0` —
    /// continuous acceleration, matches the typical "S-curve" used in
    /// industrial motion planners.
    QuinticSmooth,
}

impl InterpolationKind {
    pub const ALL: [InterpolationKind; 3] = [
        InterpolationKind::Linear,
        InterpolationKind::CubicSmooth,
        InterpolationKind::QuinticSmooth,
    ];

    pub fn label(self) -> &'static str {
        match self {
            InterpolationKind::Linear => "Linear",
            InterpolationKind::CubicSmooth => "Cubic (smooth)",
            InterpolationKind::QuinticSmooth => "Quintic (smooth)",
        }
    }
}

/// Smooth joint-space transition between two configurations.
///
/// `q_start` and `q_end` are full joint vectors (any length, indexed in robot
/// joint order). [`evaluate`](Self::evaluate) returns the interpolated joint
/// vector at sim time `t` (clamped to `[0, duration]`). Joints whose start
/// and end values are equal are passed through unchanged.
#[derive(Clone, Debug)]
pub struct PoseTransition<T: RealField> {
    pub q_start: Vec<T>,
    pub q_end: Vec<T>,
    pub duration: T,
    pub kind: InterpolationKind,
}

impl<T: RealField> PoseTransition<T> {
    pub fn new(q_start: Vec<T>, q_end: Vec<T>, duration: T, kind: InterpolationKind) -> Self {
        Self { q_start, q_end, duration, kind }
    }

    /// Has `t` reached or exceeded `duration`?
    pub fn is_done(&self, t: T) -> bool {
        t >= self.duration
    }

    /// Per-joint interpolated value at time `t`. Out-of-range `t` is clamped.
    pub fn evaluate(&self, t: T) -> Vec<T> {
        let n = self.q_start.len().min(self.q_end.len());
        let mut out = Vec::with_capacity(n);
        let zero = T::zero();
        let dur = self.duration.clone();
        let t_clamped = if t < zero.clone() {
            zero.clone()
        } else if t > dur.clone() {
            dur.clone()
        } else {
            t
        };
        // Normalised parameter s ∈ [0, 1]
        let s = if dur > zero.clone() {
            t_clamped.clone() / dur.clone()
        } else {
            T::one()
        };

        for i in 0..n {
            let p0 = self.q_start[i].clone();
            let p1 = self.q_end[i].clone();
            let v = match self.kind {
                InterpolationKind::Linear => linear_interpolate(p0, p1, s.clone()),
                InterpolationKind::CubicSmooth => cubic_hermite(
                    p0,
                    p1,
                    zero.clone(),
                    zero.clone(),
                    s.clone(),
                    dur.clone(),
                ),
                InterpolationKind::QuinticSmooth => quintic_interpolate(
                    p0,
                    p1,
                    zero.clone(),
                    zero.clone(),
                    s.clone(),
                    T::one(),
                ),
            };
            out.push(v);
        }
        out
    }
}

// =============================================================================
//  Keyframe animation — sequence of joint-space waypoints
// =============================================================================

/// One waypoint in a [`KeyframeAnimation`].
///
/// `time` is the absolute time (s) in the animation timeline at which the
/// robot should reach `q`. The `kind` controls how the segment **leading up
/// to** this keyframe is interpolated; the very first keyframe's `kind` is
/// ignored because there is no preceding segment.
#[derive(Clone, Debug)]
pub struct Keyframe<T: RealField> {
    pub time: T,
    pub q: Vec<T>,
    pub kind: InterpolationKind,
}

impl<T: RealField> Keyframe<T> {
    pub fn new(time: T, q: Vec<T>, kind: InterpolationKind) -> Self {
        Self { time, q, kind }
    }
}

/// Time-ordered sequence of joint-space keyframes evaluated as a single
/// continuous animation.
///
/// Unlike [`PoseTransition`] (point-to-point), `KeyframeAnimation` strings
/// many waypoints together so a sim or controller can replay a longer
/// motion (e.g. crouch → jump → land → recover) with one playback object.
/// Per-segment interpolation kinds let the choreographer mix linear ramps
/// (e.g. for foot-up phases) with smooth quintics (for body sway).
///
/// # Invariants
///
/// - Keyframes are kept sorted by `time` after every mutation.
/// - The first keyframe's `time` may be any value; `evaluate(t)` clamps to
///   it so callers don't have to special-case `t < t0`.
/// - All keyframes are expected to share the same joint vector length;
///   shorter vectors are padded with the previous frame's values during
///   evaluation.
#[derive(Clone, Debug, Default)]
pub struct KeyframeAnimation<T: RealField> {
    keyframes: Vec<Keyframe<T>>,
}

impl<T: RealField> KeyframeAnimation<T> {
    /// Build from a pre-sorted vector of keyframes. Sorts on insert just in
    /// case the input wasn't ordered.
    pub fn new(mut keyframes: Vec<Keyframe<T>>) -> Self {
        keyframes.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { keyframes }
    }

    /// Total number of keyframes.
    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }

    /// Read-only view of the keyframes.
    pub fn keyframes(&self) -> &[Keyframe<T>] {
        &self.keyframes
    }

    /// Append a keyframe, maintaining time ordering.
    pub fn push(&mut self, kf: Keyframe<T>) {
        self.keyframes.push(kf);
        self.keyframes.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// First keyframe's `time` (animation start). Returns 0 if empty.
    pub fn start_time(&self) -> T {
        self.keyframes
            .first()
            .map(|k| k.time.clone())
            .unwrap_or_else(T::zero)
    }

    /// Last keyframe's `time` (animation end). Returns 0 if empty.
    pub fn end_time(&self) -> T {
        self.keyframes
            .last()
            .map(|k| k.time.clone())
            .unwrap_or_else(T::zero)
    }

    /// Total wall-clock duration (`end_time - start_time`).
    pub fn duration(&self) -> T {
        self.end_time() - self.start_time()
    }

    /// `t` is past the last keyframe.
    pub fn is_done(&self, t: T) -> bool {
        match self.keyframes.last() {
            Some(last) => t >= last.time,
            None => true,
        }
    }

    /// Evaluate the joint vector at absolute time `t`.
    ///
    /// - `t` before the first keyframe → clamps to that keyframe's `q`.
    /// - `t` after the last keyframe → clamps to that keyframe's `q`.
    /// - Otherwise the segment between the bracketing keyframes is
    ///   interpolated using the *destination* keyframe's [`InterpolationKind`].
    pub fn evaluate(&self, t: T) -> Vec<T> {
        if self.keyframes.is_empty() {
            return Vec::new();
        }
        if self.keyframes.len() == 1 {
            return self.keyframes[0].q.clone();
        }
        // Find the segment [k_prev, k_next] bracketing t.
        if t <= self.keyframes[0].time {
            return self.keyframes[0].q.clone();
        }
        if t >= self.keyframes.last().unwrap().time {
            return self.keyframes.last().unwrap().q.clone();
        }
        // Linear scan is fine — these timelines are short (10s-100s of frames).
        let mut idx = 0;
        for i in 1..self.keyframes.len() {
            if t < self.keyframes[i].time {
                idx = i;
                break;
            }
        }
        let prev = &self.keyframes[idx - 1];
        let next = &self.keyframes[idx];
        let dur = next.time.clone() - prev.time.clone();
        let local_t = t - prev.time.clone();
        let traj = PoseTransition {
            q_start: prev.q.clone(),
            q_end: next.q.clone(),
            duration: dur,
            kind: next.kind,
        };
        traj.evaluate(local_t)
    }
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
    fn keyframe_anim_clamps_outside_range() {
        let kfs = vec![
            Keyframe::new(0.0, vec![0.0, 0.0], InterpolationKind::Linear),
            Keyframe::new(1.0, vec![1.0, 2.0], InterpolationKind::QuinticSmooth),
            Keyframe::new(2.0, vec![3.0, -1.0], InterpolationKind::Linear),
        ];
        let anim = KeyframeAnimation::new(kfs);
        // Before start
        let q = anim.evaluate(-1.0);
        assert_relative_eq!(q[0], 0.0);
        assert_relative_eq!(q[1], 0.0);
        // After end
        let q = anim.evaluate(10.0);
        assert_relative_eq!(q[0], 3.0);
        assert_relative_eq!(q[1], -1.0);
    }

    #[test]
    fn keyframe_anim_hits_waypoints_exactly() {
        let kfs = vec![
            Keyframe::new(0.0, vec![0.0], InterpolationKind::Linear),
            Keyframe::new(1.0, vec![5.0], InterpolationKind::QuinticSmooth),
            Keyframe::new(3.0, vec![-2.0], InterpolationKind::CubicSmooth),
        ];
        let anim = KeyframeAnimation::new(kfs);
        assert_relative_eq!(anim.evaluate(0.0)[0], 0.0, epsilon = 1e-9);
        assert_relative_eq!(anim.evaluate(1.0)[0], 5.0, epsilon = 1e-9);
        assert_relative_eq!(anim.evaluate(3.0)[0], -2.0, epsilon = 1e-9);
        assert!(anim.is_done(3.0));
        assert!(!anim.is_done(2.999));
    }

    #[test]
    fn keyframe_anim_linear_segment_midpoint() {
        let kfs = vec![
            Keyframe::new(0.0, vec![0.0], InterpolationKind::Linear),
            Keyframe::new(2.0, vec![4.0], InterpolationKind::Linear),
        ];
        let anim = KeyframeAnimation::new(kfs);
        // Linear interpolation: t=1.0 → midpoint
        assert_relative_eq!(anim.evaluate(1.0)[0], 2.0, epsilon = 1e-9);
        assert_relative_eq!(anim.duration(), 2.0, epsilon = 1e-9);
    }

    #[test]
    fn keyframe_anim_push_keeps_sorted() {
        let mut anim: KeyframeAnimation<f64> = KeyframeAnimation::default();
        anim.push(Keyframe::new(2.0, vec![1.0], InterpolationKind::Linear));
        anim.push(Keyframe::new(0.0, vec![0.0], InterpolationKind::Linear));
        anim.push(Keyframe::new(1.0, vec![0.5], InterpolationKind::Linear));
        let times: Vec<f64> = anim.keyframes().iter().map(|k| k.time).collect();
        assert_eq!(times, vec![0.0, 1.0, 2.0]);
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
