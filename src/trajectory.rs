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

/// Second derivative (acceleration) of cubic Hermite interpolation.
///
/// `t` is the normalised parameter `s ∈ [0, 1]`; `dt` is the segment duration
/// (real-time seconds). The h00/h01 basis terms (acting on `p0`, `p1`) need
/// `1/dt²` because they were originally a function of `s = t/dt`. The
/// h10/h11 terms (acting on `v0·dt`, `v1·dt`) need only `1/dt` because the
/// extra `dt` factor cancels one of them out.
pub fn cubic_hermite_second_derivative<T: RealField>(
    p0: T,
    p1: T,
    v0: T,
    v1: T,
    t: T,
    dt: T,
) -> T {
    let twelve = T::from_f64(12.0).unwrap();
    let six = T::from_f64(6.0).unwrap();
    let four = T::from_f64(4.0).unwrap();
    let two = T::from_f64(2.0).unwrap();

    let ddh00 = twelve.clone() * t.clone() - six.clone();
    let ddh10 = six.clone() * t.clone() - four;
    let ddh01 = -twelve * t.clone() + six.clone();
    let ddh11 = six * t - two;

    let dt_inv = T::one() / dt.clone();
    let dt_inv2 = dt_inv.clone() * dt_inv.clone();
    (ddh00 * p0 + ddh01 * p1) * dt_inv2 + (ddh10 * v0 + ddh11 * v1) * dt_inv
}

/// Quintic polynomial interpolation (5th order, smooth acceleration).
///
/// Solves for the unique degree-5 polynomial `p(t)` on `t ∈ [0, dt]`
/// satisfying the six boundary conditions
///
/// ```text
/// p(0) = p0    p(dt) = p1
/// p'(0) = v0   p'(dt) = v1
/// p''(0) = 0   p''(dt) = 0
/// ```
///
/// In closed form, with `τ = t / dt` and `Δp = p1 − p0`:
///
/// ```text
/// p(t) = p0 + v0·t
///      + ( 10·Δp − 6·v0·dt − 4·v1·dt) · τ³
///      + (−15·Δp + 8·v0·dt + 7·v1·dt) · τ⁴
///      + (  6·Δp − 3·v0·dt − 3·v1·dt) · τ⁵
/// ```
///
/// (The earlier implementation had each cubic-and-up coefficient doubled
/// and a sign flip on the τ⁵ velocity terms, causing a 2× overshoot at
/// `t = dt` whenever the user requested the QuinticSmooth pose curve.)
pub fn quintic_interpolate<T: RealField>(
    p0: T,
    p1: T,
    v0: T,
    v1: T,
    t: T,
    dt: T,
) -> T {
    let tau = t.clone() / dt.clone();
    let tau2 = tau.clone() * tau.clone();
    let tau3 = tau2.clone() * tau.clone();
    let tau4 = tau3.clone() * tau.clone();
    let tau5 = tau4.clone() * tau;

    let p_diff = p1 - p0.clone();
    let v0dt = v0.clone() * dt.clone();
    let v1dt = v1 * dt;

    let c10 = T::from_f64(10.0).unwrap();
    let c6 = T::from_f64(6.0).unwrap();
    let c4f = T::from_f64(4.0).unwrap();
    let c15 = T::from_f64(15.0).unwrap();
    let c8 = T::from_f64(8.0).unwrap();
    let c7 = T::from_f64(7.0).unwrap();
    let c3f = T::from_f64(3.0).unwrap();

    let k3 = c10 * p_diff.clone() - c6.clone() * v0dt.clone() - c4f * v1dt.clone();
    let k4 = -c15 * p_diff.clone() + c8 * v0dt.clone() + c7 * v1dt.clone();
    let k5 = c6 * p_diff - c3f.clone() * v0dt - c3f.clone() * v1dt;
    let _ = c3f;

    p0 + v0 * t + k3 * tau3 + k4 * tau4 + k5 * tau5
}

/// Derivative (velocity) of [`quintic_interpolate`].
///
/// Differentiating `p(t) = p0 + v0·t + Σᵢ kᵢ·τⁱ` (with `τ = t/dt`) gives
/// `p'(t) = v0 + Σᵢ (i·kᵢ/dt)·τⁱ⁻¹` for i = 3..=5. We re-derive `k₃, k₄, k₅`
/// inside this function with the same boundary-condition formulas as
/// [`quintic_interpolate`] so the two stay in lockstep.
pub fn quintic_derivative<T: RealField>(
    p0: T,
    p1: T,
    v0: T,
    v1: T,
    t: T,
    dt: T,
) -> T {
    let tau = t / dt.clone();
    let tau2 = tau.clone() * tau.clone();
    let tau3 = tau2.clone() * tau.clone();
    let tau4 = tau3.clone() * tau.clone();

    let p_diff = p1 - p0;
    let v0dt = v0.clone() * dt.clone();
    let v1dt = v1 * dt.clone();

    let c10 = T::from_f64(10.0).unwrap();
    let c6 = T::from_f64(6.0).unwrap();
    let c4f = T::from_f64(4.0).unwrap();
    let c15 = T::from_f64(15.0).unwrap();
    let c8 = T::from_f64(8.0).unwrap();
    let c7 = T::from_f64(7.0).unwrap();
    let c3f = T::from_f64(3.0).unwrap();
    let c5f = T::from_f64(5.0).unwrap();

    let k3 = c10 * p_diff.clone() - c6.clone() * v0dt.clone() - c4f.clone() * v1dt.clone();
    let k4 = -c15 * p_diff.clone() + c8 * v0dt.clone() + c7 * v1dt.clone();
    let k5 = c6 * p_diff - c3f.clone() * v0dt - c3f.clone() * v1dt;

    let inv_dt = T::one() / dt;
    v0
        + c3f.clone() * k3 * tau2 * inv_dt.clone()
        + c4f.clone() * k4 * tau3 * inv_dt.clone()
        + c5f * k5 * tau4.clone() * inv_dt
}

/// Second derivative (acceleration) of [`quintic_interpolate`].
///
/// Differentiating `p'(t) = v0 + Σᵢ (i·kᵢ/dt)·τⁱ⁻¹` once more (with respect
/// to real time, noting `dτ/dt = 1/dt`) gives
/// `p''(t) = Σᵢ i·(i-1)·kᵢ/dt²·τⁱ⁻²` for i = 3..=5. The k₃ / k₄ / k₅
/// formulas are re-derived inside this function so they stay locked with
/// [`quintic_interpolate`] and [`quintic_derivative`].
pub fn quintic_second_derivative<T: RealField>(
    p0: T,
    p1: T,
    v0: T,
    v1: T,
    t: T,
    dt: T,
) -> T {
    let tau = t / dt.clone();
    let tau2 = tau.clone() * tau.clone();
    let tau3 = tau2.clone() * tau.clone();

    let p_diff = p1 - p0;
    let v0dt = v0.clone() * dt.clone();
    let v1dt = v1 * dt.clone();

    let c10 = T::from_f64(10.0).unwrap();
    let c6 = T::from_f64(6.0).unwrap();
    let c4f = T::from_f64(4.0).unwrap();
    let c15 = T::from_f64(15.0).unwrap();
    let c8 = T::from_f64(8.0).unwrap();
    let c7 = T::from_f64(7.0).unwrap();
    let c3f = T::from_f64(3.0).unwrap();
    let c12 = T::from_f64(12.0).unwrap();
    let c20 = T::from_f64(20.0).unwrap();

    let k3 = c10 * p_diff.clone() - c6.clone() * v0dt.clone() - c4f * v1dt.clone();
    let k4 = -c15 * p_diff.clone() + c8 * v0dt.clone() + c7 * v1dt.clone();
    let k5 = c6 * p_diff - c3f.clone() * v0dt - c3f * v1dt;

    let inv_dt = T::one() / dt;
    let inv_dt2 = inv_dt.clone() * inv_dt;

    // p''(t) = (6 k3) τ / dt² + (12 k4) τ² / dt² + (20 k5) τ³ / dt²
    let c6b = T::from_f64(6.0).unwrap();
    (c6b * k3 * tau + c12 * k4 * tau2 + c20 * k5 * tau3) * inv_dt2
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

    /// Per-joint trajectory acceleration `d²q/dt²` at time `t`. Linear segments
    /// have zero interior acceleration (they're a straight line) and we return
    /// zero outside the active interval too; the smooth kinds use their
    /// closed-form second derivatives.
    ///
    /// The host's computed-torque controller multiplies `q̈*` by the inertia
    /// matrix `M(q)` to get the motor torque needed to actually realise the
    /// commanded motion. Without acceleration feedforward, the PD has to
    /// derive that torque from position error alone, which means the motion
    /// always lags the trajectory and overshoots when leaving saturation.
    pub fn evaluate_acceleration(&self, t: T) -> Vec<T> {
        let n = self.q_start.len().min(self.q_end.len());
        let mut out = Vec::with_capacity(n);
        let zero = T::zero();
        let dur = self.duration.clone();

        if dur <= zero.clone() || t < zero.clone() || t > dur.clone() {
            for _ in 0..n {
                out.push(T::zero());
            }
            return out;
        }

        let s = t.clone() / dur.clone();

        for i in 0..n {
            let p0 = self.q_start[i].clone();
            let p1 = self.q_end[i].clone();
            let v = match self.kind {
                InterpolationKind::Linear => T::zero(),
                InterpolationKind::CubicSmooth => cubic_hermite_second_derivative(
                    p0,
                    p1,
                    zero.clone(),
                    zero.clone(),
                    s.clone(),
                    dur.clone(),
                ),
                InterpolationKind::QuinticSmooth => quintic_second_derivative(
                    p0,
                    p1,
                    zero.clone(),
                    zero.clone(),
                    t.clone(),
                    dur.clone(),
                ),
            };
            out.push(v);
        }
        out
    }

    /// Per-joint trajectory velocity `dq/dt` at time `t`. Out-of-range `t`
    /// returns zero (motion already completed or not yet started). For the
    /// `Linear` kind we report the constant slope `(q_end - q_start)/duration`
    /// inside the interval — the discontinuity at the endpoints would otherwise
    /// appear as a velocity step that no PD controller can track cleanly.
    ///
    /// Both `cubic_hermite_derivative` and `quintic_derivative` return derivatives
    /// in real-time units: `cubic` takes the normalised parameter `s ∈ [0, 1]`
    /// and the actual `dur` (its closed-form already divides by `dt`), while
    /// `quintic` takes the **real** time `t` and the same `dur`. Mixing those
    /// conventions is the only subtle bit; both branches end up with units of
    /// `q/s` once dur is applied correctly.
    pub fn evaluate_velocity(&self, t: T) -> Vec<T> {
        let n = self.q_start.len().min(self.q_end.len());
        let mut out = Vec::with_capacity(n);
        let zero = T::zero();
        let dur = self.duration.clone();

        // Outside the active interval the trajectory is at rest.
        if dur <= zero.clone() || t < zero.clone() || t > dur.clone() {
            for _ in 0..n {
                out.push(T::zero());
            }
            return out;
        }

        let s = t.clone() / dur.clone();

        for i in 0..n {
            let p0 = self.q_start[i].clone();
            let p1 = self.q_end[i].clone();
            let v = match self.kind {
                InterpolationKind::Linear => {
                    (p1 - p0) / dur.clone()
                }
                InterpolationKind::CubicSmooth => cubic_hermite_derivative(
                    p0,
                    p1,
                    zero.clone(),
                    zero.clone(),
                    s.clone(),
                    dur.clone(),
                ),
                InterpolationKind::QuinticSmooth => quintic_derivative(
                    p0,
                    p1,
                    zero.clone(),
                    zero.clone(),
                    t.clone(),
                    dur.clone(),
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

    /// Per-joint trajectory velocity at time `t`. Returns zeros before the
    /// first keyframe and after the last (the timeline is at rest there).
    /// Inside an active segment the velocity comes from the same
    /// [`PoseTransition`] used for position evaluation, so position and
    /// velocity stay consistent (their relationship is exactly differentiation).
    pub fn evaluate_velocity(&self, t: T) -> Vec<T> {
        let zero_vec = |n: usize| -> Vec<T> {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(T::zero());
            }
            v
        };
        if self.keyframes.is_empty() {
            return Vec::new();
        }
        let n = self.keyframes[0].q.len();
        if self.keyframes.len() == 1 {
            return zero_vec(n);
        }
        if t <= self.keyframes[0].time {
            return zero_vec(n);
        }
        if t >= self.keyframes.last().unwrap().time {
            return zero_vec(n);
        }
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
        traj.evaluate_velocity(local_t)
    }

    /// Per-joint trajectory acceleration at time `t`. Mirrors
    /// [`Self::evaluate_velocity`]: returns zeros outside the timeline and
    /// inside Linear segments, otherwise delegates to the matching
    /// [`PoseTransition`]. The result is in the same units as `q̈` (typically
    /// rad/s² for revolute and m/s² for prismatic joints) and is suitable
    /// for direct use as the feedforward `q̈*` term in computed-torque control.
    pub fn evaluate_acceleration(&self, t: T) -> Vec<T> {
        let zero_vec = |n: usize| -> Vec<T> {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(T::zero());
            }
            v
        };
        if self.keyframes.is_empty() {
            return Vec::new();
        }
        let n = self.keyframes[0].q.len();
        if self.keyframes.len() == 1 {
            return zero_vec(n);
        }
        if t <= self.keyframes[0].time {
            return zero_vec(n);
        }
        if t >= self.keyframes.last().unwrap().time {
            return zero_vec(n);
        }
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
        traj.evaluate_acceleration(local_t)
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
    fn quintic_zero_velocity_reaches_endpoint_no_overshoot() {
        // Regression for the 2× overshoot bug: at t=dt with v0=v1=0 the
        // polynomial must land exactly on p1 and never leave [p0, p1].
        let p0 = 0.0;
        let p1 = 5.0;
        let dt = 2.0;
        let y_end = quintic_interpolate(p0, p1, 0.0, 0.0, dt, dt);
        assert_relative_eq!(y_end, p1, epsilon = 1e-9);
        for k in 0..=20 {
            let t = dt * (k as f64) / 20.0;
            let y = quintic_interpolate(p0, p1, 0.0, 0.0, t, dt);
            assert!(
                y >= p0 - 1e-9 && y <= p1 + 1e-9,
                "Quintic out of [p0, p1] at t={t}: y={y}"
            );
        }
    }

    #[test]
    fn quintic_endpoints_with_general_velocity() {
        let p0 = 0.0;
        let p1 = 10.0;
        let v0 = 1.5;
        let v1 = -0.5;
        let dt = 2.0;
        let y_start = quintic_interpolate(p0, p1, v0, v1, 0.0, dt);
        let y_end = quintic_interpolate(p0, p1, v0, v1, dt, dt);
        assert_relative_eq!(y_start, p0, epsilon = 1e-9);
        assert_relative_eq!(y_end, p1, epsilon = 1e-9);
    }

    #[test]
    fn quintic_velocity_at_both_endpoints() {
        let p0 = 0.0;
        let p1 = 10.0;
        let v0 = 1.5;
        let v1 = -0.5;
        let dt = 2.0;
        let v_start = quintic_derivative(p0, p1, v0, v1, 0.0, dt);
        let v_end = quintic_derivative(p0, p1, v0, v1, dt, dt);
        assert_relative_eq!(v_start, v0, epsilon = 1e-9);
        assert_relative_eq!(v_end, v1, epsilon = 1e-9);
    }

    #[test]
    fn quintic_zero_acceleration_at_endpoints() {
        // p''(0) = p''(dt) = 0 by construction; verify numerically.
        let p0: f64 = 0.0;
        let p1: f64 = 7.0;
        let v0: f64 = 1.0;
        let v1: f64 = -2.0;
        let dt: f64 = 1.5;
        let h: f64 = 1e-5;
        let approx_a0 = (quintic_derivative(p0, p1, v0, v1, h, dt)
            - quintic_derivative(p0, p1, v0, v1, 0.0, dt))
            / h;
        let approx_a1 = (quintic_derivative(p0, p1, v0, v1, dt, dt)
            - quintic_derivative(p0, p1, v0, v1, dt - h, dt))
            / h;
        assert!(approx_a0.abs() < 1e-3, "a(0) ≈ 0 expected, got {approx_a0}");
        assert!(approx_a1.abs() < 1e-3, "a(dt) ≈ 0 expected, got {approx_a1}");
    }

    #[test]
    fn pose_transition_quintic_no_overshoot() {
        // Direct end-to-end test through PoseTransition: this is the path
        // MuJoCo's pose playback uses, and the bug manifested as the joint
        // reaching p1 at s=0.5 then continuing to 2·p1 at s=1.
        let traj = PoseTransition::new(
            vec![0.0_f64],
            vec![1.0],
            1.0,
            InterpolationKind::QuinticSmooth,
        );
        let q_end = traj.evaluate(1.0);
        assert_relative_eq!(q_end[0], 1.0, epsilon = 1e-9);
        // Mid-curve sanity: should be between 0 and 1.
        let q_half = traj.evaluate(0.5);
        assert!(
            q_half[0] >= 0.0 && q_half[0] <= 1.0,
            "Quintic mid-point should be inside [0, 1]: got {}",
            q_half[0]
        );
        // Sweep check.
        for k in 0..=20 {
            let s = k as f64 / 20.0;
            let q = traj.evaluate(s);
            assert!(
                q[0] >= -1e-9 && q[0] <= 1.0 + 1e-9,
                "Quintic out of [0,1] at s={s}: q={}",
                q[0]
            );
        }
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

    #[test]
    fn pose_transition_velocity_quintic_zero_at_endpoints() {
        // QuinticSmooth boundary conditions are v(0) = v(dt) = 0.
        let traj = PoseTransition::new(
            vec![0.0_f64],
            vec![1.0],
            2.0,
            InterpolationKind::QuinticSmooth,
        );
        let v0 = traj.evaluate_velocity(0.0);
        let v_end = traj.evaluate_velocity(2.0);
        let v_mid = traj.evaluate_velocity(1.0);
        assert_relative_eq!(v0[0], 0.0, epsilon = 1e-9);
        assert_relative_eq!(v_end[0], 0.0, epsilon = 1e-9);
        // Peak velocity for a 0→1 quintic move over 2 s is 1.875·Δp/dur
        // = 1.875 · (1 / 2) ≈ 0.9375; just check it's positive and far from 0.
        assert!(
            v_mid[0] > 0.5,
            "Expected non-trivial mid velocity, got {}",
            v_mid[0],
        );
    }

    #[test]
    fn pose_transition_velocity_linear_constant() {
        let traj = PoseTransition::new(
            vec![1.0_f64],
            vec![5.0],
            2.0,
            InterpolationKind::Linear,
        );
        // Linear: dq/dt = (5 - 1)/2 = 2 throughout the interior.
        for s in [0.1, 0.5, 0.99] {
            let v = traj.evaluate_velocity(s * 2.0);
            assert_relative_eq!(v[0], 2.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn pose_transition_velocity_outside_returns_zero() {
        let traj = PoseTransition::new(
            vec![0.0_f64],
            vec![1.0],
            1.0,
            InterpolationKind::QuinticSmooth,
        );
        let v_neg = traj.evaluate_velocity(-0.5);
        let v_post = traj.evaluate_velocity(2.0);
        assert_relative_eq!(v_neg[0], 0.0);
        assert_relative_eq!(v_post[0], 0.0);
    }

    #[test]
    fn pose_transition_velocity_matches_finite_difference() {
        // The closed-form derivative must agree with a centred finite
        // difference of evaluate(); if these two ever drift apart we have a
        // bug in either the polynomial or its derivative.
        let traj = PoseTransition::new(
            vec![0.0_f64, -1.0],
            vec![3.0, 2.0],
            1.5,
            InterpolationKind::QuinticSmooth,
        );
        let h = 1e-5;
        for &t in &[0.05_f64, 0.4, 0.75, 1.1, 1.45] {
            let v_closed = traj.evaluate_velocity(t);
            let q_plus = traj.evaluate(t + h);
            let q_minus = traj.evaluate(t - h);
            for i in 0..2 {
                let v_fd = (q_plus[i] - q_minus[i]) / (2.0 * h);
                assert_relative_eq!(v_closed[i], v_fd, epsilon = 1e-3);
            }
        }
    }

    #[test]
    fn keyframe_anim_velocity_zero_outside_range() {
        let kfs = vec![
            Keyframe::new(0.0, vec![0.0_f64], InterpolationKind::Linear),
            Keyframe::new(1.0, vec![1.0], InterpolationKind::QuinticSmooth),
        ];
        let anim = KeyframeAnimation::new(kfs);
        assert_relative_eq!(anim.evaluate_velocity(-0.1)[0], 0.0);
        assert_relative_eq!(anim.evaluate_velocity(1.5)[0], 0.0);
    }

    #[test]
    fn pose_transition_acceleration_quintic_endpoints_zero() {
        // Quintic boundary conditions explicitly demand p''(0) = p''(dt) = 0.
        let traj = PoseTransition::new(
            vec![0.0_f64],
            vec![1.0],
            2.0,
            InterpolationKind::QuinticSmooth,
        );
        let a0 = traj.evaluate_acceleration(0.0);
        let a_end = traj.evaluate_acceleration(2.0);
        assert!(
            a0[0].abs() < 1e-9,
            "Quintic accel at t=0 should be 0, got {}",
            a0[0]
        );
        assert!(
            a_end[0].abs() < 1e-9,
            "Quintic accel at t=dur should be 0, got {}",
            a_end[0]
        );
    }

    #[test]
    fn pose_transition_acceleration_linear_zero_interior() {
        // Linear is a straight line: zero acceleration throughout the interior.
        let traj = PoseTransition::new(
            vec![1.0_f64],
            vec![5.0],
            2.0,
            InterpolationKind::Linear,
        );
        for s in [0.1, 0.5, 0.99] {
            let a = traj.evaluate_acceleration(s * 2.0);
            assert!(
                a[0].abs() < 1e-12,
                "Linear accel must be 0 in interior, got {} at s={}",
                a[0],
                s,
            );
        }
    }

    #[test]
    fn pose_transition_acceleration_matches_finite_difference() {
        // Closed-form q̈ must agree with a centred finite difference of
        // evaluate_velocity.  Sweep through both Cubic and Quintic segments.
        for kind in [InterpolationKind::CubicSmooth, InterpolationKind::QuinticSmooth] {
            let traj = PoseTransition::new(
                vec![0.0_f64, -1.0],
                vec![3.0, 2.0],
                1.5,
                kind,
            );
            let h = 1e-5;
            for &t in &[0.05_f64, 0.4, 0.75, 1.1, 1.45] {
                let a_closed = traj.evaluate_acceleration(t);
                let v_plus = traj.evaluate_velocity(t + h);
                let v_minus = traj.evaluate_velocity(t - h);
                for i in 0..2 {
                    let a_fd = (v_plus[i] - v_minus[i]) / (2.0 * h);
                    assert_relative_eq!(a_closed[i], a_fd, epsilon = 1e-2);
                }
            }
        }
    }

    #[test]
    fn keyframe_anim_acceleration_within_segment() {
        // The keyframe's `kind` controls the segment LEADING UP TO that
        // keyframe (the first keyframe's kind is ignored — it has no
        // predecessor). So for a quintic segment from t=0 to t=1, the
        // SECOND keyframe must be QuinticSmooth.
        //
        // Quintic acceleration at s=0.5 is zero (inflection point) and at
        // the segment endpoints is zero (boundary conditions). Sample at
        // s=0.25 to confirm the closed-form is producing non-trivial values
        // in the interior.
        let kfs = vec![
            Keyframe::new(0.0, vec![0.0_f64], InterpolationKind::Linear),
            Keyframe::new(1.0, vec![2.0], InterpolationKind::QuinticSmooth),
            Keyframe::new(2.0, vec![1.0], InterpolationKind::Linear),
        ];
        let anim = KeyframeAnimation::new(kfs);
        // Linear segment 1→2 (straight line) → zero accel throughout.
        let a_linear_interior = anim.evaluate_acceleration(1.5);
        assert!(a_linear_interior[0].abs() < 1e-12);
        // Quintic at s=0.25 inside the segment t∈[0,1] → t = 0.25
        let a_quintic_quarter = anim.evaluate_acceleration(0.25);
        assert!(
            a_quintic_quarter[0].abs() > 0.5,
            "Quintic segment at s=0.25 should have non-trivial accel, got {}",
            a_quintic_quarter[0],
        );
        // Inflection point at s=0.5 → t=0.5
        let a_quintic_mid = anim.evaluate_acceleration(0.5);
        assert!(
            a_quintic_mid[0].abs() < 1e-9,
            "Quintic accel must vanish at the inflection, got {}",
            a_quintic_mid[0],
        );
    }

    #[test]
    fn keyframe_anim_acceleration_outside_range_zero() {
        let kfs = vec![
            Keyframe::new(0.0, vec![0.0_f64], InterpolationKind::QuinticSmooth),
            Keyframe::new(1.0, vec![1.0], InterpolationKind::QuinticSmooth),
        ];
        let anim = KeyframeAnimation::new(kfs);
        assert_relative_eq!(anim.evaluate_acceleration(-0.1)[0], 0.0);
        assert_relative_eq!(anim.evaluate_acceleration(1.5)[0], 0.0);
    }

    #[test]
    fn keyframe_anim_velocity_within_segment() {
        // Two-segment animation: linear then quintic. Verify each segment's
        // interior velocity matches its standalone PoseTransition reading.
        let kfs = vec![
            Keyframe::new(0.0, vec![0.0_f64], InterpolationKind::Linear),
            Keyframe::new(1.0, vec![2.0], InterpolationKind::Linear),
            Keyframe::new(3.0, vec![-1.0], InterpolationKind::QuinticSmooth),
        ];
        let anim = KeyframeAnimation::new(kfs);
        // First segment (linear, 0→2 over 1 s) → constant 2.0.
        assert_relative_eq!(
            anim.evaluate_velocity(0.5)[0],
            2.0,
            epsilon = 1e-9,
        );
        // Mid of second segment (quintic, 2→-1 over 2 s) → finite, negative.
        let v = anim.evaluate_velocity(2.0)[0];
        assert!(v < -0.1, "Expected negative mid velocity, got {v}");
    }
}
