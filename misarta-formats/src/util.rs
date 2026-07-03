//! Shared helpers for the format converters (crate-internal).

use std::collections::HashMap;

use misarta::native as mn;
use nalgebra as na;

/// Format an `f64` for text emission. Values that passed through an `f32`
/// stage (most editor-derived data) print as the short f32 literal
/// ("0.1") instead of the 17-digit f64 expansion of the f32 value.
pub(crate) fn fmt(v: f64) -> String {
    let f = v as f32;
    if f as f64 == v {
        format!("{f}")
    } else {
        format!("{v}")
    }
}

pub(crate) fn parse_f64_list(s: &str) -> Vec<f64> {
    s.split_whitespace().filter_map(|t| t.parse().ok()).collect()
}

pub(crate) fn parse_vec3_or(s: &str, default: [f64; 3]) -> [f64; 3] {
    let v = parse_f64_list(s);
    [
        v.first().copied().unwrap_or(default[0]),
        v.get(1).copied().unwrap_or(default[1]),
        v.get(2).copied().unwrap_or(default[2]),
    ]
}

/// Rotation of an [`mn::Origin`] as a unit quaternion (`quat` field takes
/// precedence over `rpy`; both absent = identity).
pub(crate) fn origin_rotation(o: &mn::Origin) -> na::UnitQuaternion<f64> {
    if let Some(q) = o.quat {
        na::UnitQuaternion::from_quaternion(na::Quaternion::new(q[3], q[0], q[1], q[2]))
    } else if let Some(rpy) = o.rpy {
        na::UnitQuaternion::from_euler_angles(rpy[0], rpy[1], rpy[2])
    } else {
        na::UnitQuaternion::identity()
    }
}

/// Full isometry of an [`mn::Origin`].
pub(crate) fn origin_iso(o: &mn::Origin) -> na::Isometry3<f64> {
    na::Isometry3::from_parts(
        na::Translation3::new(o.xyz[0], o.xyz[1], o.xyz[2]),
        origin_rotation(o),
    )
}

/// Build an isometry from a translation + `[x, y, z, w]` quaternion (the
/// config-type convention used by loop closures).
pub(crate) fn config_iso(t: [f64; 3], q_xyzw: [f64; 4]) -> na::Isometry3<f64> {
    na::Isometry3::from_parts(
        na::Translation3::new(t[0], t[1], t[2]),
        na::UnitQuaternion::from_quaternion(na::Quaternion::new(
            q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2],
        )),
    )
}

pub(crate) fn color_spec_to_rgba(c: &mn::ColorSpec) -> [f32; 4] {
    match c {
        mn::ColorSpec::Rgba(v) => *v,
        mn::ColorSpec::Hex(s) => parse_hex_color(s).unwrap_or([0.8, 0.8, 0.8, 1.0]),
    }
}

pub(crate) fn parse_hex_color(s: &str) -> Option<[f32; 4]> {
    let s = s.strip_prefix('#').unwrap_or(s);
    let byte = |i: usize| -> Option<f32> {
        let pair = s.get(i..i + 2)?;
        u8::from_str_radix(pair, 16).ok().map(|b| b as f32 / 255.0)
    };
    match s.len() {
        6 => Some([byte(0)?, byte(2)?, byte(4)?, 1.0]),
        8 => Some([byte(0)?, byte(2)?, byte(4)?, byte(6)?]),
        _ => None,
    }
}

/// Resolve a visual's colour: inline `color`, then `material` reference,
/// then the neutral default.
pub(crate) fn resolve_visual_rgba(
    v: &mn::Visual,
    materials: &HashMap<&str, [f32; 4]>,
) -> [f32; 4] {
    if let Some(c) = &v.color {
        return color_spec_to_rgba(c);
    }
    if let Some(name) = &v.material {
        if let Some(c) = materials.get(name.as_str()) {
            return *c;
        }
    }
    [0.8, 0.8, 0.8, 1.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_derived_values_print_short() {
        assert_eq!(fmt(0.1_f32 as f64), "0.1");
        assert_eq!(fmt(0.1_f64), "0.1");
        assert_eq!(fmt(1.0), "1");
        // A genuine f64 that is not an f32 value keeps full precision.
        let v = 0.1_f64 + 0.2_f64;
        assert_eq!(fmt(v), format!("{v}"));
    }

    #[test]
    fn hex_colors_parse() {
        assert_eq!(parse_hex_color("#ff0000"), Some([1.0, 0.0, 0.0, 1.0]));
        assert!(parse_hex_color("#00ff0080").is_some());
        assert!(parse_hex_color("nope").is_none());
    }
}
