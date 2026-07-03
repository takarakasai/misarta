//! Foreign robot-description format conversions for misarta.
//!
//! Boundary rule (see articara `doc/refactor_20260702.md` §4.7): the
//! misarta core owns the computational model, the `.misa` master format
//! and mesh-file I/O (STL / OBJ / DAE — inseparable from
//! `native::mesh_load`); **robot description formats** authored by other
//! ecosystems live here and convert to / from
//! [`misarta::native::MisaFile`]:
//!
//! - `mjcf` — MuJoCo XML (A4, ported from articara)
//! - `usd`  — USD ASCII (A5, ported from articara)
//! - `urdf` / `sdf` — moved here from the misarta core (A5). These are
//!   direct `Model<f64>` loaders/writers; `sdf` additionally converts
//!   to / from [`misarta::native::MisaFile`].

mod util;

pub mod mjcf;
pub mod sdf;
pub mod urdf;
pub mod usd;
