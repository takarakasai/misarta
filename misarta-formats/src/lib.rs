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
//! - `usd`  — USD ASCII (A5)
//! - `urdf` / `sdf` — to be moved here from the misarta core (A5)

pub mod mjcf;
