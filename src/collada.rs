//! Collada (`.dae`) mesh reader and writer.
//!
//! Parses a subset of the Collada 1.4.1 / 1.5.0 schema that covers the vast
//! majority of robot URDF visual meshes exported from Blender, SolidWorks,
//! Fusion 360, and similar tools:
//!
//! - `<library_images>` — texture image references
//! - `<library_effects>` — Phong / Lambert / Blinn material effects
//! - `<library_materials>` — named material ↔ effect bindings
//! - `<library_geometries>` — `<triangles>` and `<polylist>` mesh primitives
//! - `<library_visual_scenes>` — instance geometry with material binding
//! - `<up_axis>` — Y_UP / Z_UP / X_UP correction
//!
//! # Example
//!
//! ```no_run
//! use misarta::collada;
//! use std::path::Path;
//!
//! let mesh = collada::load_dae(Path::new("robot.dae")).unwrap();
//! println!("{} verts, {} tris, {} materials, {} submeshes",
//!     mesh.num_vertices(), mesh.num_triangles(),
//!     mesh.num_materials(), mesh.num_submeshes());
//! ```

use crate::mesh::{Material, MeshData, SubMesh};
use nalgebra::{Point2, Point3, Vector3};
use roxmltree::{Document, Node};
use std::collections::HashMap;
use std::path::Path;

// ─── Public API ─────────────────────────────────────────────────────────────

/// Load a Collada `.dae` file into a [`MeshData`].
///
/// Materials, textures and sub-meshes are fully populated when the DAE
/// contains the corresponding library elements.
pub fn load_dae(path: &Path) -> Result<MeshData, String> {
    let xml = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read DAE file {}: {e}", path.display()))?;
    let dae_dir = path.parent().unwrap_or(Path::new("."));
    load_dae_string(&xml, dae_dir)
}

/// Load a Collada mesh from an XML string.
///
/// `dae_dir` is used to resolve relative texture paths.
pub fn load_dae_string(xml: &str, dae_dir: &Path) -> Result<MeshData, String> {
    let doc = Document::parse(xml).map_err(|e| format!("DAE XML parse error: {e}"))?;
    let root = doc.root_element();

    // ── Up-axis ────────────────────────────────────────────────────────
    let up_axis = root
        .children()
        .find(|n| n.tag_name().name() == "asset")
        .and_then(|a| a.children().find(|n| n.tag_name().name() == "up_axis"))
        .and_then(|n| n.text())
        .unwrap_or("Y_UP");

    // ── Textures (images) ──────────────────────────────────────────────
    let images = parse_library_images(&root, dae_dir);

    // ── Effects ────────────────────────────────────────────────────────
    let effects = parse_library_effects(&root, &images);

    // ── Materials → effect id ──────────────────────────────────────────
    let mat_to_effect = parse_library_materials(&root);

    // ── Geometries ─────────────────────────────────────────────────────
    let raw_geoms = parse_library_geometries(&root, up_axis);

    // ── Visual scene: bind materials to geometry instances ──────────────
    let bindings = parse_visual_scene_bindings(&root);

    // ── Merge everything into a single MeshData ────────────────────────
    merge_geometries(&raw_geoms, &effects, &mat_to_effect, &bindings)
}

/// Write a [`MeshData`] to a Collada `.dae` file.
///
/// Materials, textures and sub-meshes are preserved.  The file is always
/// written with `Z_UP` up-axis.
pub fn write_dae(mesh: &MeshData, path: &Path) -> Result<(), String> {
    let xml = write_dae_string(mesh);
    std::fs::write(path, &xml)
        .map_err(|e| format!("cannot write DAE file {}: {e}", path.display()))
}

/// Serialise a [`MeshData`] to a Collada XML string.
pub fn write_dae_string(mesh: &MeshData) -> String {
    let mut s = String::with_capacity(4096);
    s.push_str("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
    s.push_str("<COLLADA xmlns=\"http://www.collada.org/2005/11/COLLADASchema\" version=\"1.4.1\">\n");

    // ── asset ──────────────────────────────────────────────────────────
    s.push_str("  <asset>\n    <up_axis>Z_UP</up_axis>\n  </asset>\n");

    // ── library_images ─────────────────────────────────────────────────
    let has_textures = mesh.materials.iter().any(|m| m.texture_diffuse.is_some());
    if has_textures {
        s.push_str("  <library_images>\n");
        for (mi, mat) in mesh.materials.iter().enumerate() {
            if let Some(ref tex) = mat.texture_diffuse {
                s.push_str(&format!(
                    "    <image id=\"img_{mi}\" name=\"img_{mi}\">\n      <init_from>{tex}</init_from>\n    </image>\n"
                ));
            }
        }
        s.push_str("  </library_images>\n");
    }

    // ── library_effects ────────────────────────────────────────────────
    s.push_str("  <library_effects>\n");
    for (mi, mat) in mesh.materials.iter().enumerate() {
        s.push_str(&format!("    <effect id=\"effect_{mi}\">\n      <profile_COMMON>\n"));
        // If there's a diffuse texture, write a sampler.
        if mat.texture_diffuse.is_some() {
            s.push_str(&format!(
                "        <newparam sid=\"surface_{mi}\"><surface type=\"2D\"><init_from>img_{mi}</init_from></surface></newparam>\n\
                 \x20\x20\x20\x20\x20\x20\x20\x20<newparam sid=\"sampler_{mi}\"><sampler2D><source>surface_{mi}</source></sampler2D></newparam>\n"
            ));
        }
        s.push_str("        <technique sid=\"common\">\n          <phong>\n");
        write_color_element(&mut s, "emission", &mat.emission);
        write_color_element(&mut s, "ambient", &mat.ambient);
        if mat.texture_diffuse.is_some() {
            s.push_str(&format!(
                "            <diffuse><texture texture=\"sampler_{mi}\" texcoord=\"UVMap\"/></diffuse>\n"
            ));
        } else {
            write_color_element(&mut s, "diffuse", &mat.diffuse);
        }
        write_color_element(&mut s, "specular", &mat.specular);
        s.push_str(&format!(
            "            <shininess><float>{}</float></shininess>\n",
            mat.shininess
        ));
        s.push_str("          </phong>\n        </technique>\n      </profile_COMMON>\n    </effect>\n");
    }
    s.push_str("  </library_effects>\n");

    // ── library_materials ──────────────────────────────────────────────
    s.push_str("  <library_materials>\n");
    for (mi, mat) in mesh.materials.iter().enumerate() {
        let name = if mat.name.is_empty() {
            format!("material_{mi}")
        } else {
            mat.name.clone()
        };
        s.push_str(&format!(
            "    <material id=\"{name}\" name=\"{name}\"><instance_effect url=\"#effect_{mi}\"/></material>\n"
        ));
    }
    s.push_str("  </library_materials>\n");

    // ── library_geometries ─────────────────────────────────────────────
    s.push_str("  <library_geometries>\n");
    // Group triangles by submesh.  If no submeshes, write all at once.
    let submeshes: Vec<&SubMesh> = if mesh.submeshes.is_empty() {
        // Synthesise a single submesh covering everything.
        vec![]
    } else {
        mesh.submeshes.iter().collect()
    };

    if submeshes.is_empty() {
        // single geometry, no material split
        write_geometry_element(&mut s, "mesh_0", mesh, 0, mesh.num_triangles(), None);
    } else {
        for (si, sm) in submeshes.iter().enumerate() {
            let mat_sym = sm.material_index.map(|mi| {
                if mi < mesh.materials.len() && !mesh.materials[mi].name.is_empty() {
                    mesh.materials[mi].name.clone()
                } else {
                    format!("material_{mi}")
                }
            });
            write_geometry_element(&mut s, &format!("mesh_{si}"), mesh, sm.tri_start, sm.tri_count, mat_sym.as_deref());
        }
    }
    s.push_str("  </library_geometries>\n");

    // ── library_visual_scenes ──────────────────────────────────────────
    s.push_str("  <library_visual_scenes>\n    <visual_scene id=\"Scene\" name=\"Scene\">\n");
    let geom_count = if submeshes.is_empty() { 1 } else { submeshes.len() };
    for si in 0..geom_count {
        s.push_str(&format!("      <node id=\"node_{si}\" name=\"node_{si}\" type=\"NODE\">\n"));
        s.push_str(&format!("        <instance_geometry url=\"#mesh_{si}\">\n"));
        // bind_material
        let mat_id = if submeshes.is_empty() {
            None
        } else {
            submeshes[si].material_index.map(|mi| {
                if mi < mesh.materials.len() && !mesh.materials[mi].name.is_empty() {
                    mesh.materials[mi].name.clone()
                } else {
                    format!("material_{mi}")
                }
            })
        };
        if let Some(ref mid) = mat_id {
            s.push_str("          <bind_material><technique_common>\n");
            s.push_str(&format!(
                "            <instance_material symbol=\"{mid}\" target=\"#{mid}\"/>\n"
            ));
            s.push_str("          </technique_common></bind_material>\n");
        }
        s.push_str("        </instance_geometry>\n      </node>\n");
    }
    s.push_str("    </visual_scene>\n  </library_visual_scenes>\n");

    // ── scene ──────────────────────────────────────────────────────────
    s.push_str("  <scene><instance_visual_scene url=\"#Scene\"/></scene>\n");
    s.push_str("</COLLADA>\n");
    s
}

// ─── Internal: helpers ──────────────────────────────────────────────────────

fn write_color_element(s: &mut String, tag: &str, c: &[f64; 4]) {
    s.push_str(&format!(
        "            <{tag}><color>{} {} {} {}</color></{tag}>\n",
        c[0], c[1], c[2], c[3]
    ));
}

/// Write a single `<geometry>` element covering a slice of `mesh.indices`.
fn write_geometry_element(
    s: &mut String,
    id: &str,
    mesh: &MeshData,
    tri_start: usize,
    tri_count: usize,
    material_symbol: Option<&str>,
) {
    s.push_str(&format!("    <geometry id=\"{id}\" name=\"{id}\">\n      <mesh>\n"));

    // Collect unique vertex indices used by this submesh.
    let tri_end = (tri_start + tri_count).min(mesh.indices.len());
    let slice = &mesh.indices[tri_start..tri_end];

    // Source: positions
    s.push_str(&format!("        <source id=\"{id}-positions\">\n"));
    let pos_count = mesh.vertices.len();
    s.push_str(&format!(
        "          <float_array id=\"{id}-positions-array\" count=\"{}\">\n            ",
        pos_count * 3
    ));
    for (i, v) in mesh.vertices.iter().enumerate() {
        if i > 0 { s.push(' '); }
        s.push_str(&format!("{} {} {}", v.x, v.y, v.z));
    }
    s.push_str(&format!(
        "\n          </float_array>\n          <technique_common>\n            <accessor source=\"#{id}-positions-array\" count=\"{pos_count}\" stride=\"3\">\n              <param name=\"X\" type=\"float\"/><param name=\"Y\" type=\"float\"/><param name=\"Z\" type=\"float\"/>\n            </accessor>\n          </technique_common>\n        </source>\n"
    ));

    // Source: normals (use vertex_normals if available, else face_normals)
    let has_vn = mesh.has_vertex_normals();
    s.push_str(&format!("        <source id=\"{id}-normals\">\n"));
    if has_vn {
        let n_count = mesh.vertex_normals.len();
        s.push_str(&format!(
            "          <float_array id=\"{id}-normals-array\" count=\"{}\">\n            ",
            n_count * 3
        ));
        for (i, n) in mesh.vertex_normals.iter().enumerate() {
            if i > 0 { s.push(' '); }
            s.push_str(&format!("{} {} {}", n.x, n.y, n.z));
        }
    } else {
        let fn_count = mesh.face_normals.len();
        s.push_str(&format!(
            "          <float_array id=\"{id}-normals-array\" count=\"{}\">\n            ",
            fn_count * 3
        ));
        for (i, n) in mesh.face_normals.iter().enumerate() {
            if i > 0 { s.push(' '); }
            s.push_str(&format!("{} {} {}", n.x, n.y, n.z));
        }
    }
    s.push_str(&format!(
        "\n          </float_array>\n          <technique_common>\n            <accessor source=\"#{id}-normals-array\" count=\"{}\" stride=\"3\">\n              <param name=\"X\" type=\"float\"/><param name=\"Y\" type=\"float\"/><param name=\"Z\" type=\"float\"/>\n            </accessor>\n          </technique_common>\n        </source>\n",
        if has_vn { mesh.vertex_normals.len() } else { mesh.face_normals.len() }
    ));

    // Source: texcoords (optional)
    let has_uv = mesh.has_texcoords();
    if has_uv {
        let uv_count = mesh.texcoords.len();
        s.push_str(&format!("        <source id=\"{id}-texcoords\">\n"));
        s.push_str(&format!(
            "          <float_array id=\"{id}-texcoords-array\" count=\"{}\">\n            ",
            uv_count * 2
        ));
        for (i, uv) in mesh.texcoords.iter().enumerate() {
            if i > 0 { s.push(' '); }
            s.push_str(&format!("{} {}", uv.x, uv.y));
        }
        s.push_str(&format!(
            "\n          </float_array>\n          <technique_common>\n            <accessor source=\"#{id}-texcoords-array\" count=\"{uv_count}\" stride=\"2\">\n              <param name=\"S\" type=\"float\"/><param name=\"T\" type=\"float\"/>\n            </accessor>\n          </technique_common>\n        </source>\n"
        ));
    }

    // <vertices>
    s.push_str(&format!(
        "        <vertices id=\"{id}-vertices\">\n          <input semantic=\"POSITION\" source=\"#{id}-positions\"/>\n        </vertices>\n"
    ));

    // <triangles>
    let mat_attr = material_symbol.map(|m| format!(" material=\"{m}\"")).unwrap_or_default();
    let mut offset = 0;
    s.push_str(&format!(
        "        <triangles count=\"{tri_count}\"{mat_attr}>\n          <input semantic=\"VERTEX\" source=\"#{id}-vertices\" offset=\"{offset}\"/>\n"
    ));
    offset += 1;
    if has_vn {
        s.push_str(&format!(
            "          <input semantic=\"NORMAL\" source=\"#{id}-normals\" offset=\"{offset}\"/>\n"
        ));
        offset += 1;
    }
    if has_uv {
        s.push_str(&format!(
            "          <input semantic=\"TEXCOORD\" source=\"#{id}-texcoords\" offset=\"{offset}\" set=\"0\"/>\n"
        ));
        offset += 1;
    }
    let _ = offset; // suppress unused
    let _stride = 1 + if has_vn { 1 } else { 0 } + if has_uv { 1 } else { 0 };
    s.push_str("          <p>");
    for (ti, tri) in slice.iter().enumerate() {
        if ti > 0 { s.push(' '); }
        let _face_ni = tri_start + ti;
        for &vi in tri {
            s.push_str(&format!("{vi}"));
            if has_vn {
                s.push_str(&format!(" {vi}"));
            }
            if has_uv {
                s.push_str(&format!(" {vi}"));
            }
            s.push(' ');
        }
    }
    // Trim trailing space.
    if s.ends_with(' ') {
        s.pop();
    }
    s.push_str("</p>\n        </triangles>\n");

    s.push_str("      </mesh>\n    </geometry>\n");
}

// ─── Internal: DAE Reading ──────────────────────────────────────────────────

/// image-id → resolved texture file path
fn parse_library_images(root: &Node, dae_dir: &Path) -> HashMap<String, String> {
    let mut m = HashMap::new();
    let lib = match root.children().find(|n| n.tag_name().name() == "library_images") {
        Some(l) => l,
        None => return m,
    };
    for img in lib.children().filter(|n| n.tag_name().name() == "image") {
        let id = img.attribute("id").unwrap_or("").to_string();
        if let Some(init) = img.descendants().find(|n| n.tag_name().name() == "init_from") {
            if let Some(text) = init.text() {
                let path_str = text.trim();
                // Resolve relative to DAE directory.
                let resolved = if Path::new(path_str).is_relative() {
                    dae_dir.join(path_str).to_string_lossy().into_owned()
                } else {
                    path_str.to_string()
                };
                m.insert(id, resolved);
            }
        }
    }
    m
}

/// effect-id → Material
fn parse_library_effects(root: &Node, images: &HashMap<String, String>) -> HashMap<String, Material> {
    let mut m = HashMap::new();
    let lib = match root.children().find(|n| n.tag_name().name() == "library_effects") {
        Some(l) => l,
        None => return m,
    };
    for effect in lib.children().filter(|n| n.tag_name().name() == "effect") {
        let id = effect.attribute("id").unwrap_or("").to_string();
        let mut mat = Material::default();
        mat.name = id.clone();

        // Find profile_COMMON → technique → phong|lambert|blinn
        if let Some(profile) = effect.descendants().find(|n| n.tag_name().name() == "profile_COMMON") {
            // Collect newparam sampler → surface → image id mappings.
            let sampler_to_image = resolve_sampler_chain(&profile, images);

            if let Some(technique) = profile.descendants().find(|n| n.tag_name().name() == "technique") {
                for shader in technique.children() {
                    let tag = shader.tag_name().name();
                    if tag == "phong" || tag == "lambert" || tag == "blinn" {
                        mat.diffuse = parse_color_or_texture(&shader, "diffuse", &sampler_to_image, &mut mat.texture_diffuse);
                        mat.specular = parse_color_or_texture(&shader, "specular", &sampler_to_image, &mut None);
                        mat.ambient = parse_color_or_texture(&shader, "ambient", &sampler_to_image, &mut None);
                        mat.emission = parse_color_or_texture(&shader, "emission", &sampler_to_image, &mut None);
                        mat.shininess = parse_float_param(&shader, "shininess");
                        break;
                    }
                }
            }
        }

        m.insert(id, mat);
    }
    m
}

/// Follow the Collada sampler → surface → image chain.
/// Returns sampler-sid → resolved image path.
fn resolve_sampler_chain(profile: &Node, images: &HashMap<String, String>) -> HashMap<String, String> {
    let mut surface_to_image: HashMap<String, String> = HashMap::new();
    let mut sampler_to_surface: HashMap<String, String> = HashMap::new();

    for np in profile.children().filter(|n| n.tag_name().name() == "newparam") {
        let sid = np.attribute("sid").unwrap_or("").to_string();
        if let Some(surf) = np.children().find(|n| n.tag_name().name() == "surface") {
            if let Some(init) = surf.children().find(|n| n.tag_name().name() == "init_from") {
                if let Some(t) = init.text() {
                    surface_to_image.insert(sid.clone(), t.trim().to_string());
                }
            }
        }
        if let Some(sam) = np.children().find(|n| n.tag_name().name() == "sampler2D") {
            if let Some(src) = sam.children().find(|n| n.tag_name().name() == "source") {
                if let Some(t) = src.text() {
                    sampler_to_surface.insert(sid.clone(), t.trim().to_string());
                }
            }
        }
    }

    let mut result = HashMap::new();
    for (sampler_sid, surface_sid) in &sampler_to_surface {
        if let Some(image_id) = surface_to_image.get(surface_sid) {
            if let Some(path) = images.get(image_id) {
                result.insert(sampler_sid.clone(), path.clone());
            }
        }
    }
    result
}

fn parse_color_or_texture(
    shader: &Node,
    tag: &str,
    sampler_to_image: &HashMap<String, String>,
    texture_out: &mut Option<String>,
) -> [f64; 4] {
    let default = match tag {
        "diffuse" | "ambient" => [0.8, 0.8, 0.8, 1.0],
        _ => [0.0, 0.0, 0.0, 1.0],
    };
    let elem = match shader.children().find(|n| n.tag_name().name() == tag) {
        Some(e) => e,
        None => return default,
    };
    // Check for <texture>
    if let Some(tex) = elem.children().find(|n| n.tag_name().name() == "texture") {
        let tex_ref = tex.attribute("texture").unwrap_or("");
        if let Some(path) = sampler_to_image.get(tex_ref) {
            if let Some(out) = texture_out {
                let _ = out; // already set
            } else {
                *texture_out = Some(path.clone());
            }
        }
    }
    // Check for <color>
    if let Some(color) = elem.children().find(|n| n.tag_name().name() == "color") {
        return parse_color_text(color.text().unwrap_or(""));
    }
    default
}

fn parse_float_param(shader: &Node, tag: &str) -> f64 {
    shader
        .children()
        .find(|n| n.tag_name().name() == tag)
        .and_then(|e| e.children().find(|n| n.tag_name().name() == "float"))
        .and_then(|f| f.text())
        .and_then(|t| t.trim().parse::<f64>().ok())
        .unwrap_or(0.0)
}

fn parse_color_text(text: &str) -> [f64; 4] {
    let vals: Vec<f64> = text.split_whitespace().filter_map(|s| s.parse().ok()).collect();
    [
        vals.first().copied().unwrap_or(0.0),
        vals.get(1).copied().unwrap_or(0.0),
        vals.get(2).copied().unwrap_or(0.0),
        vals.get(3).copied().unwrap_or(1.0),
    ]
}

/// material-id → effect-id
fn parse_library_materials(root: &Node) -> HashMap<String, String> {
    let mut m = HashMap::new();
    let lib = match root.children().find(|n| n.tag_name().name() == "library_materials") {
        Some(l) => l,
        None => return m,
    };
    for mat in lib.children().filter(|n| n.tag_name().name() == "material") {
        let id = mat.attribute("id").unwrap_or("").to_string();
        if let Some(ie) = mat.children().find(|n| n.tag_name().name() == "instance_effect") {
            if let Some(url) = ie.attribute("url") {
                m.insert(id, url.trim_start_matches('#').to_string());
            }
        }
    }
    m
}

// ── Geometry parsing ────────────────────────────────────────────────────────

struct RawGeometry {
    id: String,
    vertices: Vec<Point3<f64>>,
    normals: Vec<Vector3<f64>>,
    texcoords: Vec<Point2<f64>>,
    /// (pos_idx, normal_idx or usize::MAX, uv_idx or usize::MAX) × 3 per triangle
    triangles: Vec<[usize; 9]>,
    material_symbol: Option<String>,
}

fn parse_library_geometries(root: &Node, up_axis: &str) -> Vec<RawGeometry> {
    let mut geoms = Vec::new();
    let lib = match root.children().find(|n| n.tag_name().name() == "library_geometries") {
        Some(l) => l,
        None => return geoms,
    };
    for geom in lib.children().filter(|n| n.tag_name().name() == "geometry") {
        let geom_id = geom.attribute("id").unwrap_or("").to_string();
        let mesh_el = match geom.children().find(|n| n.tag_name().name() == "mesh") {
            Some(m) => m,
            None => continue,
        };

        // Collect all <source>s by id.
        let mut sources: HashMap<String, Vec<f64>> = HashMap::new();
        for src in mesh_el.children().filter(|n| n.tag_name().name() == "source") {
            let src_id = src.attribute("id").unwrap_or("").to_string();
            if let Some(fa) = src.children().find(|n| n.tag_name().name() == "float_array") {
                let vals: Vec<f64> = fa
                    .text()
                    .unwrap_or("")
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                sources.insert(src_id, vals);
            }
        }

        // <vertices> → resolve POSITION semantic to actual source.
        let mut vertex_source_id = String::new();
        if let Some(verts_el) = mesh_el.children().find(|n| n.tag_name().name() == "vertices") {
            let vert_id = verts_el.attribute("id").unwrap_or("").to_string();
            for inp in verts_el.children().filter(|n| n.tag_name().name() == "input") {
                if inp.attribute("semantic") == Some("POSITION") {
                    if let Some(src_url) = inp.attribute("source") {
                        let real_id = src_url.trim_start_matches('#').to_string();
                        // Move the source data to the vertices id for later lookup.
                        if let Some(data) = sources.get(&real_id).cloned() {
                            sources.insert(vert_id.clone(), data);
                        }
                        vertex_source_id = vert_id.clone();
                    }
                }
            }
        }

        // Parse each <triangles> and <polylist>.
        for prim_el in mesh_el.children() {
            let prim_tag = prim_el.tag_name().name();
            if prim_tag != "triangles" && prim_tag != "polylist" {
                continue;
            }

            let material_symbol = prim_el.attribute("material").map(|s| s.to_string());

            // Parse <input> elements.
            struct InputDesc {
                semantic: String,
                source_id: String,
                offset: usize,
            }
            let mut inputs: Vec<InputDesc> = Vec::new();
            let mut max_offset = 0_usize;
            for inp in prim_el.children().filter(|n| n.tag_name().name() == "input") {
                let semantic = inp.attribute("semantic").unwrap_or("").to_string();
                let source_id = inp
                    .attribute("source")
                    .unwrap_or("")
                    .trim_start_matches('#')
                    .to_string();
                let offset: usize = inp
                    .attribute("offset")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                if offset > max_offset {
                    max_offset = offset;
                }
                inputs.push(InputDesc {
                    semantic,
                    source_id,
                    offset,
                });
            }
            let stride = max_offset + 1;

            // Determine which input corresponds to what.
            let vertex_offset = inputs.iter().find(|i| i.semantic == "VERTEX").map(|i| i.offset);
            let normal_input = inputs.iter().find(|i| i.semantic == "NORMAL");
            let texcoord_input = inputs.iter().find(|i| i.semantic == "TEXCOORD");

            let pos_data = sources.get(&vertex_source_id).cloned().unwrap_or_default();
            let normal_data = normal_input
                .and_then(|i| sources.get(&i.source_id))
                .cloned()
                .unwrap_or_default();
            let uv_data = texcoord_input
                .and_then(|i| sources.get(&i.source_id))
                .cloned()
                .unwrap_or_default();

            // Build vertex/normal/uv arrays.
            let raw_verts: Vec<Point3<f64>> = pos_data
                .chunks_exact(3)
                .map(|c| fix_up_axis(c[0], c[1], c[2], up_axis))
                .collect();
            let raw_normals: Vec<Vector3<f64>> = normal_data
                .chunks_exact(3)
                .map(|c| {
                    let p = fix_up_axis(c[0], c[1], c[2], up_axis);
                    Vector3::new(p.x, p.y, p.z)
                })
                .collect();
            let raw_uvs: Vec<Point2<f64>> = uv_data
                .chunks_exact(2)
                .map(|c| Point2::new(c[0], c[1]))
                .collect();

            // Read <p> index data.
            let p_text = prim_el
                .children()
                .find(|n| n.tag_name().name() == "p")
                .and_then(|n| n.text())
                .unwrap_or("");
            let p_vals: Vec<usize> = p_text
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            // Build triangle list.
            let tris = if prim_tag == "triangles" {
                build_triangles_from_p(&p_vals, stride, vertex_offset, normal_input.map(|i| i.offset), texcoord_input.map(|i| i.offset))
            } else {
                // polylist — read <vcount>
                let vcount_text = prim_el
                    .children()
                    .find(|n| n.tag_name().name() == "vcount")
                    .and_then(|n| n.text())
                    .unwrap_or("");
                let vcounts: Vec<usize> = vcount_text
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                build_triangles_from_polylist(&p_vals, &vcounts, stride, vertex_offset, normal_input.map(|i| i.offset), texcoord_input.map(|i| i.offset))
            };

            geoms.push(RawGeometry {
                id: geom_id.clone(),
                vertices: raw_verts,
                normals: raw_normals,
                texcoords: raw_uvs,
                triangles: tris,
                material_symbol,
            });
        }
    }
    geoms
}

fn fix_up_axis(x: f64, y: f64, z: f64, up: &str) -> Point3<f64> {
    match up {
        "Z_UP" => Point3::new(x, y, z),
        "X_UP" => Point3::new(y, z, x),
        _ /* Y_UP */ => Point3::new(x, z, -y),
    }
}

fn build_triangles_from_p(
    p: &[usize],
    stride: usize,
    v_off: Option<usize>,
    n_off: Option<usize>,
    uv_off: Option<usize>,
) -> Vec<[usize; 9]> {
    let mut tris = Vec::new();
    let vo = v_off.unwrap_or(0);
    let no = n_off.unwrap_or(0);
    let uo = uv_off.unwrap_or(0);
    let has_n = n_off.is_some();
    let has_uv = uv_off.is_some();
    let tri_stride = stride * 3;
    let mut i = 0;
    while i + tri_stride <= p.len() {
        let mut tri = [usize::MAX; 9];
        for k in 0..3 {
            let base = i + k * stride;
            tri[k * 3] = p[base + vo];
            tri[k * 3 + 1] = if has_n { p[base + no] } else { usize::MAX };
            tri[k * 3 + 2] = if has_uv { p[base + uo] } else { usize::MAX };
        }
        tris.push(tri);
        i += tri_stride;
    }
    tris
}

fn build_triangles_from_polylist(
    p: &[usize],
    vcounts: &[usize],
    stride: usize,
    v_off: Option<usize>,
    n_off: Option<usize>,
    uv_off: Option<usize>,
) -> Vec<[usize; 9]> {
    let mut tris = Vec::new();
    let vo = v_off.unwrap_or(0);
    let no = n_off.unwrap_or(0);
    let uo = uv_off.unwrap_or(0);
    let has_n = n_off.is_some();
    let has_uv = uv_off.is_some();

    let read_vert = |idx: usize| -> (usize, usize, usize) {
        let base = idx * stride;
        if base + stride > p.len() {
            return (0, usize::MAX, usize::MAX);
        }
        (
            p[base + vo],
            if has_n { p[base + no] } else { usize::MAX },
            if has_uv { p[base + uo] } else { usize::MAX },
        )
    };

    let mut vertex_cursor = 0usize;
    for &vc in vcounts {
        if vc < 3 {
            vertex_cursor += vc;
            continue;
        }
        // Fan triangulation: (0, k, k+1) for k in 1..vc-1.
        let v0 = read_vert(vertex_cursor);
        for k in 1..vc - 1 {
            let v1 = read_vert(vertex_cursor + k);
            let v2 = read_vert(vertex_cursor + k + 1);
            tris.push([
                v0.0, v0.1, v0.2,
                v1.0, v1.1, v1.2,
                v2.0, v2.1, v2.2,
            ]);
        }
        vertex_cursor += vc;
    }
    tris
}

/// geometry-id → (material-symbol → material-id)
fn parse_visual_scene_bindings(root: &Node) -> HashMap<String, HashMap<String, String>> {
    let mut result: HashMap<String, HashMap<String, String>> = HashMap::new();
    let lib = match root.children().find(|n| n.tag_name().name() == "library_visual_scenes") {
        Some(l) => l,
        None => return result,
    };
    for scene in lib.descendants() {
        if scene.tag_name().name() != "instance_geometry" {
            continue;
        }
        let url = scene.attribute("url").unwrap_or("").trim_start_matches('#').to_string();
        let mut bind_map: HashMap<String, String> = HashMap::new();
        for im in scene.descendants().filter(|n| n.tag_name().name() == "instance_material") {
            let symbol = im.attribute("symbol").unwrap_or("").to_string();
            let target = im.attribute("target").unwrap_or("").trim_start_matches('#').to_string();
            bind_map.insert(symbol, target);
        }
        result.insert(url, bind_map);
    }
    result
}

// ── Merge into MeshData ─────────────────────────────────────────────────────

fn merge_geometries(
    raw_geoms: &[RawGeometry],
    effects: &HashMap<String, Material>,
    mat_to_effect: &HashMap<String, String>,
    bindings: &HashMap<String, HashMap<String, String>>,
) -> Result<MeshData, String> {
    let mut all_vertices: Vec<Point3<f64>> = Vec::new();
    let mut all_normals: Vec<Vector3<f64>> = Vec::new();
    let mut all_uvs: Vec<Point2<f64>> = Vec::new();
    let mut all_indices: Vec<[u32; 3]> = Vec::new();
    let mut all_face_normals: Vec<Vector3<f64>> = Vec::new();
    let mut materials: Vec<Material> = Vec::new();
    let mut submeshes: Vec<SubMesh> = Vec::new();
    let mut material_name_to_idx: HashMap<String, usize> = HashMap::new();

    let has_any_normals = raw_geoms.iter().any(|g| !g.normals.is_empty());
    let has_any_uvs = raw_geoms.iter().any(|g| !g.texcoords.is_empty());

    for rg in raw_geoms {
        // Resolve material for this primitive.
        let resolved_mat_id = rg.material_symbol.as_ref().and_then(|sym| {
            // First check visual_scene bindings.
            for (_geom_url, bind_map) in bindings {
                if let Some(mat_id) = bind_map.get(sym) {
                    return Some(mat_id.clone());
                }
            }
            // Fall back: symbol IS the material id.
            Some(sym.clone())
        });

        let mat_idx = resolved_mat_id.as_ref().and_then(|mat_id| {
            // Already registered?
            if let Some(&idx) = material_name_to_idx.get(mat_id) {
                return Some(idx);
            }
            // Find effect.
            let effect_id = mat_to_effect.get(mat_id)?;
            let base_mat = effects.get(effect_id)?;
            let idx = materials.len();
            let mut mat = base_mat.clone();
            mat.name = mat_id.clone();
            materials.push(mat);
            material_name_to_idx.insert(mat_id.clone(), idx);
            Some(idx)
        });

        // Remap vertices.  We build a per-raw-geometry unique-vertex table.
        // Each unique combo of (pos_idx, normal_idx, uv_idx) becomes one vertex.
        let mut combo_map: HashMap<(usize, usize, usize), u32> = HashMap::new();
        let _vertex_base = all_vertices.len() as u32;
        let tri_start = all_indices.len();

        for tri9 in &rg.triangles {
            let mut idx3 = [0u32; 3];
            for k in 0..3 {
                let pi = tri9[k * 3];
                let ni = tri9[k * 3 + 1];
                let ui = tri9[k * 3 + 2];
                let key = (pi, ni, ui);
                let vi = *combo_map.entry(key).or_insert_with(|| {
                    let new_idx = all_vertices.len() as u32;
                    let pos = rg.vertices.get(pi).copied().unwrap_or(Point3::origin());
                    all_vertices.push(pos);
                    if has_any_normals {
                        let n = if ni != usize::MAX {
                            rg.normals.get(ni).copied().unwrap_or(Vector3::zeros())
                        } else {
                            Vector3::zeros()
                        };
                        all_normals.push(n);
                    }
                    if has_any_uvs {
                        let uv = if ui != usize::MAX {
                            rg.texcoords.get(ui).copied().unwrap_or(Point2::origin())
                        } else {
                            Point2::origin()
                        };
                        all_uvs.push(uv);
                    }
                    new_idx
                });
                idx3[k] = vi;
            }
            // Compute face normal.
            let v0 = &all_vertices[idx3[0] as usize];
            let v1 = &all_vertices[idx3[1] as usize];
            let v2 = &all_vertices[idx3[2] as usize];
            let n = (v1 - v0).cross(&(v2 - v0));
            let len = n.norm();
            all_face_normals.push(if len > 1e-30 { n / len } else { Vector3::z() });
            all_indices.push(idx3);
        }

        let tri_count = all_indices.len() - tri_start;
        if tri_count > 0 {
            submeshes.push(SubMesh {
                name: rg.id.clone(),
                tri_start,
                tri_count,
                material_index: mat_idx,
            });
        }
    }

    Ok(MeshData {
        vertices: all_vertices,
        indices: all_indices,
        face_normals: all_face_normals,
        vertex_normals: all_normals,
        texcoords: all_uvs,
        materials,
        submeshes,
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Minimal Collada with one triangle, one material, one diffuse colour.
    const MINIMAL_DAE: &str = r##"<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><up_axis>Z_UP</up_axis></asset>
  <library_effects>
    <effect id="eff0">
      <profile_COMMON>
        <technique sid="common">
          <phong>
            <diffuse><color>1 0 0 1</color></diffuse>
            <specular><color>0.5 0.5 0.5 1</color></specular>
            <shininess><float>50</float></shininess>
          </phong>
        </technique>
      </profile_COMMON>
    </effect>
  </library_effects>
  <library_materials>
    <material id="red_mat" name="red_mat">
      <instance_effect url="#eff0"/>
    </material>
  </library_materials>
  <library_geometries>
    <geometry id="geom0" name="geom0">
      <mesh>
        <source id="geom0-positions">
          <float_array id="geom0-positions-array" count="9">0 0 0 1 0 0 0 1 0</float_array>
          <technique_common>
            <accessor source="#geom0-positions-array" count="3" stride="3">
              <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <source id="geom0-normals">
          <float_array id="geom0-normals-array" count="3">0 0 1</float_array>
          <technique_common>
            <accessor source="#geom0-normals-array" count="1" stride="3">
              <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="geom0-vertices">
          <input semantic="POSITION" source="#geom0-positions"/>
        </vertices>
        <triangles count="1" material="red_mat">
          <input semantic="VERTEX" source="#geom0-vertices" offset="0"/>
          <input semantic="NORMAL" source="#geom0-normals" offset="1"/>
          <p>0 0 1 0 2 0</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
      <node id="node0" type="NODE">
        <instance_geometry url="#geom0">
          <bind_material><technique_common>
            <instance_material symbol="red_mat" target="#red_mat"/>
          </technique_common></bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>"##;

    #[test]
    fn parse_minimal_dae() {
        let mesh = load_dae_string(MINIMAL_DAE, Path::new(".")).unwrap();
        assert_eq!(mesh.num_vertices(), 3);
        assert_eq!(mesh.num_triangles(), 1);
        assert_eq!(mesh.num_materials(), 1);
        assert_eq!(mesh.num_submeshes(), 1);

        // Check material
        let mat = &mesh.materials[0];
        assert_eq!(mat.name, "red_mat");
        assert_relative_eq!(mat.diffuse[0], 1.0);
        assert_relative_eq!(mat.diffuse[1], 0.0);
        assert_relative_eq!(mat.specular[0], 0.5);
        assert_relative_eq!(mat.shininess, 50.0);

        // Check submesh
        let sm = &mesh.submeshes[0];
        assert_eq!(sm.tri_start, 0);
        assert_eq!(sm.tri_count, 1);
        assert_eq!(sm.material_index, Some(0));
    }

    #[test]
    fn roundtrip_write_read() {
        let mesh = load_dae_string(MINIMAL_DAE, Path::new(".")).unwrap();
        let xml = write_dae_string(&mesh);
        let mesh2 = load_dae_string(&xml, Path::new(".")).unwrap();

        assert_eq!(mesh.num_vertices(), mesh2.num_vertices());
        assert_eq!(mesh.num_triangles(), mesh2.num_triangles());
        assert_eq!(mesh.num_materials(), mesh2.num_materials());

        // Material properties should survive roundtrip.
        assert_relative_eq!(mesh2.materials[0].diffuse[0], 1.0);
        assert_relative_eq!(mesh2.materials[0].shininess, 50.0);
    }

    #[test]
    fn up_axis_y_up() {
        // Same mesh but Y_UP — vertex (0, 0, 1) in Z_UP becomes (0, 1, 0) in Y_UP.
        let xml = MINIMAL_DAE.replace("Z_UP", "Y_UP")
            .replace("0 0 0 1 0 0 0 1 0", "0 0 0 1 0 0 0 0 -1");
        let mesh = load_dae_string(&xml, Path::new(".")).unwrap();
        // After up-axis correction (Y_UP → Z_UP internal): (x, z, -y)
        // (0,0,-1) → (0,-1,0) ... wait, let's check the actual vertex.
        // Original verts in Y_UP: (0,0,0), (1,0,0), (0,0,-1)
        // fix_up_axis(0, 0, -1, "Y_UP") → (0, -1, 0)
        let v2 = &mesh.vertices[2];
        assert_relative_eq!(v2.y, -1.0, epsilon = 1e-10);
        assert_relative_eq!(v2.z, 0.0, epsilon = 1e-10);
    }

    /// Two submeshes with different materials in one DAE.
    const TWO_MATERIAL_DAE: &str = r##"<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><up_axis>Z_UP</up_axis></asset>
  <library_effects>
    <effect id="eff_red"><profile_COMMON><technique sid="common"><phong>
      <diffuse><color>1 0 0 1</color></diffuse>
    </phong></technique></profile_COMMON></effect>
    <effect id="eff_blue"><profile_COMMON><technique sid="common"><phong>
      <diffuse><color>0 0 1 1</color></diffuse>
    </phong></technique></profile_COMMON></effect>
  </library_effects>
  <library_materials>
    <material id="mat_red"><instance_effect url="#eff_red"/></material>
    <material id="mat_blue"><instance_effect url="#eff_blue"/></material>
  </library_materials>
  <library_geometries>
    <geometry id="g0">
      <mesh>
        <source id="g0-pos"><float_array id="g0-pos-arr" count="9">0 0 0 1 0 0 0 1 0</float_array>
          <technique_common><accessor source="#g0-pos-arr" count="3" stride="3">
            <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
          </accessor></technique_common>
        </source>
        <vertices id="g0-v"><input semantic="POSITION" source="#g0-pos"/></vertices>
        <triangles count="1" material="mat_red">
          <input semantic="VERTEX" source="#g0-v" offset="0"/>
          <p>0 1 2</p>
        </triangles>
      </mesh>
    </geometry>
    <geometry id="g1">
      <mesh>
        <source id="g1-pos"><float_array id="g1-pos-arr" count="9">2 0 0 3 0 0 2 1 0</float_array>
          <technique_common><accessor source="#g1-pos-arr" count="3" stride="3">
            <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
          </accessor></technique_common>
        </source>
        <vertices id="g1-v"><input semantic="POSITION" source="#g1-pos"/></vertices>
        <triangles count="1" material="mat_blue">
          <input semantic="VERTEX" source="#g1-v" offset="0"/>
          <p>0 1 2</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene">
      <node id="n0" type="NODE">
        <instance_geometry url="#g0">
          <bind_material><technique_common>
            <instance_material symbol="mat_red" target="#mat_red"/>
          </technique_common></bind_material>
        </instance_geometry>
      </node>
      <node id="n1" type="NODE">
        <instance_geometry url="#g1">
          <bind_material><technique_common>
            <instance_material symbol="mat_blue" target="#mat_blue"/>
          </technique_common></bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>"##;

    #[test]
    fn two_submeshes_two_materials() {
        let mesh = load_dae_string(TWO_MATERIAL_DAE, Path::new(".")).unwrap();
        assert_eq!(mesh.num_triangles(), 2);
        assert_eq!(mesh.num_materials(), 2);
        assert_eq!(mesh.num_submeshes(), 2);

        // First submesh → red.
        let sm0 = &mesh.submeshes[0];
        assert_eq!(sm0.tri_count, 1);
        let mat0 = &mesh.materials[sm0.material_index.unwrap()];
        assert_relative_eq!(mat0.diffuse[0], 1.0);
        assert_relative_eq!(mat0.diffuse[2], 0.0);

        // Second submesh → blue.
        let sm1 = &mesh.submeshes[1];
        assert_eq!(sm1.tri_count, 1);
        let mat1 = &mesh.materials[sm1.material_index.unwrap()];
        assert_relative_eq!(mat1.diffuse[0], 0.0);
        assert_relative_eq!(mat1.diffuse[2], 1.0);

        // material_for_triangle
        let m_t0 = mesh.material_for_triangle(0).unwrap();
        assert_relative_eq!(m_t0.diffuse[0], 1.0);
        let m_t1 = mesh.material_for_triangle(1).unwrap();
        assert_relative_eq!(m_t1.diffuse[2], 1.0);
    }

    #[test]
    fn texture_reference_parsing() {
        let xml = r##"<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><up_axis>Z_UP</up_axis></asset>
  <library_images>
    <image id="tex0"><init_from>textures/diffuse.png</init_from></image>
  </library_images>
  <library_effects>
    <effect id="eff0"><profile_COMMON>
      <newparam sid="surf0"><surface type="2D"><init_from>tex0</init_from></surface></newparam>
      <newparam sid="samp0"><sampler2D><source>surf0</source></sampler2D></newparam>
      <technique sid="common"><phong>
        <diffuse><texture texture="samp0" texcoord="UVMap"/></diffuse>
      </phong></technique>
    </profile_COMMON></effect>
  </library_effects>
  <library_materials>
    <material id="textured_mat"><instance_effect url="#eff0"/></material>
  </library_materials>
  <library_geometries>
    <geometry id="g0"><mesh>
      <source id="g0-pos"><float_array id="g0-pa" count="9">0 0 0 1 0 0 0 1 0</float_array>
        <technique_common><accessor source="#g0-pa" count="3" stride="3">
          <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
        </accessor></technique_common>
      </source>
      <source id="g0-uv"><float_array id="g0-uva" count="6">0 0 1 0 0 1</float_array>
        <technique_common><accessor source="#g0-uva" count="3" stride="2">
          <param name="S" type="float"/><param name="T" type="float"/>
        </accessor></technique_common>
      </source>
      <vertices id="g0-v"><input semantic="POSITION" source="#g0-pos"/></vertices>
      <triangles count="1" material="textured_mat">
        <input semantic="VERTEX" source="#g0-v" offset="0"/>
        <input semantic="TEXCOORD" source="#g0-uv" offset="1" set="0"/>
        <p>0 0 1 1 2 2</p>
      </triangles>
    </mesh></geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene">
      <node id="n0" type="NODE">
        <instance_geometry url="#g0">
          <bind_material><technique_common>
            <instance_material symbol="textured_mat" target="#textured_mat"/>
          </technique_common></bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>"##;

        let mesh = load_dae_string(xml, Path::new("/robot/meshes")).unwrap();
        assert_eq!(mesh.num_vertices(), 3);
        assert!(mesh.has_texcoords());
        assert_eq!(mesh.num_materials(), 1);

        let mat = &mesh.materials[0];
        assert!(mat.texture_diffuse.is_some());
        let tex_path = mat.texture_diffuse.as_ref().unwrap();
        assert!(tex_path.contains("textures/diffuse.png"));

        // UV coords
        assert_relative_eq!(mesh.texcoords[0].x, 0.0);
        assert_relative_eq!(mesh.texcoords[1].x, 1.0);
    }

    #[test]
    fn polylist_parsing() {
        let xml = r##"<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><up_axis>Z_UP</up_axis></asset>
  <library_geometries>
    <geometry id="g0"><mesh>
      <source id="g0-pos"><float_array id="g0-pa" count="12">0 0 0  1 0 0  1 1 0  0 1 0</float_array>
        <technique_common><accessor source="#g0-pa" count="4" stride="3">
          <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
        </accessor></technique_common>
      </source>
      <vertices id="g0-v"><input semantic="POSITION" source="#g0-pos"/></vertices>
      <polylist count="1">
        <input semantic="VERTEX" source="#g0-v" offset="0"/>
        <vcount>4</vcount>
        <p>0 1 2 3</p>
      </polylist>
    </mesh></geometry>
  </library_geometries>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>"##;

        let mesh = load_dae_string(xml, Path::new(".")).unwrap();
        // Quad fan-triangulated → 2 triangles.
        assert_eq!(mesh.num_triangles(), 2);
        assert_eq!(mesh.num_vertices(), 4);
    }

    #[test]
    fn two_material_roundtrip() {
        let mesh = load_dae_string(TWO_MATERIAL_DAE, Path::new(".")).unwrap();
        let xml2 = write_dae_string(&mesh);
        let mesh2 = load_dae_string(&xml2, Path::new(".")).unwrap();

        assert_eq!(mesh2.num_triangles(), mesh.num_triangles());
        assert_eq!(mesh2.num_materials(), mesh.num_materials());
        assert_eq!(mesh2.num_submeshes(), mesh.num_submeshes());

        // Verify materials survived.
        for mi in 0..mesh.num_materials() {
            assert_relative_eq!(mesh2.materials[mi].diffuse[0], mesh.materials[mi].diffuse[0], epsilon = 1e-6);
            assert_relative_eq!(mesh2.materials[mi].diffuse[2], mesh.materials[mi].diffuse[2], epsilon = 1e-6);
        }
    }
}
