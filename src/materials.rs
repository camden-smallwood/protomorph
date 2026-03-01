use asset_importer::{Material, TextureType};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::path::Path;

// ---------------------------------------------------------------------------
// Texture usage types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaterialTextureUsage {
    Diffuse,
    Specular,
    Ambient,
    Emissive,
    Height,
    Normal,
    Shininess,
    Opacity,
    Displacement,
    Lightmap,
    Reflection,
    BaseColor,
    NormalCamera,
    EmissionColor,
    Metalness,
    DiffuseRoughness,
    AmbientOcclusion,
    Sheen,
    Clearcoat,
    Transmission,
    Unknown,
}

pub struct MaterialTexture {
    pub usage: MaterialTextureUsage,
    pub path: String,
}

// ---------------------------------------------------------------------------
// Material property types
// ---------------------------------------------------------------------------

pub struct MaterialBaseProperties {
    pub name: String,
    pub two_sided: bool,
    pub opacity: f32,
    pub bump_scaling: f32,
    pub shininess: f32,
    pub reflectivity: f32,
    pub shininess_strength: f32,
    pub refracti: f32,
    pub color_diffuse: Vec3,
    pub color_ambient: Vec3,
    pub color_specular: Vec3,
    pub color_emissive: Vec3,
    pub color_transparent: Vec3,
    pub color_reflective: Vec3,
}

impl Default for MaterialBaseProperties {
    fn default() -> Self {
        Self {
            name: String::new(),
            two_sided: false,
            opacity: 1.0,
            bump_scaling: 1.0,
            shininess: 1.0,
            reflectivity: 1.0,
            shininess_strength: 1.0,
            refracti: 1.0,
            color_diffuse: Vec3::ONE,
            color_ambient: Vec3::ONE,
            color_specular: Vec3::ONE,
            color_emissive: Vec3::ZERO,
            color_transparent: Vec3::ZERO,
            color_reflective: Vec3::ZERO,
        }
    }
}

pub struct MaterialPbrProperties {
    pub base_color: Vec3,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub anisotropy_factor: f32,
}

impl Default for MaterialPbrProperties {
    fn default() -> Self {
        Self {
            base_color: Vec3::ONE,
            metallic_factor: 0.0,
            roughness_factor: 0.0,
            anisotropy_factor: 0.0,
        }
    }
}

#[derive(Default)]
pub struct MaterialSpecularProperties {
    pub specular_factor: f32,
    pub glossiness_factor: f32,
}

impl MaterialSpecularProperties {
    fn new() -> Self {
        Self {
            specular_factor: 0.5,
            glossiness_factor: 32.0,
        }
    }
}

pub struct MaterialEmissiveProperties {
    pub intensity: f32,
}

impl Default for MaterialEmissiveProperties {
    fn default() -> Self {
        Self { intensity: 1.0 }
    }
}

pub struct MaterialSheenProperties {
    pub color_factor: f32,
    pub roughness_factor: f32,
}

impl Default for MaterialSheenProperties {
    fn default() -> Self {
        Self {
            color_factor: 1.0,
            roughness_factor: 0.0,
        }
    }
}

pub struct MaterialClearcoatProperties {
    pub clearcoat_factor: f32,
    pub roughness_factor: f32,
}

impl Default for MaterialClearcoatProperties {
    fn default() -> Self {
        Self {
            clearcoat_factor: 1.0,
            roughness_factor: 0.0,
        }
    }
}

pub struct MaterialTransmissionProperties {
    pub transmission_factor: f32,
}

impl Default for MaterialTransmissionProperties {
    fn default() -> Self {
        Self {
            transmission_factor: 1.0,
        }
    }
}

pub struct MaterialVolumeProperties {
    pub thickness_factor: f32,
    pub attenuation_distance: f32,
    pub attenuation_color: Vec3,
}

impl Default for MaterialVolumeProperties {
    fn default() -> Self {
        Self {
            thickness_factor: 1.0,
            attenuation_distance: 1.0,
            attenuation_color: Vec3::ZERO,
        }
    }
}

#[derive(Default)]
pub struct MaterialAmbientOcclusionProperties {
    pub use_ao_texture: bool,
}

pub struct MaterialData {
    pub textures: Vec<MaterialTexture>,
    pub base: MaterialBaseProperties,
    pub pbr: MaterialPbrProperties,
    pub specular: MaterialSpecularProperties,
    pub emissive: MaterialEmissiveProperties,
    pub sheen: MaterialSheenProperties,
    pub clearcoat: MaterialClearcoatProperties,
    pub transmission: MaterialTransmissionProperties,
    pub volume: MaterialVolumeProperties,
    pub ambient_occlusion: MaterialAmbientOcclusionProperties,
    pub has_transparency: bool,
}

// ---------------------------------------------------------------------------
// Assimp property extraction helpers
// ---------------------------------------------------------------------------

fn find_float_property(material: &Material, key: &str, default: f32) -> f32 {
    material
        .get_float_property_str(key)
        .ok()
        .flatten()
        .unwrap_or(default)
}

fn find_color_property(material: &Material, key: &str, default: Vec3) -> Vec3 {
    material
        .get_color_property_str(key)
        .ok()
        .flatten()
        .map(|c| Vec3::new(c.x, c.y, c.z))
        .unwrap_or(default)
}

fn find_int_property(material: &Material, key: &str, default: i32) -> i32 {
    material
        .get_integer_property_str(key)
        .ok()
        .flatten()
        .unwrap_or(default)
}

fn find_string_property(material: &Material, key: &str) -> Option<String> {
    material.get_string_property_str(key).ok().flatten()
}

fn find_texture_path(material: &Material, tex_type: TextureType) -> Option<String> {
    material.texture(tex_type, 0).map(|t| t.path)
}

// ---------------------------------------------------------------------------
// MaterialData loading from assimp
// ---------------------------------------------------------------------------

impl MaterialData {
    pub fn from_assimp(
        mat: &Material,
        model_dir: &Path,
        resolve_fn: &dyn Fn(&Path, &str) -> Option<String>,
    ) -> Self {
        // Load base properties
        let base = MaterialBaseProperties {
            name: find_string_property(mat, "?mat.name").unwrap_or_default(),
            two_sided: find_int_property(mat, "$mat.twosided", 0) != 0,
            opacity: find_float_property(mat, "$mat.opacity", 1.0),
            bump_scaling: find_float_property(mat, "$mat.bumpscaling", 1.0),
            shininess: find_float_property(mat, "$mat.shininess", 1.0),
            reflectivity: find_float_property(mat, "$mat.reflectivity", 1.0),
            shininess_strength: find_float_property(mat, "$mat.shinpercent", 1.0),
            refracti: find_float_property(mat, "$mat.refracti", 1.0),
            color_diffuse: find_color_property(mat, "$clr.diffuse", Vec3::ONE),
            color_ambient: find_color_property(mat, "$clr.ambient", Vec3::ONE),
            color_specular: find_color_property(mat, "$clr.specular", Vec3::ONE),
            color_emissive: find_color_property(mat, "$clr.emissive", Vec3::ZERO),
            color_transparent: find_color_property(mat, "$clr.transparent", Vec3::ZERO),
            color_reflective: find_color_property(mat, "$clr.reflective", Vec3::ZERO),
        };

        let specular = MaterialSpecularProperties {
            specular_factor: find_float_property(mat, "$mat.specularFactor", 0.5),
            glossiness_factor: find_float_property(mat, "$mat.glossinessFactor", 32.0),
        };

        let pbr = MaterialPbrProperties {
            base_color: find_color_property(mat, "$clr.base", Vec3::ONE),
            metallic_factor: find_float_property(mat, "$mat.metallicFactor", 0.0),
            roughness_factor: find_float_property(mat, "$mat.roughnessFactor", 0.0),
            anisotropy_factor: find_float_property(mat, "$mat.anisotropyFactor", 0.0),
        };

        let emissive = MaterialEmissiveProperties {
            intensity: find_float_property(mat, "$mat.emissiveIntensity", 1.0),
        };

        let sheen = MaterialSheenProperties {
            color_factor: find_float_property(mat, "$mat.sheenColorFactor", 1.0),
            roughness_factor: find_float_property(mat, "$mat.sheenRoughnessFactor", 0.0),
        };

        let clearcoat = MaterialClearcoatProperties {
            clearcoat_factor: find_float_property(mat, "$mat.clearcoatFactor", 1.0),
            roughness_factor: find_float_property(mat, "$mat.clearcoatRoughnessFactor", 0.0),
        };

        let transmission = MaterialTransmissionProperties {
            transmission_factor: find_float_property(mat, "$mat.transmissionFactor", 1.0),
        };

        let volume = MaterialVolumeProperties {
            thickness_factor: find_float_property(mat, "$mat.thicknessFactor", 1.0),
            attenuation_distance: find_float_property(mat, "$mat.attenuationDistance", 1.0),
            attenuation_color: find_color_property(mat, "$mat.attenuationColor", Vec3::ZERO),
        };

        let ambient_occlusion = MaterialAmbientOcclusionProperties {
            use_ao_texture: find_int_property(mat, "$mat.useAOMap", 0) != 0,
        };

        // Load textures
        let texture_mappings: &[(TextureType, MaterialTextureUsage)] = &[
            (TextureType::Diffuse, MaterialTextureUsage::Diffuse),
            (TextureType::Specular, MaterialTextureUsage::Specular),
            (TextureType::Ambient, MaterialTextureUsage::Ambient),
            (TextureType::Emissive, MaterialTextureUsage::Emissive),
            (TextureType::Normals, MaterialTextureUsage::Normal),
            (TextureType::Height, MaterialTextureUsage::Height),
            (TextureType::Shininess, MaterialTextureUsage::Shininess),
            (TextureType::Opacity, MaterialTextureUsage::Opacity),
            (
                TextureType::Displacement,
                MaterialTextureUsage::Displacement,
            ),
            (TextureType::Lightmap, MaterialTextureUsage::Lightmap),
            (TextureType::Reflection, MaterialTextureUsage::Reflection),
        ];

        let mut textures = Vec::new();
        let mut has_transparency = false;

        for &(ref tex_type, usage) in texture_mappings {
            if let Some(raw_path) = find_texture_path(mat, *tex_type) {
                if let Some(resolved) = resolve_fn(model_dir, &raw_path) {
                    textures.push(MaterialTexture {
                        usage,
                        path: resolved,
                    });
                    if usage == MaterialTextureUsage::Opacity {
                        has_transparency = true;
                    }
                } else {
                    eprintln!("WARNING: could not resolve texture path: {raw_path}");
                }
            }
        }

        Self {
            textures,
            base,
            pbr,
            specular,
            emissive,
            sheen,
            clearcoat,
            transmission,
            volume,
            ambient_occlusion,
            has_transparency,
        }
    }

    pub fn find_texture(&self, usage: MaterialTextureUsage) -> Option<&str> {
        self.textures
            .iter()
            .find(|t| t.usage == usage)
            .map(|t| t.path.as_str())
    }
}

// ---------------------------------------------------------------------------
// GPU material uniform (matches geometry.wgsl)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuMaterialProps {
    pub diffuse_color: [f32; 3],
    pub bump_scaling: f32,
    pub ambient_color: [f32; 3],
    pub ambient_amount: f32,
    pub specular_color: [f32; 3],
    pub specular_amount: f32,
    pub emissive_color: [f32; 3],
    pub emissive_intensity: f32,
    pub specular_shininess: f32,
    pub _pad: [f32; 3],
}

impl GpuMaterialProps {
    pub fn from_material(mat: &MaterialData) -> Self {
        Self {
            diffuse_color: mat.base.color_diffuse.into(),
            bump_scaling: mat.base.bump_scaling,
            ambient_color: mat.base.color_ambient.into(),
            ambient_amount: 0.1, // hardcoded, matching C render.c:472
            specular_color: mat.base.color_specular.into(),
            specular_amount: mat.specular.specular_factor,
            emissive_color: if mat.base.color_emissive == Vec3::ZERO
                && mat.find_texture(MaterialTextureUsage::Emissive).is_some()
            {
                [1.0; 3]
            } else {
                mat.base.color_emissive.into()
            },
            emissive_intensity: mat.emissive.intensity,
            specular_shininess: mat.specular.glossiness_factor,
            _pad: [0.0; 3],
        }
    }
}
