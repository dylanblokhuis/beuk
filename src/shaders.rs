use std::{
    collections::{BTreeMap, HashMap},
    ffi::CString,
    path::Path,
};

use ash::vk::{self, Filter, SamplerAddressMode, SamplerMipmapMode};
use rspirv_reflect::BindingCount;
use shaderc::CompilationArtifact;

use crate::{chunky_list::TempList, ctx::SamplerDesc, memory::ImmutableShaderInfo};

#[derive(Clone)]
pub struct Shader {
    pub kind: ShaderKind,
    pub spirv_descripor_set_layouts: StageDescriptorSetLayouts,
    pub entry_point: String,
    pub entry_point_cstr: CString,
    pub module: vk::ShaderModule,
    pub spirv: Vec<u8>,
}

#[derive(Clone)]
pub enum ShaderKind {
    Vertex,
    Fragment,
    Compute,
}
impl ShaderKind {
    pub fn to_shaderc_kind(&self) -> shaderc::ShaderKind {
        match self {
            Self::Vertex => shaderc::ShaderKind::Vertex,
            Self::Fragment => shaderc::ShaderKind::Fragment,
            Self::Compute => shaderc::ShaderKind::Compute,
        }
    }

    pub fn to_vk_shader_stage_flag(&self) -> vk::ShaderStageFlags {
        match self {
            Self::Vertex => vk::ShaderStageFlags::VERTEX,
            Self::Fragment => vk::ShaderStageFlags::FRAGMENT,
            Self::Compute => vk::ShaderStageFlags::COMPUTE,
        }
    }
}

type DescriptorSetLayout = BTreeMap<u32, rspirv_reflect::DescriptorInfo>;
type StageDescriptorSetLayouts = BTreeMap<u32, DescriptorSetLayout>;

impl Shader {
    pub fn new(
        device: &ash::Device,
        spirv: CompilationArtifact,
        kind: ShaderKind,
        entry_point: &str,
    ) -> Self {
        let refl_info = rspirv_reflect::Reflection::new_from_spirv(spirv.as_binary_u8()).unwrap();
        let descriptor_sets = refl_info.get_descriptor_sets().unwrap();

        let module = unsafe {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(spirv.as_binary()),
                    None,
                )
                .expect("Vertex shader module error")
        };

        Self {
            kind,
            spirv_descripor_set_layouts: descriptor_sets,
            entry_point: entry_point.to_string(),
            entry_point_cstr: CString::new(entry_point).unwrap(),
            module,
            spirv: spirv.as_binary_u8().to_vec(),
        }
    }

    pub fn create_descriptor_sets(
        &self,
        device: &ash::Device,
        shader_info: &ImmutableShaderInfo,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        set_layout_info: &[HashMap<u32, vk::DescriptorType>],
    ) -> Vec<vk::DescriptorSet> {
        let mut descriptor_pool_sizes: Vec<vk::DescriptorPoolSize> = Vec::new();
        for bindings in set_layout_info.iter() {
            for ty in bindings.values() {
                if let Some(mut dps) = descriptor_pool_sizes.iter_mut().find(|item| item.ty == *ty)
                {
                    dps.descriptor_count += 1;
                } else {
                    descriptor_pool_sizes.push(vk::DescriptorPoolSize {
                        ty: *ty,
                        descriptor_count: shader_info.max_descriptor_count,
                    })
                }
            }
        }

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&descriptor_pool_sizes)
            .max_sets(2);

        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap()
        };

        let desc_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(descriptor_set_layouts);
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&desc_alloc_info).unwrap() };

        descriptor_sets
    }

    // pub fn ext_shader_create_info(&self) -> ShaderCreateInfoEXT {
    //     ShaderCreateInfoEXT::default()
    //         .name(self.entry_point_cstr.as_c_str())
    //         .code(&self.spirv)
    //         .code_type(ShaderCodeTypeEXT::SPIRV)
    //         .stage(self.kind.to_vk_shader_stage_flag())
    // }

    pub fn create_descriptor_set_layouts(
        &self,
        device: &ash::Device,
        shader_info: &mut ImmutableShaderInfo,
    ) -> (
        Vec<vk::DescriptorSetLayout>,
        Vec<HashMap<u32, vk::DescriptorType>>,
    ) {
        let samplers = TempList::new();
        let set_count = self
            .spirv_descripor_set_layouts
            .keys()
            .map(|set_index| *set_index + 1)
            .max()
            .unwrap_or(0u32);

        let mut set_layouts: Vec<vk::DescriptorSetLayout> = Vec::with_capacity(set_count as usize);
        let mut set_layout_info: Vec<HashMap<u32, vk::DescriptorType>> =
            Vec::with_capacity(set_count as usize);

        for set_index in 0..set_count {
            let stage_flags = vk::ShaderStageFlags::ALL;
            let set = self.spirv_descripor_set_layouts.get(&set_index);

            if let Some(set) = set {
                let mut bindings: Vec<vk::DescriptorSetLayoutBinding> =
                    Vec::with_capacity(set.len());
                let mut binding_flags: Vec<vk::DescriptorBindingFlags> =
                    vec![vk::DescriptorBindingFlags::PARTIALLY_BOUND; set.len()];

                let mut set_layout_create_flags = vk::DescriptorSetLayoutCreateFlags::empty();

                let yuv_2_plane = *samplers.add(
                    shader_info
                        .get_yuv_conversion_sampler(
                            device,
                            SamplerDesc {
                                texel_filter: Filter::LINEAR,
                                mipmap_mode: SamplerMipmapMode::LINEAR,
                                // Must always be CLAMP_TO_EDGE
                                address_modes: SamplerAddressMode::CLAMP_TO_EDGE,
                            },
                            vk::Format::G8_B8R8_2PLANE_420_UNORM,
                        )
                        .1,
                );

                let yuv_3_plane = *samplers.add(
                    shader_info
                        .get_yuv_conversion_sampler(
                            device,
                            SamplerDesc {
                                texel_filter: Filter::LINEAR,
                                mipmap_mode: SamplerMipmapMode::LINEAR,
                                // Must always be CLAMP_TO_EDGE
                                address_modes: SamplerAddressMode::CLAMP_TO_EDGE,
                            },
                            vk::Format::G8_B8_R8_3PLANE_420_UNORM,
                        )
                        .1,
                );

                let yuv_2_plane_hdr = *samplers.add(
                    shader_info
                        .get_yuv_conversion_sampler(
                            device,
                            SamplerDesc {
                                texel_filter: Filter::LINEAR,
                                mipmap_mode: SamplerMipmapMode::LINEAR,
                                // Must always be CLAMP_TO_EDGE
                                address_modes: SamplerAddressMode::CLAMP_TO_EDGE,
                            },
                            vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
                        )
                        .1,
                );

                let yuv_3_plane_hdr = *samplers.add(
                    shader_info
                        .get_yuv_conversion_sampler(
                            device,
                            SamplerDesc {
                                texel_filter: Filter::LINEAR,
                                mipmap_mode: SamplerMipmapMode::LINEAR,
                                // Must always be CLAMP_TO_EDGE
                                address_modes: SamplerAddressMode::CLAMP_TO_EDGE,
                            },
                            vk::Format::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16,
                        )
                        .1,
                );

                let yuv_samplers = &[yuv_2_plane, yuv_3_plane, yuv_2_plane_hdr, yuv_3_plane_hdr];

                for (binding_index, binding) in set.iter() {
                    // if binding.name.starts_with("u_") {
                    //     binding_flags[bindings.len()] =
                    //         vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                    //             | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING
                    //             | vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    //             | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;

                    //     set_layout_create_flags |=
                    //         vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL;
                    // }

                    let descriptor_count: u32 = if binding.name.starts_with("u_") {
                        shader_info.max_descriptor_count
                    } else {
                        match binding.binding_count {
                            BindingCount::One => 1,
                            BindingCount::StaticSized(size) => size.try_into().unwrap(),
                            BindingCount::Unbounded => shader_info.max_descriptor_count,
                        }
                    };

                    println!(
                        "{} binding: {:?} {}",
                        binding_index, binding, descriptor_count
                    );

                    match binding.ty {
                        rspirv_reflect::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                            if binding.name.contains("LinearYUV420P") {
                                bindings.push(
                                    vk::DescriptorSetLayoutBinding::default()
                                        .binding(*binding_index)
                                        .descriptor_count(descriptor_count) // TODO
                                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                        .stage_flags(stage_flags)
                                        .immutable_samplers(yuv_samplers),
                                );
                            } else {
                                bindings.push(
                                    vk::DescriptorSetLayoutBinding::default()
                                        .binding(*binding_index)
                                        .descriptor_count(descriptor_count) // TODO
                                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                        .stage_flags(stage_flags),
                                );
                            }
                        }
                        rspirv_reflect::DescriptorType::UNIFORM_BUFFER
                        | rspirv_reflect::DescriptorType::UNIFORM_TEXEL_BUFFER
                        | rspirv_reflect::DescriptorType::STORAGE_IMAGE
                        | rspirv_reflect::DescriptorType::STORAGE_BUFFER
                        | rspirv_reflect::DescriptorType::STORAGE_BUFFER_DYNAMIC
                        | rspirv_reflect::DescriptorType::SAMPLED_IMAGE => {
                            bindings.push(
                                vk::DescriptorSetLayoutBinding::default()
                                    .binding(*binding_index)
                                    .descriptor_count(descriptor_count) // TODO
                                    .descriptor_type(match binding.ty {
                                        rspirv_reflect::DescriptorType::UNIFORM_BUFFER => {
                                            vk::DescriptorType::UNIFORM_BUFFER
                                        }
                                        rspirv_reflect::DescriptorType::UNIFORM_BUFFER_DYNAMIC => {
                                            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                                        }
                                        rspirv_reflect::DescriptorType::UNIFORM_TEXEL_BUFFER => {
                                            vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                                        }
                                        rspirv_reflect::DescriptorType::STORAGE_IMAGE => {
                                            vk::DescriptorType::STORAGE_IMAGE
                                        }
                                        rspirv_reflect::DescriptorType::STORAGE_BUFFER => {
                                            if binding.name.ends_with("_dyn") {
                                                vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                                            } else {
                                                vk::DescriptorType::STORAGE_BUFFER
                                            }
                                        }
                                        rspirv_reflect::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                                            vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                                        }
                                        rspirv_reflect::DescriptorType::SAMPLED_IMAGE => {
                                            vk::DescriptorType::SAMPLED_IMAGE
                                        }
                                        _ => unimplemented!("{:?}", binding),
                                    })
                                    .stage_flags(stage_flags),
                            )
                        }

                        rspirv_reflect::DescriptorType::SAMPLER => {
                            let name_prefix = "sampler_";
                            if let Some(mut spec) = binding.name.strip_prefix(name_prefix) {
                                let texel_filter = match &spec[..1] {
                                    "n" => vk::Filter::NEAREST,
                                    "l" => vk::Filter::LINEAR,
                                    _ => panic!("{}", &spec[..1]),
                                };
                                spec = &spec[1..];

                                let mipmap_mode = match &spec[..1] {
                                    "n" => vk::SamplerMipmapMode::NEAREST,
                                    "l" => vk::SamplerMipmapMode::LINEAR,
                                    _ => panic!("{}", &spec[..1]),
                                };
                                spec = &spec[1..];

                                let address_modes = match spec {
                                    "r" => vk::SamplerAddressMode::REPEAT,
                                    "mr" => vk::SamplerAddressMode::MIRRORED_REPEAT,
                                    "c" => vk::SamplerAddressMode::CLAMP_TO_EDGE,
                                    "cb" => vk::SamplerAddressMode::CLAMP_TO_BORDER,
                                    _ => panic!("{}", spec),
                                };

                                bindings.push(
                                    vk::DescriptorSetLayoutBinding::default()
                                        .descriptor_count(descriptor_count)
                                        .descriptor_type(vk::DescriptorType::SAMPLER)
                                        .stage_flags(stage_flags)
                                        .binding(*binding_index)
                                        .immutable_samplers(std::slice::from_ref(samplers.add(
                                            shader_info.get_sampler(&SamplerDesc {
                                                texel_filter,
                                                mipmap_mode,
                                                address_modes,
                                            }),
                                        ))),
                                );
                            } else {
                                panic!("{}", binding.name);
                            }
                        }
                        rspirv_reflect::DescriptorType::ACCELERATION_STRUCTURE_KHR => bindings
                            .push(
                                vk::DescriptorSetLayoutBinding::default()
                                    .binding(*binding_index)
                                    .descriptor_count(descriptor_count) // TODO
                                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                                    .stage_flags(stage_flags),
                            ),

                        _ => unimplemented!("{:?}", binding),
                    }
                }

                let mut binding_flags_create_info =
                    vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                        .binding_flags(&binding_flags);

                let set_layout = unsafe {
                    device
                        .create_descriptor_set_layout(
                            &vk::DescriptorSetLayoutCreateInfo::default()
                                .flags(set_layout_create_flags)
                                .bindings(&bindings)
                                .push_next(&mut binding_flags_create_info),
                            None,
                        )
                        .unwrap()
                };

                set_layouts.push(set_layout);
                set_layout_info.push(
                    bindings
                        .iter()
                        .map(|binding| (binding.binding, binding.descriptor_type))
                        .collect(),
                );
            } else {
                let set_layout = unsafe {
                    device
                        .create_descriptor_set_layout(
                            &vk::DescriptorSetLayoutCreateInfo::default(),
                            None,
                        )
                        .unwrap()
                };

                set_layouts.push(set_layout);
                set_layout_info.push(Default::default());
            }
        }

        (set_layouts, set_layout_info)
    }

    pub fn from_source_text(
        device: &ash::Device,
        spirv_text: &str,
        debug_filename: &str,
        kind: ShaderKind,
        entry_point: &str,
    ) -> Self {
        let compiler = shaderc::Compiler::new().unwrap();
        let mut options = shaderc::CompileOptions::new().unwrap();
        options.add_macro_definition("EP", Some("main"));
        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_2 as u32,
        );
        options.set_optimization_level(shaderc::OptimizationLevel::Zero);
        options.set_generate_debug_info();
        options.set_include_callback(|name, include_type, source_file, _depth| {
            let path = if include_type == shaderc::IncludeType::Relative {
                Path::new(Path::new(source_file).parent().unwrap()).join(name)
            } else {
                Path::new("shader").join(name)
            };

            match std::fs::read_to_string(&path) {
                Ok(glsl_code) => Ok(shaderc::ResolvedInclude {
                    resolved_name: String::from(name),
                    content: glsl_code,
                }),
                Err(err) => Err(format!(
                    "Failed to resolve include to {} in {} (was looking for {:?}): {}",
                    name, source_file, path, err
                )),
            }
        });

        let spirv = compiler
            .compile_into_spirv(
                spirv_text,
                kind.to_shaderc_kind(),
                debug_filename,
                entry_point,
                Some(&options),
            )
            .unwrap();

        Self::new(device, spirv, kind, entry_point)
    }

    pub fn from_file(
        device: &ash::Device,
        path: &str,
        kind: ShaderKind,
        entry_point: &str,
    ) -> Self {
        Self::from_source_text(
            device,
            &std::fs::read_to_string(path).unwrap(),
            path,
            kind,
            entry_point,
        )
    }
}
