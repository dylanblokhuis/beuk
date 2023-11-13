use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
    env::current_dir,
    ffi::CString,
    hash::{Hash, Hasher},
    io::Read,
    path::Path,
    sync::Arc,
};

use ash::vk::{self, Filter, SamplerAddressMode, SamplerMipmapMode};
use rspirv_reflect::BindingCount;
use rustc_hash::FxHashMap;
use shaderc::{CompilationArtifact, OptimizationLevel};
use smallvec::SmallVec;

use crate::{chunky_list::TempList, ctx::SamplerDesc, memory::ResourceHooks};

impl ResourceHooks for Shader {
    fn cleanup(
        &mut self,
        device: Arc<ash::Device>,
        _allocator: Arc<std::sync::Mutex<gpu_allocator::vulkan::Allocator>>,
    ) {
        unsafe {
            device.destroy_shader_module(self.module, None);
        }
    }

    fn on_swapchain_resize(
        &mut self,
        _ctx: &crate::ctx::RenderContext,
        _old_surface_resolution: vk::Extent2D,
        _new_surface_resolution: vk::Extent2D,
    ) {
    }
}

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub enum ShaderOptimization {
    None,
    Size,
    #[default]
    Performance,
}

impl From<ShaderOptimization> for OptimizationLevel {
    fn from(opt: ShaderOptimization) -> Self {
        match opt {
            ShaderOptimization::None => OptimizationLevel::Zero,
            ShaderOptimization::Size => OptimizationLevel::Size,
            ShaderOptimization::Performance => OptimizationLevel::Performance,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ShaderSource {
    Text(Cow<'static, str>),
    File(Cow<'static, Path>),
}

impl Default for ShaderSource {
    fn default() -> Self {
        Self::Text(Cow::Borrowed(""))
    }
}

impl From<&'static str> for ShaderSource {
    fn from(text: &'static str) -> Self {
        if text.contains('\n') {
            Self::Text(Cow::Borrowed(text))
        } else {
            Self::File(Cow::Borrowed(Path::new(text)))
        }
    }
}

#[derive(Clone, Debug, Default, Hash, PartialEq, Eq)]
pub struct ShaderDescriptor {
    pub label: &'static str,
    pub source: ShaderSource,
    pub kind: ShaderKind,
    pub defines: Vec<(String, Option<String>)>,
    pub entry_point: Cow<'static, str>,
    pub optimization: ShaderOptimization,
    // pub path: Option<Cow<'static, Path>>,
}

#[derive(Clone, Debug, Default, Eq)]
pub struct Shader {
    pub kind: ShaderKind,
    pub spirv_descripor_set_layouts: StageDescriptorSetLayouts,
    pub entry_point: String,
    pub entry_point_cstr: CString,
    pub module: vk::ShaderModule,
    pub spirv: Vec<u8>,
}

impl Hash for Shader {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
        self.spirv.hash(state);
    }
}

impl PartialEq for Shader {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.spirv == other.spirv
    }
}

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub enum ShaderKind {
    #[default]
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
        set_layout_info: &[FxHashMap<u32, vk::DescriptorType>],
    ) -> (Vec<vk::DescriptorSet>, vk::DescriptorPool) {
        let mut descriptor_pool_sizes: Vec<vk::DescriptorPoolSize> = Vec::new();
        for (_set, bindings) in set_layout_info.iter().enumerate() {
            for ty in bindings.values() {
                if let Some(dps) = descriptor_pool_sizes.iter_mut().find(|item| item.ty == *ty) {
                    dps.descriptor_count += 1;
                } else {
                    descriptor_pool_sizes.push(vk::DescriptorPoolSize {
                        ty: *ty,
                        descriptor_count: shader_info.max_descriptor_count,
                    })
                }
            }
        }

        if descriptor_pool_sizes.is_empty() {
            return (vec![], vk::DescriptorPool::null());
        }

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .pool_sizes(&descriptor_pool_sizes)
            .max_sets(1024);

        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_info, None)
                .unwrap()
        };

        let desc_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_set_layouts);

        (
            unsafe { device.allocate_descriptor_sets(&desc_alloc_info).unwrap() },
            descriptor_pool,
        )
    }

    pub fn create_descriptor_set_layouts(
        &self,
        device: &ash::Device,
        shader_info: &ImmutableShaderInfo,
        skip_sets: &[u32],
    ) -> (
        SmallVec<[vk::DescriptorSetLayout; 8]>,
        SmallVec<[FxHashMap<u32, vk::DescriptorType>; 8]>,
    ) {
        let samplers = TempList::new();
        let set_count = self
            .spirv_descripor_set_layouts
            .keys()
            .map(|set_index| *set_index + 1)
            .max()
            .unwrap_or(0u32);

        let mut set_layouts = SmallVec::new();
        let mut set_layout_info = SmallVec::new();

        for set_index in 0..set_count {
            if skip_sets.contains(&set_index) {
                continue;
            }

            let stage_flags = vk::ShaderStageFlags::ALL;
            let set = self.spirv_descripor_set_layouts.get(&set_index);

            if let Some(set) = set {
                let mut bindings: Vec<vk::DescriptorSetLayoutBinding> =
                    Vec::with_capacity(set.len());
                let binding_flags: Vec<vk::DescriptorBindingFlags> =
                    vec![vk::DescriptorBindingFlags::PARTIALLY_BOUND; set.len()];

                let set_layout_create_flags = vk::DescriptorSetLayoutCreateFlags::empty();

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

                    log::debug!(
                        "{} binding: {:?} {}",
                        binding_index,
                        binding,
                        descriptor_count
                    );

                    match binding.ty {
                        rspirv_reflect::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                            if binding.name.contains("LinearYUV") {
                                bindings.push(
                                    vk::DescriptorSetLayoutBinding::default()
                                        .binding(*binding_index)
                                        .descriptor_count(descriptor_count) // TODO
                                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                        .stage_flags(stage_flags)
                                        .immutable_samplers(std::slice::from_ref(
                                            samplers.add(
                                                shader_info
                                                    .get_yuv_conversion_sampler(
                                                        SamplerDesc {
                                                            texel_filter: Filter::LINEAR,
                                                            mipmap_mode: SamplerMipmapMode::LINEAR,
                                                            address_modes:
                                                                SamplerAddressMode::CLAMP_TO_EDGE,
                                                        },
                                                        if binding.name.contains(
                                                            "LinearYUV420SP"
                                                        ) {
                                                            vk::Format::G8_B8R8_2PLANE_420_UNORM
                                                        } else if binding.name.contains("LinearYUV420SP10") {
                                                            vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
                                                        } else if binding.name.contains("LinearYUV420P10") {
                                                            vk::Format::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16
                                                        } else if binding.name.contains("LinearYUV420P") {
                                                            vk::Format::G8_B8_R8_3PLANE_420_UNORM
                                                        } else {
                                                            panic!("Unknown YUV format in shader: {}", binding.name);
                                                        }).1,
                                                    )
                                            ),
                                        ),
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

        log::debug!("set_layout_info: {:#?}", set_layout_info);

        (set_layouts, set_layout_info)
    }

    pub fn from_source_text(ctx: &crate::ctx::RenderContext, desc: &ShaderDescriptor) -> Self {
        let compiler = shaderc::Compiler::new().unwrap();
        let mut options = shaderc::CompileOptions::new().unwrap();
        options.add_macro_definition("EP", Some("main"));

        for (name, value) in desc.defines.iter() {
            options.add_macro_definition(
                name,
                if let Some(value) = value {
                    Some(value)
                } else {
                    None
                },
            );
        }

        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_2 as u32,
        );
        options.set_optimization_level(desc.optimization.into());
        options.set_generate_debug_info();

        options.set_include_callback(|name, include_type, source_file, _depth| {
            let shader_include_dir: std::path::PathBuf = match desc.source {
                ShaderSource::Text(_) => current_dir().unwrap(),
                ShaderSource::File(ref path) => path.parent().unwrap().to_path_buf(),
            };

            let path = if include_type == shaderc::IncludeType::Relative {
                Path::new(Path::new(source_file).parent().unwrap()).join(name)
            } else {
                Path::new(&shader_include_dir).join(name)
            };

            match std::fs::read_to_string(&path) {
                Ok(glsl_code) => {
                    #[cfg(feature = "hot-reload")]
                    if let ShaderSource::File(shader_file_to_reload) = &desc.source {
                        crate::hot_reload::INCLUDED_SHADERS.lock().unwrap().insert(
                            path.canonicalize().unwrap(),
                            shader_file_to_reload.to_path_buf(),
                        );
                    }

                    Ok(shaderc::ResolvedInclude {
                        resolved_name: String::from(name),
                        content: glsl_code,
                    })
                }
                Err(err) => Err(format!(
                    "Failed to resolve include to {} in {} (was looking for {:?}): {}",
                    name, source_file, path, err
                )),
            }
        });

        let source: String = match desc.source {
            ShaderSource::Text(ref source) => source.to_string(),
            ShaderSource::File(ref path) => {
                let mut file = std::fs::File::open(path).unwrap();
                let mut source = String::new();
                file.read_to_string(&mut source).unwrap();
                source
            }
        };

        let spirv = compiler
            .compile_into_spirv(
                &source,
                desc.kind.to_shaderc_kind(),
                desc.label,
                &desc.entry_point,
                Some(&options),
            )
            .unwrap();

        Self::new(&ctx.device, spirv, desc.kind, &desc.entry_point)
    }

    // pub fn from_file(
    //     device: &ash::Device,
    //     path: &str,
    //     kind: ShaderKind,
    //     entry_point: &str,
    //     defines: &[(String, Option<String>)],
    //     optimization: OptimizationLevel,
    // ) -> Self {
    //     Self::from_source_text(
    //         device,
    //         &std::fs::read_to_string(path).unwrap(),
    //         path,
    //         kind,
    //         entry_point,
    //         defines,
    //         optimization,
    //     )
    // }
}

pub struct ImmutableShaderInfo {
    pub immutable_samplers: Arc<HashMap<SamplerDesc, vk::Sampler>>,
    pub yuv_conversion_samplers:
        Arc<HashMap<(vk::Format, SamplerDesc), (vk::SamplerYcbcrConversion, vk::Sampler)>>,
    pub max_descriptor_count: u32,
}
impl ImmutableShaderInfo {
    pub fn get_sampler(&self, desc: &SamplerDesc) -> vk::Sampler {
        *self
            .immutable_samplers
            .get(desc)
            .expect("Tried to get an immutable sampler that doesn't exist.")
    }

    pub fn get_yuv_conversion_sampler(
        &self,
        desc: SamplerDesc,
        format: vk::Format,
    ) -> (vk::SamplerYcbcrConversion, vk::Sampler) {
        *self
            .yuv_conversion_samplers
            .get(&(format, desc))
            .expect("Tried to get an immutable yuv sampler that doesn't exist.")
    }
}
