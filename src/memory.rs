use ash::vk::{self};
use std::{
    collections::HashMap,
    fmt::Debug,
    hash::Hash,
    sync::{Arc, RwLock},
};

use crate::{
    ctx::SamplerDesc,
    pipeline::{GraphicsPipeline, GraphicsPipelineDescriptor},
};

pub type MemoryLocation = gpu_allocator::MemoryLocation;

#[derive(Debug, Clone)]
pub struct BufferDescriptor {
    pub debug_name: &'static str,
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
    /// GpuOnly will be a slower allocation, but it will be faster on the gpu
    pub location: MemoryLocation,
}

impl Default for BufferDescriptor {
    fn default() -> Self {
        Self {
            debug_name: "unnamed",
            size: 0,
            usage: vk::BufferUsageFlags::empty(),
            location: MemoryLocation::CpuToGpu,
        }
    }
}

// pipelines

pub struct ImmutableShaderInfo {
    pub immutable_samplers: HashMap<SamplerDesc, vk::Sampler>,
    pub yuv_conversion_samplers:
        HashMap<(vk::Format, SamplerDesc), (vk::SamplerYcbcrConversion, vk::Sampler)>,
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
        &mut self,
        device: &ash::Device,
        desc: SamplerDesc,
        format: vk::Format,
    ) -> (vk::SamplerYcbcrConversion, vk::Sampler) {
        // to avoid creating the same sampler twice
        if let Some(samplers) = self.yuv_conversion_samplers.get(&(format, desc)) {
            return *samplers;
        }

        let sampler_conversion = unsafe {
            device
                .create_sampler_ycbcr_conversion(
                    &vk::SamplerYcbcrConversionCreateInfo::default()
                        .ycbcr_model(vk::SamplerYcbcrModelConversion::YCBCR_709)
                        .ycbcr_range(vk::SamplerYcbcrRange::ITU_NARROW)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::IDENTITY,
                            g: vk::ComponentSwizzle::IDENTITY,
                            b: vk::ComponentSwizzle::IDENTITY,
                            a: vk::ComponentSwizzle::IDENTITY,
                        })
                        .chroma_filter(vk::Filter::LINEAR)
                        .x_chroma_offset(vk::ChromaLocation::MIDPOINT)
                        .y_chroma_offset(vk::ChromaLocation::MIDPOINT)
                        .force_explicit_reconstruction(false)
                        .format(format),
                    None,
                )
                .unwrap()
        };

        let mut conversion_info =
            vk::SamplerYcbcrConversionInfo::default().conversion(sampler_conversion);

        let sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(desc.texel_filter)
                        .min_filter(desc.texel_filter)
                        .mipmap_mode(desc.mipmap_mode)
                        .address_mode_u(desc.address_modes)
                        .address_mode_v(desc.address_modes)
                        .address_mode_w(desc.address_modes)
                        .max_lod(vk::LOD_CLAMP_NONE)
                        .max_anisotropy(16.0)
                        .anisotropy_enable(false)
                        .unnormalized_coordinates(false)
                        .push_next(&mut conversion_info),
                    None,
                )
                .unwrap()
        };

        self.yuv_conversion_samplers
            .insert((format, desc), (sampler_conversion, sampler));

        *self.yuv_conversion_samplers.get(&(format, desc)).unwrap()
    }
}

#[derive(Debug)]
pub struct PipelineHandle {
    id: PipelineId,
    manager: Arc<RwLock<PipelineManager>>,
}

impl PipelineHandle {
    pub fn new(id: PipelineId, manager: Arc<RwLock<PipelineManager>>) -> Self {
        Self { id, manager }
    }

    #[inline]
    pub fn id(&self) -> PipelineId {
        self.id.clone()
    }
}

impl Clone for PipelineHandle {
    #[tracing::instrument]
    fn clone(&self) -> Self {
        self.manager.write().unwrap().retain(self.id.clone());
        Self {
            id: self.id.clone(),
            manager: self.manager.clone(),
        }
    }
}

impl Drop for PipelineHandle {
    fn drop(&mut self) {
        self.manager
            .write()
            .unwrap()
            .remove_graphics_pipeline(self.id.clone());
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineId(serde_hashkey::Key);

pub struct PipelineManager {
    graphics_pipelines: HashMap<PipelineId, GraphicsPipeline>,
    counters: HashMap<PipelineId, usize>,
    device: ash::Device,
    pub immutable_shader_info: ImmutableShaderInfo,
    swapchain_size: vk::Extent2D,
}

impl Debug for PipelineManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineManager")
            .field("graphics_pipelines", &self.graphics_pipelines)
            .field("counters", &self.counters)
            .finish()
    }
}

impl PipelineManager {
    pub fn new(
        device: ash::Device,
        device_properties: vk::PhysicalDeviceProperties,
        swapchain_size: vk::Extent2D,
    ) -> Self {
        Self {
            device: device.clone(),
            graphics_pipelines: HashMap::new(),
            counters: HashMap::new(),
            immutable_shader_info: ImmutableShaderInfo {
                immutable_samplers: Self::create_immutable_samplers(&device),
                max_descriptor_count: device_properties.limits.max_descriptor_set_samplers,
                yuv_conversion_samplers: HashMap::new(),
            },
            swapchain_size,
        }
    }

    #[tracing::instrument(skip(self))]
    pub fn create_graphics_pipeline(&mut self, desc: &GraphicsPipelineDescriptor) -> PipelineId {
        let id = PipelineId(serde_hashkey::to_key(&desc).unwrap());

        let pipeline: GraphicsPipeline = GraphicsPipeline::new(
            &self.device,
            desc,
            &mut self.immutable_shader_info,
            self.swapchain_size,
        );
        self.graphics_pipelines.insert(id.clone(), pipeline);
        self.counters.insert(id.clone(), 1);

        id
    }

    #[tracing::instrument(skip(self))]
    pub fn get_graphics_pipeline(&self, key: &PipelineId) -> &GraphicsPipeline {
        self.graphics_pipelines.get(key).unwrap()
    }

    #[tracing::instrument(skip(self))]
    pub fn retain(&mut self, id: PipelineId) {
        let counter = self
            .counters
            .get_mut(&id)
            .expect("Tried to retain a pipeline that doesn't exist");
        *counter += 1;
    }

    /// will resize all pipelines that are using the swapchain size
    pub fn resize_all_graphics_pipelines(&mut self, extent: vk::Extent2D) {
        for (_, pipeline) in self.graphics_pipelines.iter_mut() {
            // if the pipeline is using the swapchain size, update it
            if pipeline.viewports[0].width == self.swapchain_size.width as f32
                && pipeline.viewports[0].height == self.swapchain_size.height as f32
            {
                pipeline.viewports[0].width = extent.width as f32;
                pipeline.viewports[0].height = extent.height as f32;
                pipeline.scissors[0].extent = extent;
            }
        }
        self.swapchain_size = extent;
    }

    fn create_immutable_samplers(device: &ash::Device) -> HashMap<SamplerDesc, vk::Sampler> {
        let texel_filters = [vk::Filter::NEAREST, vk::Filter::LINEAR];
        let mipmap_modes = [
            vk::SamplerMipmapMode::NEAREST,
            vk::SamplerMipmapMode::LINEAR,
        ];
        let address_modes = [
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
        ];

        let mut result = HashMap::new();

        for &texel_filter in &texel_filters {
            for &mipmap_mode in &mipmap_modes {
                for &address_modes in &address_modes {
                    let anisotropy_enable = texel_filter == vk::Filter::LINEAR;

                    result.insert(
                        SamplerDesc {
                            texel_filter,
                            mipmap_mode,
                            address_modes,
                        },
                        unsafe {
                            device.create_sampler(
                                &vk::SamplerCreateInfo::default()
                                    .mag_filter(texel_filter)
                                    .min_filter(texel_filter)
                                    .mipmap_mode(mipmap_mode)
                                    .address_mode_u(address_modes)
                                    .address_mode_v(address_modes)
                                    .address_mode_w(address_modes)
                                    .max_lod(vk::LOD_CLAMP_NONE)
                                    .max_anisotropy(16.0)
                                    .anisotropy_enable(anisotropy_enable),
                                None,
                            )
                        }
                        .expect("create_sampler"),
                    );
                }
            }
        }

        result
    }

    #[tracing::instrument]
    pub fn remove_graphics_pipeline(&mut self, id: PipelineId) {
        let pipeline = self.get_graphics_pipeline(&id);
        pipeline.destroy(&self.device);
        self.graphics_pipelines.remove(&id);
    }

    pub fn clear(&mut self) {
        unsafe {
            for sampler in self.immutable_shader_info.immutable_samplers.values() {
                self.device.destroy_sampler(*sampler, None);
            }
            for (conv, sampler) in self.immutable_shader_info.yuv_conversion_samplers.values() {
                self.device.destroy_sampler_ycbcr_conversion(*conv, None);
                self.device.destroy_sampler(*sampler, None);
            }
        }

        for (_, pipeline) in self.graphics_pipelines.iter() {
            pipeline.destroy(&self.device);
        }
        self.graphics_pipelines.clear();
    }
}

impl Drop for PipelineManager {
    fn drop(&mut self) {
        self.clear();
    }
}
