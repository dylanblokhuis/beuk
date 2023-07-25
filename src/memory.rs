use ash::vk::{self};
use std::{
    collections::{BTreeMap, HashMap},
    mem::size_of_val,
};

use crate::{
    buffer::Buffer,
    ctx::SamplerDesc,
    pipeline::{GraphicsPipeline, GraphicsPipelineDescriptor},
    texture::Texture,
};

pub type MemoryLocation = gpu_allocator::MemoryLocation;

/// the handle is also the device address of the buffer
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct BufferHandle(u64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TextureHandle(usize);

pub struct BufferManager {
    device: ash::Device,
    allocator: gpu_allocator::vulkan::Allocator,
    buffers: HashMap<BufferHandle, Buffer>,
}

impl BufferManager {
    pub fn new(device: ash::Device, allocator: gpu_allocator::vulkan::Allocator) -> Self {
        Self {
            device,
            allocator,
            buffers: HashMap::new(),
        }
    }

    pub fn create_buffer(
        &mut self,
        debug_name: &str,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> BufferHandle {
        let buffer = Buffer::new(
            &self.device,
            &mut self.allocator,
            debug_name,
            size,
            usage,
            location,
        );
        let handle = BufferHandle(buffer.device_addr);
        self.buffers.insert(handle, buffer);
        handle
    }

    pub fn create_buffer_with_data(
        &mut self,
        debug_name: &str,
        data: &[u8],
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> BufferHandle {
        let mut buffer = Buffer::new(
            &self.device,
            &mut self.allocator,
            debug_name,
            size_of_val(data) as vk::DeviceSize,
            usage,
            location,
        );
        buffer.copy_from_slice(data, 0);
        let handle = BufferHandle(buffer.device_addr);
        self.buffers.insert(handle, buffer);
        handle
    }

    pub fn get_buffer(&self, handle: BufferHandle) -> &Buffer {
        self.buffers.get(&handle).unwrap()
    }

    pub fn get_buffer_mut(&mut self, handle: BufferHandle) -> &mut Buffer {
        self.buffers.get_mut(&handle).unwrap()
    }

    pub fn remove_buffer(&mut self, handle: BufferHandle) {
        {
            let buffer = self.buffers.get_mut(&handle).unwrap();

            buffer.destroy(&self.device, &mut self.allocator);
        }

        self.buffers.remove(&handle);
    }
}

impl Drop for BufferManager {
    fn drop(&mut self) {
        for (_, buffer) in self.buffers.iter_mut() {
            buffer.destroy(&self.device, &mut self.allocator);
        }
    }
}

// Textures

pub struct TextureManager {
    device: ash::Device,
    allocator: gpu_allocator::vulkan::Allocator,
    textures: BTreeMap<TextureHandle, Texture>,
}

impl TextureManager {
    pub fn new(device: ash::Device, allocator: gpu_allocator::vulkan::Allocator) -> Self {
        Self {
            device,
            allocator,
            textures: BTreeMap::new(),
        }
    }

    pub fn create_texture(
        &mut self,
        debug_name: &str,
        image_info: &vk::ImageCreateInfo,
    ) -> TextureHandle {
        let buffer = Texture::new(&self.device, &mut self.allocator, debug_name, image_info);
        let handle = TextureHandle(self.textures.len());
        self.textures.insert(handle, buffer);
        handle
    }

    pub fn get_texture(&self, handle: TextureHandle) -> &Texture {
        self.textures.get(&handle).unwrap()
    }

    pub fn get_texture_mut(&mut self, handle: TextureHandle) -> &mut Texture {
        self.textures.get_mut(&handle).unwrap()
    }

    pub fn remove_texture(&mut self, handle: TextureHandle) {
        {
            let buffer = self.textures.get_mut(&handle).unwrap();

            buffer.destroy(&self.device, &mut self.allocator);
        }

        self.textures.remove(&handle);
    }
}

impl Drop for TextureManager {
    fn drop(&mut self) {
        for (_, texture) in self.textures.iter_mut() {
            texture.destroy(&self.device, &mut self.allocator);
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
        *self.immutable_samplers.get(desc).unwrap()
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineHandle(serde_hashkey::Key);

pub struct PipelineManager {
    device: ash::Device,
    // handle is serialized pipeline state?
    graphics_pipelines: HashMap<PipelineHandle, GraphicsPipeline>,
    pub immutable_shader_info: ImmutableShaderInfo,
}

impl PipelineManager {
    pub fn new(device: ash::Device, device_properties: vk::PhysicalDeviceProperties) -> Self {
        Self {
            device: device.clone(),
            graphics_pipelines: HashMap::new(),
            immutable_shader_info: ImmutableShaderInfo {
                immutable_samplers: Self::create_immutable_samplers(&device),
                max_descriptor_count: device_properties.limits.max_descriptor_set_samplers,
                yuv_conversion_samplers: HashMap::new(),
            },
        }
    }

    pub fn create_graphics_pipeline(&mut self, desc: GraphicsPipelineDescriptor) -> PipelineHandle {
        let key = PipelineHandle(serde_hashkey::to_key(&desc).unwrap());
        let pipeline: GraphicsPipeline =
            GraphicsPipeline::new(&self.device, desc, &mut self.immutable_shader_info);
        self.graphics_pipelines.insert(key.clone(), pipeline);
        self.graphics_pipelines.get(&key).unwrap();

        key
    }

    pub fn get_graphics_pipeline(&self, key: &PipelineHandle) -> &GraphicsPipeline {
        self.graphics_pipelines.get(key).unwrap()
    }
    pub fn resize_all_graphics_pipelines(&mut self, extent: vk::Extent2D) {
        for (_, pipeline) in self.graphics_pipelines.iter_mut() {
            pipeline.viewports[0].width = extent.width as f32;
            pipeline.viewports[0].height = extent.height as f32;
            pipeline.scissors[0].extent = extent;
        }
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

    pub fn destroy(&mut self) {
        unsafe {
            for sampler in self.immutable_shader_info.immutable_samplers.values() {
                self.device.destroy_sampler(*sampler, None);
            }
            for (_, pipeline) in self.graphics_pipelines.iter_mut() {
                self.device.destroy_pipeline_layout(pipeline.layout, None);
                self.device.destroy_pipeline(pipeline.pipeline, None);
            }
        }
    }
}

impl Drop for PipelineManager {
    fn drop(&mut self) {
        self.destroy();
    }
}
