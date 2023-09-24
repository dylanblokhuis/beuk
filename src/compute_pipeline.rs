use ::smallvec::smallvec;
use ash::vk::{self, DescriptorType, WriteDescriptorSet};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;

use crate::{
    ctx::RenderContext,
    graphics_pipeline::{PrependDescriptorSets, PushConstantRange},
    memory::{ResourceHandle, ResourceHooks},
    shaders::{ImmutableShaderInfo, Shader},
    smallvec,
};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ComputePipelineDescriptor {
    pub shader: ResourceHandle<Shader>,
    pub push_constant_range: Option<PushConstantRange>,
    pub prepend_descriptor_sets: Option<Arc<PrependDescriptorSets>>,
}

impl ResourceHooks for ComputePipeline {
    fn cleanup(
        &mut self,
        device: std::sync::Arc<ash::Device>,
        allocator: std::sync::Arc<std::sync::Mutex<gpu_allocator::vulkan::Allocator>>,
    ) {
        drop(allocator);
        self.destroy(&device)
    }
    fn on_swapchain_resize(
        &mut self,
        _ctx: &RenderContext,
        _old_surface_resolution: vk::Extent2D,
        _new_surface_resolution: vk::Extent2D,
    ) {
    }
}

#[derive(Debug)]
pub enum DescriptorWriteType {
    Buffer(vk::DescriptorBufferInfo),
    Image(vk::DescriptorImageInfo),
}

#[derive(Debug)]
pub struct DescriptorWrite {
    pub set: u32,
    pub binding: u32,
    pub array_index: u32,
    pub descriptor_buffer: DescriptorWriteType,
}

#[derive(Debug, Default)]
pub struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub descriptor_set_layouts: SmallVec<[vk::DescriptorSetLayout; 8]>,
    pub descriptor_pool: vk::DescriptorPool,
    pub set_layout_info: SmallVec<[FxHashMap<u32, DescriptorType>; 8]>,
    pub prepended_descriptor_sets: Option<Arc<PrependDescriptorSets>>,
    queued_descriptor_writes: Vec<DescriptorWrite>,
}

impl ComputePipeline {
    pub fn new(ctx: &RenderContext, desc: &ComputePipelineDescriptor) -> Self {
        let immutable_shader_info = &ImmutableShaderInfo {
            immutable_samplers: ctx.immutable_samplers.clone(),
            yuv_conversion_samplers: ctx.yuv_immutable_samplers.clone(),
            max_descriptor_count: ctx.max_descriptor_count,
        };
        let shader = ctx.shader_manager.get(&desc.shader).unwrap();
        let sets_prepended = if let Some(prepend_descriptor_sets) = &desc.prepend_descriptor_sets {
            (0..prepend_descriptor_sets.sets.len() as u32).collect()
        } else {
            Vec::new()
        };

        println!("sets_prepended: {:?}", sets_prepended);

        let mut descriptor_set_layouts = desc
            .prepend_descriptor_sets
            .as_ref()
            .map_or(SmallVec::new(), |sets| sets.layouts.clone());

        let (mut shader_descriptor_set_layouts, mut set_layout_info) = shader
            .create_descriptor_set_layouts(&ctx.device, immutable_shader_info, &sets_prepended);

        let (descriptor_sets, descriptor_pool) = if !set_layout_info.is_empty() {
            shader.create_descriptor_sets(
                &ctx.device,
                immutable_shader_info,
                &shader_descriptor_set_layouts,
                &set_layout_info,
            )
        } else {
            (vec![], vk::DescriptorPool::null())
        };

        descriptor_set_layouts.append(&mut shader_descriptor_set_layouts);

        let push_constant_ranges: SmallVec<[vk::PushConstantRange; 1]> = desc
            .push_constant_range
            .map_or(smallvec![], |range| smallvec![range.into()]);

        let pipeline_layout = unsafe {
            ctx.device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&descriptor_set_layouts)
                        .push_constant_ranges(&push_constant_ranges),
                    None,
                )
                .unwrap()
        };

        let shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .name(&shader.entry_point_cstr)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader.module);

        let compute_pipeline_info = vk::ComputePipelineCreateInfo::default()
            .layout(pipeline_layout)
            .stage(shader_stage);

        let pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[compute_pipeline_info], None)
                .unwrap()[0]
        };

        Self {
            pipeline,
            layout: pipeline_layout,
            set_layout_info,
            descriptor_sets,
            descriptor_pool,
            descriptor_set_layouts,
            prepended_descriptor_sets: desc.prepend_descriptor_sets.clone(),
            ..Default::default()
        }
    }

    pub fn update_descriptors(&mut self, ctx: &RenderContext) {
        unsafe {
            if self.queued_descriptor_writes.is_empty() {
                return;
            }

            let writes: Vec<WriteDescriptorSet> = self
                .queued_descriptor_writes
                .iter()
                .map(|w| {
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(
                            *self
                                .descriptor_sets
                                .get(w.set as usize)
                                .expect("Descriptor set not found"),
                        )
                        .dst_binding(w.binding)
                        .dst_array_element(w.array_index)
                        .descriptor_type(self.set_layout_info[w.set as usize][&w.binding]);

                    match &w.descriptor_buffer {
                        DescriptorWriteType::Buffer(buf) => {
                            write.buffer_info(std::slice::from_ref(buf))
                        }
                        DescriptorWriteType::Image(tex) => {
                            write.image_info(std::slice::from_ref(tex))
                        }
                    }
                })
                .collect();

            ctx.device.update_descriptor_sets(&writes, &[]);
            self.queued_descriptor_writes.clear();
        }
    }

    pub fn queue_descriptor_buffer(
        &mut self,
        set: u32,
        binding: u32,
        array_index: u32,
        descriptor_buffer: vk::DescriptorBufferInfo,
    ) {
        self.queued_descriptor_writes.push(DescriptorWrite {
            set,
            binding,
            array_index,
            descriptor_buffer: DescriptorWriteType::Buffer(descriptor_buffer),
        });
    }

    pub fn queue_descriptor_image(
        &mut self,
        set: u32,
        binding: u32,
        array_index: u32,
        image_info: vk::DescriptorImageInfo,
    ) {
        self.queued_descriptor_writes.push(DescriptorWrite {
            set,
            binding,
            array_index,
            descriptor_buffer: DescriptorWriteType::Image(image_info),
        });
    }

    pub fn bind_descriptor_sets(&mut self, ctx: &RenderContext, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.update_descriptors(ctx);

            let offset = if let Some(prepend_descriptor_sets) = &self.prepended_descriptor_sets {
                ctx.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    0,
                    &prepend_descriptor_sets.sets,
                    &[],
                );
                prepend_descriptor_sets.sets.len() as u32
            } else {
                0
            };

            if self.descriptor_sets.is_empty() {
                return;
            }

            ctx.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                offset,
                &self.descriptor_sets,
                &[],
            );
        }
    }

    pub fn bind_pipeline(&self, ctx: &RenderContext, command_buffer: vk::CommandBuffer) {
        unsafe {
            ctx.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
        }
    }
    pub fn destroy(&mut self, device: &ash::Device) {
        if std::thread::panicking() {
            return;
        }
        if self.pipeline == Default::default() {
            return;
        }

        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            if !self.descriptor_sets.is_empty() {
                device
                    .free_descriptor_sets(self.descriptor_pool, &self.descriptor_sets)
                    .unwrap();
            }
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            for layout in &self.descriptor_set_layouts {
                device.destroy_descriptor_set_layout(*layout, None);
            }
        }
    }
}
