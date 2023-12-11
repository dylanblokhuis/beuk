use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use ash::vk::{self, CommandBuffer, ShaderStageFlags};
use petgraph::{graphmap::DiGraphMap, visit::IntoNodeReferences};

use crate::{
    buffer::{Buffer, BufferDescriptor},
    compute_pipeline::{ComputePipeline, ComputePipelineDescriptor},
    ctx::{RenderContext, SamplerDesc},
    graphics_pipeline::{GraphicsPipeline, GraphicsPipelineDescriptor},
    memory::{ResourceHandle, ResourceId, ResourceManager},
    texture::Texture,
};

#[derive(Hash, PartialEq, Eq, Clone, PartialOrd, Ord, Copy, Debug)]
pub enum PassType {
    Compute,
    Graphics,
}

#[derive(Hash, PartialEq, Eq, Clone, PartialOrd, Ord, Copy, Debug)]
pub struct PassId {
    pub label: &'static str,
    pub pass_type: PassType,
}

pub struct RenderGraph<W> {
    pub ctx: Arc<RenderContext>,
    pub(super) compute_passes: BTreeMap<PassId, ComputePass<W>>,
    pub(super) graphics_passes: BTreeMap<PassId, GraphicsPass<W>>,
    // track in which order the resources are accessed by the added passes
    current_built_graph: Option<DiGraphMap<PassId, ()>>,
    built_entry_nodes: Vec<PassId>,
}

unsafe impl<W> Send for RenderGraph<W> {}
unsafe impl<W> Sync for RenderGraph<W> {}

impl<W> RenderGraph<W> {
    pub fn new(ctx: Arc<RenderContext>) -> Self {
        Self {
            ctx,
            compute_passes: BTreeMap::new(),
            graphics_passes: BTreeMap::new(),
            current_built_graph: None,
            built_entry_nodes: vec![],
        }
    }


    pub fn order_and_build_graph(&mut self) {
        // Create a new directed graph
        let mut graph = DiGraphMap::<PassId, ()>::new();

        // Add nodes for each compute pass
        for label in self.compute_passes.keys() {
            graph.add_node(*label);
        }

        // Add nodes for each graphics pass
        for label in self.graphics_passes.keys() {
            graph.add_node(*label);
        }

        let mut resource_readers: HashMap<ResourceId, Vec<PassId>> = HashMap::new();
        let mut resource_writers: HashMap<ResourceId, Vec<PassId>> = HashMap::new();

        // Populate resource usage maps
        for (&pass_id, pass) in self.compute_passes.iter() {
            for resource in &pass.read_buffers {
                resource_readers
                    .entry(resource.id())
                    .or_default()
                    .push(pass_id);
            }
            for resource in &pass.write_buffers {
                resource_writers
                    .entry(resource.id())
                    .or_default()
                    .push(pass_id);
            }

            for resource in &pass.read_textures {
                resource_readers
                    .entry(resource.0.id())
                    .or_default()
                    .push(pass_id);
            }

            for resource in &pass.write_textures {
                resource_writers
                    .entry(resource.id())
                    .or_default()
                    .push(pass_id);
            }
        }

        for (&pass_id, pass) in self.graphics_passes.iter() {
            for resource in &pass.read_buffers {
                resource_readers
                    .entry(resource.id())
                    .or_default()
                    .push(pass_id);
            }
            for resource in &pass.write_buffers {
                resource_writers
                    .entry(resource.id())
                    .or_default()
                    .push(pass_id);
            }

            for resource in &pass.read_textures {
                resource_readers
                    .entry(resource.0.id())
                    .or_default()
                    .push(pass_id);
            }

            for resource in &pass.write_textures {
                resource_writers
                    .entry(resource.id())
                    .or_default()
                    .push(pass_id);
            }
        }

        for (_, passes) in &resource_writers {
            for (i, &pass) in passes.iter().enumerate() {
                for &dependent_pass in &passes[i + 1..] {
                    graph.add_edge(pass, dependent_pass, ());
                }
            }
        }

        for (resource, read_passes) in &resource_readers {
            if let Some(write_passes) = resource_writers.get(resource) {
                for &write_pass in write_passes {
                    for &read_pass in read_passes {
                        if !write_passes.contains(&read_pass) {
                            graph.add_edge(write_pass, read_pass, ());
                        }
                    }
                }
            }
        }

        // bind all the resources to the compute passes
        for pass in self.compute_passes.values() {
            let mut pipeline = pass.pipeline.get();

            for (binding_index, resource_id) in pass.binding_order.iter().enumerate() {
                println!(
                    "binding_index: {} {}",
                    binding_index, resource_id.manager_id
                );

                let set: u32 = 0;
                if resource_id.manager_id == self.ctx.buffer_manager.id {
                    let handle = ResourceManager::<Buffer>::handle_from_id(
                        self.ctx.buffer_manager.clone(),
                        *resource_id,
                    )
                    .unwrap();
                    if pipeline.set_layout_info[set as usize]
                        .get(&(binding_index as u32))
                        .is_none()
                    {
                        log::error!(
                            "RenderGraph: Descriptor set layout does not support binding index {}",
                            binding_index
                        );
                        continue;
                    }

                    let buffer = self.ctx.buffer_manager.get_mut(&handle).unwrap();

                    pipeline.queue_descriptor_buffer(
                        set,
                        binding_index as u32,
                        0,
                        vk::DescriptorBufferInfo::default()
                            .buffer(buffer.buffer)
                            .offset(0)
                            .range(buffer.size),
                    );
                } else if resource_id.manager_id == self.ctx.texture_manager.id {
                    let handle = ResourceManager::<Texture>::handle_from_id(
                        self.ctx.texture_manager.clone(),
                        *resource_id,
                    )
                    .unwrap();
                    let mut texture = self.ctx.texture_manager.get_mut(&handle).unwrap();
                    let view = texture.create_view(&self.ctx.device);

                    if pipeline.set_layout_info[set as usize]
                        .get(&(binding_index as u32))
                        .is_none()
                    {
                        log::error!(
                            "RenderGraph: Descriptor set layout does not support binding index {}",
                            binding_index
                        );
                        continue;
                    }

                    let is_write = pass
                        .write_textures
                        .iter()
                        .any(|texture| texture.id() == *resource_id);

                    if is_write {
                        pipeline.queue_descriptor_image(
                            set,
                            binding_index as u32,
                            0,
                            vk::DescriptorImageInfo::default()
                                .image_view(*view)
                                .image_layout(vk::ImageLayout::GENERAL),
                        );
                    } else {
                        let sampler = pass.read_textures.iter().find_map(|(texture, sampler)| {
                            if texture.id() == *resource_id && sampler.is_some() {
                                Some(*self.ctx.immutable_samplers.get(&sampler.unwrap()).unwrap())
                            } else {
                                None
                            }
                        });
                        pipeline.queue_descriptor_image(
                            set,
                            binding_index as u32,
                            0,
                            vk::DescriptorImageInfo::default()
                                .image_view(*view)
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .sampler(sampler.unwrap_or(vk::Sampler::null())),
                        );
                    }
                }
            }

            pipeline.update_descriptors(&self.ctx);
        }

        for pass in self.graphics_passes.values() {
            let mut pipeline = pass.pipeline.get();

            for (binding_index, resource_id) in pass.binding_order.iter().enumerate() {
                let set: u32 = 0;
                if resource_id.manager_id == self.ctx.buffer_manager.id {
                    let handle = ResourceManager::<Buffer>::handle_from_id(
                        self.ctx.buffer_manager.clone(),
                        *resource_id,
                    )
                    .unwrap();

                    if pipeline.set_layout_info[set as usize]
                        .get(&(binding_index as u32))
                        .is_none()
                    {
                        log::error!(
                            "RenderGraph: Descriptor set layout does not support binding index {}",
                            binding_index
                        );
                        continue;
                    }

                    let buffer = self.ctx.buffer_manager.get_mut(&handle).unwrap();
                    pipeline.queue_descriptor_buffer(
                        set,
                        binding_index as u32,
                        0,
                        vk::DescriptorBufferInfo::default()
                            .buffer(buffer.buffer)
                            .offset(0)
                            .range(buffer.size),
                    );
                } else if resource_id.manager_id == self.ctx.texture_manager.id {
                    let handle = ResourceManager::<Texture>::handle_from_id(
                        self.ctx.texture_manager.clone(),
                        *resource_id,
                    )
                    .unwrap();
                    let mut texture = self.ctx.texture_manager.get_mut(&handle).unwrap();
                    let view = texture.create_view(&self.ctx.device);

                    if pipeline.set_layout_info[set as usize]
                        .get(&(binding_index as u32))
                        .is_none()
                    {
                        log::error!(
                            "RenderGraph: Descriptor set layout does not support binding index {}",
                            binding_index
                        );
                        continue;
                    }

                    let is_write = pass
                        .write_textures
                        .iter()
                        .any(|texture| texture.id() == *resource_id);

                    if is_write {
                        pipeline.queue_descriptor_image(
                            set,
                            binding_index as u32,
                            0,
                            vk::DescriptorImageInfo::default()
                                .image_view(*view)
                                .image_layout(vk::ImageLayout::GENERAL),
                        );
                    } else {
                        let sampler = pass.read_textures.iter().find_map(|(texture, sampler)| {
                            if texture.id() == *resource_id && sampler.is_some() {
                                Some(*self.ctx.immutable_samplers.get(&sampler.unwrap()).unwrap())
                            } else {
                                None
                            }
                        });
                        pipeline.queue_descriptor_image(
                            set,
                            binding_index as u32,
                            0,
                            vk::DescriptorImageInfo::default()
                                .image_view(*view)
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .sampler(sampler.unwrap_or(vk::Sampler::null())),
                        );
                    }
                }
            }

            pipeline.update_descriptors(&self.ctx);
        }

        let sorted_passes = petgraph::algo::toposort(&graph, None).unwrap();
        println!("sorted_passes: {:#?}", sorted_passes);

        self.built_entry_nodes = graph
            .node_references()
            .filter(|(_, node)| {
                graph
                    .neighbors_directed(**node, petgraph::Direction::Incoming)
                    .count()
                    == 0
            })
            .map(|(node, _)| node)
            .collect::<Vec<_>>();
        self.current_built_graph = Some(graph);
    }

    pub fn is_built(&self) -> bool {
        self.current_built_graph.is_some()
    }

    pub fn run(&mut self, w: &mut W) {
        let Some(graph) = &self.current_built_graph else {
            panic!("RenderGraph: Cannot run graph without building it first");
        };

        // first we run all the update passes in the graph
        // TODO: parallelize this?
        for nodes in graph.node_references() {
            let pass_id = &nodes.0;
            match pass_id.pass_type {
                PassType::Compute => {
                    let node = self.compute_passes.get(pass_id).unwrap();
                    if let Some(update_callback) = &node.update_callback {
                        update_callback(self, node, w);
                    }
                }
                PassType::Graphics => {
                    let node = self.graphics_passes.get(pass_id).unwrap();
                    if let Some(update_callback) = &node.update_callback {
                        update_callback(self, node, w);
                    }
                }
            };
        }

        let present_pass = PassId {
            label: "present",
            pass_type: PassType::Graphics,
        };

        let present_index = self.ctx.acquire_present_index();
        let command_buffer = self.ctx.get_command_buffer();
        let fence = command_buffer.fence;
        self.ctx.record(&command_buffer, |command_buffer| {
            let mut last_stage = None;
            // check if present is an entry node, which means there are no other nodes needed to be ran
            if self.built_entry_nodes.contains(&&present_pass) {
                self.run_present_pass(command_buffer, present_index, &mut last_stage, w);
                return;
            }

            for entry_node in self.built_entry_nodes.iter() {
                // println!("entry_node: {:?}", entry_node);
                self.run_pass(&entry_node, command_buffer, &mut last_stage, w);

                for pass in graph.neighbors_directed(*entry_node, petgraph::Direction::Outgoing) {
                    // println!("pass: {:?}", pass);
                    if pass == present_pass {
                        self.run_present_pass(command_buffer, present_index, &mut last_stage, w);
                    } else {
                        self.run_pass(&pass, command_buffer, &mut last_stage, w);
                    }
                }
            }
        });

        unsafe {
            let swapchain = self.ctx.get_swapchain();
            // submit to the present queue
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(std::slice::from_ref(&self.ctx.present_complete_semaphore))
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                .command_buffers(std::slice::from_ref(&command_buffer.command_buffer))
                .signal_semaphores(std::slice::from_ref(&self.ctx.rendering_complete_semaphore));

            self.ctx
                .device
                .queue_submit(self.ctx.present_queue, &[submit_info], fence)
                .expect("queue submit failed.");
            self.ctx
                .device
                .wait_for_fences(&[fence], true, std::u64::MAX)
                .unwrap();

            let wait_semaphors = [self.ctx.rendering_complete_semaphore];
            let swapchains = [swapchain.swapchain];
            let image_indices = [present_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&wait_semaphors)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            let result = swapchain
                .swapchain_loader
                .queue_present(self.ctx.present_queue, &present_info);

            match result {
                Ok(_) => false,
                Err(vk_result) => match vk_result {
                    vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
                    _ => panic!("Failed to execute queue present."),
                },
            };
        }
    }

    fn place_barriers(
        &self,
        pass_id: &PassId,
        command_buffer: CommandBuffer,
        last_stage: &mut Option<vk::PipelineStageFlags>,
    ) {
        let (read_textures, write_textures, read_buffers, write_buffers) = match pass_id.pass_type {
            PassType::Compute => {
                let node = self.compute_passes.get(pass_id).unwrap();
                (
                    &node.read_textures,
                    &node.write_textures,
                    &node.read_buffers,
                    &node.write_buffers,
                )
            }
            PassType::Graphics => {
                let node = self.graphics_passes.get(pass_id).unwrap();
                (
                    &node.read_textures,
                    &node.write_textures,
                    &node.read_buffers,
                    &node.write_buffers,
                )
            }
        };

        let shader_stages = match pass_id.pass_type {
            PassType::Compute => vk::PipelineStageFlags::COMPUTE_SHADER,
            PassType::Graphics => vk::PipelineStageFlags::FRAGMENT_SHADER,
        };

        let mut image_barriers = vec![];
        for (texture, _) in read_textures {
            let mut texture = texture.get();

            if texture.access_mask == vk::AccessFlags::SHADER_READ
                && texture.stage_mask == shader_stages
                && texture.layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            {
                continue;
            }

            let image_memory_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(texture.access_mask)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(texture.layout)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(texture.image)
                .subresource_range(texture.subresource_range);

            texture.access_mask = vk::AccessFlags::SHADER_READ;
            texture.layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            texture.stage_mask = shader_stages;
            image_barriers.push(image_memory_barrier);
        }

        for texture in write_textures {
            let mut texture = texture.get();

            let image_memory_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(texture.access_mask)
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .old_layout(texture.layout)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(texture.image)
                .subresource_range(texture.subresource_range);

            texture.access_mask = vk::AccessFlags::SHADER_WRITE;
            texture.layout = vk::ImageLayout::GENERAL;
            texture.stage_mask = shader_stages;

            image_barriers.push(image_memory_barrier);
        }

        let mut buffer_barriers = vec![];

        for buffer in read_buffers {
            let mut buffer = buffer.get();

            if (buffer.access_mask == vk::AccessFlags::SHADER_READ)
                && (buffer.stage_mask == shader_stages)
            {
                continue;
            }

            let buffer_memory_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(buffer.access_mask)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .buffer(buffer.buffer)
                .offset(0)
                .size(buffer.size);

            buffer.access_mask = vk::AccessFlags::SHADER_READ;
            buffer.stage_mask = shader_stages;

            buffer_barriers.push(buffer_memory_barrier);
        }

        for buffer in write_buffers {
            let mut buffer = buffer.get();

            let buffer_memory_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(buffer.access_mask)
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .buffer(buffer.buffer)
                .offset(0)
                .size(buffer.size);

            buffer.access_mask = vk::AccessFlags::SHADER_WRITE;
            buffer.stage_mask = shader_stages;

            buffer_barriers.push(buffer_memory_barrier);
        }

        if !image_barriers.is_empty() || !buffer_barriers.is_empty() {
            unsafe {
                self.ctx.device.cmd_pipeline_barrier(
                    command_buffer,
                    if let Some(stage) = last_stage {
                        *stage
                    } else {
                        vk::PipelineStageFlags::ALL_COMMANDS
                    },
                    shader_stages,
                    vk::DependencyFlags::empty(),
                    &[],
                    &buffer_barriers,
                    &image_barriers,
                );
            }
        }
    }

    pub fn run_pass(
        &self,
        pass_id: &PassId,
        command_buffer: CommandBuffer,
        last_stage: &mut Option<vk::PipelineStageFlags>,
        w: &mut W,
    ) {
        self.place_barriers(pass_id, command_buffer, last_stage);

        match pass_id.pass_type {
            PassType::Compute => {
                let node = self.compute_passes.get(pass_id).unwrap();
                *last_stage = Some(vk::PipelineStageFlags::COMPUTE_SHADER);
                (node.record_callback)(&self, &node, command_buffer, w);
            }
            PassType::Graphics => {
                let node = self.graphics_passes.get(pass_id).unwrap();
                *last_stage = Some(vk::PipelineStageFlags::FRAGMENT_SHADER);
                (node.record_callback)(&self, &node, command_buffer, w);
            }
        }
    }

    pub fn run_present_pass(
        &self,
        command_buffer: CommandBuffer,
        present_index: u32,
        last_stage: &mut Option<vk::PipelineStageFlags>,
        w: &mut W,
    ) {
        let pass_id = PassId {
            label: "present",
            pass_type: PassType::Graphics,
        };
        // we place barriers here for the dependencies
        self.place_barriers(&pass_id, command_buffer, last_stage);

        // here we handle the swapchain image layout transition
        unsafe {
            let render_swapchain = self.ctx.get_swapchain();

            let layout_transition_barriers = vk::ImageMemoryBarrier::default()
                .image(render_swapchain.present_images[present_index as usize])
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1)
                        .level_count(1),
                );

            self.ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[layout_transition_barriers],
            );

            let present_image_view = render_swapchain.present_image_views[present_index as usize];
            let depth_image_view = self
                .ctx
                .get_texture_view(&render_swapchain.depth_image_handle)
                .unwrap();

            let color_attachments = &[vk::RenderingAttachmentInfo::default()
                .image_view(present_image_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.1, 0.1, 0.1, 1.0],
                    },
                })];

            let depth_attachment = &vk::RenderingAttachmentInfo::default()
                .image_view(*depth_image_view)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                });

            self.ctx
                .begin_rendering(command_buffer, color_attachments, Some(depth_attachment));

            let pass = self
                .graphics_passes
                .get(&pass_id)
                .expect("Present pass not found");

            (pass.record_callback)(&self, &pass, command_buffer, w);

            self.ctx.end_rendering(command_buffer);

            let layout_transition_barriers = vk::ImageMemoryBarrier::default()
                .image(render_swapchain.present_images[present_index as usize])
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1)
                        .level_count(1),
                );

            self.ctx.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[layout_transition_barriers],
            );
        }
    }

    pub fn add_buffer_to_pass(&mut self, pass_id: &PassId, buffer: ResourceHandle<Buffer>, write: bool) {
        match pass_id.pass_type {
            PassType::Compute => {
                let node = self.compute_passes.get_mut(pass_id).unwrap();
                if write {
                    node.write_buffers.push(buffer);
                } else {
                    node.read_buffers.push(buffer);
                }
            }
            PassType::Graphics => {
                let node = self.graphics_passes.get_mut(pass_id).unwrap();
                if write {
                    node.write_buffers.push(buffer);
                } else {
                    node.read_buffers.push(buffer);
                }
            }
        }
    }

    pub fn add_texture_to_pass(
        &mut self,
        pass_id: &PassId,
        texture: ResourceHandle<Texture>,
        sampler_desc: Option<SamplerDesc>,
        write: bool,
    ) {
        match pass_id.pass_type {
            PassType::Compute => {
                let node = self.compute_passes.get_mut(pass_id).unwrap();
                if write {
                    node.write_textures.push(texture);
                } else {
                    node.read_textures.push((texture, sampler_desc));
                }
            }
            PassType::Graphics => {
                let node = self.graphics_passes.get_mut(pass_id).unwrap();
                if write {
                    node.write_textures.push(texture);
                } else {
                    node.read_textures.push((texture, sampler_desc));
                }
            }
        }
    }

    /// The input buffer is a host visible buffer that has already been written to
    /// this function will queue the buffer to be copied to a device local buffer
    pub fn queue_buffer(&mut self, staging_buffer: &mut Buffer) {
        assert!(
            staging_buffer.has_been_written_to,
            "Buffer has not been written to"
        );
        assert!(
            staging_buffer
                .usage
                .contains(vk::BufferUsageFlags::TRANSFER_SRC),
            "Buffer usage must contain TRANSFER_SRC"
        );
        assert!(
            staging_buffer.location == gpu_allocator::MemoryLocation::CpuToGpu,
            "Buffer must be in CpuToGpu memory location for it be copied to a device local buffer"
        );

        let mut device_local_usage = staging_buffer.usage;
        device_local_usage |= vk::BufferUsageFlags::TRANSFER_DST;
        device_local_usage &= !vk::BufferUsageFlags::TRANSFER_SRC;

        let buffer = self.ctx.create_buffer(&BufferDescriptor {
            debug_name: "device_local_buffer".into(),
            location: gpu_allocator::MemoryLocation::GpuOnly,
            usage: device_local_usage,
            size: staging_buffer.size,
        });
    }
}

pub struct ComputePass<W> {
    pub id: PassId,
    pub pipeline: ResourceHandle<ComputePipeline>,
    record_callback: ComputePassRecordCallback<W>,
    update_callback: Option<ComputePassUpdateCallback<W>>,
    read_textures: Vec<(ResourceHandle<Texture>, Option<SamplerDesc>)>,
    write_textures: Vec<ResourceHandle<Texture>>,
    read_buffers: Vec<ResourceHandle<Buffer>>,
    write_buffers: Vec<ResourceHandle<Buffer>>,
    binding_order: Vec<ResourceId>,
}

impl<W> ComputePass<W> {
    pub fn execute(
        &self,
        ctx: &RenderContext,
        command_buffer: CommandBuffer,
        push_constants: &[u8],
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        unsafe {
            let mut pipeline = self.pipeline.get();

            pipeline.bind_descriptor_sets(ctx, command_buffer);
            pipeline.bind_pipeline(ctx, command_buffer);

            if !push_constants.is_empty() {
                ctx.device.cmd_push_constants(
                    command_buffer,
                    pipeline.layout,
                    ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }

            ctx.device
                .cmd_dispatch(command_buffer, group_count_x, group_count_y, group_count_z);
        }
    }
}

type ComputePassRecordCallback<W> =
    Box<dyn Fn(&RenderGraph<W>, &ComputePass<W>, CommandBuffer, &mut W)>;

type ComputePassUpdateCallback<W> = Box<dyn Fn(&RenderGraph<W>, &ComputePass<W>, &mut W)>;

pub struct ComputePassBuilder<'a, W> {
    id: PassId,
    graph: &'a mut RenderGraph<W>,
    pipeline: Option<ResourceHandle<ComputePipeline>>,
    record_callback: Option<ComputePassRecordCallback<W>>,
    update_callback: Option<ComputePassUpdateCallback<W>>,
    read_textures: Vec<(ResourceHandle<Texture>, Option<SamplerDesc>)>,
    write_textures: Vec<ResourceHandle<Texture>>,
    read_buffers: Vec<ResourceHandle<Buffer>>,
    write_buffers: Vec<ResourceHandle<Buffer>>,
    binding_order: Vec<ResourceId>,
}

impl<'a, W> ComputePassBuilder<'a, W> {
    pub fn new(id: &'static str, graph: &'a mut RenderGraph<W>) -> Self {
        Self {
            id: PassId {
                label: id,
                pass_type: PassType::Compute,
            },
            graph,
            pipeline: None,
            record_callback: None,
            update_callback: None,
            binding_order: vec![],
            read_buffers: vec![],
            read_textures: vec![],
            write_buffers: vec![],
            write_textures: vec![],
        }
    }

    pub fn pipeline(mut self, pipeline_descriptor: ComputePipelineDescriptor) -> Self {
        self.pipeline = Some(
            self.graph
                .ctx
                .create_compute_pipeline(self.id.label, pipeline_descriptor),
        );
        self
    }

    pub fn read_buffer(mut self, read_buffer_deps: ResourceHandle<Buffer>) -> Self {
        self.read_buffers.push(read_buffer_deps.clone());
        self.binding_order.push(read_buffer_deps.id());
        self
    }

    pub fn write_buffer(mut self, write_buffer_deps: ResourceHandle<Buffer>) -> Self {
        self.write_buffers.push(write_buffer_deps.clone());
        self.binding_order.push(write_buffer_deps.id());
        self
    }

    pub fn read_texture(
        mut self,
        read_texture_deps: ResourceHandle<Texture>,
        sampler_desc: Option<SamplerDesc>,
    ) -> Self {
        self.read_textures
            .push((read_texture_deps.clone(), sampler_desc));
        self.binding_order.push(read_texture_deps.id());
        self
    }

    pub fn write_texture(mut self, write_texture_deps: ResourceHandle<Texture>) -> Self {
        self.write_textures.push(write_texture_deps.clone());
        self.binding_order.push(write_texture_deps.id());
        self
    }

    /// This callback is ran when the command buffer is being recorded
    pub fn record_callback(
        mut self,
        callback: impl Fn(&RenderGraph<W>, &ComputePass<W>, CommandBuffer, &mut W) + 'static,
    ) -> Self {
        self.record_callback = Some(Box::new(callback));
        self
    }

    /// This callback is ran before any command buffer recording is started, so you can update descriptor sets etc.
    pub fn update_callback(
        mut self,
        callback: impl Fn(&RenderGraph<W>, &ComputePass<W>, &mut W) + 'static,
    ) -> Self {
        self.update_callback = Some(Box::new(callback));
        self
    }

    pub fn build(self) {
        let pass = ComputePass {
            id: self.id,
            pipeline: self
                .pipeline
                .expect("Pipeline is required to build a ComputePass"),
            record_callback: self.record_callback.expect(
                "A record callback is required to build a ComputePass, cannot be ran without",
            ),
            update_callback: self.update_callback,
            binding_order: self.binding_order,
            read_buffers: self.read_buffers,
            read_textures: self.read_textures,
            write_buffers: self.write_buffers,
            write_textures: self.write_textures,
        };
        self.graph.compute_passes.insert(self.id, pass);
    }
}

type GraphicsPassRecordCallback<W> =
    Box<dyn Fn(&RenderGraph<W>, &GraphicsPass<W>, CommandBuffer, &mut W)>;
type GraphicsPassUpdateCallback<W> = Box<dyn Fn(&RenderGraph<W>, &GraphicsPass<W>, &mut W)>;

pub struct GraphicsPass<W> {
    pub id: PassId,
    pub pipeline: ResourceHandle<GraphicsPipeline>,
    record_callback: GraphicsPassRecordCallback<W>,
    update_callback: Option<GraphicsPassUpdateCallback<W>>,
    read_textures: Vec<(ResourceHandle<Texture>, Option<SamplerDesc>)>,
    write_textures: Vec<ResourceHandle<Texture>>,
    read_buffers: Vec<ResourceHandle<Buffer>>,
    write_buffers: Vec<ResourceHandle<Buffer>>,
    binding_order: Vec<ResourceId>,
}

impl<W> GraphicsPass<W> {
    pub fn execute(
        &self,
        ctx: &RenderContext,
        command_buffer: CommandBuffer,
        push_constants: &[u8],
    ) {
        unsafe {
            let mut pipeline = self.pipeline.get();

            pipeline.bind_descriptor_sets(ctx, command_buffer);
            pipeline.bind_pipeline(ctx, command_buffer);

            if !push_constants.is_empty() {
                ctx.device.cmd_push_constants(
                    command_buffer,
                    pipeline.layout,
                    ShaderStageFlags::ALL_GRAPHICS,
                    0,
                    push_constants,
                );
            }
        }
    }
}

pub struct GraphicsPassBuilder<'a, W> {
    id: PassId,
    graph: &'a mut RenderGraph<W>,
    pipeline: Option<ResourceHandle<GraphicsPipeline>>,
    record_callback: Option<GraphicsPassRecordCallback<W>>,
    update_callback: Option<GraphicsPassUpdateCallback<W>>,
    read_textures: Vec<(ResourceHandle<Texture>, Option<SamplerDesc>)>,
    write_textures: Vec<ResourceHandle<Texture>>,
    read_buffers: Vec<ResourceHandle<Buffer>>,
    write_buffers: Vec<ResourceHandle<Buffer>>,
    binding_order: Vec<ResourceId>,
}

impl<'a, W> GraphicsPassBuilder<'a, W> {
    pub fn new(id: &'static str, graph: &'a mut RenderGraph<W>) -> Self {
        Self {
            id: PassId {
                label: id,
                pass_type: PassType::Graphics,
            },
            graph,
            pipeline: None,
            record_callback: None,
            update_callback: None,
            read_buffers: vec![],
            write_buffers: vec![],
            read_textures: vec![],
            write_textures: vec![],
            binding_order: vec![],
        }
    }

    pub fn pipeline(mut self, pipeline_descriptor: GraphicsPipelineDescriptor) -> Self {
        self.pipeline = Some(
            self.graph
                .ctx
                .create_graphics_pipeline(self.id.label, pipeline_descriptor),
        );
        self
    }

    pub fn read_buffer(mut self, read_buffer_deps: ResourceHandle<Buffer>) -> Self {
        self.read_buffers.push(read_buffer_deps.clone());
        self.binding_order.push(read_buffer_deps.id());

        self
    }

    pub fn write_buffer(mut self, write_buffer_deps: ResourceHandle<Buffer>) -> Self {
        self.write_buffers.push(write_buffer_deps.clone());
        self.binding_order.push(write_buffer_deps.id());

        self
    }

    pub fn read_texture(
        mut self,
        read_texture_deps: ResourceHandle<Texture>,
        sampler_desc: Option<SamplerDesc>,
    ) -> Self {
        self.read_textures
            .push((read_texture_deps.clone(), sampler_desc));
        self.binding_order.push(read_texture_deps.id());

        self
    }

    pub fn write_texture(mut self, write_texture_deps: ResourceHandle<Texture>) -> Self {
        self.write_textures.push(write_texture_deps.clone());
        self.binding_order.push(write_texture_deps.id());
        self
    }

    /// This callback is ran when the command buffer is being recorded
    pub fn record_callback(
        mut self,
        callback: impl Fn(&RenderGraph<W>, &GraphicsPass<W>, CommandBuffer, &mut W) + 'static,
    ) -> Self {
        self.record_callback = Some(Box::new(callback));
        self
    }

    /// This callback is ran before any command buffer recording is started, so you can update descriptor sets etc.
    pub fn update_callback(
        mut self,
        callback: impl Fn(&RenderGraph<W>, &GraphicsPass<W>, &mut W) + 'static,
    ) -> Self {
        self.update_callback = Some(Box::new(callback));
        self
    }

    pub fn build(self) {
        let pass = GraphicsPass {
            id: self.id,
            pipeline: self
                .pipeline
                .expect("Pipeline is required to build a ComputePass"),
            record_callback: self.record_callback.expect(
                "A record callback is required to build a ComputePass, cannot be ran without",
            ),
            update_callback: self.update_callback,
            binding_order: self.binding_order,
            read_buffers: self.read_buffers,
            read_textures: self.read_textures,
            write_buffers: self.write_buffers,
            write_textures: self.write_textures,
        };
        self.graph.graphics_passes.insert(self.id, pass);
    }
}
