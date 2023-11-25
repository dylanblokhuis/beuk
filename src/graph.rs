use std::{collections::HashMap, sync::Arc};

use ash::vk::{self, CommandBuffer, ShaderStageFlags};
use petgraph::{
    graphmap::{DiGraphMap, NodeTrait},
    visit::IntoNodeReferences,
};

use crate::{
    buffer::Buffer,
    compute_pipeline::{ComputePipeline, ComputePipelineDescriptor},
    ctx::RenderContext,
    graphics_pipeline::{GraphicsPipeline, GraphicsPipelineDescriptor},
    memory::{ResourceHandle, ResourceId, ResourceManager},
    texture::{Texture, TransitionDesc},
};

#[derive(Hash, PartialEq, Eq, Clone, PartialOrd, Ord, Copy, Debug)]
pub enum PassType {
    Compute,
    Graphics,
}

#[derive(Hash, PartialEq, Eq, Clone, PartialOrd, Ord, Copy, Debug)]
pub struct PassId {
    label: &'static str,
    pass_type: PassType,
}

pub struct RenderGraph {
    pub ctx: Arc<RenderContext>,
    pub(super) compute_passes: HashMap<PassId, ComputePass>,
    pub(super) graphics_passes: HashMap<PassId, GraphicsPass>,
    // track in which order the resources are accessed by the added passes
    pub read_buffer_dependencies: HashMap<ResourceHandle<Buffer>, Vec<PassId>>,
    pub write_buffer_dependencies: HashMap<ResourceHandle<Buffer>, Vec<PassId>>,
    pub read_texture_dependencies: HashMap<ResourceHandle<Texture>, Vec<PassId>>,
    pub write_texture_dependencies: HashMap<ResourceHandle<Texture>, Vec<PassId>>,
    pub binding_order: HashMap<PassId, Vec<ResourceId>>,

    current_built_graph: Option<DiGraphMap<PassId, ()>>,
}

impl RenderGraph {
    pub fn new(ctx: Arc<RenderContext>) -> Self {
        Self {
            ctx,
            compute_passes: HashMap::new(),
            graphics_passes: HashMap::new(),
            read_buffer_dependencies: HashMap::new(),
            write_buffer_dependencies: HashMap::new(),
            read_texture_dependencies: HashMap::new(),
            write_texture_dependencies: HashMap::new(),
            binding_order: HashMap::new(),
            current_built_graph: None,
        }
    }

    pub fn order_and_build_graph(&mut self) {
        // Create a new directed graph
        let mut graph = DiGraphMap::<PassId, ()>::new();

        // Add nodes for each compute pass
        for (&label, _) in &self.compute_passes {
            graph.add_node(label);
        }

        // Add nodes for each graphics pass
        for (&label, _) in &self.graphics_passes {
            graph.add_node(label);
        }

        for (_, passes) in &self.write_buffer_dependencies {
            for (i, &pass) in passes.iter().enumerate() {
                for &dependent_pass in &passes[i + 1..] {
                    graph.add_edge(pass, dependent_pass, ());
                }
            }
        }

        for (_, passes) in &self.write_texture_dependencies {
            for (i, &pass) in passes.iter().enumerate() {
                for &dependent_pass in &passes[i + 1..] {
                    graph.add_edge(pass, dependent_pass, ());
                }
            }
        }

        // Add edges for read after write dependencies
        for (resource, read_passes) in &self.read_buffer_dependencies {
            if let Some(write_passes) = self.write_buffer_dependencies.get(resource) {
                for &write_pass in write_passes {
                    for &read_pass in read_passes {
                        if !write_passes.contains(&read_pass) {
                            graph.add_edge(write_pass, read_pass, ());
                        }
                    }
                }
            }
        }

        for (resource, read_passes) in &self.read_texture_dependencies {
            if let Some(write_passes) = self.write_texture_dependencies.get(resource) {
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
            for (binding_index, resource_id) in
                self.binding_order.get(&pass.id).unwrap().iter().enumerate()
            {
                println!(
                    "binding_index: {} {}",
                    binding_index, resource_id.manager_id
                );
                if resource_id.manager_id == self.ctx.buffer_manager.id {
                    let handle = ResourceManager::<Buffer>::handle_from_id(
                        self.ctx.buffer_manager.clone(),
                        *resource_id,
                    )
                    .unwrap();
                    let buffer = self.ctx.buffer_manager.get_mut(&handle).unwrap();

                    pipeline.queue_descriptor_buffer(
                        0,
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
                    pipeline.queue_descriptor_image(
                        0,
                        binding_index as u32,
                        0,
                        vk::DescriptorImageInfo::default()
                            .image_view(*view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    );
                }
            }

            pipeline.update_descriptors(&self.ctx);
        }

        for pass in self.graphics_passes.values() {
            let mut pipeline = pass.pipeline.get();
            let Some(bindings) = self.binding_order.get(&pass.id) else {
                continue;
            };
            for (binding_index, resource_id) in bindings.iter().enumerate() {
                println!(
                    "binding_index: {} {}",
                    binding_index, resource_id.manager_id
                );
                if resource_id.manager_id == self.ctx.buffer_manager.id {
                    let handle = ResourceManager::<Buffer>::handle_from_id(
                        self.ctx.buffer_manager.clone(),
                        *resource_id,
                    )
                    .unwrap();
                    let buffer = self.ctx.buffer_manager.get_mut(&handle).unwrap();

                    pipeline.queue_descriptor_buffer(
                        0,
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
                    pipeline.queue_descriptor_image(
                        0,
                        binding_index as u32,
                        0,
                        vk::DescriptorImageInfo::default()
                            .image_view(*view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    );
                }
            }

            pipeline.update_descriptors(&self.ctx);
        }

        self.current_built_graph = Some(graph);
    }

    pub fn run<T>(&mut self, data: T) {
        let Some(graph) = &self.current_built_graph else {
            return;
        };

        let graph_entry_nodes = graph
            .node_references()
            .filter(|(_, node)| {
                graph
                    .neighbors_directed(**node, petgraph::Direction::Incoming)
                    .count()
                    == 0
            })
            .map(|(node, _)| node)
            .collect::<Vec<_>>();

        println!("{:?}", graph_entry_nodes);

        for entry_node in graph_entry_nodes {
            println!("Running entry node: {:?}", entry_node);
            self.run_pass(&entry_node);

            for pass in graph.neighbors_directed(entry_node, petgraph::Direction::Outgoing) {
                // let node = self.compute_passes.get(pass).unwrap();
                self.run_pass(&pass);
            }
        }
    }

    pub fn run_pass(&self, pass_id: &PassId) {
        let node_id = match pass_id.pass_type {
            PassType::Compute => self.compute_passes.get(pass_id).unwrap().id,
            PassType::Graphics => self.graphics_passes.get(pass_id).unwrap().id,
        };

        self.ctx.record_submit(|command_buffer| {
            let read_textures = self
                .read_texture_dependencies
                .iter()
                .filter_map(|(resource, pass_id)| {
                    for &pass in pass_id {
                        if pass == node_id {
                            return Some(resource);
                        }
                    }

                    return None;
                })
                .collect::<Vec<_>>();

            let write_textures = self
                .write_texture_dependencies
                .iter()
                .filter_map(|(resource, pass_id)| {
                    for &pass in pass_id {
                        if pass == node_id {
                            return Some(resource);
                        }
                    }

                    return None;
                })
                .collect::<Vec<_>>();

            // println!("read_buffers: {:?}", read_buffers.len());
            // println!("write_buffers: {:?}", write_buffers.len());
            println!("read_textures: {:?}", read_textures.len());
            println!("write_textures: {:?}", write_textures.len());

            let mut read_barriers = vec![];
            for texture in read_textures {
                let mut texture = texture.get();
                read_barriers.push(texture.transition_without_barrier(
                    &self.ctx.device,
                    command_buffer,
                    &TransitionDesc {
                        new_access_mask: vk::AccessFlags::SHADER_READ,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        new_stage_mask: match pass_id.pass_type {
                            PassType::Compute => vk::PipelineStageFlags::COMPUTE_SHADER,
                            PassType::Graphics => vk::PipelineStageFlags::FRAGMENT_SHADER,
                        },
                    },
                ));
            }

            let mut write_barriers = vec![];
            for texture in write_textures {
                let mut texture = texture.get();
                write_barriers.push(texture.transition_without_barrier(
                    &self.ctx.device,
                    command_buffer,
                    &TransitionDesc {
                        new_access_mask: vk::AccessFlags::SHADER_WRITE,
                        new_layout: vk::ImageLayout::GENERAL,
                        new_stage_mask: match pass_id.pass_type {
                            PassType::Compute => vk::PipelineStageFlags::COMPUTE_SHADER,
                            PassType::Graphics => vk::PipelineStageFlags::FRAGMENT_SHADER,
                        },
                    },
                ));
            }

            let shader_stages =  match pass_id.pass_type {
                PassType::Compute => vk::PipelineStageFlags::COMPUTE_SHADER,
                PassType::Graphics => vk::PipelineStageFlags::FRAGMENT_SHADER,
            };

            
            unsafe {
                self.ctx.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers,
                );
            }

            match pass_id.pass_type {
                PassType::Compute => {
                    let node = self.compute_passes.get(pass_id).unwrap();
                    (node.callback)(&self, &node, &command_buffer);
                }
                PassType::Graphics => {
                    let node = self.graphics_passes.get(pass_id).unwrap();
                    (node.callback)(&self, &node, &command_buffer);
                }
            }
        });
    }

    pub fn run_present_pass(&self) {
        let present_index = self.ctx.acquire_present_index();

        self.ctx.present_record(
            present_index,
            |command_buffer, color_view, depth_view| unsafe {
                let color_attachments = &[vk::RenderingAttachmentInfo::default()
                    .image_view(color_view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.1, 0.1, 0.1, 1.0],
                        },
                    })];

                let depth_attachment = &vk::RenderingAttachmentInfo::default()
                    .image_view(depth_view)
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

                let pipeline = self
                    .ctx
                    .graphics_pipelines
                    .get_mut(&self.pipeline_handle)
                    .unwrap();
                pipeline.bind_pipeline(ctx, command_buffer);
                drop(pipeline);

                ctx.device.cmd_bind_vertex_buffers(
                    command_buffer,
                    0,
                    std::slice::from_ref(
                        &ctx.buffer_manager
                            .get(&self.vertex_buffer)
                            .unwrap()
                            .buffer(),
                    ),
                    &[0],
                );
                self.ctx.device.cmd_bind_index_buffer(
                    command_buffer,
                    self.ctx
                        .buffer_manager
                        .get(&self.index_buffer)
                        .unwrap()
                        .buffer(),
                    0,
                    vk::IndexType::UINT16,
                );
                self.ctx
                    .device
                    .cmd_draw_indexed(command_buffer, 3, 1, 0, 0, 1);

                self.ctx.end_rendering(command_buffer);
            },
        );

        self.ctx.present_submit(present_index);
    }
}

pub struct ComputePass {
    pub id: PassId,
    pub pipeline: ResourceHandle<ComputePipeline>,
    callback: ComputePassCallback,
}

impl ComputePass {
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
            ctx.device.cmd_push_constants(
                command_buffer,
                pipeline.layout,
                ShaderStageFlags::COMPUTE,
                0,
                push_constants,
            );

            ctx.device
                .cmd_dispatch(command_buffer, group_count_x, group_count_y, group_count_z);
        }
    }
}

type ComputePassCallback = Box<dyn Fn(&RenderGraph, &ComputePass, &CommandBuffer)>;

pub struct ComputePassBuilder<'a> {
    id: PassId,
    graph: &'a mut RenderGraph,
    pipeline: Option<ResourceHandle<ComputePipeline>>,
    callback: Option<ComputePassCallback>,
}

impl<'a> ComputePassBuilder<'a> {
    pub fn new(id: &'static str, graph: &'a mut RenderGraph) -> Self {
        Self {
            id: PassId {
                label: id,
                pass_type: PassType::Compute,
            },
            graph,
            pipeline: None,
            callback: None,
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
        self.graph
            .read_buffer_dependencies
            .entry(read_buffer_deps.clone())
            .or_default()
            .push(self.id);
        self.graph
            .binding_order
            .entry(self.id)
            .or_default()
            .push(read_buffer_deps.id());
        self
    }

    pub fn write_buffer(mut self, write_buffer_deps: ResourceHandle<Buffer>) -> Self {
        self.graph
            .write_buffer_dependencies
            .entry(write_buffer_deps.clone())
            .or_default()
            .push(self.id);

        self.graph
            .binding_order
            .entry(self.id)
            .or_default()
            .push(write_buffer_deps.id());

        self
    }

    pub fn read_texture(mut self, read_texture_deps: ResourceHandle<Texture>) -> Self {
        self.graph
            .read_texture_dependencies
            .entry(read_texture_deps.clone())
            .or_default()
            .push(self.id);

        self.graph
            .binding_order
            .entry(self.id)
            .or_default()
            .push(read_texture_deps.id());
        self
    }

    pub fn write_texture(mut self, write_texture_deps: ResourceHandle<Texture>) -> Self {
        self.graph
            .write_texture_dependencies
            .entry(write_texture_deps.clone())
            .or_default()
            .push(self.id);

        self.graph
            .binding_order
            .entry(self.id)
            .or_default()
            .push(write_texture_deps.id());
        self
    }

    pub fn callback(
        mut self,
        callback: impl Fn(&RenderGraph, &ComputePass, &CommandBuffer) + 'static,
    ) -> Self {
        self.callback = Some(Box::new(callback));
        self
    }

    pub fn build(self) {
        let pass = ComputePass {
            id: self.id,
            pipeline: self
                .pipeline
                .expect("Pipeline is required to build a ComputePass"),
            callback: self
                .callback
                .expect("Callback is required to build a ComputePass, cannot be ran without"),
        };
        self.graph.compute_passes.insert(self.id, pass);
    }
}

type GraphicsPassCallback = Box<dyn Fn(&RenderGraph, &GraphicsPass, &CommandBuffer)>;

pub struct GraphicsPass {
    pub id: PassId,
    pub pipeline: ResourceHandle<GraphicsPipeline>,
    callback: GraphicsPassCallback,
}

pub struct GraphicsPassBuilder<'a> {
    id: PassId,
    graph: &'a mut RenderGraph,
    pipeline: Option<ResourceHandle<GraphicsPipeline>>,
    callback: Option<GraphicsPassCallback>,
}

impl<'a> GraphicsPassBuilder<'a> {
    pub fn new(id: &'static str, graph: &'a mut RenderGraph) -> Self {
        Self {
            id: PassId {
                label: id,
                pass_type: PassType::Graphics,
            },
            graph,
            pipeline: None,
            callback: None,
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
        self.graph
            .read_buffer_dependencies
            .entry(read_buffer_deps.clone())
            .or_default()
            .push(self.id);
        self.graph
            .binding_order
            .entry(self.id)
            .or_default()
            .push(read_buffer_deps.id());
        self
    }

    pub fn write_buffer(mut self, write_buffer_deps: ResourceHandle<Buffer>) -> Self {
        self.graph
            .read_buffer_dependencies
            .entry(write_buffer_deps.clone())
            .or_default()
            .push(self.id);

        self.graph
            .write_buffer_dependencies
            .entry(write_buffer_deps.clone())
            .or_default()
            .push(self.id);

        self.graph
            .binding_order
            .entry(self.id)
            .or_default()
            .push(write_buffer_deps.id());

        self
    }

    pub fn read_texture(mut self, read_texture_deps: ResourceHandle<Texture>) -> Self {
        self.graph
            .read_texture_dependencies
            .entry(read_texture_deps.clone())
            .or_default()
            .push(self.id);

        self.graph
            .binding_order
            .entry(self.id)
            .or_default()
            .push(read_texture_deps.id());
        self
    }

    pub fn write_texture(mut self, write_texture_deps: ResourceHandle<Texture>) -> Self {
        self.graph
            .read_texture_dependencies
            .entry(write_texture_deps.clone())
            .or_default()
            .push(self.id);
        self.graph
            .write_texture_dependencies
            .entry(write_texture_deps.clone())
            .or_default()
            .push(self.id);

        self.graph
            .binding_order
            .entry(self.id)
            .or_default()
            .push(write_texture_deps.id());
        self
    }

    pub fn callback(
        mut self,
        callback: impl Fn(&RenderGraph, &GraphicsPass, &CommandBuffer) + 'static,
    ) -> Self {
        self.callback = Some(Box::new(callback));
        self
    }

    pub fn build(self) {
        let pass = GraphicsPass {
            id: self.id,
            pipeline: self
                .pipeline
                .expect("Pipeline is required to build a ComputePass"),
            callback: self
                .callback
                .expect("Callback is required to build a ComputePass, cannot be ran without"),
        };
        self.graph.graphics_passes.insert(self.id, pass);
    }
}
