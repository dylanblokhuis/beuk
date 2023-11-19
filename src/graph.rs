use std::{any::Any, collections::HashMap, sync::Arc};

use ash::vk::ShaderStageFlags;
use petgraph::graphmap::{DiGraphMap, NodeTrait};

use crate::{
    buffer::Buffer,
    compute_pipeline::{ComputePipeline, ComputePipelineDescriptor},
    ctx::RenderContext,
    memory::{ResourceHandle, ResourceId},
    texture::Texture,
};

type PassId = &'static str;

pub struct RenderGraph {
    ctx: Arc<RenderContext>,
    pub(super) compute_passes: HashMap<PassId, ComputePass>,
    // track in which order the resources are accessed by the added passes
    pub read_buffer_dependencies: HashMap<ResourceHandle<Buffer>, Vec<PassId>>,
    pub write_buffer_dependencies: HashMap<ResourceHandle<Buffer>, Vec<PassId>>,
    pub read_texture_dependencies: HashMap<ResourceHandle<Texture>, Vec<PassId>>,
    pub write_texture_dependencies: HashMap<ResourceHandle<Texture>, Vec<PassId>>,
}

impl RenderGraph {
    pub fn new(ctx: Arc<RenderContext>) -> Self {
        Self {
            ctx,
            compute_passes: HashMap::new(),
            read_buffer_dependencies: HashMap::new(),
            write_buffer_dependencies: HashMap::new(),
            read_texture_dependencies: HashMap::new(),
            write_texture_dependencies: HashMap::new(),
        }
    }

    pub fn run(&mut self) {
        // self.schedule.run(&mut self.world)
        // for (_, pass) in self.compute_passes.iter() {
        //     pass.
        // }
    }

    pub fn order_and_build_graph(&self) {
        // Create a new directed graph
        let mut graph = DiGraphMap::<&'static str, ()>::new();

        // Add nodes for each compute pass
        for (&label, _) in &self.compute_passes {
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

        let sorted_passes = petgraph::algo::toposort(&graph, None).unwrap();

        println!("Sorted passes: {:?}", sorted_passes);
    }
}

pub struct ComputePass {
    pipeline: ResourceHandle<ComputePipeline>,
}

impl ComputePass {
    // TODO: this will called once, just a temp function, should be done whenever the pass is added
    pub fn setup(&self) {
        let pipeline = self.pipeline.get();

        // do descriptors here
    }

    pub fn execute(
        &self,
        ctx: &RenderContext,
        push_constants: &[u8],
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        ctx.record_submit(|command_buffer| unsafe {
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
        });
    }
}

pub struct ComputePassBuilder<'a> {
    id: PassId,
    graph: &'a mut RenderGraph,
    pipeline: Option<ResourceHandle<ComputePipeline>>,
}

impl<'a> ComputePassBuilder<'a> {
    pub fn new(id: PassId, graph: &'a mut RenderGraph) -> Self {
        Self {
            id,
            graph,
            pipeline: None,
        }
    }

    pub fn pipeline(mut self, pipeline_descriptor: ComputePipelineDescriptor) -> Self {
        self.pipeline = Some(
            self.graph
                .ctx
                .create_compute_pipeline(self.id, pipeline_descriptor),
        );
        self
    }

    pub fn read_buffer(mut self, read_buffer_deps: ResourceHandle<Buffer>) -> Self {
        self.graph
            .read_buffer_dependencies
            .entry(read_buffer_deps.clone())
            .or_default()
            .push(self.id);
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

        self
    }

    pub fn read_texture(mut self, read_texture_deps: ResourceHandle<Texture>) -> Self {
        self.graph
            .read_texture_dependencies
            .entry(read_texture_deps.clone())
            .or_default()
            .push(self.id);
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
        self
    }

    // pub fn system<M>(mut self, system: impl IntoSystemConfigs<M>) -> Self {
    //     self.graph.schedule.add_systems(system.before);
    //     self
    // }

    pub fn build(self) {
        let pass = ComputePass {
            pipeline: self
                .pipeline
                .expect("Pipeline is required to build a ComputePass"),
        };
        pass.setup();
        self.graph.compute_passes.insert(self.id, pass);
    }
}
