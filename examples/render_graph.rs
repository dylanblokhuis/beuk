use std::sync::Arc;

use ::smallvec::smallvec;
use ash::vk::{self, PresentModeKHR};
use beuk::{
    compute_pipeline::ComputePipelineDescriptor,
    ctx::RenderContextDescriptor,
    graph::{ComputePass, ComputePassBuilder, GraphicsPass, GraphicsPassBuilder, RenderGraph},
    graphics_pipeline::{
        BlendState, FragmentState, GraphicsPipelineDescriptor, PrimitiveState, VertexBufferLayout,
        VertexState,
    },
    shaders::ShaderDescriptor,
    smallvec,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

#[repr(C, align(16))]
#[derive(Clone, Debug, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 4],
    color: [f32; 4],
}

fn main() {
    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("A fantastic window!")
        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0))
        .build(&event_loop)
        .unwrap();

    let ctx = Arc::new(beuk::ctx::RenderContext::new(RenderContextDescriptor {
        display_handle: window.raw_display_handle(),
        window_handle: window.raw_window_handle(),
        present_mode: PresentModeKHR::default(),
    }));

    let mut graph = beuk::graph::RenderGraph::new(ctx.clone());

    let swapchain = ctx.get_swapchain().clone();
    let attachment_handle = ctx.create_texture(
        "media",
        &vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: swapchain.surface_format.format,
            extent: vk::Extent3D {
                width: swapchain.surface_resolution.width,
                height: swapchain.surface_resolution.height,
                depth: 1,
            },
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE,
            samples: vk::SampleCountFlags::TYPE_1,
            mip_levels: 1,
            array_layers: 1,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        },
        true,
    );

    let albedo_handle = ctx.create_texture(
        "albedo",
        &vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: swapchain.surface_format.format,
            extent: vk::Extent3D {
                width: swapchain.surface_resolution.width,
                height: swapchain.surface_resolution.height,
                depth: 1,
            },
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE,
            samples: vk::SampleCountFlags::TYPE_1,
            mip_levels: 1,
            array_layers: 1,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        },
        true,
    );

    ComputePassBuilder::new("raycast", &mut graph)
        .pipeline(ComputePipelineDescriptor {
            shader: ctx.create_shader(ShaderDescriptor {
                entry_point: "main".into(),
                kind: beuk::shaders::ShaderKind::Compute,
                optimization: beuk::shaders::ShaderOptimization::None,
                source: include_str!("./render_graph/raycast.comp").into(),
                ..Default::default()
            }),
            prepend_descriptor_sets: None,
            push_constant_range: None,
        })
        .write_texture(attachment_handle.clone())
        .callback(run_raycast)
        .build();

    ComputePassBuilder::new("raycast-2", &mut graph)
        .pipeline(ComputePipelineDescriptor {
            shader: ctx.create_shader(ShaderDescriptor {
                entry_point: "main".into(),
                kind: beuk::shaders::ShaderKind::Compute,
                optimization: beuk::shaders::ShaderOptimization::None,
                source: include_str!("./render_graph/raycast.comp").into(),
                ..Default::default()
            }),
            prepend_descriptor_sets: None,
            push_constant_range: None,
        })
        .write_texture(attachment_handle.clone())
        .callback(run_raycast)
        .build();

    ComputePassBuilder::new("raycast-3", &mut graph)
        .pipeline(ComputePipelineDescriptor {
            shader: ctx.create_shader(ShaderDescriptor {
                entry_point: "main".into(),
                kind: beuk::shaders::ShaderKind::Compute,
                optimization: beuk::shaders::ShaderOptimization::None,
                source: include_str!("./render_graph/raycast.comp").into(),
                ..Default::default()
            }),
            prepend_descriptor_sets: None,
            push_constant_range: None,
        })
        .write_texture(albedo_handle.clone())
        .callback(run_raycast)
        .build();

    ComputePassBuilder::new("raycast-4", &mut graph)
        .pipeline(ComputePipelineDescriptor {
            shader: ctx.create_shader(ShaderDescriptor {
                entry_point: "main".into(),
                kind: beuk::shaders::ShaderKind::Compute,
                optimization: beuk::shaders::ShaderOptimization::None,
                source: include_str!("./render_graph/raycast.comp").into(),
                ..Default::default()
            }),
            prepend_descriptor_sets: None,
            push_constant_range: None,
        })
        .write_texture(albedo_handle.clone())
        .callback(run_raycast)
        .build();

    ComputePassBuilder::new("raycast-5", &mut graph)
        .pipeline(ComputePipelineDescriptor {
            shader: ctx.create_shader(ShaderDescriptor {
                entry_point: "main".into(),
                kind: beuk::shaders::ShaderKind::Compute,
                optimization: beuk::shaders::ShaderOptimization::None,
                source: include_str!("./render_graph/raycast.comp").into(),
                ..Default::default()
            }),
            prepend_descriptor_sets: None,
            push_constant_range: None,
        })
        .write_texture(albedo_handle.clone())
        .callback(run_raycast)
        .build();

    GraphicsPassBuilder::new("present", &mut graph)
        .pipeline(GraphicsPipelineDescriptor {
            vertex: VertexState {
                shader: ctx.create_shader(ShaderDescriptor {
                    kind: beuk::shaders::ShaderKind::Vertex,
                    entry_point: "main".into(),
                    source: "./examples/triangle/shader.vert".into(),
                    ..Default::default()
                }),
                buffers: smallvec![VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u32,
                    step_mode: beuk::graphics_pipeline::VertexStepMode::Vertex,
                    attributes: smallvec![
                        beuk::graphics_pipeline::VertexAttribute {
                            shader_location: 0,
                            format: vk::Format::R32G32B32A32_SFLOAT,
                            offset: bytemuck::offset_of!(Vertex, pos) as u32,
                        },
                        beuk::graphics_pipeline::VertexAttribute {
                            shader_location: 1,
                            format: vk::Format::R32G32B32A32_SFLOAT,
                            offset: bytemuck::offset_of!(Vertex, color) as u32,
                        },
                    ],
                }],
            },
            fragment: FragmentState {
                shader: ctx.create_shader(ShaderDescriptor {
                    kind: beuk::shaders::ShaderKind::Fragment,
                    entry_point: "main".into(),
                    source: "./examples/triangle/shader.frag".into(),
                    ..Default::default()
                }),
                color_attachment_formats: smallvec![swapchain.surface_format.format],
                depth_attachment_format: swapchain.depth_image_format,
            },

            viewport: None,
            primitive: PrimitiveState {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            },
            depth_stencil: Default::default(),
            push_constant_range: None,
            blend: vec![BlendState::ALPHA_BLENDING],
            multisample: beuk::graphics_pipeline::MultisampleState::default(),
            prepend_descriptor_sets: None,
        })
        // .read_texture(albedo_handle.clone())
        // .read_texture(attachment_handle.clone())
        .callback(run_present)
        .build();

    graph.order_and_build_graph();
    // let ordered_passes = petgraph::algo::toposort(&built_graph, None).unwrap();
    graph.run();

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            window_id,
        } if window_id == window.id() => control_flow.set_exit(),

        Event::WindowEvent { event, .. } => match event {
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                ctx.recreate_swapchain(new_inner_size.width, new_inner_size.height);
            }
            WindowEvent::Resized(size) => {
                ctx.recreate_swapchain(size.width, size.height);
            }
            _ => (),
        },

        Event::MainEventsCleared => {
            window.request_redraw();
        }
        Event::RedrawRequested(_) => {}
        _ => (),
    });

    // ComputePassBuilder::new(ctx, "raycast").pipeline(ComputePipelineDescriptor {

    // });

    // graph.add_compute_pass("raycast");
}

fn run_raycast(rg: &RenderGraph, pass: &ComputePass, command_buffer: &vk::CommandBuffer) {
    // println!("Hey!");
    // std::thread::sleep(std::time::Duration::from_secs(1));
    // println!("Hey!!");

    // pass.execute(&rg.ctx, *command_buffer, &[], 8, 8, 1);

    println!("Running raycast pass: {:?}", pass.id);
    // rg.ctx.record_submit(|command_buffer| unsafe {
    //     let pipeline = pass.pipeline.get();
    //     pipeline.bind_pipeline(&rg.ctx, command_buffer);
    // });
}

fn run_present(rg: &RenderGraph, pass: &GraphicsPass, command_buffer: &vk::CommandBuffer) {
    println!("Running present pass: {:?}", pass.id);
    // rg.ctx.record_submit(|command_buffer| unsafe {
    //     let pipeline = pass.pipeline.get();
    //     pipeline.bind_pipeline(&rg.ctx, command_buffer);
    // });
}
