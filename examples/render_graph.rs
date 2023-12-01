use std::sync::Arc;

use ::smallvec::smallvec;
use ash::vk::{self, PresentModeKHR};
use beuk::{
    buffer::BufferDescriptor,
    compute_pipeline::ComputePipelineDescriptor,
    ctx::{RenderContextDescriptor, SamplerDesc},
    graph::{ComputePass, ComputePassBuilder, GraphicsPass, GraphicsPassBuilder, RenderGraph},
    graphics_pipeline::{
        FragmentState, GraphicsPipelineDescriptor, PrimitiveState, PushConstantRange, VertexState,
    },
    shaders::ShaderDescriptor,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    color: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TestBuffer {
    color: [f32; 4],
}

fn main() {
    simple_logger::SimpleLogger::new().init().unwrap();
    let mut event_loop = EventLoop::new();

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

    #[cfg(feature = "hot-reload")]
    let _watcher = beuk::hot_reload::ShaderHotReload::new(
        ctx.clone(),
        &[std::path::Path::new(r#"./examples/render_graph"#).into()],
    );

    let mut graph = beuk::graph::RenderGraph::<()>::new(ctx.clone());

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

    let buffer = ctx.create_buffer_with_data(
        &BufferDescriptor {
            debug_name: "test".into(),
            location: gpu_allocator::MemoryLocation::GpuOnly,
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            ..Default::default()
        },
        bytemuck::bytes_of(&TestBuffer {
            color: [1.0, 0.0, 1.0, 1.0],
        }),
        0,
    );

    ComputePassBuilder::new("raycast", &mut graph)
        .pipeline(ComputePipelineDescriptor {
            shader: ctx.create_shader(ShaderDescriptor {
                entry_point: "main".into(),
                kind: beuk::shaders::ShaderKind::Compute,
                source: "./examples/render_graph/raycast.comp".into(),
                ..Default::default()
            }),
            prepend_descriptor_sets: None,
            push_constant_range: Some(PushConstantRange {
                stages: beuk::graphics_pipeline::ShaderStages::Compute,
                offset: 0,
                range: std::mem::size_of::<PushConstants>() as u32,
            }),
        })
        .write_texture(attachment_handle.clone())
        .read_buffer(buffer.clone())
        .callback(run_raycast)
        .build();

    ComputePassBuilder::new("raycast-2", &mut graph)
        .pipeline(ComputePipelineDescriptor {
            shader: ctx.create_shader(ShaderDescriptor {
                entry_point: "main".into(),
                kind: beuk::shaders::ShaderKind::Compute,
                source: include_str!("./render_graph/raycast.comp").into(),
                ..Default::default()
            }),
            prepend_descriptor_sets: None,
            push_constant_range: Some(PushConstantRange {
                stages: beuk::graphics_pipeline::ShaderStages::Compute,
                offset: 0,
                range: std::mem::size_of::<PushConstants>() as u32,
            }),
        })
        .write_texture(attachment_handle.clone())
        .read_buffer(buffer.clone())
        .callback(run_raycast_two)
        .build();

    ComputePassBuilder::new("raycast-3", &mut graph)
        .pipeline(ComputePipelineDescriptor {
            shader: ctx.create_shader(ShaderDescriptor {
                entry_point: "main".into(),
                kind: beuk::shaders::ShaderKind::Compute,
                source: include_str!("./render_graph/raycast.comp").into(),
                ..Default::default()
            }),
            prepend_descriptor_sets: None,
            push_constant_range: Some(PushConstantRange {
                stages: beuk::graphics_pipeline::ShaderStages::Compute,
                offset: 0,
                range: std::mem::size_of::<PushConstants>() as u32,
            }),
        })
        .write_texture(attachment_handle.clone())
        .read_buffer(buffer.clone())
        .callback(run_raycast_three)
        .build();

    GraphicsPassBuilder::new("present", &mut graph)
        .pipeline(GraphicsPipelineDescriptor {
            vertex: VertexState {
                shader: ctx.create_shader(ShaderDescriptor {
                    kind: beuk::shaders::ShaderKind::Vertex,
                    entry_point: "main".into(),
                    source: "./examples/render_graph/blit.vert".into(),
                    ..Default::default()
                }),
                buffers: smallvec![],
            },
            fragment: FragmentState {
                shader: ctx.create_shader(ShaderDescriptor {
                    kind: beuk::shaders::ShaderKind::Fragment,
                    entry_point: "main".into(),
                    source: "./examples/render_graph/blit.frag".into(),
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
            blend: vec![],
            multisample: beuk::graphics_pipeline::MultisampleState::default(),
            prepend_descriptor_sets: None,
        })
        // .read_texture(albedo_handle.clone())
        .read_texture(
            attachment_handle.clone(),
            Some(SamplerDesc {
                address_modes: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                texel_filter: vk::Filter::NEAREST,
            }),
        )
        .callback(run_present)
        .build();

    graph.order_and_build_graph();
    // graph.run(&());

    event_loop.run_return(|event, _, control_flow| match event {
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
        Event::RedrawRequested(_) => {
            graph.run(&());
        }
        _ => (),
    });
}

fn run_raycast(rg: &RenderGraph<()>, pass: &ComputePass<()>, command_buffer: vk::CommandBuffer) {
    pass.execute(
        &rg.ctx,
        command_buffer,
        &bytemuck::bytes_of(&PushConstants {
            color: [1.0, 0.0, 0.0, 1.0],
        }),
        1280 / 16,
        720 / 16,
        1,
    );
}

fn run_raycast_three(
    rg: &RenderGraph<()>,
    pass: &ComputePass<()>,
    command_buffer: vk::CommandBuffer,
) {
    pass.execute(
        &rg.ctx,
        command_buffer,
        &bytemuck::bytes_of(&PushConstants {
            color: [0.0, 1.0, 0.0, 1.0],
        }),
        8,
        8,
        1,
    );
}

fn run_raycast_two(
    rg: &RenderGraph<()>,
    pass: &ComputePass<()>,
    command_buffer: vk::CommandBuffer,
) {
    pass.execute(
        &rg.ctx,
        command_buffer,
        &bytemuck::bytes_of(&PushConstants {
            color: [0.0, 0.0, 1.0, 1.0],
        }),
        16,
        16,
        1,
    );
}

fn run_present(rg: &RenderGraph<()>, pass: &GraphicsPass<()>, command_buffer: vk::CommandBuffer) {
    unsafe {
        pass.execute(&rg.ctx, command_buffer, &[]);

        rg.ctx.device.cmd_draw(command_buffer, 3, 1, 0, 0);
    }
}
