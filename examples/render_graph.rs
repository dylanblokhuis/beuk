use std::sync::Arc;

use ash::vk::{self, PresentModeKHR};
use beuk::{
    compute_pipeline::ComputePipelineDescriptor,
    ctx::RenderContextDescriptor,
    graph::{ComputePass, ComputePassBuilder},
    shaders::ShaderDescriptor,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

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
        .build();

    graph.order_and_build_graph();

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
        Event::RedrawRequested(_) => {
            graph.run();
        }
        _ => (),
    });

    // ComputePassBuilder::new(ctx, "raycast").pipeline(ComputePipelineDescriptor {

    // });

    // graph.add_compute_pass("raycast");
}

fn run_raycast() {
    println!("Hey!");
    std::thread::sleep(std::time::Duration::from_secs(1));
    println!("Hey!!");
}
