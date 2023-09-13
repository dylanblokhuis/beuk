use beuk::ash::vk::{self, BufferUsageFlags};
use beuk::buffer::MemoryLocation;
use beuk::buffer::{Buffer, BufferDescriptor};
use beuk::ctx::RenderContextDescriptor;
use beuk::graphics_pipeline::{
    BlendState, FragmentState, GraphicsPipeline, VertexBufferLayout, VertexState,
};
use beuk::memory::ResourceHandle;
use beuk::shaders::ShaderDescriptor;
use beuk::{
    ctx::RenderContext,
    graphics_pipeline::{GraphicsPipelineDescriptor, PrimitiveState},
};
use image::EncodableLayout;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use smallvec::smallvec;
use tracing_subscriber::prelude::__tracing_subscriber_SubscriberExt;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

fn main() {
    simple_logger::SimpleLogger::new().init().unwrap();
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(tracing_tracy::TracyLayer::new()),
    )
    .expect("set up the subscriber");

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("A fantastic window!")
        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0))
        .build(&event_loop)
        .unwrap();

    let ctx = beuk::ctx::RenderContext::new(RenderContextDescriptor {
        display_handle: window.raw_display_handle(),
        window_handle: window.raw_window_handle(),
        present_mode: vk::PresentModeKHR::default(),
    });

    let canvas = Canvas::new(&ctx);

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
            canvas.draw(&ctx);
        }
        _ => (),
    });
}

struct Canvas {
    pipeline_handle: ResourceHandle<GraphicsPipeline>,
    vertex_buffer: ResourceHandle<Buffer>,
    index_buffer: ResourceHandle<Buffer>,
}

#[repr(C, align(16))]
#[derive(Clone, Debug, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 4],
    color: [f32; 4],
}

impl Canvas {
    fn new(ctx: &RenderContext) -> Self {
        let vertex_buffer = ctx.create_buffer_with_data(
            &BufferDescriptor {
                debug_name: "vertices",
                location: MemoryLocation::GpuOnly,
                size: (std::mem::size_of::<Vertex>() * 3) as u64,
                usage: BufferUsageFlags::VERTEX_BUFFER,
            },
            bytemuck::cast_slice(&[
                Vertex {
                    pos: [-1.0, 1.0, 0.0, 1.0],
                    color: [0.0, 1.0, 0.0, 0.5],
                },
                Vertex {
                    pos: [1.0, 1.0, 0.0, 1.0],
                    color: [0.0, 0.0, 1.0, 0.5],
                },
                Vertex {
                    pos: [0.0, -1.0, 0.0, 1.0],
                    color: [1.0, 0.0, 0.0, 0.5],
                },
            ]),
            0,
        );

        let index_buffer = ctx.create_buffer_with_data(
            &BufferDescriptor {
                debug_name: "indices",
                location: MemoryLocation::GpuOnly,
                size: (std::mem::size_of::<u32>() * 3) as u64,
                usage: BufferUsageFlags::INDEX_BUFFER,
            },
            bytemuck::cast_slice(&[0u32, 1, 2]),
            0,
        );

        let swapchain = ctx.get_swapchain();
        let pipeline_handle = ctx.create_graphics_pipeline(GraphicsPipelineDescriptor {
            vertex: VertexState {
                shader: ctx.create_shader(ShaderDescriptor {
                    kind: beuk::shaders::ShaderKind::Vertex,
                    entry_point: "main".into(),
                    source: include_str!("./texture/shader.vert").into(),
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
                    source: include_str!("./texture/shader.frag").into(),
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
        });

        let wallpaper_bytes = include_bytes!("./texture/95.jpg");
        let image = image::load_from_memory(wallpaper_bytes).unwrap();
        let texture_handle = ctx.create_texture_with_data(
            "texture",
            &vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_SRGB,
                extent: vk::Extent3D {
                    width: image.width() as u32,
                    height: image.height() as u32,
                    depth: 1,
                },
                usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            },
            image.to_rgba8().as_bytes(),
            0,
            false,
        );

        let view = ctx.get_texture_view(&texture_handle).unwrap();

        unsafe {
            let pipeline = ctx.graphics_pipelines.get(&pipeline_handle).unwrap();
            ctx.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(pipeline.descriptor_sets[0])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(std::slice::from_ref(
                        &vk::DescriptorImageInfo::default()
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image_view(*view),
                    ))],
                &[],
            );
        }

        Self {
            pipeline_handle,
            vertex_buffer,
            index_buffer,
        }
    }

    pub fn draw(&self, ctx: &RenderContext) {
        let present_index = ctx.acquire_present_index();

        ctx.present_record(
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

                ctx.begin_rendering(command_buffer, color_attachments, Some(depth_attachment));

                let pipeline = ctx
                    .graphics_pipelines
                    .get_mut(&self.pipeline_handle)
                    .unwrap();
                pipeline.bind_pipeline(ctx, command_buffer);
                pipeline.bind_descriptor_sets(ctx, command_buffer);

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
                ctx.device.cmd_bind_index_buffer(
                    command_buffer,
                    ctx.buffer_manager.get(&self.index_buffer).unwrap().buffer(),
                    0,
                    vk::IndexType::UINT32,
                );
                ctx.device.cmd_draw_indexed(command_buffer, 3, 1, 0, 0, 1);

                ctx.end_rendering(command_buffer);
            },
        );

        ctx.present_submit(present_index);
    }
}
