use ash::vk::Extent3D;
use beuk::ash::vk::{self, BufferUsageFlags};
use beuk::buffer::{Buffer, BufferDescriptor};
use beuk::compute_pipeline::{ComputePipeline, ComputePipelineDescriptor};
use beuk::ctx::RenderContextDescriptor;

use beuk::graphics_pipeline::{
    BlendState, FragmentState, GraphicsPipeline, VertexBufferLayout, VertexState,
};
#[cfg(feature = "hot-reload")]
use beuk::hot_reload::ShaderHotReload;
use beuk::memory::ResourceHandle;
use beuk::shaders::ShaderDescriptor;
use beuk::texture::Texture;
use beuk::{
    ctx::RenderContext,
    graphics_pipeline::{GraphicsPipelineDescriptor, PrimitiveState},
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use smallvec::smallvec;
use std::sync::Arc;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

fn main() {
    simple_logger::SimpleLogger::new().init().unwrap();
    // use tracing_subscriber::layer::SubscriberExt;
    // tracing::subscriber::set_global_default(
    //     tracing_subscriber::registry().with(tracing_tracy::TracyLayer::new()),
    // )
    // .expect("set up the subscriber");

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("A fantastic window!")
        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0))
        .build(&event_loop)
        .unwrap();

    let ctx = Arc::new(beuk::ctx::RenderContext::new(RenderContextDescriptor {
        display_handle: window.raw_display_handle(),
        window_handle: window.raw_window_handle(),
        present_mode: vk::PresentModeKHR::default(),
    }));

    #[cfg(feature = "hot-reload")]
    let _watcher = ShaderHotReload::new(
        ctx.clone(),
        &[std::path::Path::new(r#"./examples/triangle"#).into()],
    );

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
    compute_handle: ResourceHandle<ComputePipeline>,
    vertex_buffer: ResourceHandle<Buffer>,
    index_buffer: ResourceHandle<Buffer>,
    attachment_handle: ResourceHandle<Texture>,
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
                location: gpu_allocator::MemoryLocation::GpuOnly,
                usage: BufferUsageFlags::VERTEX_BUFFER,
                ..Default::default()
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
                location: gpu_allocator::MemoryLocation::GpuOnly,
                usage: BufferUsageFlags::INDEX_BUFFER,
                ..Default::default()
            },
            bytemuck::cast_slice(&[0u16, 1, 2]),
            0,
        );

        let swapchain: std::sync::RwLockReadGuard<'_, beuk::ctx::RenderSwapchain> =
            ctx.get_swapchain();

        let pipeline_handle = ctx.create_graphics_pipeline(
            "triangle",
            GraphicsPipelineDescriptor {
                vertex: VertexState {
                    shader: ctx.create_shader(ShaderDescriptor {
                        kind: beuk::shaders::ShaderKind::Vertex,
                        entry_point: "main".into(),
                        source: include_str!("./triangle/shader.vert").into(),
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
                        source: include_str!("./triangle/shader.frag").into(),
                        ..Default::default()
                    }),
                    color_attachment_formats: smallvec![swapchain.surface_format.format],
                    depth_attachment_format: vk::Format::UNDEFINED,
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
            },
        );

        let compute_handle = ctx.create_compute_pipeline(
            "invert colors",
            ComputePipelineDescriptor {
                shader: ctx.create_shader(ShaderDescriptor {
                    kind: beuk::shaders::ShaderKind::Compute,
                    entry_point: "main".into(),
                    source: "./examples/triangle/shader.comp".into(),
                    ..Default::default()
                }),
                push_constant_range: None,
                prepend_descriptor_sets: None,
            },
        );

        let attachment_format = swapchain.surface_format.format;
        let attachment_handle = ctx.create_texture(
            "media",
            &vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: attachment_format,
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

        Self {
            pipeline_handle,
            compute_handle,
            vertex_buffer,
            index_buffer,
            attachment_handle,
        }
    }

    pub fn draw(&self, ctx: &RenderContext) {
        ctx.record_submit(|command_buffer| unsafe {
            let attachment_view = ctx.get_texture_view(&self.attachment_handle).unwrap();
            let color_attachments = &[vk::RenderingAttachmentInfo::default()
                .image_view(*attachment_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.1, 0.1, 0.1, 1.0],
                    },
                })];

            // let depth_attachment = &vk::RenderingAttachmentInfo::default()
            //     .image_view(depth_view)
            //     .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            //     .load_op(vk::AttachmentLoadOp::CLEAR)
            //     .store_op(vk::AttachmentStoreOp::STORE)
            //     .clear_value(vk::ClearValue {
            //         depth_stencil: vk::ClearDepthStencilValue {
            //             depth: 1.0,
            //             stencil: 0,
            //         },
            //     });

            ctx.begin_rendering(command_buffer, color_attachments, None);

            let mut pipeline = ctx
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
                vk::IndexType::UINT16,
            );
            ctx.device.cmd_draw_indexed(command_buffer, 3, 1, 0, 0, 1);
            ctx.end_rendering(command_buffer);

            // flip colors with compute
            let mut compute_pipeline = ctx.compute_pipelines.get_mut(&self.compute_handle).unwrap();
            compute_pipeline.queue_descriptor_image(
                0,
                0,
                0,
                vk::DescriptorImageInfo::default()
                    .image_view(*attachment_view)
                    .image_layout(vk::ImageLayout::GENERAL),
            );
            compute_pipeline.queue_descriptor_image(
                0,
                1,
                0,
                vk::DescriptorImageInfo::default()
                    .image_view(*attachment_view)
                    .image_layout(vk::ImageLayout::GENERAL),
            );
            compute_pipeline.bind_descriptor_sets(ctx, command_buffer);
            compute_pipeline.bind_pipeline(ctx, command_buffer);
            let texture = ctx.texture_manager.get(&self.attachment_handle).unwrap();
            ctx.device.cmd_dispatch(
                command_buffer,
                texture.extent.width / 16,
                texture.extent.height / 16,
                1,
            );
        });

        let present_index = ctx.acquire_present_index();

        ctx.present_record(present_index, |command_buffer, _image_view, _depth_view| {
            ctx.copy_texture_to_texture(
                command_buffer,
                &mut ctx
                    .texture_manager
                    .get_mut(&self.attachment_handle)
                    .unwrap(),
                &mut Texture::from_swapchain_image(
                    ctx.get_swapchain().present_images[present_index as usize],
                ),
                Extent3D {
                    width: ctx.get_swapchain().surface_resolution.width,
                    height: ctx.get_swapchain().surface_resolution.height,
                    depth: 1,
                }
                .into(),
            );
        });

        ctx.present_submit(present_index);
    }
}
