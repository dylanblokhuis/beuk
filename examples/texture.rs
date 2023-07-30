use beuk::ash::vk::{self, BufferUsageFlags, PipelineVertexInputStateCreateInfo};
use beuk::ctx::{RenderContextDescriptor, SamplerDesc};
use beuk::memory::{BufferDescriptor, MemoryLocation};
use beuk::pipeline::BlendState;
use beuk::{
    ctx::RenderContext,
    memory::{BufferHandle, PipelineHandle},
    pipeline::{GraphicsPipelineDescriptor, PrimitiveState},
    shaders::Shader,
};
use image::EncodableLayout;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
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
    pipeline_handle: PipelineHandle,
    vertex_buffer: BufferHandle,
    index_buffer: BufferHandle,
}

#[repr(C, align(16))]
#[derive(Clone, Debug, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 4],
    color: [f32; 4],
}

impl Canvas {
    fn new(ctx: &RenderContext) -> Self {
        let vertex_shader = Shader::from_source_text(
            &ctx.device,
            include_str!("./texture/shader.vert"),
            "triangle.vert",
            beuk::shaders::ShaderKind::Vertex,
            "main",
        );

        let fragment_shader = Shader::from_source_text(
            &ctx.device,
            include_str!("./texture/shader.frag"),
            "triangle.frag",
            beuk::shaders::ShaderKind::Fragment,
            "main",
        );

        let vertex_buffer = ctx.create_buffer_with_data(
            &BufferDescriptor {
                debug_name: "vertices",
                location: MemoryLocation::CpuToGpu,
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

        // let index_buffer = ctx.buffer_manager.create_buffer_with_data(
        //     "indices",
        //     bytemuck::cast_slice(&[0u32, 1, 2]),
        //     BufferUsageFlags::INDEX_BUFFER,
        //     gpu_allocator::MemoryLocation::CpuToGpu,
        // );
        let index_buffer = ctx.create_buffer_with_data(
            &BufferDescriptor {
                debug_name: "indices",
                location: MemoryLocation::CpuToGpu,
                size: (std::mem::size_of::<u32>() * 3) as u64,
                usage: BufferUsageFlags::INDEX_BUFFER,
            },
            bytemuck::cast_slice(&[0u32, 1, 2]),
            0,
        );

        let swapchain = ctx.get_swapchain();
        let pipeline_handle = ctx.create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader,
            fragment_shader,
            vertex_input: PipelineVertexInputStateCreateInfo::default()
                .vertex_attribute_descriptions(&[
                    vk::VertexInputAttributeDescription {
                        location: 0,
                        binding: 0,
                        format: vk::Format::R32G32B32A32_SFLOAT,
                        offset: bytemuck::offset_of!(Vertex, pos) as u32,
                    },
                    vk::VertexInputAttributeDescription {
                        location: 1,
                        binding: 0,
                        format: vk::Format::R32G32B32A32_SFLOAT,
                        offset: bytemuck::offset_of!(Vertex, color) as u32,
                    },
                ])
                .vertex_binding_descriptions(&[vk::VertexInputBindingDescription {
                    binding: 0,
                    stride: std::mem::size_of::<Vertex>() as u32,
                    input_rate: vk::VertexInputRate::VERTEX,
                }]),
            color_attachment_formats: &[swapchain.surface_format.format],
            depth_attachment_format: swapchain.depth_image_format,
            viewport: None,
            primitive: PrimitiveState {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            },
            depth_stencil: Default::default(),
            push_constant_range: None,
            blend: vec![BlendState::ALPHA_BLENDING],
            multisample: beuk::pipeline::MultisampleState::default(),
        });

        let wallpaper_bytes = include_bytes!("./texture/95.jpg");
        let image = image::load_from_memory(wallpaper_bytes).unwrap();
        let texture_handle = ctx.create_texture(
            "fonts",
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
        );

        let buffer_handle = ctx.create_buffer_with_data(
            &BufferDescriptor {
                debug_name: "fonts",
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                location: MemoryLocation::CpuToGpu,
                ..Default::default()
            },
            bytemuck::cast_slice(image.to_rgba8().as_bytes()),
            0,
        );
        ctx.copy_buffer_to_texture(&buffer_handle, &texture_handle);
        let view = ctx.get_texture_view(&texture_handle);

        unsafe {
            let pipelines = ctx.get_pipeline_manager();
            let pipeline = pipelines.get_graphics_pipeline(&pipeline_handle.id());
            ctx.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(pipeline.descriptor_sets[0])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(
                        &vk::DescriptorImageInfo::default()
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image_view(*view)
                            .sampler(pipelines.immutable_shader_info.get_sampler(&SamplerDesc {
                                address_modes: vk::SamplerAddressMode::REPEAT,
                                mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                                texel_filter: vk::Filter::LINEAR,
                            })),
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
            |ctx, command_buffer, color_view, depth_view| unsafe {
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

                ctx.get_pipeline_manager()
                    .get_graphics_pipeline(&self.pipeline_handle.id())
                    .bind(&ctx.device, command_buffer);

                let manager = ctx.get_buffer_manager();
                ctx.device.cmd_bind_vertex_buffers(
                    command_buffer,
                    0,
                    std::slice::from_ref(&manager.get(self.vertex_buffer.id()).buffer),
                    &[0],
                );
                ctx.device.cmd_bind_index_buffer(
                    command_buffer,
                    manager.get(self.index_buffer.id()).buffer(),
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
