use beuk::ash::vk::{self, BufferUsageFlags, PipelineVertexInputStateCreateInfo};
use beuk::buffer::{Buffer, BufferDescriptor, MemoryLocation};
use beuk::ctx::{RenderContextDescriptor, SamplerDesc};

use beuk::memory::ResourceHandle;
use beuk::pipeline::{BlendState, GraphicsPipeline};
use beuk::texture::Texture;
use beuk::{
    ctx::RenderContext,
    pipeline::{GraphicsPipelineDescriptor, PrimitiveState},
    shaders::Shader,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::sync::Arc;
use std::time::{Instant, Duration};
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

    let pass_one = Pass::new(&ctx, TriangleSide::LEFT, [1.0, 0.0, 0.0, 1.0]);
    let pass_two = Pass::new(&ctx, TriangleSide::RIGHT, [0.0, 1.0, 0.0, 1.0]);

    let pass_one_attachment = pass_one.attachment.clone();
    let pass_two_attachment = pass_two.attachment.clone();
    let present_pass = PresentRenderPass::new(&ctx);

    ctx.command_thread_pool.spawn({
        let ctx = ctx.clone();
        move || loop {
            pass_one.draw(&ctx);
        }
    });

    ctx.command_thread_pool.spawn({
        let ctx = ctx.clone();

        move || loop {
            pass_two.draw(&ctx);
        }
    });

   
let mut last_resize_time = Instant::now();
let mut pending_resize: Option<(u32, u32)> = None;
const RESIZE_THROTTLE_DURATION: Duration = Duration::from_millis(100);  // Throttle to 100ms

event_loop.run(move |event, _, control_flow| match event {
    Event::WindowEvent {
        event: WindowEvent::CloseRequested,
        window_id,
    } if window_id == window.id() => control_flow.set_exit(),

    Event::WindowEvent { event, .. } => match event {
        WindowEvent::ScaleFactorChanged { new_inner_size, .. } | WindowEvent::Resized(new_inner_size) => {
            pending_resize = Some((new_inner_size.width, new_inner_size.height));
        }
        _ => (),
    },

    Event::MainEventsCleared => {
        if let Some((width, height)) = pending_resize {
            if last_resize_time.elapsed() > RESIZE_THROTTLE_DURATION {
                ctx.recreate_swapchain(width, height);
                last_resize_time = Instant::now();
                pending_resize = None;
            }
        }
        window.request_redraw();
    }
    Event::RedrawRequested(_) => {
        present_pass.combine_and_draw(&ctx, &pass_one_attachment, &pass_two_attachment)
    }
    _ => (),
});
}

#[repr(C, align(16))]
#[derive(Clone, Debug, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 4],
    color: [f32; 4],
}

pub struct Pass {
    pipeline_handle: ResourceHandle<GraphicsPipeline>,
    vertex_buffer: ResourceHandle<Buffer>,
    index_buffer: ResourceHandle<Buffer>,
    pub attachment: ResourceHandle<Texture>,
}

pub enum TriangleSide {
    LEFT,
    RIGHT,
}

impl Pass {
    pub fn new(ctx: &RenderContext, side: TriangleSide, color: [f32; 4]) -> Self {
        let triangle_left = [
            Vertex {
                pos: [-1.0, 1.0, 0.0, 1.0],
                color,
            },
            Vertex {
                pos: [-0.5, 1.0, 0.0, 1.0],
                color,
            },
            Vertex {
                pos: [-0.75, -1.0, 0.0, 1.0],
                color,
            },
        ];

        let triangle_right = [
            Vertex {
                pos: [0.5, 1.0, 0.0, 1.0],
                color,
            },
            Vertex {
                pos: [1.0, 1.0, 0.0, 1.0],
                color,
            },
            Vertex {
                pos: [0.75, -1.0, 0.0, 1.0],
                color,
            },
        ];

        let vertex_buffer = ctx.create_buffer_with_data(
            &BufferDescriptor {
                debug_name: "vertices",
                size: std::mem::size_of::<[Vertex; 3]>() as vk::DeviceSize,
                usage: BufferUsageFlags::VERTEX_BUFFER,
                location: MemoryLocation::GpuOnly,
            },
            match side {
                TriangleSide::LEFT => bytemuck::cast_slice(&triangle_left),
                TriangleSide::RIGHT => bytemuck::cast_slice(&triangle_right),
            },
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
                    | vk::ImageUsageFlags::SAMPLED,
                samples: vk::SampleCountFlags::TYPE_1,
                mip_levels: 1,
                array_layers: 1,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            },
        );

        let swapchain = ctx.get_swapchain();
        let pipeline_handle = ctx.create_graphics_pipeline(&GraphicsPipelineDescriptor {
            vertex_shader: Shader::from_source_text(
                &ctx.device,
                include_str!("./triangle/shader.vert"),
                "shader.vert",
                beuk::shaders::ShaderKind::Vertex,
                "main",
            ),
            fragment_shader: Shader::from_source_text(
                &ctx.device,
                include_str!("./triangle/shader.frag"),
                "shader.frag",
                beuk::shaders::ShaderKind::Fragment,
                "main",
            ),
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
            depth_attachment_format: vk::Format::UNDEFINED,
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

        Self {
            pipeline_handle,
            vertex_buffer,
            index_buffer,
            attachment: attachment_handle,
        }
    }

    pub fn draw(&self, ctx: &RenderContext) {
        ctx.record_submit(|ctx, command_buffer| unsafe {
            let color_attachments = &[vk::RenderingAttachmentInfo::default()
                .image_view(*ctx.get_texture_view(&self.attachment).unwrap())
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                })];

            ctx.begin_rendering(command_buffer, color_attachments, None);

            ctx.graphics_pipelines
                .get(&self.pipeline_handle)
                .unwrap()
                .bind(&ctx.device, command_buffer);
            ctx.device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                std::slice::from_ref(&ctx.buffer_manager.get(&self.vertex_buffer).unwrap().buffer),
                &[0],
            );
            ctx.device.cmd_bind_index_buffer(
                command_buffer,
                ctx.buffer_manager.get(&self.index_buffer).unwrap().buffer,
                0,
                vk::IndexType::UINT32,
            );
            ctx.device.cmd_draw_indexed(command_buffer, 3, 1, 0, 0, 1);
            ctx.end_rendering(command_buffer);
        });
    }
}

pub struct PresentRenderPass {
    pipeline_handle: ResourceHandle<GraphicsPipeline>,
    vertex_buffer: ResourceHandle<Buffer>,
    index_buffer: ResourceHandle<Buffer>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct PresentVertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

impl PresentRenderPass {
    pub fn new(ctx: &RenderContext) -> Self {
        let vertex_shader = beuk::shaders::Shader::from_source_text(
            &ctx.device,
            r#"
            #version 450
            #extension GL_ARB_separate_shader_objects : enable
            #extension GL_ARB_shading_language_420pack : enable
            
            layout (location = 0) in vec2 pos;
            layout (location = 1) in vec2 uv;
            
            layout (location = 0) out vec2 o_uv;
            
            void main() {
                o_uv = uv;
                gl_Position = vec4(pos, 0.0, 1.0);
            }
            "#,
            "shader.vert",
            beuk::shaders::ShaderKind::Vertex,
            "main",
        );

        let fragment_shader = beuk::shaders::Shader::from_source_text(
            &ctx.device,
            r#"
            #version 450
            #extension GL_ARB_separate_shader_objects : enable
            #extension GL_ARB_shading_language_420pack : enable

            layout (set = 0, binding = 0) uniform sampler2D ui_texture;
            layout (set = 0, binding = 1) uniform sampler2D media_texture;

            layout (location = 0) in vec2 o_uv;
            layout (location = 0) out vec4 a_frag_color;

            void main() {
                vec4 ui_data = texture(ui_texture, o_uv);
                if (ui_data.a == 0.0) {
                    a_frag_color = texture(media_texture, o_uv);
                } else {
                    a_frag_color = ui_data;
                }
            }
            "#,
            "shader.frag",
            beuk::shaders::ShaderKind::Fragment,
            "main",
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
                        format: vk::Format::R32G32_SFLOAT,
                        offset: bytemuck::offset_of!(PresentVertex, pos) as u32,
                    },
                    vk::VertexInputAttributeDescription {
                        location: 1,
                        binding: 0,
                        format: vk::Format::R32G32_SFLOAT,
                        offset: bytemuck::offset_of!(PresentVertex, uv) as u32,
                    },
                ])
                .vertex_binding_descriptions(&[vk::VertexInputBindingDescription {
                    binding: 0,
                    stride: std::mem::size_of::<PresentVertex>() as u32,
                    input_rate: vk::VertexInputRate::VERTEX,
                }]),
            color_attachment_formats: &[swapchain.surface_format.format],
            depth_attachment_format: vk::Format::UNDEFINED,
            viewport: None,
            primitive: PrimitiveState {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            },
            depth_stencil: Default::default(),
            push_constant_range: None,
            blend: Default::default(),
            multisample: Default::default(),
        });

        let vertex_buffer = ctx.create_buffer_with_data(
            &BufferDescriptor {
                debug_name: "vertices",
                size: std::mem::size_of::<[PresentVertex; 6]>() as vk::DeviceSize,
                usage: BufferUsageFlags::VERTEX_BUFFER,
                location: MemoryLocation::GpuOnly,
            },
            bytemuck::cast_slice(&[
                PresentVertex {
                    pos: [-1.0, -1.0],
                    uv: [0.0, 0.0],
                },
                PresentVertex {
                    pos: [1.0, -1.0],
                    uv: [1.0, 0.0],
                },
                PresentVertex {
                    pos: [1.0, 1.0],
                    uv: [1.0, 1.0],
                },
                PresentVertex {
                    pos: [-1.0, -1.0],
                    uv: [0.0, 0.0],
                },
                PresentVertex {
                    pos: [1.0, 1.0],
                    uv: [1.0, 1.0],
                },
                PresentVertex {
                    pos: [-1.0, 1.0],
                    uv: [0.0, 1.0],
                },
            ]),
            0,
        );

        let index_buffer = ctx.create_buffer_with_data(
            &BufferDescriptor {
                debug_name: "indices",
                size: std::mem::size_of::<[u32; 6]>() as vk::DeviceSize,
                location: MemoryLocation::GpuOnly,
                usage: BufferUsageFlags::INDEX_BUFFER,
            },
            bytemuck::cast_slice(&[0u32, 1, 2, 3, 4, 5]),
            0,
        );

        Self {
            pipeline_handle,
            index_buffer,
            vertex_buffer,
        }
    }

    pub fn combine_and_draw(
        &self,
        ctx: &RenderContext,
        pass_one: &ResourceHandle<Texture>,
        pass_two: &ResourceHandle<Texture>,
    ) {
        let present_index = ctx.acquire_present_index();
        let pipeline = ctx.graphics_pipelines.get(&self.pipeline_handle).unwrap();
        unsafe {
            ctx.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(pipeline.descriptor_sets[0])
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(std::slice::from_ref(
                            &vk::DescriptorImageInfo::default()
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .image_view(*ctx.get_texture_view(pass_one).unwrap())
                                .sampler(
                                    *ctx.immutable_samplers
                                        .get(&SamplerDesc {
                                            address_modes: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                                            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                                            texel_filter: vk::Filter::LINEAR,
                                        })
                                        .unwrap(),
                                ),
                        )),
                    vk::WriteDescriptorSet::default()
                        .dst_set(pipeline.descriptor_sets[0])
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(std::slice::from_ref(
                            &vk::DescriptorImageInfo::default()
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                .image_view(*ctx.get_texture_view(pass_two).unwrap())
                                .sampler(
                                    *ctx.immutable_samplers
                                        .get(&SamplerDesc {
                                            address_modes: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                                            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                                            texel_filter: vk::Filter::LINEAR,
                                        })
                                        .unwrap(),
                                ),
                        )),
                ],
                &[],
            );
        };
        ctx.present_record(
            present_index,
            |ctx, command_buffer, color_view, _depth_view| unsafe {
                let color_attachments = &[vk::RenderingAttachmentInfo::default()
                    .image_view(color_view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    })];

                ctx.begin_rendering(command_buffer, color_attachments, None);

                pipeline.bind(&ctx.device, command_buffer);
                ctx.device.cmd_bind_vertex_buffers(
                    command_buffer,
                    0,
                    std::slice::from_ref(
                        &ctx.buffer_manager.get(&self.vertex_buffer).unwrap().buffer,
                    ),
                    &[0],
                );
                ctx.device.cmd_bind_index_buffer(
                    command_buffer,
                    ctx.buffer_manager.get(&self.index_buffer).unwrap().buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                ctx.device.cmd_draw_indexed(command_buffer, 6, 1, 0, 0, 1);
                ctx.end_rendering(command_buffer);
            },
        );
        ctx.present_submit(present_index);
    }
}
