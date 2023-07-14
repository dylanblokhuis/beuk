use ash::vk::{self, BufferUsageFlags, PipelineVertexInputStateCreateInfo};
use beuk::{
    ctx::RenderContext,
    memory::{BufferHandle, PipelineHandle},
    pipeline::{GraphicsPipelineDescriptor, PrimitiveState},
    shaders::Shader,
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

    let mut ctx = beuk::ctx::RenderContext::new(
        window.raw_display_handle(),
        window.raw_window_handle(),
        |dc| {
            // example of enabling an extension
            // let names = unsafe {
            //     std::slice::from_raw_parts_mut(
            //         dc.pp_enabled_extension_names.cast_mut(),
            //         dc.enabled_extension_count as usize,
            //     )
            // };
            // names[dc.enabled_extension_count as usize - 1] =
            //     ash::vk::KhrSamplerYcbcrConversionFn::NAME.as_ptr();

            // dc.enabled_extension_names(&*names)

            dc
        },
    );

    let canvas = Canvas::new(&mut ctx);

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            window_id,
        } if window_id == window.id() => control_flow.set_exit(),
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            canvas.draw(&mut ctx);
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
    fn new(ctx: &mut RenderContext) -> Self {
        let vertex_shader = Shader::from_source_text(
            &ctx.device,
            include_str!("./triangle.vert"),
            "triangle.vert",
            beuk::shaders::ShaderKind::Vertex,
            "main",
        );

        let fragment_shader = Shader::from_source_text(
            &ctx.device,
            include_str!("./triangle.frag"),
            "triangle.frag",
            beuk::shaders::ShaderKind::Fragment,
            "main",
        );

        let vertex_buffer = ctx.buffer_manager.create_buffer_with_data(
            "vertices",
            bytemuck::cast_slice(&[
                Vertex {
                    pos: [-1.0, 1.0, 0.0, 1.0],
                    color: [0.0, 1.0, 0.0, 1.0],
                },
                Vertex {
                    pos: [1.0, 1.0, 0.0, 1.0],
                    color: [0.0, 0.0, 1.0, 1.0],
                },
                Vertex {
                    pos: [0.0, -1.0, 0.0, 1.0],
                    color: [1.0, 0.0, 0.0, 1.0],
                },
            ]),
            BufferUsageFlags::VERTEX_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );

        let index_buffer = ctx.buffer_manager.create_buffer_with_data(
            "indices",
            bytemuck::cast_slice(&[0u32, 1, 2]),
            BufferUsageFlags::INDEX_BUFFER,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );

        let pipeline_handle =
            ctx.pipeline_manager
                .create_graphics_pipeline(GraphicsPipelineDescriptor {
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
                    color_attachment_formats: &[ctx.render_swapchain.surface_format.format],
                    depth_attachment_format: ctx.render_swapchain.depth_image_format,
                    viewport: ctx.render_swapchain.surface_resolution,
                    primitive: PrimitiveState {
                        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                        ..Default::default()
                    },
                    depth_stencil: Default::default(),
                    push_constant_range: None,
                });
        Self {
            pipeline_handle,
            vertex_buffer,
            index_buffer,
        }
    }

    pub fn draw(&self, ctx: &mut RenderContext) {
        ctx.record(
            ctx.draw_command_buffer,
            Some(ctx.draw_commands_reuse_fence),
            |ctx, command_buffer| unsafe {
                ctx.begin_rendering(command_buffer);
                let pipeline = ctx
                    .pipeline_manager
                    .get_graphics_pipeline(&self.pipeline_handle);
                pipeline.bind(&ctx.device, command_buffer);

                ctx.device.cmd_bind_vertex_buffers(
                    command_buffer,
                    0,
                    std::slice::from_ref(&ctx.buffer_manager.get_buffer(self.vertex_buffer).buffer),
                    &[0],
                );
                ctx.device.cmd_bind_index_buffer(
                    command_buffer,
                    ctx.buffer_manager.get_buffer(self.index_buffer).buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                ctx.device.cmd_draw_indexed(command_buffer, 3, 1, 0, 0, 1);

                ctx.end_rendering(command_buffer);
            },
        );
        ctx.present_submit(ctx.draw_command_buffer, ctx.draw_commands_reuse_fence);
    }
}
