use beuk::ctx::RenderContext;
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

    let mut ctx =
        beuk::ctx::RenderContext::new(window.raw_display_handle(), window.raw_window_handle());

    let canvas = Canvas::new();

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

struct Canvas {}

impl Canvas {
    fn new() -> Self {
        Self {}
    }

    pub fn draw(&self, ctx: &mut RenderContext) {
        ctx.record(
            ctx.draw_command_buffer,
            Some(ctx.draw_commands_reuse_fence),
            |ctx, command_buffer| {
                ctx.begin_rendering(command_buffer);

                ctx.end_rendering(command_buffer);
            },
        );

        ctx.present_submit(ctx.draw_command_buffer, ctx.draw_commands_reuse_fence);
    }
}
