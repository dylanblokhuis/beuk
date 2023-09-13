/**
 * Tests require to run on the main thread so we create this test in a separate folder and disabling the rust test harness
 */
use std::{
    mem::size_of,
    sync::{Arc, Mutex},
};

use ash::vk::{Buffer, BufferUsageFlags, Extent2D};
use beuk::{
    buffer::{BufferDescriptor, MemoryLocation},
    ctx::RenderContext,
    memory::{ResourceHandle, ResourceHooks, ResourceManager},
};
use gpu_allocator::vulkan::Allocator;
use simple_logger::SimpleLogger;
use winit::event_loop::ControlFlow;

fn main() {
    test_capacity_growing();
}

#[repr(C, align(16))]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Test {
    yo: u32,
    _pad: [u8; 12],
}

impl ResourceHooks for Test {
    fn cleanup(&mut self, _device: Arc<ash::Device>, _allocator: Arc<Mutex<Allocator>>) {}
    fn on_swapchain_resize(
        &mut self,
        _ctx: &RenderContext,
        _old_surface_resolution: Extent2D,
        _new_surface_resolution: Extent2D,
    ) {
    }
}

fn test_capacity_growing() {
    use beuk::ctx::RenderContextDescriptor;
    use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

    SimpleLogger::new().init().unwrap();

    let event_loop = winit::event_loop::EventLoopBuilder::<()>::with_user_event().build();
    let window = winit::window::WindowBuilder::new()
        .with_title("Test")
        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0))
        .build(&event_loop)
        .unwrap();

    let ctx = Arc::new(RenderContext::new(RenderContextDescriptor {
        display_handle: window.raw_display_handle(),
        present_mode: Default::default(),
        window_handle: window.raw_window_handle(),
    }));

    std::thread::spawn({
        let ctx = ctx.clone();
        let proxy = event_loop.create_proxy();
        move || {
            test_capacity_with_vulkan_buffers(ctx.clone());
            test_capacity(ctx.clone());

            // requests a close
            proxy.send_event(()).unwrap();
        }
    });

    // let mut got_any_event = false;
    event_loop.run(move |event, _, control_flow| match event {
        winit::event::Event::WindowEvent {
            event: winit::event::WindowEvent::CloseRequested,
            ..
        } => *control_flow = winit::event_loop::ControlFlow::Exit,
        winit::event::Event::UserEvent(_) => {
            *control_flow = winit::event_loop::ControlFlow::Exit;
        }
        _ => {
            ControlFlow::WaitUntil(
                std::time::Instant::now() + std::time::Duration::from_millis(100),
            );
            // if handle.is_finished() {
            //   *control_flow = winit::event_loop::ControlFlow::Exit;
            //   assert!(got_any_event, "Didn't get any event");
            // }
        }
    });
}

fn test_capacity_with_vulkan_buffers(ctx: Arc<RenderContext>) {
    let mut handles = vec![];
    for i in 0..2000 {
        let handle = ctx.create_buffer_with_data(
            &BufferDescriptor {
                location: MemoryLocation::CpuToGpu,
                usage: BufferUsageFlags::UNIFORM_BUFFER,
                size: size_of::<Test>() as u64,
                ..Default::default()
            },
            &bytemuck::cast_slice(&[Test {
                yo: i,
                _pad: [0; 12],
            }]),
            0,
        );

        handles.push(handle.clone());
    }

    for i in 0..2000 {
        let buffer = ctx.buffer_manager.get(&handles[i]).unwrap();
        let resource: &Test = buffer.cast();
        assert_eq!(resource.yo, i as u32, "Resource {} has wrong value", i);
    }
}

fn test_capacity(ctx: Arc<RenderContext>) {
    let resource_manager = Arc::new(ResourceManager::<Test>::new(
        ctx.device.clone(),
        ctx.allocator.clone(),
        100,
    ));

    let mut handles = vec![];
    for i in 0..20000 {
        let id = resource_manager.create(Test {
            yo: i,
            _pad: [0; 12],
        });
        handles.push(ResourceHandle::new(id, resource_manager.clone()));
    }

    for i in 0..20000 {
        let resource = resource_manager.get(&handles[i]).unwrap();
        assert_eq!(resource.yo, i as u32, "Resource {} has wrong value", i);
    }
}
