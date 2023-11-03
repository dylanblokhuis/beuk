use std::{
    sync::{Arc, Mutex},
};

use ash::vk::{Extent2D};
use beuk::{
    ctx::{RenderContext, RenderContextDescriptor},
    memory::{
        ResourceHandle, ResourceHooks, ResourceManager, UnsafeResourceHandle, UnsafeResourceManager,
    },
};
use gpu_allocator::vulkan::Allocator;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};


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

use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
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

    let resource_manager = Arc::new(ResourceManager::<Test>::new(
        ctx.device.clone(),
        ctx.allocator.clone(),
        100,
    ));

    c.bench_function("create", |b| {
        b.iter(|| create(black_box(resource_manager.clone())))
    });

    let resource_manager = Arc::new(ResourceManager::<Test>::new(
        ctx.device.clone(),
        ctx.allocator.clone(),
        100,
    ));
    c.bench_function("create and get", |b| {
        b.iter(|| create_and_get(black_box(resource_manager.clone())))
    });

    let resource_manager = Arc::new(UnsafeResourceManager::<Test>::new(
        ctx.device.clone(),
        ctx.allocator.clone(),
        100,
    ));
    c.bench_function("unsafe create", |b| {
        b.iter(|| unsafe_create(black_box(resource_manager.clone())))
    });

    let resource_manager = Arc::new(UnsafeResourceManager::<Test>::new(
        ctx.device.clone(),
        ctx.allocator.clone(),
        100,
    ));
    c.bench_function("unsafe create and get", |b| {
        b.iter(|| unsafe_create_and_get(black_box(resource_manager.clone())))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

fn create(manager: Arc<ResourceManager<Test>>) {
    let id = manager.create(
        "test",
        Test {
            yo: 0,
            _pad: [0; 12],
        },
    );
    let handle = ResourceHandle::new(id, manager.clone());
    drop(handle);
}

fn create_and_get(manager: Arc<ResourceManager<Test>>) {
    let id = manager.create(
        "test",
        Test {
            yo: 0,
            _pad: [0; 12],
        },
    );

    let handle = ResourceHandle::new(id, manager.clone());
    let _ = manager.get(&handle);
}

fn unsafe_create(manager: Arc<UnsafeResourceManager<Test>>) {
    let id = manager.create(Test {
        yo: 0,
        _pad: [0; 12],
    });
    let handle = UnsafeResourceHandle::new(id, manager.clone());
    drop(handle);
}

fn unsafe_create_and_get(manager: Arc<UnsafeResourceManager<Test>>) {
    let id = manager.create(Test {
        yo: 0,
        _pad: [0; 12],
    });

    let handle = UnsafeResourceHandle::new(id, manager.clone());
    let _ = manager.get(&handle);
}
