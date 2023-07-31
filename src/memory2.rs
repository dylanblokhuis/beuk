use std::{
    cell::UnsafeCell,
    fmt::{Debug, Formatter},
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, Result};
use crossbeam_queue::ArrayQueue;

use crate::{buffer::Buffer, memory::BufferDescriptor};

pub type Generation = usize;
pub type Index = usize;

struct Resource {
    buffer: Buffer,
    generation: Generation,
    retain_count: usize,
}

#[derive(Debug)]
struct ResourceInner(UnsafeCell<Resource>);
unsafe impl Send for ResourceInner {}
unsafe impl Sync for ResourceInner {}

#[derive(Debug)]
pub struct BufferHandle {
    id: BufferId,
    manager: Arc<ResourceManager>,
}

impl BufferHandle {
    pub fn new(id: BufferId, manager: Arc<ResourceManager>) -> Self {
        Self { id, manager }
    }

    #[inline]
    pub fn id(&self) -> BufferId {
        self.id
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BufferId {
    index: Index,
    generation: Generation,
}

impl Clone for BufferHandle {
    fn clone(&self) -> Self {
        let resource = unsafe { &mut *self.manager.resources[self.id.index].0.get() };
        if self.id.generation != resource.generation {
            panic!("Buffer generation mismatch, cannot clone");
        }
        resource.retain_count += 1;
        Self {
            id: self.id,
            manager: self.manager.clone(),
        }
    }
}

impl Drop for BufferHandle {
    fn drop(&mut self) {
        self.manager.destroy(self.id).unwrap();
    }
}

pub struct ResourceManager {
    resources: Arc<boxcar::Vec<ResourceInner>>,
    free_indices: ArrayQueue<Index>,
    device: Arc<ash::Device>,
    allocator: Arc<Mutex<gpu_allocator::vulkan::Allocator>>,
}

unsafe impl Send for ResourceManager {}
unsafe impl Sync for ResourceManager {}

impl Debug for ResourceManager {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceManager")
            .field("resources", &self.resources)
            .field("free_indices", &self.free_indices)
            .field("allocator", &self.allocator)
            .finish()
    }
}

impl ResourceManager {
    pub fn new(device: Arc<ash::Device>, allocator: gpu_allocator::vulkan::Allocator) -> Self {
        let queue = ArrayQueue::new(1024);
        let resources = Arc::new(boxcar::Vec::with_capacity(1024));
        for i in 0..1024 {
            queue.push(i).unwrap();
            resources.push(ResourceInner(UnsafeCell::new(Resource {
                buffer: Buffer::default(),
                generation: 0,
                retain_count: 0,
            })));
        }

        Self {
            device,
            resources,
            free_indices: queue,
            allocator: Arc::new(Mutex::new(allocator)),
        }
    }

    #[tracing::instrument(skip(self))]
    pub fn get(&self, handle: BufferId) -> Option<&Buffer> {
        let resource = self.resources.get(handle.index);
        let Some(lock) = resource else {
            return None;
        };

        let data = unsafe { &*lock.0.get() };
        if handle.generation != data.generation {
            return None;
        }

        Some(&data.buffer)
    }

    #[tracing::instrument(skip(self))]
    pub fn get_mut(&self, handle: BufferId) -> Option<&mut Buffer> {
        let resource = self.resources.get(handle.index);
        let Some(lock) = resource else {
            return None;
        };

        let data = unsafe { &mut *lock.0.get() };
        if handle.generation != data.generation {
            return None;
        }

        Some(&mut data.buffer)
    }

    #[tracing::instrument(skip(self))]
    pub fn create(&self, desc: &BufferDescriptor) -> BufferId {
        let Some(index) = self.free_indices.pop() else {
            panic!("No more free indices");
        };

        println!("creating {:?}", index);

        let buffer = Buffer::new(
            &self.device,
            &mut self.allocator.lock().unwrap(),
            desc.debug_name,
            desc.size,
            desc.usage,
            desc.location,
        );
        let old_generation = unsafe { &*self.resources[index].0.get() }.generation;
        let new_generation = old_generation + 1;

        unsafe {
            *self.resources[index].0.get() = Resource {
                buffer,
                generation: new_generation,
                retain_count: 1,
            }
        }

        BufferId {
            index,
            generation: new_generation,
        }
    }

    #[tracing::instrument(skip(self))]
    pub fn destroy(&self, handle: BufferId) -> Result<()> {
        let resource = self.resources.get(handle.index);
        let Some(cell) = resource else {
            return Err(anyhow!("No resource at index {:?}", handle.index));
        };

        let data = unsafe { &mut *cell.0.get() };

        if handle.generation != data.generation {
            return Err(anyhow!("Generation mismatch"));
        }
        data.retain_count -= 1;
        if data.retain_count > 0 {
            return Ok(());
        }

        data.buffer
            .destroy(&self.device, &mut self.allocator.lock().unwrap());

        let _ = self.free_indices.force_push(handle.index);
        Ok(())
    }
}
