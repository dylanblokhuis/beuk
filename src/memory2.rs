use std::{
    cell::UnsafeCell,
    fmt::{Debug, Formatter},
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, Result};
use crossbeam_queue::ArrayQueue;
use gpu_allocator::vulkan::Allocator;

pub type Generation = usize;
pub type Index = usize;

struct Resource<T> {
    inner: T,
    generation: Generation,
    retain_count: usize,
}

#[derive(Debug)]
struct ResourceInner<T>(UnsafeCell<Resource<T>>);
unsafe impl<T> Send for ResourceInner<T> {}
unsafe impl<T> Sync for ResourceInner<T> {}

/// Trait for resources that need to be cleaned up when they are destroyed.
pub trait ResourceCleanup {
    fn cleanup(&mut self, device: Arc<ash::Device>, allocator: Arc<Mutex<Allocator>>);
}

#[derive(Debug)]
pub struct ResourceHandle<T: Default + Debug + ResourceCleanup> {
    id: ResourceId,
    manager: Arc<ResourceManager<T>>,
}

impl<T: Default + Debug + ResourceCleanup> ResourceHandle<T> {
    pub fn new(id: ResourceId, manager: Arc<ResourceManager<T>>) -> Self {
        Self { id, manager }
    }

    #[inline]
    pub fn id(&self) -> ResourceId {
        self.id
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ResourceId {
    index: Index,
    generation: Generation,
}

impl<T: Default + Debug + ResourceCleanup> Clone for ResourceHandle<T> {
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

impl<T: Default + Debug + ResourceCleanup> Drop for ResourceHandle<T> {
    fn drop(&mut self) {
        self.manager.destroy(self.id).unwrap();
    }
}

pub struct ResourceManager<T> {
    resources: Arc<boxcar::Vec<ResourceInner<T>>>,
    free_indices: ArrayQueue<Index>,
    device: Arc<ash::Device>,
    allocator: Arc<Mutex<Allocator>>,
}

impl<T: Debug> Debug for ResourceManager<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceManager")
            .field("resources", &self.resources)
            .field("free_indices", &self.free_indices)
            .finish()
    }
}

impl<T: Default + Debug + ResourceCleanup> ResourceManager<T> {
    pub fn new(device: Arc<ash::Device>, allocator: Arc<Mutex<Allocator>>) -> Self {
        let queue = ArrayQueue::new(1024);
        let resources = Arc::new(boxcar::Vec::with_capacity(1024));
        for i in 0..1024 {
            queue.push(i).unwrap();
            resources.push(ResourceInner(UnsafeCell::new(Resource {
                inner: Default::default(),
                generation: 0,
                retain_count: 0,
            })));
        }

        Self {
            resources,
            free_indices: queue,
            device,
            allocator,
        }
    }

    #[tracing::instrument(skip(self))]
    pub fn get(&self, handle: ResourceId) -> Option<&T> {
        let resource = self.resources.get(handle.index);
        let Some(lock) = resource else {
            return None;
        };

        let data = unsafe { &*lock.0.get() };
        if handle.generation != data.generation {
            return None;
        }

        Some(&data.inner)
    }

    #[tracing::instrument(skip(self))]
    pub fn get_mut(&self, handle: ResourceId) -> Option<&mut T> {
        let resource = self.resources.get(handle.index);
        let Some(lock) = resource else {
            return None;
        };

        let data = unsafe { &mut *lock.0.get() };
        if handle.generation != data.generation {
            return None;
        }

        Some(&mut data.inner)
    }

    #[tracing::instrument(skip(self))]
    pub fn create(&self, resource: T) -> ResourceId {
        let Some(index) = self.free_indices.pop() else {
            panic!("No more free indices");
        };

        println!("creating {:?}", index);

        let old_generation = unsafe { &*self.resources[index].0.get() }.generation;
        let new_generation = old_generation + 1;

        unsafe {
            *self.resources[index].0.get() = Resource {
                inner: resource,
                generation: new_generation,
                retain_count: 1,
            }
        }

        ResourceId {
            index,
            generation: new_generation,
        }
    }

    #[tracing::instrument(skip(self))]
    pub fn destroy(&self, handle: ResourceId) -> Result<()> {
        let resource = self.resources.get(handle.index);
        let Some(cell) = resource else {
            return Err(anyhow!("No resource at index {:?}", handle.index));
        };
        let data = unsafe { &mut *cell.0.get() };

        if handle.generation != data.generation {
            // its OK to destroy a handle thats no longer valid, since it has already been destroyed.
            return Ok(());
        }
        data.retain_count -= 1;
        if data.retain_count > 0 {
            return Ok(());
        }

        data.inner
            .cleanup(self.device.clone(), self.allocator.clone());

        let _ = self.free_indices.force_push(handle.index);
        Ok(())
    }

    pub fn clear_all(&self) {
        for (_, resource) in self.resources.iter() {
            let data = unsafe { &mut *resource.0.get() };
            data.inner
                .cleanup(self.device.clone(), self.allocator.clone());
            data.inner = Default::default();
        }
    }
}