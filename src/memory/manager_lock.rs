use std::any::type_name;
use std::{
    collections::VecDeque,
    fmt::{Debug, Formatter},
    hash::Hash,
    sync::{Arc, Mutex, MutexGuard},
};

use anyhow::{anyhow, Result};
use ash::vk::Extent2D;
use gpu_allocator::vulkan::Allocator;

use crate::ctx::RenderContext;

pub type Generation = usize;
pub type Index = usize;

#[derive(Debug)]
pub struct Resource<T> {
    pub(crate) inner: T,
    label: &'static str,
    generation: Generation,
    retain_count: usize,
}

impl<T> std::ops::Deref for Resource<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> std::ops::DerefMut for Resource<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[derive(Debug)]
struct ResourceInner<T>(Mutex<Resource<T>>);

/// Trait for resources that need to be cleaned up when they are destroyed.
pub trait ResourceHooks {
    fn cleanup(&mut self, device: Arc<ash::Device>, allocator: Arc<Mutex<Allocator>>);
    fn on_swapchain_resize(
        &mut self,
        _ctx: &RenderContext,
        _old_surface_resolution: Extent2D,
        _new_surface_resolution: Extent2D,
    ) {
    }
}

#[derive(Debug)]
pub struct ResourceHandle<T: Default + Debug + ResourceHooks> {
    id: ResourceId,
    manager: Arc<ResourceManager<T>>,
}

impl<T: Default + Debug + ResourceHooks> PartialEq for ResourceHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T: Default + Debug + ResourceHooks> Eq for ResourceHandle<T> {}

impl<T: Default + Debug + ResourceHooks> Hash for ResourceHandle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T: Default + Debug + ResourceHooks> ResourceHandle<T> {
    pub fn new(id: ResourceId, manager: Arc<ResourceManager<T>>) -> Self {
        Self { id, manager }
    }

    #[inline]
    pub fn id(&self) -> ResourceId {
        self.id
    }

    #[inline]
    pub fn get(&self) -> MutexGuard<Resource<T>> {
        self.manager.get(self).unwrap()
    }

    #[inline]
    pub fn get_mut(&self) -> MutexGuard<Resource<T>> {
        self.manager.get_mut(self).unwrap()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResourceId {
    pub index: Index,
    pub(super) generation: Generation,
}

impl<T: Default + Debug + ResourceHooks> Clone for ResourceHandle<T> {
    fn clone(&self) -> Self {
        let mut resource = self.manager.resources[self.id.index].0.lock().unwrap();
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

impl<T: Default + Debug + ResourceHooks> Drop for ResourceHandle<T> {
    fn drop(&mut self) {
        self.manager.destroy(self.id).unwrap();
    }
}

pub struct ResourceManager<T> {
    resources: Arc<boxcar::Vec<ResourceInner<T>>>,
    pub free_indices: Arc<Mutex<VecDeque<usize>>>,
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

impl<T: Default + Debug + ResourceHooks> ResourceManager<T> {
    pub fn new(
        device: Arc<ash::Device>,
        allocator: Arc<Mutex<Allocator>>,
        capacity: usize,
    ) -> Self {
        let mut queue = VecDeque::with_capacity(capacity);
        let resources = Arc::new(boxcar::Vec::with_capacity(capacity));
        for i in 0..capacity {
            queue.push_back(i);
            resources.push(ResourceInner(Mutex::new(Resource {
                label: Default::default(),
                inner: Default::default(),
                generation: 0,
                retain_count: 0,
            })));
        }

        Self {
            resources,
            free_indices: Arc::new(Mutex::new(queue)),
            device,
            allocator,
        }
    }

    #[tracing::instrument(skip(self, handle))]
    pub fn get(&self, handle: &ResourceHandle<T>) -> Option<MutexGuard<Resource<T>>> {
        let ResourceId { index, generation } = handle.id();
        let resource = self.resources.get(index);
        let Some(lock) = resource else {
            return None;
        };

        let data = lock.0.lock().unwrap();
        if generation != data.generation {
            return None;
        }

        Some(data)
    }

    #[tracing::instrument(skip(self, handle))]
    pub fn get_mut(&self, handle: &ResourceHandle<T>) -> Option<MutexGuard<Resource<T>>> {
        let ResourceId { index, generation } = handle.id();
        let resource = self.resources.get(index);
        let Some(lock) = resource else {
            return None;
        };

        let data = lock.0.lock().unwrap();
        if generation != data.generation {
            return None;
        }

        Some(data)
    }

    #[tracing::instrument(skip_all, name = "create")]
    pub fn create(&self, label: &'static str, resource: T) -> ResourceId {
        let index = self.get_index();

        log::debug!(
            "Creating {:?} | {:?} at index {} ",
            if label.is_empty() { "Unnamed" } else { label },
            type_name::<T>(),
            index,
        );

        let new_generation = {
            let old_resource = self.resources[index].0.lock().unwrap();
            assert_eq!(
                old_resource.retain_count, 0,
                "Resource at index {} retain count must be 0 when creating a new resource",
                index
            );
            old_resource.generation + 1
        };

        *self.resources[index].0.lock().unwrap() = Resource {
            inner: resource,
            label,
            generation: new_generation,
            retain_count: 1,
        };

        ResourceId {
            index,
            generation: new_generation,
        }
    }

    #[tracing::instrument(skip(self, handle))]
    pub fn destroy(&self, handle: ResourceId) -> Result<()> {
        let resource = self.resources.get(handle.index);
        let Some(lock) = resource else {
            return Err(anyhow!("No resource at index {:?}", handle.index));
        };

        let mut data = lock.0.lock().unwrap();

        if handle.generation != data.generation {
            // its OK to destroy a handle thats no longer valid, since it has already been destroyed.
            return Ok(());
        }
        data.retain_count -= 1;
        if data.retain_count > 0 {
            return Ok(());
        }

        log::debug!(
            "Destroying {:?} | {:?} at index {} ",
            if data.label.is_empty() {
                "Unnamed"
            } else {
                data.label
            },
            type_name::<T>(),
            handle.index,
        );
        data.inner
            .cleanup(self.device.clone(), self.allocator.clone());
        data.inner = Default::default();
        data.label = Default::default();

        // push to front to keep memory compact
        self.free_indices.lock().unwrap().push_front(handle.index);
        Ok(())
    }

    pub fn get_index(&self) -> usize {
        // Attempt to pop a free index.
        let mut free_indices_guard = self.free_indices.lock().unwrap();
        if let Some(index) = free_indices_guard.pop_front() {
            index
        } else {
            let current_capacity = self.resources.count();
            let new_capacity = current_capacity * 2; // Double the capacity. You can choose any other growth factor.
            self.resources.reserve(current_capacity);
            free_indices_guard.reserve(current_capacity);

            log::debug!(
                "Growing resource manager capacity from {} to {}",
                current_capacity,
                new_capacity
            );

            // Populate the resources and free_indices with the new capacity.
            for i in current_capacity..new_capacity {
                self.resources.push(ResourceInner(Mutex::new(Resource {
                    label: Default::default(),
                    inner: Default::default(),
                    generation: 0,
                    retain_count: 0,
                })));
                free_indices_guard.push_back(i);
            }

            // Reserve the new capacity for the resources.
            // At this point, we should be able to pop a new free index.
            free_indices_guard
                .pop_front()
                .expect("Failed to get a new free index after expanding capacity.")
        }
    }

    pub fn clear_all(&self) {
        for (_, resource) in self.resources.iter() {
            let mut data = resource.0.lock().unwrap();

            // data.inner
            //     .cleanup(self.device.clone(), self.allocator.clone());
            data.inner = Default::default();
        }
    }

    /// Call the swapchain resize hooks for all resources.
    #[tracing::instrument(skip(self, ctx))]
    pub fn call_swapchain_resize_hooks(
        &self,
        ctx: &RenderContext,
        old_surface_resolution: Extent2D,
        new_surface_resolution: Extent2D,
    ) {
        for (_, resource) in self.resources.iter() {
            let mut data = resource.0.lock().unwrap();
            data.inner
                .on_swapchain_resize(ctx, old_surface_resolution, new_surface_resolution);
        }
    }
}
