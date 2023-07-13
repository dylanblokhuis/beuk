use std::collections::{BTreeMap, HashMap};

use ash::vk;

use crate::{buffer::Buffer, texture::Texture};

pub type MemoryLocation = gpu_allocator::MemoryLocation;

/// the handle is also the device address of the buffer
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct BufferHandle(u64);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TextureHandle(usize);

pub struct BufferManager {
    device: ash::Device,
    allocator: gpu_allocator::vulkan::Allocator,
    buffers: HashMap<BufferHandle, Buffer>,
}

impl BufferManager {
    pub fn new(device: ash::Device, allocator: gpu_allocator::vulkan::Allocator) -> Self {
        Self {
            device,
            allocator,
            buffers: HashMap::new(),
        }
    }

    pub fn create_buffer(
        &mut self,
        debug_name: &str,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> (BufferHandle, &Buffer) {
        let buffer = Buffer::new(
            &self.device,
            &mut self.allocator,
            debug_name,
            size,
            usage,
            location,
        );
        let handle = BufferHandle(buffer.device_addr);
        self.buffers.insert(handle, buffer);
        (handle, self.buffers.get(&handle).unwrap())
    }

    pub fn get_buffer(&self, handle: BufferHandle) -> &Buffer {
        self.buffers.get(&handle).unwrap()
    }

    pub fn get_buffer_mut(&mut self, handle: BufferHandle) -> &mut Buffer {
        self.buffers.get_mut(&handle).unwrap()
    }

    pub fn remove_buffer(&mut self, handle: BufferHandle) {
        {
            let buffer = self.buffers.get_mut(&handle).unwrap();

            buffer.destroy(&self.device, &mut self.allocator);
        }

        self.buffers.remove(&handle);
    }
}

impl Drop for BufferManager {
    fn drop(&mut self) {
        for (_, buffer) in self.buffers.iter_mut() {
            buffer.destroy(&self.device, &mut self.allocator);
        }
    }
}

// Textures

pub struct TextureManager {
    device: ash::Device,
    allocator: gpu_allocator::vulkan::Allocator,
    textures: BTreeMap<TextureHandle, Texture>,
}

impl TextureManager {
    pub fn new(device: ash::Device, allocator: gpu_allocator::vulkan::Allocator) -> Self {
        Self {
            device,
            allocator,
            textures: BTreeMap::new(),
        }
    }

    pub fn create_texture(
        &mut self,
        debug_name: &str,
        image_info: &vk::ImageCreateInfo,
    ) -> (TextureHandle, &Texture) {
        let buffer = Texture::new(&self.device, &mut self.allocator, debug_name, image_info);
        let handle = TextureHandle(self.textures.len());
        self.textures.insert(handle, buffer);
        (handle, self.textures.get(&handle).unwrap())
    }

    pub fn get_buffer(&self, handle: TextureHandle) -> &Texture {
        self.textures.get(&handle).unwrap()
    }

    pub fn get_buffer_mut(&mut self, handle: TextureHandle) -> &mut Texture {
        self.textures.get_mut(&handle).unwrap()
    }

    pub fn remove_buffer(&mut self, handle: TextureHandle) {
        {
            let buffer = self.textures.get_mut(&handle).unwrap();

            buffer.destroy(&self.device, &mut self.allocator);
        }

        self.textures.remove(&handle);
    }
}

impl Drop for TextureManager {
    fn drop(&mut self) {
        for (_, texture) in self.textures.iter_mut() {
            texture.destroy(&self.device, &mut self.allocator);
        }
    }
}
