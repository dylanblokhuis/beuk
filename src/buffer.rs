use ash::vk::{self, BufferCreateInfo};
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, Allocator},
    MemoryLocation,
};
use std::sync::Arc;

use crate::memory2::ResourceCleanup;

#[derive(Debug, Default)]
pub struct Buffer {
    pub buffer: vk::Buffer,
    pub size: u64,
    pub device_addr: u64,
    pub has_been_written_to: bool,
    pub offset: u64,
    pub allocation: Option<Allocation>,
}

impl ResourceCleanup for Buffer {
    fn cleanup(
        &mut self,
        device: Arc<ash::Device>,
        allocator: std::sync::Arc<std::sync::Mutex<Allocator>>,
    ) {
        self.destroy(&device, &mut allocator.lock().unwrap());
    }
}

impl Buffer {
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        debug_name: &str,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Buffer {
        let size = size;
        let mut usage = usage;
        if !usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            usage |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        }

        let buffer = unsafe {
            device.create_buffer(
                &BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .unwrap();
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: debug_name,
                requirements,
                location,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        let offset = allocation.offset();
        let device_addr: u64;
        unsafe {
            device
                .bind_buffer_memory(buffer, allocation.memory(), offset)
                .unwrap();

            device_addr = device.get_buffer_device_address(&vk::BufferDeviceAddressInfo {
                buffer,
                s_type: vk::StructureType::BUFFER_DEVICE_ADDRESS_INFO,
                p_next: std::ptr::null(),
                ..Default::default()
            });
        };

        Self {
            buffer,
            allocation: Some(allocation),
            size,
            device_addr,
            has_been_written_to: false,
            offset,
        }
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        let Some(allocation) = self.allocation.take() else {
            return;
        };
        allocator.free(allocation).unwrap();
        unsafe { device.destroy_buffer(self.buffer, None) };
    }

    pub fn copy_from_slice<T>(&mut self, slice: &[T], offset: usize)
    where
        T: Copy,
    {
        let Some(allocation) = self.allocation.as_ref() else {
            panic!("Tried writing to buffer but buffer not allocated");
        };

        unsafe {
            let ptr = allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
            let mem_ptr = ptr.add(offset);
            let mapped_slice = std::slice::from_raw_parts_mut(mem_ptr as *mut T, slice.len());
            mapped_slice.copy_from_slice(slice);
        }
        self.has_been_written_to = true;
    }

    #[inline]
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }
}
