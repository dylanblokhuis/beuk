use ash::vk::{self, BufferCreateInfo};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};
use std::{ptr::NonNull, sync::Arc};

use crate::memory::ResourceHooks;

pub type MemoryLocation = gpu_allocator::MemoryLocation;

#[derive(Debug, Default)]
pub struct Buffer {
    pub debug_name: &'static str,
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size: u64,
    pub device_addr: u64,
    pub has_been_written_to: bool,
    pub offset: u64,
    pub access_mask: vk::AccessFlags,
    pub stage_mask: vk::PipelineStageFlags,
}

impl ResourceHooks for Buffer {
    fn cleanup(
        &mut self,
        device: Arc<ash::Device>,
        allocator: std::sync::Arc<std::sync::Mutex<Allocator>>,
    ) {
        self.destroy(&device, &mut allocator.lock().unwrap());
    }
}

#[derive(Debug, Clone)]
pub struct BufferDescriptor {
    pub debug_name: &'static str,
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
    /// GpuOnly will be a slower allocation, but it will be faster on the gpu
    pub location: MemoryLocation,
}

impl Default for BufferDescriptor {
    fn default() -> Self {
        Self {
            debug_name: "unnamed",
            size: 0,
            usage: vk::BufferUsageFlags::empty(),
            location: MemoryLocation::CpuToGpu,
        }
    }
}

impl Buffer {
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        debug_name: &'static str,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Buffer {
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

        log::debug!("Allocating buffer: {}", debug_name);
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
            debug_name,
            buffer,
            allocation: Some(allocation),
            size,
            device_addr,
            has_been_written_to: false,
            offset,
            access_mask: vk::AccessFlags::empty(),
            stage_mask: vk::PipelineStageFlags::empty(),
        }
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        if std::thread::panicking() {
            return;
        }

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

    pub fn cast<T>(&self) -> &T {
        let Some(allocation) = self.allocation.as_ref() else {
            panic!(
                "Tried reading from {} but buffer not allocated",
                self.debug_name
            );
        };

        let Some(ptr) = allocation.mapped_ptr() else {
            panic!(
                "Tried reading from {} but buffer not mapped",
                self.debug_name
            );
        };

        let non_null_t_ptr: NonNull<T> = unsafe { NonNull::new_unchecked(ptr.as_ptr() as *mut T) };
        unsafe { non_null_t_ptr.as_ref() }
    }

    pub fn cast_slice<T>(&self) -> &[T] {
        let Some(allocation) = self.allocation.as_ref() else {
            panic!("Tried reading from buffer but buffer not allocated");
        };

        let Some(ptr) = allocation.mapped_ptr() else {
            panic!("Tried reading from buffer but buffer not mapped");
        };

        let non_null_t_ptr: NonNull<T> = unsafe { NonNull::new_unchecked(ptr.as_ptr() as *mut T) };
        unsafe { std::slice::from_raw_parts(non_null_t_ptr.as_ptr(), self.size as usize) }
    }

    #[inline]
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn transition_memory(
        &mut self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        dst_access_mask: vk::AccessFlags,
        dst_stage_mask: vk::PipelineStageFlags,
    ) {
        let memory_barrier = vk::MemoryBarrier::default()
            .src_access_mask(self.access_mask)
            .dst_access_mask(dst_access_mask);
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                self.stage_mask,
                dst_stage_mask,
                vk::DependencyFlags::empty(),
                &[memory_barrier],
                &[],
                &[],
            );
        }
    }
}
