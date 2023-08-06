#[cfg(any(target_os = "macos", target_os = "ios"))]
use ash::vk::{
    KhrGetMemoryRequirements2Fn, KhrGetPhysicalDeviceProperties2Fn, KhrPortabilityEnumerationFn,
    KhrPortabilitySubsetFn,
};
use ash::{
    extensions::{
        ext::{BufferDeviceAddress, DebugUtils},
        khr::{DynamicRendering, Surface, Swapchain},
    },
    vk::{
        CommandBuffer, DebugUtilsMessageSeverityFlagsEXT, DeviceSize, ExtDescriptorIndexingFn,
        Fence, ImageView, KhrSamplerYcbcrConversionFn,
        PhysicalDeviceBufferDeviceAddressFeaturesKHR, PhysicalDeviceDescriptorIndexingFeatures,
        PhysicalDeviceSamplerYcbcrConversionFeatures, PresentModeKHR, SurfaceKHR, API_VERSION_1_2,
    },
};
use ash::{vk, Entry};
use ash::{Device, Instance};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use rayon::ThreadPool;
use std::{borrow::Cow, collections::HashMap, mem::size_of_val};
use std::{cell::RefCell, default::Default};
use std::{ffi::CStr, sync::Mutex};
use std::{ops::Drop, sync::RwLock};
use std::{os::raw::c_char, sync::Arc};

use crate::{
    buffer::{Buffer, BufferDescriptor, MemoryLocation},
    memory::{ResourceHandle, ResourceManager},
    pipeline::{GraphicsPipeline, GraphicsPipelineDescriptor},
    shaders::ImmutableShaderInfo,
    texture::{Texture, TransitionDesc},
};

pub const RESERVED_DESCRIPTOR_COUNT: u32 = 32;

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    match message_severity {
        DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            log::error!("{message}");
        }
        DebugUtilsMessageSeverityFlagsEXT::INFO => {
            log::info!("{message}");
        }
        DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            log::debug!("{message}");
        }
        DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            log::warn!("{message}");
        }
        _ => {
            log::info!("{message}");
        }
    }

    vk::FALSE
}

pub fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_req.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(index, _memory_type)| index as _)
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct SamplerDesc {
    pub texel_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_modes: vk::SamplerAddressMode,
}

#[derive(Clone)]
pub struct RenderSwapchain {
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_loader: Swapchain,
    pub present_mode: vk::PresentModeKHR,
    pub present_images: Vec<vk::Image>,
    pub present_image_views: Vec<vk::ImageView>,
    pub depth_image: vk::Image,
    pub depth_image_view: vk::ImageView,
    pub depth_image_memory: vk::DeviceMemory,
    pub depth_image_format: vk::Format,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_resolution: vk::Extent2D,
}

thread_local! {
    pub static COMMAND_POOL: RefCell<vk::CommandPool> = RefCell::new(vk::CommandPool::null());
}

pub struct RenderContext {
    pub buffer_manager: Arc<ResourceManager<Buffer>>,
    pub texture_manager: Arc<ResourceManager<Texture>>,
    pub graphics_pipelines: Arc<ResourceManager<GraphicsPipeline>>,
    pub immutable_samplers: Arc<HashMap<SamplerDesc, vk::Sampler>>,
    pub yuv_immutable_samplers:
        Arc<HashMap<(vk::Format, SamplerDesc), (vk::SamplerYcbcrConversion, vk::Sampler)>>,
    pub allocator: Arc<Mutex<Allocator>>,
    pub threaded_command_buffers: Arc<RwLock<HashMap<usize, (CommandBuffer, vk::Fence)>>>,
    pub render_swapchain: Arc<RwLock<RenderSwapchain>>,
    pub thread_id: std::thread::ThreadId,

    pub entry: Entry,
    pub instance: Instance,
    pub device: Device,
    pub dynamic_rendering: DynamicRendering,
    pub surface_loader: Surface,
    pub debug_utils_loader: DebugUtils,
    pub debug_call_back: vk::DebugUtilsMessengerEXT,
    pub max_descriptor_count: u32,
    pub command_thread_pool: Arc<ThreadPool>,
    pub pdevice: vk::PhysicalDevice,
    pub queue_family_index: u32,
    pub present_queue: Arc<Mutex<vk::Queue>>,

    pub surface: vk::SurfaceKHR,

    pub pool: vk::CommandPool,
    pub main_command_buffer: vk::CommandBuffer,
    pub main_commands_reuse_fence: vk::Fence,

    pub present_complete_semaphore: vk::Semaphore,
    pub rendering_complete_semaphore: vk::Semaphore,
}

#[derive(Clone, Copy)]
pub struct RenderContextDescriptor {
    pub display_handle: raw_window_handle::RawDisplayHandle,
    pub window_handle: raw_window_handle::RawWindowHandle,
    pub present_mode: PresentModeKHR,
}

impl RenderContext {
    pub fn new(desc: RenderContextDescriptor) -> Self {
        unsafe {
            let entry = Entry::linked();
            let app_name = CStr::from_bytes_with_nul_unchecked(b"beuk\0");

            let mut layer_names = vec![];

            if cfg!(debug_assertions) {
                layer_names.push(CStr::from_bytes_with_nul_unchecked(
                    b"VK_LAYER_KHRONOS_validation\0",
                ))
            }

            let layers_names_raw: Vec<*const c_char> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let mut extension_names =
                ash_window::enumerate_required_extensions(desc.display_handle)
                    .unwrap()
                    .to_vec();
            extension_names.push(DebugUtils::NAME.as_ptr());
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                extension_names.push(KhrPortabilityEnumerationFn::NAME.as_ptr());
                // Enabling this extension is a requirement when using `VK_KHR_portability_subset`
                extension_names.push(KhrGetPhysicalDeviceProperties2Fn::NAME.as_ptr());
            }

            let appinfo = vk::ApplicationInfo::default()
                .application_name(app_name)
                .application_version(0)
                .engine_name(app_name)
                .engine_version(0)
                .api_version(API_VERSION_1_2);

            let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
                vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
            } else {
                vk::InstanceCreateFlags::default()
            };

            let create_info = vk::InstanceCreateInfo::default()
                .application_info(&appinfo)
                .enabled_layer_names(&layers_names_raw)
                .enabled_extension_names(&extension_names)
                .flags(create_flags);

            let instance: Instance = entry
                .create_instance(&create_info, None)
                .expect("Instance creation error");

            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));

            let debug_utils_loader = DebugUtils::new(&entry, &instance);
            let debug_call_back = debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap();
            let surface = ash_window::create_surface(
                &entry,
                &instance,
                desc.display_handle,
                desc.window_handle,
                None,
            )
            .unwrap();
            let pdevices = instance
                .enumerate_physical_devices()
                .expect("Physical device error");
            let surface_loader = Surface::new(&entry, &instance);
            let (pdevice, queue_family_index) = pdevices
                .iter()
                .find_map(|pdevice| {
                    instance
                        .get_physical_device_queue_family_properties(*pdevice)
                        .iter()
                        .enumerate()
                        .find_map(|(index, info)| {
                            let supports_graphic_and_surface =
                                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                    && surface_loader
                                        .get_physical_device_surface_support(
                                            *pdevice,
                                            index as u32,
                                            surface,
                                        )
                                        .unwrap();
                            if supports_graphic_and_surface {
                                Some((*pdevice, index))
                            } else {
                                None
                            }
                        })
                })
                .expect("Couldn't find suitable device.");

            let device_properties = instance.get_physical_device_properties(pdevice);
            let queue_family_index = queue_family_index as u32;
            let device_extension_names_raw = [
                Swapchain::NAME.as_ptr(),
                DynamicRendering::NAME.as_ptr(),
                ExtDescriptorIndexingFn::NAME.as_ptr(),
                BufferDeviceAddress::NAME.as_ptr(),
                KhrSamplerYcbcrConversionFn::NAME.as_ptr(),
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                KhrPortabilitySubsetFn::NAME.as_ptr(),
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                KhrGetMemoryRequirements2Fn::NAME.as_ptr(),
            ];
            let features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: 1,
                sampler_anisotropy: 1,
                depth_clamp: 1,
                ..Default::default()
            };
            let mut dynamic_rendering_features =
                vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

            let mut buffer_features =
                PhysicalDeviceBufferDeviceAddressFeaturesKHR::default().buffer_device_address(true);

            let mut indexing_features = PhysicalDeviceDescriptorIndexingFeatures::default()
                .descriptor_binding_partially_bound(true)
                .runtime_descriptor_array(true)
                .shader_sampled_image_array_non_uniform_indexing(true)
                // .shader_uniform_buffer_array_non_uniform_indexing(true)
                // .shader_storage_buffer_array_non_uniform_indexing(true)
                // after bind
                .descriptor_binding_sampled_image_update_after_bind(true)
                .descriptor_binding_uniform_buffer_update_after_bind(true)
                .descriptor_binding_storage_buffer_update_after_bind(true)
                // dynamic indexing
                .shader_input_attachment_array_dynamic_indexing(true)
                .shader_storage_texel_buffer_array_dynamic_indexing(true)
                .shader_uniform_texel_buffer_array_dynamic_indexing(true);

            let mut yuv_features = PhysicalDeviceSamplerYcbcrConversionFeatures::default()
                .sampler_ycbcr_conversion(true);

            let queue_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&[1.0]);

            let device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(&device_extension_names_raw)
                .enabled_features(&features)
                .push_next(&mut dynamic_rendering_features)
                .push_next(&mut buffer_features)
                .push_next(&mut indexing_features)
                .push_next(&mut yuv_features);

            let device: Device = instance
                .create_device(pdevice, &device_create_info, None)
                .unwrap();

            let present_queue = device.get_device_queue(queue_family_index, 0);

            let pool_create_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);

            let pool = device.create_command_pool(&pool_create_info, None).unwrap();

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let command_buffers = device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap();
            let main_command_buffer = command_buffers[0];

            let render_swapchain = Self::create_swapchain(
                &instance,
                &device,
                pdevice,
                &surface_loader,
                surface,
                desc.present_mode,
            );

            let main_commands_reuse_fence = device
                .create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
                .expect("Create fence failed.");

            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            let present_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();
            let rendering_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();

            let (command_thread_pool, threaded_command_buffers) =
                Self::create_command_thread_pool(device.clone(), queue_family_index);

            let command_thread_pool = Arc::new(command_thread_pool);

            let dynamic_rendering = DynamicRendering::new(&instance, &device);

            let allocator = Arc::new(Mutex::new(
                Allocator::new(&AllocatorCreateDesc {
                    instance: instance.clone(),
                    device: device.clone(),
                    physical_device: pdevice,
                    debug_settings: Default::default(),
                    buffer_device_address: true,
                    allocation_sizes: Default::default(),
                })
                .unwrap(),
            ));

            let arc_device = Arc::new(device.clone());
            let buffer_manager = ResourceManager::new(arc_device.clone(), allocator.clone(), 512);
            let texture_manager = ResourceManager::new(arc_device.clone(), allocator.clone(), 512);
            let graphics_pipelines =
                ResourceManager::new(arc_device.clone(), allocator.clone(), 64);

            let present_queue = Arc::new(Mutex::new(present_queue));
            let thread_id = std::thread::current().id();

            Self {
                allocator,
                entry,
                instance,
                device: device.clone(),
                dynamic_rendering,
                queue_family_index,
                pdevice,
                command_thread_pool,
                threaded_command_buffers,
                render_swapchain: Arc::new(RwLock::new(render_swapchain)),
                buffer_manager: Arc::new(buffer_manager),
                texture_manager: Arc::new(texture_manager),
                graphics_pipelines: Arc::new(graphics_pipelines),
                immutable_samplers: Arc::new(create_immutable_samplers(&device)),
                yuv_immutable_samplers: Arc::new(create_immutable_yuv_samplers(&device)),
                thread_id,
                // TODO: fetch from device
                max_descriptor_count: {
                    (512 * 1024).min(
                        device_properties
                            .limits
                            .max_per_stage_descriptor_sampled_images
                            - RESERVED_DESCRIPTOR_COUNT,
                    )
                },
                surface_loader,
                present_queue,

                pool,
                main_command_buffer,

                present_complete_semaphore,
                rendering_complete_semaphore,
                main_commands_reuse_fence,
                surface,
                debug_call_back,
                debug_utils_loader,
            }
        }
    }

    pub fn create_swapchain(
        instance: &ash::Instance,
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        surface_loader: &Surface,
        surface: SurfaceKHR,
        present_mode: PresentModeKHR,
    ) -> RenderSwapchain {
        unsafe {
            let swapchain_loader = Swapchain::new(instance, device);

            let surface_format = surface_loader
                .get_physical_device_surface_formats(pdevice, surface)
                .unwrap()[0];

            let surface_capabilities = surface_loader
                .get_physical_device_surface_capabilities(pdevice, surface)
                .unwrap();
            let mut desired_image_count = surface_capabilities.min_image_count + 1;
            if surface_capabilities.max_image_count > 0
                && desired_image_count > surface_capabilities.max_image_count
            {
                desired_image_count = surface_capabilities.max_image_count;
            }

            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };
            let surface_resolution = surface_capabilities.current_extent;

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(surface)
                .min_image_count(desired_image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(surface_resolution)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(pre_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1);

            let swapchain = swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap();

            let present_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
            let present_image_views: Vec<vk::ImageView> = present_images
                .iter()
                .map(|&image| {
                    let create_view_info = vk::ImageViewCreateInfo::default()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image(image);
                    device.create_image_view(&create_view_info, None).unwrap()
                })
                .collect();

            let device_memory_properties = instance.get_physical_device_memory_properties(pdevice);
            let depth_image_create_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D16_UNORM)
                .extent(surface_resolution.into())
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let depth_image = device.create_image(&depth_image_create_info, None).unwrap();
            let depth_image_memory_req = device.get_image_memory_requirements(depth_image);
            let depth_image_memory_index = find_memorytype_index(
                &depth_image_memory_req,
                &device_memory_properties,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .expect("Unable to find suitable memory index for depth image.");

            let depth_image_allocate_info = vk::MemoryAllocateInfo::default()
                .allocation_size(depth_image_memory_req.size)
                .memory_type_index(depth_image_memory_index);

            let depth_image_memory = device
                .allocate_memory(&depth_image_allocate_info, None)
                .unwrap();

            device
                .bind_image_memory(depth_image, depth_image_memory, 0)
                .expect("Unable to bind depth image memory");

            // transition depth?

            let depth_image_view_info = vk::ImageViewCreateInfo::default()
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1),
                )
                .image(depth_image)
                .format(depth_image_create_info.format)
                .view_type(vk::ImageViewType::TYPE_2D);

            let depth_image_view = device
                .create_image_view(&depth_image_view_info, None)
                .unwrap();

            RenderSwapchain {
                swapchain,
                swapchain_loader,
                present_images,
                present_image_views,
                depth_image,
                depth_image_view,
                depth_image_memory,
                depth_image_format: depth_image_create_info.format,
                surface_format,
                surface_resolution,
                present_mode,
            }
        }
    }

    pub fn cleanup_swapchain(&self) {
        unsafe {
            let render_swapchain = self.get_swapchain();
            self.device
                .destroy_image_view(render_swapchain.depth_image_view, None);
            self.device
                .free_memory(render_swapchain.depth_image_memory, None);
            self.device
                .destroy_image(render_swapchain.depth_image, None);

            for &image_view in render_swapchain.present_image_views.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            render_swapchain
                .swapchain_loader
                .destroy_swapchain(render_swapchain.swapchain, None);
        }
    }

    #[tracing::instrument(skip(self))]
    pub fn recreate_swapchain(&self, width: u32, height: u32) {
        {
            let swapchain = self.get_swapchain();
            if width == swapchain.surface_resolution.width
                && height == swapchain.surface_resolution.height
            {
                return;
            }
        }

        let (old_surface_resolution, new_surface_resolution) = {
            let _guard = self.present_queue.lock().unwrap();
            unsafe {
                self.device
                    .device_wait_idle()
                    .expect("Failed to wait device idle!")
            };
            let old_surface_resolution = self.get_swapchain().surface_resolution;
            self.cleanup_swapchain();
            let render_swapchain = {
                let present_mode = self.get_swapchain().present_mode;
                Self::create_swapchain(
                    &self.instance,
                    &self.device,
                    self.pdevice,
                    &self.surface_loader,
                    self.surface,
                    present_mode,
                )
            };
            let new_surface_resolution = render_swapchain.surface_resolution;
            *self.render_swapchain.write().unwrap() = render_swapchain;
            (old_surface_resolution, new_surface_resolution)
        };

        self.graphics_pipelines
            .call_swapchain_resize_hooks(old_surface_resolution, new_surface_resolution);
        self.texture_manager
            .call_swapchain_resize_hooks(old_surface_resolution, new_surface_resolution);
    }

    pub fn record(
        &self,
        command_buffer: vk::CommandBuffer,
        fence: vk::Fence,
        f: impl FnOnce(&Self, vk::CommandBuffer),
    ) {
        unsafe {
            self.device
                .wait_for_fences(&[fence], true, std::u64::MAX)
                .expect("Wait for fence failed.");
            self.device
                .reset_fences(&[fence])
                .expect("Reset fences failed.");

            self.device
                .reset_command_buffer(
                    command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .expect("Reset command buffer failed.");

            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Begin commandbuffer");
            f(self, command_buffer);
            self.device
                .end_command_buffer(command_buffer)
                .expect("End commandbuffer");
        }
    }

    /// this can only run on a thread created by the thread pool
    pub fn get_command_buffer(&self) -> (vk::CommandBuffer, Fence) {
        if let Some(thread_index) = rayon::current_thread_index() {
            let threaded_command_buffers = self.threaded_command_buffers.read().unwrap();
            *threaded_command_buffers
                .get(&thread_index)
                .expect("Command buffer not found inside thread pool.")
        } else if std::thread::current().id() == self.thread_id {
            (self.main_command_buffer, self.main_commands_reuse_fence)
        } else {
            panic!("Command buffer not found inside thread pool.")
        }
    }

    /// Submits the command buffer to the queue and waits for it to finish.
    pub fn submit(&self, command_buffer: vk::CommandBuffer, fence: vk::Fence) {
        let queue = self.present_queue.lock().unwrap();
        unsafe {
            let submit_info =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
            self.device
                .queue_submit(*queue, &[submit_info], fence)
                .expect("queue submit failed.");
            self.device.queue_wait_idle(*queue).unwrap();
        }
    }

    pub fn record_submit(&self, f: impl FnOnce(&Self, vk::CommandBuffer)) {
        let (command_buffer, fence) = self.get_command_buffer();
        self.record(command_buffer, fence, f);
        self.submit(command_buffer, fence);
    }

    /// Submits the command buffer to the present queue with corresponding semaphores and waits for it to finish.
    #[tracing::instrument(skip(self))]
    pub fn present_submit(&self, present_index: u32) {
        let (command_buffer, fence) = self.get_command_buffer();
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(std::slice::from_ref(&self.present_complete_semaphore))
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .command_buffers(std::slice::from_ref(&command_buffer))
            .signal_semaphores(std::slice::from_ref(&self.rendering_complete_semaphore));
        unsafe {
            let queue = self.present_queue.lock().unwrap();
            self.device
                .queue_submit(*queue, &[submit_info], fence)
                .expect("queue submit failed.");
            self.device.queue_wait_idle(*queue).unwrap();

            let wait_semaphors = [self.rendering_complete_semaphore];
            let swapchains = [self.get_swapchain().swapchain];
            let image_indices = [present_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&wait_semaphors)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            let result = self
                .get_swapchain()
                .swapchain_loader
                .queue_present(*queue, &present_info);

            drop(queue);

            let is_resized = match result {
                Ok(_) => false,
                Err(vk_result) => match vk_result {
                    vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
                    _ => panic!("Failed to execute queue present."),
                },
            };

            if is_resized {
                self.recreate_swapchain(0, 0);
            }
        }
    }

    pub fn acquire_present_index(&self) -> u32 {
        unsafe {
            let render_swapchain = self.get_swapchain();

            render_swapchain
                .swapchain_loader
                .acquire_next_image(
                    render_swapchain.swapchain,
                    std::u64::MAX,
                    self.present_complete_semaphore,
                    vk::Fence::null(),
                )
                .unwrap()
                .0
        }
    }

    /// Acquires the next image, transitions the image from the swapchain and records the draw command buffer. Returns the present_index
    #[tracing::instrument(skip_all)]
    pub fn present_record<
        F: FnOnce(&Self, vk::CommandBuffer, ImageView, ImageView) + Send + Sync,
    >(
        &self,
        present_index: u32,
        f: F,
    ) {
        let (command_buffer, fence) = self.get_command_buffer();
        self.record(command_buffer, fence, |ctx, command_buffer| unsafe {
            let render_swapchain = ctx.get_swapchain();
            let layout_transition_barriers = vk::ImageMemoryBarrier::default()
                .image(render_swapchain.present_images[present_index as usize])
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1)
                        .level_count(1),
                );

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[layout_transition_barriers],
            );

            let present_image_view = render_swapchain.present_image_views[present_index as usize];
            let depth_image_view = render_swapchain.depth_image_view;

            f(ctx, command_buffer, present_image_view, depth_image_view);

            let layout_transition_barriers = vk::ImageMemoryBarrier::default()
                .image(render_swapchain.present_images[present_index as usize])
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1)
                        .level_count(1),
                );

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[layout_transition_barriers],
            );
        });
    }

    /// Example
    /// ```rs
    /// let color_attachments = &[vk::RenderingAttachmentInfo::default()
    ///    .image_view(self.render_swapchain.present_image_views[present_index as usize])
    ///    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
    ///    .load_op(vk::AttachmentLoadOp::CLEAR)
    ///    .store_op(vk::AttachmentStoreOp::STORE)
    ///    .clear_value(vk::ClearValue {
    ///        color: vk::ClearColorValue {
    ///            float32: [0.1, 0.1, 0.1, 1.0],
    ///        },
    ///    })];
    ///
    /// let depth_attachment = &vk::RenderingAttachmentInfo::default()
    ///    .image_view(self.render_swapchain.depth_image_view)
    ///    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    ///    .load_op(vk::AttachmentLoadOp::CLEAR)
    ///    .store_op(vk::AttachmentStoreOp::STORE)
    ///    .clear_value(vk::ClearValue {
    ///        depth_stencil: vk::ClearDepthStencilValue {
    ///            depth: 1.0,
    ///            stencil: 0,
    ///        },
    ///    });
    ///
    /// ctx.begin_rendering(
    ///     ctx.draw_command_buffer,
    ///     color_attachments,
    ///     Some(depth_attachment),
    /// );
    /// ```
    pub fn begin_rendering(
        &self,
        command_buffer: vk::CommandBuffer,
        color_attachments: &[vk::RenderingAttachmentInfo],
        depth_attachment: Option<&vk::RenderingAttachmentInfo>,
    ) {
        let depth_fallback = vk::RenderingAttachmentInfo::default();
        let depth_attachment = match depth_attachment {
            Some(depth_attachment) => Some(depth_attachment),
            None => Some(&depth_fallback),
        };
        let surface_resolution = self.get_swapchain().surface_resolution;
        let render_pass_begin_info = vk::RenderingInfo::default()
            // .flags(vk::RenderingFlags::CONTENTS_SECONDARY_COMMAND_BUFFERS)
            .render_area(surface_resolution.into())
            .layer_count(1)
            .color_attachments(color_attachments)
            .depth_attachment(depth_attachment.as_ref().unwrap());

        unsafe {
            self.dynamic_rendering
                .cmd_begin_rendering(command_buffer, &render_pass_begin_info);
        }
    }

    pub fn end_rendering(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.dynamic_rendering.cmd_end_rendering(command_buffer);
        }
    }

    fn create_samplers(device: &ash::Device) -> HashMap<SamplerDesc, vk::Sampler> {
        let texel_filters = [vk::Filter::NEAREST, vk::Filter::LINEAR];
        let mipmap_modes = [
            vk::SamplerMipmapMode::NEAREST,
            vk::SamplerMipmapMode::LINEAR,
        ];
        let address_modes = [
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
        ];

        let mut result = HashMap::new();

        for &texel_filter in &texel_filters {
            for &mipmap_mode in &mipmap_modes {
                for &address_modes in &address_modes {
                    let anisotropy_enable = texel_filter == vk::Filter::LINEAR;

                    result.insert(
                        SamplerDesc {
                            texel_filter,
                            mipmap_mode,
                            address_modes,
                        },
                        unsafe {
                            device.create_sampler(
                                &vk::SamplerCreateInfo::default()
                                    .mag_filter(texel_filter)
                                    .min_filter(texel_filter)
                                    .mipmap_mode(mipmap_mode)
                                    .address_mode_u(address_modes)
                                    .address_mode_v(address_modes)
                                    .address_mode_w(address_modes)
                                    .max_lod(vk::LOD_CLAMP_NONE)
                                    .max_anisotropy(16.0)
                                    .anisotropy_enable(anisotropy_enable),
                                None,
                            )
                        }
                        .expect("create_sampler"),
                    );
                }
            }
        }

        result
    }

    pub fn create_command_thread_pool(
        device: Device,
        queue_family_index: u32,
    ) -> (
        ThreadPool,
        Arc<RwLock<HashMap<usize, (CommandBuffer, Fence)>>>,
    ) {
        let m_command_buffers: Arc<RwLock<HashMap<usize, (CommandBuffer, Fence)>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let m_command_buffers_clone = m_command_buffers.clone();

        let pool = rayon::ThreadPoolBuilder::new()
            .thread_name(|x| format!("Command buffer generation thread {}", x))
            .start_handler(move |x| {
                let pool_create_info = vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(queue_family_index);

                COMMAND_POOL.with(|pool| {
                    *pool.borrow_mut() =
                        unsafe { device.create_command_pool(&pool_create_info, None).unwrap() };

                    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                        .command_buffer_count(1)
                        .command_pool(*pool.borrow())
                        .level(vk::CommandBufferLevel::PRIMARY);

                    let command_buffers = unsafe {
                        device
                            .allocate_command_buffers(&command_buffer_allocate_info)
                            .unwrap()
                    };

                    let fence = unsafe {
                        device
                            .create_fence(
                                &vk::FenceCreateInfo::default()
                                    .flags(vk::FenceCreateFlags::SIGNALED),
                                None,
                            )
                            .expect("Create fence failed.")
                    };

                    m_command_buffers
                        .write()
                        .unwrap()
                        .insert(x, (command_buffers[0], fence));
                });
            })
            .build()
            .unwrap();

        (pool, m_command_buffers_clone)
    }

    #[tracing::instrument(skip_all, name = "copy_buffer_to_texture")]
    pub fn copy_buffer_to_texture(
        &self,
        buffer: &ResourceHandle<Buffer>,
        texture: &ResourceHandle<Texture>,
    ) {
        self.record_submit(|ctx, command_buffer| unsafe {
            let texture = ctx.texture_manager.get_mut(texture).unwrap();
            texture.transition(
                &ctx.device,
                command_buffer,
                &TransitionDesc {
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    new_stage_mask: vk::PipelineStageFlags::TRANSFER,
                },
            );

            ctx.device.cmd_copy_buffer_to_image(
                command_buffer,
                ctx.buffer_manager.get(buffer).unwrap().buffer(),
                texture.image,
                texture.layout,
                &[vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(texture.extent.width)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(texture.extent)],
            );
        });
    }

    // pub fn copy_buffer_to_texture(&self, buffer: &Buffer, texture: &Image) {
    //     unsafe {
    //         record_submit_commandbuffer(
    //             &self.device,
    //             self.setup_command_buffer,
    //             self.setup_commands_reuse_fence,
    //             self.present_queue,
    //             &[],
    //             &[],
    //             &[],
    //             |device, setup_command_buffer| {
    //                 {
    //                     let image_memory_barrier = vk::ImageMemoryBarrier2::default()
    //                         .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
    //                         .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
    //                         .src_access_mask(vk::AccessFlags2::empty())
    //                         .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
    //                         .old_layout(vk::ImageLayout::UNDEFINED)
    //                         .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
    //                         .image(texture.image)
    //                         .subresource_range(vk::ImageSubresourceRange {
    //                             aspect_mask: vk::ImageAspectFlags::COLOR,
    //                             layer_count: 1,
    //                             level_count: 1,
    //                             ..Default::default()
    //                         });

    //                     let dependency_info = vk::DependencyInfo::default()
    //                         .image_memory_barriers(std::slice::from_ref(&image_memory_barrier));

    //                     self.synchronization2
    //                         .cmd_pipeline_barrier2(setup_command_buffer, &dependency_info);
    //                 }

    //                 // println!(
    //                 //     "{:?} {:?} {:?}",
    //                 //     buffer.size,
    //                 //     texture.extent.width * texture.bytes_per_texel(),
    //                 //     texture.extent.width
    //                 // );

    //                 device.cmd_copy_buffer_to_image(
    //                     setup_command_buffer,
    //                     buffer.buffer,
    //                     texture.image,
    //                     ImageLayout::TRANSFER_DST_OPTIMAL,
    //                     &[BufferImageCopy::default()
    //                         .buffer_offset(0)
    //                         .buffer_row_length(texture.extent.width)
    //                         .buffer_image_height(0)
    //                         .image_subresource(vk::ImageSubresourceLayers {
    //                             aspect_mask: vk::ImageAspectFlags::COLOR,
    //                             mip_level: 0,
    //                             base_array_layer: 0,
    //                             layer_count: 1,
    //                         })
    //                         .image_extent(texture.extent)],
    //                 );

    //                 // {

    //                 // }
    //             },
    //         );
    //     }
    // }

    /// Returns a read lock to the buffer manager.
    // pub fn get_buffer_manager(&self) -> std::sync::RwLockReadGuard<BufferManager> {
    //     self.buffer_manager
    // }

    // /// Returns a write lock to the buffer manager.
    // pub fn get_buffer_manager_mut(&self) -> std::sync::RwLockWriteGuard<BufferManager> {
    //     self.buffer_manager.write().unwrap()
    // }

    pub fn create_buffer(&self, desc: &BufferDescriptor) -> ResourceHandle<Buffer> {
        let buffer = Buffer::new(
            &self.device,
            &mut self.allocator.lock().unwrap(),
            desc.debug_name,
            desc.size,
            desc.usage,
            desc.location,
        );
        let id = self.buffer_manager.create(buffer);
        ResourceHandle::new(id, self.buffer_manager.clone())
    }

    pub fn create_buffer_with_data(
        &self,
        desc: &BufferDescriptor,
        data: &[u8],
        offset: usize,
    ) -> ResourceHandle<Buffer> {
        let mut desc = desc.clone();
        if desc.location == MemoryLocation::GpuOnly {
            desc.usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        if desc.size == Default::default() {
            desc.size = size_of_val(data) as DeviceSize;
        }

        let handle = self.create_buffer(&desc);
        if desc.location == MemoryLocation::GpuOnly {
            let staging_handle = self.create_buffer(&BufferDescriptor {
                size: desc.size,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                location: MemoryLocation::CpuToGpu,
                debug_name: "Staging buffer",
            });
            let staging_buffer = self.buffer_manager.get_mut(&staging_handle).unwrap();
            staging_buffer.copy_from_slice(data, offset);

            self.record_submit(|ctx: &RenderContext, command_buffer| unsafe {
                let buffer = self.buffer_manager.get(&handle).unwrap();
                ctx.device.cmd_copy_buffer(
                    command_buffer,
                    staging_buffer.buffer(),
                    buffer.buffer(),
                    &[vk::BufferCopy {
                        size: desc.size,
                        src_offset: 0,
                        dst_offset: offset as DeviceSize,
                    }],
                )
            });
        } else {
            self.buffer_manager
                .get_mut(&handle)
                .unwrap()
                .copy_from_slice(data, offset);
        }

        handle
    }

    pub fn create_texture(
        &self,
        debug_name: &str,
        create_info: &vk::ImageCreateInfo,
    ) -> ResourceHandle<Texture> {
        let texture = Texture::new(
            &self.device,
            &mut self.allocator.lock().unwrap(),
            debug_name,
            create_info,
        );
        let id = self.texture_manager.create(texture);
        ResourceHandle::new(id, self.texture_manager.clone())
    }

    /// Creates or gets (if already created) a texture view, which can be cloned cheaply.
    pub fn get_texture_view(
        &self,
        texture: &ResourceHandle<Texture>,
    ) -> Option<Arc<vk::ImageView>> {
        let texture = self.texture_manager.get_mut(texture)?;
        Some(texture.create_view(&self.device))
    }

    pub fn create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDescriptor,
    ) -> ResourceHandle<GraphicsPipeline> {
        let pipeline = GraphicsPipeline::new(
            &self.device,
            desc,
            &ImmutableShaderInfo {
                immutable_samplers: self.immutable_samplers.clone(),
                yuv_conversion_samplers: self.yuv_immutable_samplers.clone(),
                max_descriptor_count: self.max_descriptor_count,
            },
            self.get_swapchain().surface_resolution,
        );
        let id = self.graphics_pipelines.create(pipeline);
        ResourceHandle::new(id, self.graphics_pipelines.clone())
    }

    /// Returns a read lock to the render swapchain.
    pub fn get_swapchain(&self) -> std::sync::RwLockReadGuard<RenderSwapchain> {
        self.render_swapchain.read().unwrap()
    }
}

impl Drop for RenderContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            self.device
                .destroy_semaphore(self.present_complete_semaphore, None);
            self.device
                .destroy_semaphore(self.rendering_complete_semaphore, None);
            self.device
                .destroy_fence(self.main_commands_reuse_fence, None);

            self.cleanup_swapchain();

            self.command_thread_pool.broadcast(|ctx| {
                COMMAND_POOL.with(|pool| {
                    let pool = pool.take();
                    let mut command_buffer_g = self.threaded_command_buffers.write().unwrap();
                    let command_buffer = command_buffer_g.get(&ctx.index());
                    if let Some((command_buffer, fence)) = command_buffer {
                        self.device.free_command_buffers(pool, &[*command_buffer]);
                        self.device.destroy_fence(*fence, None);
                    }
                    self.device.destroy_command_pool(pool, None);
                    command_buffer_g.remove(&ctx.index());
                });
            });

            // self.graphics_pipelines.clear_all();
            // self.buffer_manager.clear_all();
            // self.texture_manager.clear_all();

            for sampler in self.immutable_samplers.values() {
                self.device.destroy_sampler(*sampler, None);
            }

            for (conversion, sampler) in self.yuv_immutable_samplers.values() {
                self.device
                    .destroy_sampler_ycbcr_conversion(*conversion, None);
                self.device.destroy_sampler(*sampler, None);
            }

            self.device.destroy_command_pool(self.pool, None);
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle!");
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_call_back, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn create_immutable_samplers(device: &ash::Device) -> HashMap<SamplerDesc, vk::Sampler> {
    let texel_filters = [vk::Filter::NEAREST, vk::Filter::LINEAR];
    let mipmap_modes = [
        vk::SamplerMipmapMode::NEAREST,
        vk::SamplerMipmapMode::LINEAR,
    ];
    let address_modes = [
        vk::SamplerAddressMode::REPEAT,
        vk::SamplerAddressMode::CLAMP_TO_EDGE,
    ];

    let mut result = HashMap::new();

    for &texel_filter in &texel_filters {
        for &mipmap_mode in &mipmap_modes {
            for &address_modes in &address_modes {
                let anisotropy_enable = texel_filter == vk::Filter::LINEAR;

                result.insert(
                    SamplerDesc {
                        texel_filter,
                        mipmap_mode,
                        address_modes,
                    },
                    unsafe {
                        device.create_sampler(
                            &vk::SamplerCreateInfo::default()
                                .mag_filter(texel_filter)
                                .min_filter(texel_filter)
                                .mipmap_mode(mipmap_mode)
                                .address_mode_u(address_modes)
                                .address_mode_v(address_modes)
                                .address_mode_w(address_modes)
                                .max_lod(vk::LOD_CLAMP_NONE)
                                .max_anisotropy(16.0)
                                .anisotropy_enable(anisotropy_enable),
                            None,
                        )
                    }
                    .expect("create_sampler"),
                );
            }
        }
    }

    result
}

fn create_immutable_yuv_samplers(
    device: &ash::Device,
) -> HashMap<(vk::Format, SamplerDesc), (vk::SamplerYcbcrConversion, vk::Sampler)> {
    let mut result = HashMap::new();

    let formats = [
        vk::Format::G8_B8R8_2PLANE_420_UNORM,
        vk::Format::G8_B8_R8_3PLANE_420_UNORM,
        vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
        vk::Format::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16,
    ];

    for format in formats {
        let sampler_conversion = unsafe {
            device
                .create_sampler_ycbcr_conversion(
                    &vk::SamplerYcbcrConversionCreateInfo::default()
                        .ycbcr_model(vk::SamplerYcbcrModelConversion::YCBCR_709)
                        .ycbcr_range(vk::SamplerYcbcrRange::ITU_NARROW)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::IDENTITY,
                            g: vk::ComponentSwizzle::IDENTITY,
                            b: vk::ComponentSwizzle::IDENTITY,
                            a: vk::ComponentSwizzle::IDENTITY,
                        })
                        .chroma_filter(vk::Filter::LINEAR)
                        .x_chroma_offset(vk::ChromaLocation::MIDPOINT)
                        .y_chroma_offset(vk::ChromaLocation::MIDPOINT)
                        .force_explicit_reconstruction(false)
                        .format(format),
                    None,
                )
                .unwrap()
        };

        let mut conversion_info =
            vk::SamplerYcbcrConversionInfo::default().conversion(sampler_conversion);

        let desc = SamplerDesc {
            texel_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_modes: vk::SamplerAddressMode::CLAMP_TO_EDGE,
        };

        let sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(desc.texel_filter)
                        .min_filter(desc.texel_filter)
                        .mipmap_mode(desc.mipmap_mode)
                        .address_mode_u(desc.address_modes)
                        .address_mode_v(desc.address_modes)
                        .address_mode_w(desc.address_modes)
                        .max_lod(vk::LOD_CLAMP_NONE)
                        .max_anisotropy(16.0)
                        .anisotropy_enable(false)
                        .unnormalized_coordinates(false)
                        .push_next(&mut conversion_info),
                    None,
                )
                .unwrap()
        };
        result.insert((format, desc), (sampler_conversion, sampler));
    }

    result
}
