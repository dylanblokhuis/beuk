use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, Allocator},
    MemoryLocation,
};

#[derive(Debug)]
pub struct Texture {
    pub image: vk::Image,
    pub allocation: Option<Allocation>,
    pub view: Option<vk::ImageView>,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub offset: u64,
}

impl Texture {
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        debug_name: &str,
        image_info: &vk::ImageCreateInfo,
    ) -> Texture {
        let image = unsafe { device.create_image(image_info, None) }.unwrap();
        let requirements = unsafe { device.get_image_memory_requirements(image) };

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: debug_name,
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();
        let offset = allocation.offset();

        unsafe {
            device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .unwrap()
        };

        Self {
            image,
            allocation: Some(allocation),
            view: None,
            format: image_info.format,
            extent: image_info.extent,
            offset,
        }
    }

    pub fn create_view(&mut self, device: &ash::Device) -> vk::ImageView {
        if self.view.is_some() {
            return self.view.unwrap();
        }
        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo {
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: self.format,
                    components: vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    },
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        level_count: 1,
                        layer_count: 1,
                        ..Default::default()
                    },
                    image: self.image,
                    ..Default::default()
                },
                None,
            )
        }
        .unwrap();
        self.view = Some(view);
        view
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(view) = self.view.take() {
            unsafe { device.destroy_image_view(view, None) };
        }
        allocator.free(self.allocation.take().unwrap()).unwrap();
        unsafe { device.destroy_image(self.image, None) };
    }

    // pub fn from_image_buffer(
    //     render_instance: &RenderInstance,
    //     render_allocator: &mut RenderAllocator,
    //     image: DynamicImage,
    //     format: vk::Format,
    // ) -> Self {
    //     let texture = Self::new(
    //         render_instance.device(),
    //         render_allocator.allocator(),
    //         &vk::ImageCreateInfo::default()
    //             .image_type(vk::ImageType::TYPE_2D)
    //             .format(format)
    //             .extent(vk::Extent3D {
    //                 width: image.width(),
    //                 height: image.height(),
    //                 depth: 1,
    //             })
    //             .mip_levels(1)
    //             .array_layers(1)
    //             .samples(vk::SampleCountFlags::TYPE_1)
    //             .tiling(vk::ImageTiling::OPTIMAL)
    //             .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
    //             .sharing_mode(vk::SharingMode::EXCLUSIVE),
    //     );

    //     {
    //         // let image_data = match format {
    //         //     vk::Format::R8G8B8A8_SRGB => image.to_rgba8().into_raw(),
    //         //     vk::Format::R8G8B8_SRGB => image.to_rgb8().into_raw(),
    //         //     _ => unimplemented!("Format not supported yet"),
    //         // };
    //         let image_data = image.to_rgba8().into_raw();
    //         let mut img_buffer = Buffer::new(
    //             render_instance.device(),
    //             render_allocator.allocator(),
    //             &vk::BufferCreateInfo::default()
    //                 .size(image_data.len() as DeviceSize)
    //                 .usage(vk::BufferUsageFlags::TRANSFER_SRC)
    //                 .sharing_mode(vk::SharingMode::EXCLUSIVE),
    //             MemoryLocation::CpuToGpu,
    //         );
    //         img_buffer.copy_from_slice(&image_data, 0);

    //         render_instance
    //             .0
    //             .copy_buffer_to_texture(&img_buffer, &texture);

    //         img_buffer.destroy(render_instance.device(), render_allocator.allocator());
    //     }

    //     texture
    // }

    pub fn bytes_per_texel(&self) -> u32 {
        match self.format {
            vk::Format::R8G8B8A8_UNORM => 4,
            vk::Format::R8G8B8A8_SRGB => 4,
            vk::Format::B8G8R8A8_SRGB => 4,
            vk::Format::R8G8B8A8_SNORM => 4,
            vk::Format::R16G16B16A16_SFLOAT => 8,
            vk::Format::R32G32B32A32_SFLOAT => 16,
            _ => panic!("Block info format hasn't been supplied yet, please add it"),
            // vk::Format::R32_SFLOAT => uncompressed(4),
            // vk::Format::R16G16_SFLOAT => uncompressed(8),
            // vk::Format::Rgba32Float => uncompressed(16),
            // vk::Format::R32Uint => uncompressed(4),
            // vk::Format::Rg32Uint => uncompressed(8),
            // vk::Format::Rgba32Uint => uncompressed(16),
            // vk::Format::Depth32Float => uncompressed(4),
            // vk::Format::Bc1Unorm => cx_bc(8),
            // vk::Format::Bc1UnormSrgb => cx_bc(8),
            // vk::Format::Bc2Unorm => cx_bc(16),
            // vk::Format::Bc2UnormSrgb => cx_bc(16),
            // vk::Format::Bc3Unorm => cx_bc(16),
            // vk::Format::Bc3UnormSrgb => cx_bc(16),
            // vk::Format::Bc4Unorm => cx_bc(8),
            // vk::Format::Bc4Snorm => cx_bc(8),
            // vk::Format::Bc5Unorm => cx_bc(16),
            // vk::Format::Bc5Snorm => cx_bc(16),
        }
    }
}
