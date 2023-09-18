use std::mem::size_of;

use ash::extensions::khr::AccelerationStructure;
use ash::vk::{
    AccelerationStructureBuildSizesInfoKHR, AccelerationStructureCreateInfoKHR,
    AccelerationStructureKHR, Extent3D,
};
use beuk::ash::vk::{self, BufferUsageFlags};
use beuk::buffer::MemoryLocation;
use beuk::buffer::{Buffer, BufferDescriptor};
use beuk::compute_pipeline::{ComputePipeline, ComputePipelineDescriptor};
use beuk::ctx::RenderContextDescriptor;
use beuk::graphics_pipeline::{
    BlendState, FragmentState, GraphicsPipeline, VertexBufferLayout, VertexState,
};
use beuk::memory::ResourceHandle;
use beuk::shaders::ShaderDescriptor;
use beuk::texture::Texture;
use beuk::{
    ctx::RenderContext,
    graphics_pipeline::{GraphicsPipelineDescriptor, PrimitiveState},
};
use glam::{Mat4, Vec3, Vec4};
use image::EncodableLayout;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use smallvec::smallvec;
use tracing_subscriber::prelude::__tracing_subscriber_SubscriberExt;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraBuffer {
    view_inverse: Mat4,
    proj_inverse: Mat4,
}

fn main() {
    simple_logger::SimpleLogger::new().init().unwrap();
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(tracing_tracy::TracyLayer::new()),
    )
    .expect("set up the subscriber");

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("A fantastic window!")
        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0))
        .build(&event_loop)
        .unwrap();

    let ctx = beuk::ctx::RenderContext::new(RenderContextDescriptor {
        display_handle: window.raw_display_handle(),
        window_handle: window.raw_window_handle(),
        present_mode: vk::PresentModeKHR::default(),
    });

    let canvas = Canvas::new(&ctx);

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            window_id,
        } if window_id == window.id() => control_flow.set_exit(),

        Event::WindowEvent { event, .. } => match event {
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                ctx.recreate_swapchain(new_inner_size.width, new_inner_size.height);
            }
            WindowEvent::Resized(size) => {
                ctx.recreate_swapchain(size.width, size.height);
            }
            _ => (),
        },

        Event::MainEventsCleared => {
            window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            canvas.draw(&ctx);
        }
        _ => (),
    });
}

struct Canvas {
    compute_handle: ResourceHandle<ComputePipeline>,
    vertex_buffer: ResourceHandle<Buffer>,
    index_buffer: ResourceHandle<Buffer>,
    bottom_as: (ResourceHandle<Buffer>, AccelerationStructureKHR, u64),
    top_as: (ResourceHandle<Buffer>, AccelerationStructureKHR, u64),
    output_texture_handle: ResourceHandle<Texture>,
}

#[repr(C, align(16))]
#[derive(Clone, Debug, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 4],
}

fn mat4_to_transform_matrix_khr(mat: &Mat4) -> [f32; 12] {
    let m = mat.to_cols_array_2d();
    [
        m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3], m[2][0], m[2][1],
        m[2][2], m[2][3],
    ]
}

impl Canvas {
    fn new(ctx: &RenderContext) -> Self {
        let vertices = &[
            Vertex {
                pos: [-0.5, -0.5, -0.5, 1.0],
            }, // 0
            Vertex {
                pos: [0.5, -0.5, -0.5, 1.0],
            }, // 1
            Vertex {
                pos: [0.5, 0.5, -0.5, 1.0],
            }, // 2
            Vertex {
                pos: [-0.5, 0.5, -0.5, 1.0],
            }, // 3
            Vertex {
                pos: [-0.5, -0.5, 0.5, 1.0],
            }, // 4
            Vertex {
                pos: [0.5, -0.5, 0.5, 1.0],
            }, // 5
            Vertex {
                pos: [0.5, 0.5, 0.5, 1.0],
            }, // 6
            Vertex {
                pos: [-0.5, 0.5, 0.5, 1.0],
            }, // 7
        ];
        let vertex_buffer_handle = ctx.create_buffer_with_data(
            &BufferDescriptor {
                debug_name: "vertices",
                location: MemoryLocation::GpuOnly,
                usage: BufferUsageFlags::VERTEX_BUFFER
                    | BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                ..Default::default()
            },
            bytemuck::cast_slice(vertices),
            0,
        );

        let indices = &[
            0, 1, 2, 2, 3, 0, // front
            4, 5, 6, 6, 7, 4, // back
            0, 4, 7, 7, 3, 0, // left
            1, 5, 6, 6, 2, 1, // right
            3, 2, 6, 6, 7, 3, // top
            0, 1, 5, 5, 4, 0, // bottom
        ];

        let index_buffer_handle = ctx.create_buffer_with_data(
            &BufferDescriptor {
                debug_name: "indices",
                location: MemoryLocation::GpuOnly,
                size: (std::mem::size_of::<u32>() * 36) as u64,
                usage: BufferUsageFlags::INDEX_BUFFER
                    | BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            },
            bytemuck::cast_slice(indices),
            0,
        );

        let vertex_buffer = ctx.buffer_manager.get(&vertex_buffer_handle).unwrap();
        let index_buffer = ctx.buffer_manager.get(&index_buffer_handle).unwrap();

        let accel = AccelerationStructure::new(&ctx.instance, &ctx.device);

        // blas
        let (bottom_as_buffer_handle, bottom_as, bottom_as_buffer_addr) = unsafe {
            let geometry = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                        .vertex_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: vertex_buffer.device_addr,
                        })
                        .max_vertex(vertices.len() as u32)
                        .vertex_stride(size_of::<Vertex>() as u64)
                        .vertex_format(vk::Format::R32G32B32A32_SFLOAT)
                        .index_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: index_buffer.device_addr,
                        })
                        .index_type(vk::IndexType::UINT32),
                })
                .flags(vk::GeometryFlagsKHR::OPAQUE);

            let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::default()
                .first_vertex(0)
                .primitive_count(indices.len() as u32 / 3)
                .primitive_offset(0)
                .transform_offset(0);

            let geometries = &[geometry];
            let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .geometries(geometries)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);

            let mut size_info = AccelerationStructureBuildSizesInfoKHR::default();
            accel.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[build_range_info.primitive_count],
                &mut size_info,
            );

            let bottom_as_buffer_handle = ctx.create_buffer(&BufferDescriptor {
                debug_name: "bottom_as",
                size: size_info.acceleration_structure_size,
                usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
                location: MemoryLocation::CpuToGpu,
            });

            let bottom_as_buffer = ctx.buffer_manager.get(&bottom_as_buffer_handle).unwrap();

            let as_create_info = vk::AccelerationStructureCreateInfoKHR::default()
                .ty(build_info.ty)
                .size(size_info.acceleration_structure_size)
                .buffer(bottom_as_buffer.buffer())
                .offset(0);

            let bottom_as = accel
                .create_acceleration_structure(&as_create_info, None)
                .unwrap();
            build_info.dst_acceleration_structure = bottom_as;

            let scratch_buffer = ctx.create_buffer(&BufferDescriptor {
                debug_name: "scratch",
                size: size_info.build_scratch_size,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                location: MemoryLocation::CpuToGpu,
            });
            let scratch_buffer = ctx.buffer_manager.get(&scratch_buffer).unwrap();
            build_info.scratch_data = vk::DeviceOrHostAddressKHR {
                device_address: scratch_buffer.device_addr,
            };

            ctx.record_submit(|command_buffer| {
                accel.cmd_build_acceleration_structures(
                    command_buffer,
                    &[build_info],
                    &[&[build_range_info]],
                );
            });

            let bottom_as_buffer_addr = {
                accel.get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(bottom_as),
                )
            };

            (bottom_as_buffer_handle, bottom_as, bottom_as_buffer_addr)
        };

        let cube_transform = Transform::from_translation(Vec3::new(-10.0, 3.0, 1.0));

        let (top_as_buffer, top_as, top_as_buffer_addr) = unsafe {
            let mut instances = Vec::new();
            instances.push(vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR {
                    matrix: mat4_to_transform_matrix_khr(&cube_transform.compute_matrix()),
                },
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: bottom_as_buffer_addr,
                },
                instance_custom_index_and_mask: ash::vk::Packed24_8::new(0, 0xFF),
                instance_shader_binding_table_record_offset_and_flags: ash::vk::Packed24_8::new(
                    0,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                ),
            });

            let instance_buffer_size =
                std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() * instances.len();

            let instances_slice = &instances[..];
            let instances_byte_slice = std::slice::from_raw_parts(
                instances_slice.as_ptr() as *const u8,
                std::mem::size_of_val(instances_slice),
            );

            let instance_buffer = ctx.create_buffer_with_data(
                &BufferDescriptor {
                    debug_name: "instance",
                    location: MemoryLocation::CpuToGpu,
                    size: instance_buffer_size as u64,
                    usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                },
                instances_byte_slice,
                0,
            );

            let instance_buffer = ctx.buffer_manager.get(&instance_buffer).unwrap();
            let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::default()
                .first_vertex(0)
                .primitive_count(instances.len() as u32)
                .primitive_offset(0)
                .transform_offset(0);

            let instances = vk::AccelerationStructureGeometryInstancesDataKHR::default()
                .array_of_pointers(false)
                .data(vk::DeviceOrHostAddressConstKHR {
                    device_address: instance_buffer.device_addr,
                });

            let geometry = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                .geometry(vk::AccelerationStructureGeometryDataKHR { instances });

            let geometries = [geometry];

            let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .geometries(&geometries)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

            let mut size_info = AccelerationStructureBuildSizesInfoKHR::default();
            accel.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[build_range_info.primitive_count],
                &mut size_info,
            );

            let top_as_buffer_handle = ctx.create_buffer(&BufferDescriptor {
                debug_name: "top_as",
                size: size_info.acceleration_structure_size,
                usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
                location: MemoryLocation::CpuToGpu,
            });

            let top_as_buffer = ctx.buffer_manager.get(&top_as_buffer_handle).unwrap();
            let as_create_info = vk::AccelerationStructureCreateInfoKHR::default()
                .ty(build_info.ty)
                .size(size_info.acceleration_structure_size)
                .buffer(top_as_buffer.buffer())
                .offset(0);

            let top_as = accel
                .create_acceleration_structure(&as_create_info, None)
                .unwrap();

            build_info.dst_acceleration_structure = top_as;

            let scratch_buffer = ctx.create_buffer(&BufferDescriptor {
                debug_name: "scratch",
                size: size_info.build_scratch_size,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                location: MemoryLocation::CpuToGpu,
            });

            let scratch_buffer = ctx.buffer_manager.get(&scratch_buffer).unwrap();
            build_info.scratch_data = vk::DeviceOrHostAddressKHR {
                device_address: scratch_buffer.device_addr,
            };

            ctx.record_submit(|command_buffer| {
                accel.cmd_build_acceleration_structures(
                    command_buffer,
                    &[build_info],
                    &[&[build_range_info]],
                );
            });

            let top_as_buffer_addr = {
                accel.get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(top_as),
                )
            };

            (top_as_buffer_handle, top_as, top_as_buffer_addr)
        };

        let swapchain = ctx.get_swapchain();
        let output_texture_handle = ctx.create_texture(
            "rt-result",
            &vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: swapchain.surface_format.format,
                extent: vk::Extent3D {
                    width: swapchain.surface_resolution.width,
                    height: swapchain.surface_resolution.height,
                    depth: 1,
                },
                usage: vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::STORAGE,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            },
            false,
        );

        let compute_handle = ctx.create_compute_pipeline(
            "compute",
            ComputePipelineDescriptor {
                shader: ctx.create_shader(ShaderDescriptor {
                    kind: beuk::shaders::ShaderKind::Compute,
                    entry_point: "main".into(),
                    source: include_str!("./rt/shader.comp").into(),
                    ..Default::default()
                }),
                push_constant_range: None,
                prepend_descriptor_sets: None,
            },
        );

        let camera_position = Transform::from_xyz(-2.0, 0.2, 7.0).looking_at(Vec3::ZERO, Vec3::Y);

        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 2.5), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(59.0_f32.to_radians(), 16.0 / 9.0, 0.001, 1000.0);

        let camera_buffer = ctx.create_buffer_with_data(
            &BufferDescriptor {
                debug_name: "camera",
                location: MemoryLocation::CpuToGpu,
                size: std::mem::size_of::<CameraBuffer>() as u64,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            },
            bytemuck::cast_slice(&[CameraBuffer {
                view_inverse: view.inverse(),
                proj_inverse: proj.inverse(),
            }]),
            0,
        );

        unsafe {
            let camera_buffer = ctx.buffer_manager.get(&camera_buffer).unwrap();
            let pipeline = ctx.compute_pipelines.get(&compute_handle).unwrap();
            let mut accel_info = vk::WriteDescriptorSetAccelerationStructureKHR::default()
                .acceleration_structures(std::slice::from_ref(&top_as));
            let mut accel_write = vk::WriteDescriptorSet::default()
                .dst_set(pipeline.descriptor_sets[0])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .push_next(&mut accel_info);

            accel_write.descriptor_count = 1;

            ctx.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(pipeline.descriptor_sets[0])
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(std::slice::from_ref(
                            &vk::DescriptorImageInfo::default()
                                .image_layout(vk::ImageLayout::GENERAL)
                                .image_view(*ctx.get_texture_view(&output_texture_handle).unwrap()),
                        )),
                    accel_write,
                    vk::WriteDescriptorSet::default()
                        .dst_set(pipeline.descriptor_sets[0])
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(std::slice::from_ref(
                            &vk::DescriptorBufferInfo::default()
                                .buffer(camera_buffer.buffer())
                                .offset(0)
                                .range(vk::WHOLE_SIZE),
                        )),
                ],
                &[],
            );
        }

        Self {
            compute_handle,
            vertex_buffer: vertex_buffer_handle,
            index_buffer: index_buffer_handle,
            bottom_as: (bottom_as_buffer_handle, bottom_as, bottom_as_buffer_addr),
            top_as: (top_as_buffer, top_as, top_as_buffer_addr),
            output_texture_handle,
        }
    }

    pub fn draw(&self, ctx: &RenderContext) {
        ctx.record_submit(|command_buffer| unsafe {
            let mut compute_pipeline = ctx.compute_pipelines.get_mut(&self.compute_handle).unwrap();
            let texture = ctx
                .texture_manager
                .get(&self.output_texture_handle)
                .unwrap();
            compute_pipeline.bind_descriptor_sets(ctx, command_buffer);
            compute_pipeline.bind_pipeline(ctx, command_buffer);
            ctx.device.cmd_dispatch(
                command_buffer,
                texture.extent.width / 16,
                texture.extent.height / 16,
                1,
            );
        });

        let present_index = ctx.acquire_present_index();

        ctx.present_record(present_index, |command_buffer, image_view, depth_view| {
            // ctx.copy_texture_to_texture(
            //     command_buffer,
            //     &mut ctx
            //         .texture_manager
            //         .get_mut(&self.output_texture_handle)
            //         .unwrap(),
            //     &mut Texture::from_swapchain_image(
            //         ctx.get_swapchain().present_images[present_index as usize],
            //     ),
            //     Extent3D {
            //         width: ctx.get_swapchain().surface_resolution.width,
            //         height: ctx.get_swapchain().surface_resolution.height,
            //         depth: 1,
            //     }
            //     .into(),
            // );
        });

        ctx.present_submit(present_index);
    }
}

use glam::{Affine3A, Mat3, Quat};
use std::ops::Mul;

/// Describe the position of an entity. If the entity has a parent, the position is relative
/// to its parent position.
///
/// * To place or move an entity, you should set its [`Transform`].
/// * To get the global transform of an entity, you should get its [`GlobalTransform`].
/// * To be displayed, an entity must have both a [`Transform`] and a [`GlobalTransform`].
///   * You may use the [`TransformBundle`](crate::TransformBundle) to guarantee this.
///
/// ## [`Transform`] and [`GlobalTransform`]
///
/// [`Transform`] is the position of an entity relative to its parent position, or the reference
/// frame if it doesn't have a [`Parent`](bevy_hierarchy::Parent).
///
/// [`GlobalTransform`] is the position of an entity relative to the reference frame.
///
/// [`GlobalTransform`] is updated from [`Transform`] by systems in the system set
/// [`TransformPropagate`](crate::TransformSystem::TransformPropagate).
///
/// This system runs during [`PostUpdate`](bevy_app::PostUpdate). If you
/// update the [`Transform`] of an entity during this set or after, you will notice a 1 frame lag
/// before the [`GlobalTransform`] is updated.
///
/// # Examples
///
/// - [`transform`]
/// - [`global_vs_local_translation`]
///
/// [`global_vs_local_translation`]: https://github.com/bevyengine/bevy/blob/latest/examples/transforms/global_vs_local_translation.rs
/// [`transform`]: https://github.com/bevyengine/bevy/blob/latest/examples/transforms/transform.rs
/// [`Transform`]: super::Transform
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Transform {
    /// Position of the entity. In 2d, the last value of the `Vec3` is used for z-ordering.
    ///
    /// See the [`translations`] example for usage.
    ///
    /// [`translations`]: https://github.com/bevyengine/bevy/blob/latest/examples/transforms/translation.rs
    pub translation: Vec3,
    /// Rotation of the entity.
    ///
    /// See the [`3d_rotation`] example for usage.
    ///
    /// [`3d_rotation`]: https://github.com/bevyengine/bevy/blob/latest/examples/transforms/3d_rotation.rs
    pub rotation: Quat,
    /// Scale of the entity.
    ///
    /// See the [`scale`] example for usage.
    ///
    /// [`scale`]: https://github.com/bevyengine/bevy/blob/latest/examples/transforms/scale.rs
    pub scale: Vec3,
}

impl Transform {
    /// An identity [`Transform`] with no translation, rotation, and a scale of 1 on all axes.
    pub const IDENTITY: Self = Transform {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    /// Creates a new [`Transform`] at the position `(x, y, z)`. In 2d, the `z` component
    /// is used for z-ordering elements: higher `z`-value will be in front of lower
    /// `z`-value.
    #[inline]
    pub const fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        Self::from_translation(Vec3::new(x, y, z))
    }

    /// Extracts the translation, rotation, and scale from `matrix`. It must be a 3d affine
    /// transformation matrix.
    #[inline]
    pub fn from_matrix(matrix: Mat4) -> Self {
        let (scale, rotation, translation) = matrix.to_scale_rotation_translation();

        Transform {
            translation,
            rotation,
            scale,
        }
    }

    /// Creates a new [`Transform`], with `translation`. Rotation will be 0 and scale 1 on
    /// all axes.
    #[inline]
    pub const fn from_translation(translation: Vec3) -> Self {
        Transform {
            translation,
            ..Self::IDENTITY
        }
    }

    /// Creates a new [`Transform`], with `rotation`. Translation will be 0 and scale 1 on
    /// all axes.
    #[inline]
    pub const fn from_rotation(rotation: Quat) -> Self {
        Transform {
            rotation,
            ..Self::IDENTITY
        }
    }

    /// Creates a new [`Transform`], with `scale`. Translation will be 0 and rotation 0 on
    /// all axes.
    #[inline]
    pub const fn from_scale(scale: Vec3) -> Self {
        Transform {
            scale,
            ..Self::IDENTITY
        }
    }

    /// Returns this [`Transform`] with a new rotation so that [`Transform::forward`]
    /// points towards the `target` position and [`Transform::up`] points towards `up`.
    ///
    /// In some cases it's not possible to construct a rotation. Another axis will be picked in those cases:
    /// * if `target` is the same as the transform translation, `Vec3::Z` is used instead
    /// * if `up` is zero, `Vec3::Y` is used instead
    /// * if the resulting forward direction is parallel with `up`, an orthogonal vector is used as the "right" direction
    #[inline]
    #[must_use]
    pub fn looking_at(mut self, target: Vec3, up: Vec3) -> Self {
        self.look_at(target, up);
        self
    }

    /// Returns this [`Transform`] with a new rotation so that [`Transform::forward`]
    /// points in the given `direction` and [`Transform::up`] points towards `up`.
    ///
    /// In some cases it's not possible to construct a rotation. Another axis will be picked in those cases:
    /// * if `direction` is zero, `Vec3::Z` is used instead
    /// * if `up` is zero, `Vec3::Y` is used instead
    /// * if `direction` is parallel with `up`, an orthogonal vector is used as the "right" direction
    #[inline]
    #[must_use]
    pub fn looking_to(mut self, direction: Vec3, up: Vec3) -> Self {
        self.look_to(direction, up);
        self
    }

    /// Returns this [`Transform`] with a new translation.
    #[inline]
    #[must_use]
    pub const fn with_translation(mut self, translation: Vec3) -> Self {
        self.translation = translation;
        self
    }

    /// Returns this [`Transform`] with a new rotation.
    #[inline]
    #[must_use]
    pub const fn with_rotation(mut self, rotation: Quat) -> Self {
        self.rotation = rotation;
        self
    }

    /// Returns this [`Transform`] with a new scale.
    #[inline]
    #[must_use]
    pub const fn with_scale(mut self, scale: Vec3) -> Self {
        self.scale = scale;
        self
    }

    /// Returns the 3d affine transformation matrix from this transforms translation,
    /// rotation, and scale.
    #[inline]
    pub fn compute_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// Returns the 3d affine transformation matrix from this transforms translation,
    /// rotation, and scale.
    #[inline]
    pub fn compute_affine(&self) -> Affine3A {
        Affine3A::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// Get the unit vector in the local `X` direction.
    #[inline]
    pub fn local_x(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    /// Equivalent to [`-local_x()`][Transform::local_x()]
    #[inline]
    pub fn left(&self) -> Vec3 {
        -self.local_x()
    }

    /// Equivalent to [`local_x()`][Transform::local_x()]
    #[inline]
    pub fn right(&self) -> Vec3 {
        self.local_x()
    }

    /// Get the unit vector in the local `Y` direction.
    #[inline]
    pub fn local_y(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }

    /// Equivalent to [`local_y()`][Transform::local_y]
    #[inline]
    pub fn up(&self) -> Vec3 {
        self.local_y()
    }

    /// Equivalent to [`-local_y()`][Transform::local_y]
    #[inline]
    pub fn down(&self) -> Vec3 {
        -self.local_y()
    }

    /// Get the unit vector in the local `Z` direction.
    #[inline]
    pub fn local_z(&self) -> Vec3 {
        self.rotation * Vec3::Z
    }

    /// Equivalent to [`-local_z()`][Transform::local_z]
    #[inline]
    pub fn forward(&self) -> Vec3 {
        -self.local_z()
    }

    /// Equivalent to [`local_z()`][Transform::local_z]
    #[inline]
    pub fn back(&self) -> Vec3 {
        self.local_z()
    }

    /// Rotates this [`Transform`] by the given rotation.
    ///
    /// If this [`Transform`] has a parent, the `rotation` is relative to the rotation of the parent.
    ///
    /// # Examples
    ///
    /// - [`3d_rotation`]
    ///
    /// [`3d_rotation`]: https://github.com/bevyengine/bevy/blob/latest/examples/transforms/3d_rotation.rs
    #[inline]
    pub fn rotate(&mut self, rotation: Quat) {
        self.rotation = rotation * self.rotation;
    }

    /// Rotates this [`Transform`] around the given `axis` by `angle` (in radians).
    ///
    /// If this [`Transform`] has a parent, the `axis` is relative to the rotation of the parent.
    #[inline]
    pub fn rotate_axis(&mut self, axis: Vec3, angle: f32) {
        self.rotate(Quat::from_axis_angle(axis, angle));
    }

    /// Rotates this [`Transform`] around the `X` axis by `angle` (in radians).
    ///
    /// If this [`Transform`] has a parent, the axis is relative to the rotation of the parent.
    #[inline]
    pub fn rotate_x(&mut self, angle: f32) {
        self.rotate(Quat::from_rotation_x(angle));
    }

    /// Rotates this [`Transform`] around the `Y` axis by `angle` (in radians).
    ///
    /// If this [`Transform`] has a parent, the axis is relative to the rotation of the parent.
    #[inline]
    pub fn rotate_y(&mut self, angle: f32) {
        self.rotate(Quat::from_rotation_y(angle));
    }

    /// Rotates this [`Transform`] around the `Z` axis by `angle` (in radians).
    ///
    /// If this [`Transform`] has a parent, the axis is relative to the rotation of the parent.
    #[inline]
    pub fn rotate_z(&mut self, angle: f32) {
        self.rotate(Quat::from_rotation_z(angle));
    }

    /// Rotates this [`Transform`] by the given `rotation`.
    ///
    /// The `rotation` is relative to this [`Transform`]'s current rotation.
    #[inline]
    pub fn rotate_local(&mut self, rotation: Quat) {
        self.rotation *= rotation;
    }

    /// Rotates this [`Transform`] around its local `axis` by `angle` (in radians).
    #[inline]
    pub fn rotate_local_axis(&mut self, axis: Vec3, angle: f32) {
        self.rotate_local(Quat::from_axis_angle(axis, angle));
    }

    /// Rotates this [`Transform`] around its local `X` axis by `angle` (in radians).
    #[inline]
    pub fn rotate_local_x(&mut self, angle: f32) {
        self.rotate_local(Quat::from_rotation_x(angle));
    }

    /// Rotates this [`Transform`] around its local `Y` axis by `angle` (in radians).
    #[inline]
    pub fn rotate_local_y(&mut self, angle: f32) {
        self.rotate_local(Quat::from_rotation_y(angle));
    }

    /// Rotates this [`Transform`] around its local `Z` axis by `angle` (in radians).
    #[inline]
    pub fn rotate_local_z(&mut self, angle: f32) {
        self.rotate_local(Quat::from_rotation_z(angle));
    }

    /// Translates this [`Transform`] around a `point` in space.
    ///
    /// If this [`Transform`] has a parent, the `point` is relative to the [`Transform`] of the parent.
    #[inline]
    pub fn translate_around(&mut self, point: Vec3, rotation: Quat) {
        self.translation = point + rotation * (self.translation - point);
    }

    /// Rotates this [`Transform`] around a `point` in space.
    ///
    /// If this [`Transform`] has a parent, the `point` is relative to the [`Transform`] of the parent.
    #[inline]
    pub fn rotate_around(&mut self, point: Vec3, rotation: Quat) {
        self.translate_around(point, rotation);
        self.rotate(rotation);
    }

    /// Rotates this [`Transform`] so that [`Transform::forward`] points towards the `target` position,
    /// and [`Transform::up`] points towards `up`.
    ///
    /// In some cases it's not possible to construct a rotation. Another axis will be picked in those cases:
    /// * if `target` is the same as the transform translation, `Vec3::Z` is used instead
    /// * if `up` is zero, `Vec3::Y` is used instead
    /// * if the resulting forward direction is parallel with `up`, an orthogonal vector is used as the "right" direction
    #[inline]
    pub fn look_at(&mut self, target: Vec3, up: Vec3) {
        self.look_to(target - self.translation, up);
    }

    /// Rotates this [`Transform`] so that [`Transform::forward`] points in the given `direction`
    /// and [`Transform::up`] points towards `up`.
    ///
    /// In some cases it's not possible to construct a rotation. Another axis will be picked in those cases:
    /// * if `direction` is zero, `Vec3::NEG_Z` is used instead
    /// * if `up` is zero, `Vec3::Y` is used instead
    /// * if `direction` is parallel with `up`, an orthogonal vector is used as the "right" direction
    #[inline]
    pub fn look_to(&mut self, direction: Vec3, up: Vec3) {
        let back = -direction.try_normalize().unwrap_or(Vec3::NEG_Z);
        let up = up.try_normalize().unwrap_or(Vec3::Y);
        let right = up
            .cross(back)
            .try_normalize()
            .unwrap_or_else(|| up.any_orthonormal_vector());
        let up = back.cross(right);
        self.rotation = Quat::from_mat3(&Mat3::from_cols(right, up, back));
    }

    /// Multiplies `self` with `transform` component by component, returning the
    /// resulting [`Transform`]
    #[inline]
    #[must_use]
    pub fn mul_transform(&self, transform: Transform) -> Self {
        let translation = self.transform_point(transform.translation);
        let rotation = self.rotation * transform.rotation;
        let scale = self.scale * transform.scale;
        Transform {
            translation,
            rotation,
            scale,
        }
    }

    /// Transforms the given `point`, applying scale, rotation and translation.
    ///
    /// If this [`Transform`] has a parent, this will transform a `point` that is
    /// relative to the parent's [`Transform`] into one relative to this [`Transform`].
    ///
    /// If this [`Transform`] does not have a parent, this will transform a `point`
    /// that is in global space into one relative to this [`Transform`].
    ///
    /// If you want to transform a `point` in global space to the local space of this [`Transform`],
    /// consider using [`GlobalTransform::transform_point()`] instead.
    #[inline]
    pub fn transform_point(&self, mut point: Vec3) -> Vec3 {
        point = self.scale * point;
        point = self.rotation * point;
        point += self.translation;
        point
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mul<Transform> for Transform {
    type Output = Transform;

    fn mul(self, transform: Transform) -> Self::Output {
        self.mul_transform(transform)
    }
}

impl Mul<Vec3> for Transform {
    type Output = Vec3;

    fn mul(self, value: Vec3) -> Self::Output {
        self.transform_point(value)
    }
}
