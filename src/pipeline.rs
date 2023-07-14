use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
};

use ash::{
    vk::{self, CullModeFlags, DescriptorType, FrontFace, PolygonMode, PrimitiveTopology},
    Device,
};
use serde::ser::SerializeStruct;

use crate::memory::ImmutableShaderInfo;

use super::shaders::Shader;

#[repr(C)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct StencilState {
    /// Front face mode.
    pub front: StencilFaceState,
    /// Back face mode.
    pub back: StencilFaceState,
    /// Stencil values are AND'd with this mask when reading and writing from the stencil buffer. Only low 8 bits are used.
    pub read_mask: u32,
    /// Stencil values are AND'd with this mask when writing to the stencil buffer. Only low 8 bits are used.
    pub write_mask: u32,
}

impl StencilState {
    /// Returns true if the stencil test is enabled.
    pub fn is_enabled(&self) -> bool {
        (self.front != StencilFaceState::IGNORE || self.back != StencilFaceState::IGNORE)
            && (self.read_mask != 0 || self.write_mask != 0)
    }
    /// Returns true if the state doesn't mutate the target values.
    pub fn is_read_only(&self, cull_mode: CullModeFlags) -> bool {
        if self.write_mask == 0 {
            return true;
        }

        let front_ro = cull_mode == CullModeFlags::FRONT || self.front.is_read_only();
        let back_ro = cull_mode == CullModeFlags::BACK || self.back.is_read_only();

        front_ro && back_ro
    }
    /// Returns true if the stencil state uses the reference value for testing.
    pub fn needs_ref_value(&self) -> bool {
        self.front.needs_ref_value() || self.back.needs_ref_value()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DepthBiasState {
    /// Constant depth biasing factor, in basic units of the depth format.
    pub constant: i32,
    /// Slope depth biasing factor.
    pub slope_scale: f32,
    /// Depth bias clamp value (absolute).
    pub clamp: f32,
}

impl DepthBiasState {
    /// Returns true if the depth biasing is enabled.
    pub fn is_enabled(&self) -> bool {
        self.constant != 0 || self.slope_scale != 0.0
    }
}

impl Hash for DepthBiasState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.constant.hash(state);
        self.slope_scale.to_bits().hash(state);
        self.clamp.to_bits().hash(state);
    }
}

impl PartialEq for DepthBiasState {
    fn eq(&self, other: &Self) -> bool {
        (self.constant == other.constant)
            && (self.slope_scale.to_bits() == other.slope_scale.to_bits())
            && (self.clamp.to_bits() == other.clamp.to_bits())
    }
}

impl Eq for DepthBiasState {}

#[repr(C)]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct DepthStencilState {
    pub format: vk::Format,
    /// If disabled, depth will not be written to.
    pub depth_write_enabled: bool,
    /// Comparison function used to compare depth values in the depth test.
    pub depth_compare: CompareFunction,
    /// Stencil state.
    #[cfg_attr(any(feature = "trace", feature = "replay"), serde(default))]
    pub stencil: StencilState,
    /// Depth bias state.
    #[cfg_attr(any(feature = "trace", feature = "replay"), serde(default))]
    pub bias: DepthBiasState,
}

impl DepthStencilState {
    /// Returns true if the depth testing is enabled.
    pub fn is_depth_enabled(&self) -> bool {
        self.depth_compare != CompareFunction::Always || self.depth_write_enabled
    }

    /// Returns true if the state doesn't mutate the depth buffer.
    pub fn is_depth_read_only(&self) -> bool {
        !self.depth_write_enabled
    }

    /// Returns true if the state doesn't mutate the stencil.
    pub fn is_stencil_read_only(&self, cull_mode: CullModeFlags) -> bool {
        self.stencil.is_read_only(cull_mode)
    }

    /// Returns true if the state doesn't mutate either depth or stencil of the target.
    pub fn is_read_only(&self, cull_mode: CullModeFlags) -> bool {
        self.is_depth_read_only() && self.is_stencil_read_only(cull_mode)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub enum IndexFormat {
    /// Indices are 16 bit unsigned integers.
    Uint16 = 0,
    /// Indices are 32 bit unsigned integers.
    #[default]
    Uint32 = 1,
}

/// Operation to perform on the stencil value.
///
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub enum StencilOperation {
    /// Keep stencil value unchanged.
    #[default]
    Keep = 0,
    /// Set stencil value to zero.
    Zero = 1,
    /// Replace stencil value with value provided in most recent call to
    /// [`RenderPass::set_stencil_reference`][RPssr].
    ///
    /// [RPssr]: ../wgpu/struct.RenderPass.html#method.set_stencil_reference
    Replace = 2,
    /// Bitwise inverts stencil value.
    Invert = 3,
    /// Increments stencil value by one, clamping on overflow.
    IncrementClamp = 4,
    /// Decrements stencil value by one, clamping on underflow.
    DecrementClamp = 5,
    /// Increments stencil value by one, wrapping on overflow.
    IncrementWrap = 6,
    /// Decrements stencil value by one, wrapping on underflow.
    DecrementWrap = 7,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StencilFaceState {
    /// Comparison function that determines if the fail_op or pass_op is used on the stencil buffer.
    pub compare: CompareFunction,
    /// Operation that is preformed when stencil test fails.
    pub fail_op: StencilOperation,
    /// Operation that is performed when depth test fails but stencil test succeeds.
    pub depth_fail_op: StencilOperation,
    /// Operation that is performed when stencil test success.
    pub pass_op: StencilOperation,
}

impl StencilFaceState {
    /// Ignore the stencil state for the face.
    pub const IGNORE: Self = StencilFaceState {
        compare: CompareFunction::Always,
        fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op: StencilOperation::Keep,
    };

    /// Returns true if the face state uses the reference value for testing or operation.
    pub fn needs_ref_value(&self) -> bool {
        self.compare.needs_ref_value()
            || self.fail_op == StencilOperation::Replace
            || self.depth_fail_op == StencilOperation::Replace
            || self.pass_op == StencilOperation::Replace
    }

    /// Returns true if the face state doesn't mutate the target values.
    pub fn is_read_only(&self) -> bool {
        self.pass_op == StencilOperation::Keep
            && self.depth_fail_op == StencilOperation::Keep
            && self.fail_op == StencilOperation::Keep
    }
}

impl Default for StencilFaceState {
    fn default() -> Self {
        Self::IGNORE
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum CompareFunction {
    /// Function never passes
    Never = 1,
    /// Function passes if new value less than existing value
    Less = 2,
    /// Function passes if new value is equal to existing value. When using
    /// this compare function, make sure to mark your Vertex Shader's `@builtin(position)`
    /// output as `@invariant` to prevent artifacting.
    Equal = 3,
    /// Function passes if new value is less than or equal to existing value
    LessEqual = 4,
    /// Function passes if new value is greater than existing value
    Greater = 5,
    /// Function passes if new value is not equal to existing value. When using
    /// this compare function, make sure to mark your Vertex Shader's `@builtin(position)`
    /// output as `@invariant` to prevent artifacting.
    NotEqual = 6,
    /// Function passes if new value is greater than or equal to existing value
    GreaterEqual = 7,
    /// Function always passes
    Always = 8,
}

impl CompareFunction {
    /// Returns true if the comparison depends on the reference value.
    pub fn needs_ref_value(self) -> bool {
        match self {
            Self::Never | Self::Always => false,
            _ => true,
        }
    }
}

pub fn map_comparison(fun: CompareFunction) -> vk::CompareOp {
    match fun {
        CompareFunction::Never => vk::CompareOp::NEVER,
        CompareFunction::Less => vk::CompareOp::LESS,
        CompareFunction::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        CompareFunction::Equal => vk::CompareOp::EQUAL,
        CompareFunction::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        CompareFunction::Greater => vk::CompareOp::GREATER,
        CompareFunction::NotEqual => vk::CompareOp::NOT_EQUAL,
        CompareFunction::Always => vk::CompareOp::ALWAYS,
    }
}

pub fn map_stencil_op(op: StencilOperation) -> vk::StencilOp {
    match op {
        StencilOperation::Keep => vk::StencilOp::KEEP,
        StencilOperation::Zero => vk::StencilOp::ZERO,
        StencilOperation::Replace => vk::StencilOp::REPLACE,
        StencilOperation::Invert => vk::StencilOp::INVERT,
        StencilOperation::IncrementClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
        StencilOperation::IncrementWrap => vk::StencilOp::INCREMENT_AND_WRAP,
        StencilOperation::DecrementClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
        StencilOperation::DecrementWrap => vk::StencilOp::DECREMENT_AND_WRAP,
    }
}

pub fn map_stencil_face(
    face: &StencilFaceState,
    compare_mask: u32,
    write_mask: u32,
) -> vk::StencilOpState {
    vk::StencilOpState {
        fail_op: map_stencil_op(face.fail_op),
        pass_op: map_stencil_op(face.pass_op),
        depth_fail_op: map_stencil_op(face.depth_fail_op),
        compare_op: map_comparison(face.compare),
        compare_mask,
        write_mask,
        reference: 0,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PrimitiveState {
    /// The primitive topology used to interpret vertices.
    pub topology: PrimitiveTopology,
    /// The face to consider the front for the purpose of culling and stencil operations.
    pub front_face: FrontFace,
    /// The face culling mode.
    pub cull_mode: CullModeFlags,
    pub unclipped_depth: bool,
    pub polygon_mode: PolygonMode,
    pub conservative: bool,
}

pub struct GraphicsPipelineDescriptor<'a> {
    pub vertex_shader: Shader,
    pub fragment_shader: Shader,
    pub vertex_input: vk::PipelineVertexInputStateCreateInfo<'a>,
    pub color_attachment_formats: &'a [vk::Format],
    pub depth_attachment_format: vk::Format,
    pub viewport: vk::Extent2D,
    pub primitive: PrimitiveState,
    pub depth_stencil: Option<DepthStencilState>,
    pub push_constant_range: Option<vk::PushConstantRange>,
}

impl serde::ser::Serialize for GraphicsPipelineDescriptor<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_struct("GraphicsPipelineDescriptor", 3)?;
        seq.serialize_field("vertex_shader", &self.vertex_shader.spirv)
            .unwrap();
        seq.serialize_field("fragment_shader", &self.fragment_shader.spirv)
            .unwrap();
        seq.end()
    }
}

#[derive(Debug)]
pub struct GraphicsPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pub set_layout_info: Vec<HashMap<u32, DescriptorType>>,
    pub color_attachments: Vec<vk::Format>,
    pub depth_attachment: vk::Format,
    pub viewports: Vec<vk::Viewport>,
    pub scissors: Vec<vk::Rect2D>,
}

impl GraphicsPipeline {
    pub fn new(
        device: &Device,
        desc: GraphicsPipelineDescriptor,
        shader_info: &ImmutableShaderInfo,
    ) -> Self {
        let multisample_state_info: vk::PipelineMultisampleStateCreateInfo<'_> =
            vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };

        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
            blend_enable: 0,
            src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ZERO,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        }];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op(vk::LogicOp::CLEAR)
            .attachments(&color_blend_attachment_states);

        let dynamic_state = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::BLEND_CONSTANTS,
            vk::DynamicState::STENCIL_REFERENCE,
        ];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_state);

        let (descriptor_set_layouts, set_layout_info) = desc
            .fragment_shader
            .create_descriptor_set_layouts(device, shader_info);

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&descriptor_set_layouts)
                        .push_constant_ranges(
                            desc.push_constant_range
                                .as_ref()
                                .map_or(&[], |range| std::slice::from_ref(range)),
                        ),
                    None,
                )
                .unwrap()
        };

        let mut rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(desc.primitive.polygon_mode)
            .front_face(desc.primitive.front_face)
            .line_width(1.0)
            .depth_clamp_enable(desc.primitive.unclipped_depth)
            .cull_mode(desc.primitive.cull_mode);

        let mut rasterization_conservative_state =
            vk::PipelineRasterizationConservativeStateCreateInfoEXT::default()
                .conservative_rasterization_mode(
                    vk::ConservativeRasterizationModeEXT::OVERESTIMATE,
                );

        if desc.primitive.conservative {
            rasterization = rasterization.push_next(&mut rasterization_conservative_state);
        }

        let mut depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
        if let Some(ref ds) = desc.depth_stencil {
            if ds.is_depth_enabled() {
                depth_stencil = depth_stencil
                    .depth_test_enable(true)
                    .depth_write_enable(ds.depth_write_enabled)
                    .depth_compare_op(map_comparison(ds.depth_compare));
            }
            if ds.stencil.is_enabled() {
                let s = &ds.stencil;
                let front = map_stencil_face(&s.front, s.read_mask, s.write_mask);
                let back = map_stencil_face(&s.back, s.read_mask, s.write_mask);
                depth_stencil = depth_stencil
                    .stencil_test_enable(true)
                    .front(front)
                    .back(back);
            }

            if ds.bias.is_enabled() {
                rasterization = rasterization
                    .depth_bias_enable(true)
                    .depth_bias_constant_factor(ds.bias.constant as f32)
                    .depth_bias_clamp(ds.bias.clamp)
                    .depth_bias_slope_factor(ds.bias.slope_scale);
            }
        }

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .name(&desc.vertex_shader.entry_point_cstr)
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(desc.vertex_shader.module),
            vk::PipelineShaderStageCreateInfo::default()
                .name(&desc.fragment_shader.entry_point_cstr)
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(desc.fragment_shader.module),
        ];

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(desc.primitive.topology)
            .primitive_restart_enable(false);

        let viewports = vec![vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: desc.viewport.width as f32,
            height: desc.viewport.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        let scissors = vec![desc.viewport.into()];
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .scissors(&scissors)
            .viewports(&viewports);

        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(desc.color_attachment_formats)
            .depth_attachment_format(desc.depth_attachment_format);

        let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&desc.vertex_input)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .depth_stencil_state(&depth_stencil)
            .dynamic_state(&dynamic_state_info)
            .layout(pipeline_layout)
            // todo
            .multisample_state(&multisample_state_info)
            .color_blend_state(&color_blend_state)
            .push_next(&mut rendering_info);

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[graphic_pipeline_info],
                    None,
                )
                .unwrap()[0]
        };

        let descriptor_sets = if !set_layout_info.is_empty() {
            desc.fragment_shader.create_descriptor_sets(
                device,
                shader_info,
                &descriptor_set_layouts,
                &set_layout_info,
            )
        } else {
            vec![]
        };

        Self {
            pipeline,
            layout: pipeline_layout,
            descriptor_set_layouts,
            set_layout_info,
            descriptor_sets,
            color_attachments: desc.color_attachment_formats.to_vec(),
            depth_attachment: desc.depth_attachment_format,
            viewports,
            scissors,
        }
    }

    pub fn bind(&self, device: &Device, command_buffer: vk::CommandBuffer) {
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            device.cmd_set_viewport(command_buffer, 0, &self.viewports);
            device.cmd_set_scissor(command_buffer, 0, &self.scissors);
        }
    }
}
