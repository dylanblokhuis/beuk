use std::{
    borrow::Cow,
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::Arc,
};

use ash::{
    vk::{self, CullModeFlags, DescriptorType, FrontFace, PolygonMode, PrimitiveTopology},
    Device,
};
use smallvec::SmallVec;

use crate::{
    ctx::RenderContext,
    memory::{ResourceHandle, ResourceHooks, ResourceManager},
    shaders::ImmutableShaderInfo,
};

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

impl From<IndexFormat> for vk::IndexType {
    fn from(val: IndexFormat) -> Self {
        match val {
            IndexFormat::Uint16 => vk::IndexType::UINT16,
            IndexFormat::Uint32 => vk::IndexType::UINT32,
        }
    }
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
        !matches!(self, Self::Never | Self::Always)
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

/// Alpha blend factor.
///
/// Alpha blending is very complicated: see the OpenGL or Vulkan spec for more information.
///
/// Corresponds to [WebGPU `GPUBlendFactor`](
/// https://gpuweb.github.io/gpuweb/#enumdef-gpublendfactor).
#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum BlendFactor {
    /// 0.0
    Zero = 0,
    /// 1.0
    One = 1,
    /// S.component
    Src = 2,
    /// 1.0 - S.component
    OneMinusSrc = 3,
    /// S.alpha
    SrcAlpha = 4,
    /// 1.0 - S.alpha
    OneMinusSrcAlpha = 5,
    /// D.component
    Dst = 6,
    /// 1.0 - D.component
    OneMinusDst = 7,
    /// D.alpha
    DstAlpha = 8,
    /// 1.0 - D.alpha
    OneMinusDstAlpha = 9,
    /// min(S.alpha, 1.0 - D.alpha)
    SrcAlphaSaturated = 10,
    /// Constant
    Constant = 11,
    /// 1.0 - Constant
    OneMinusConstant = 12,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub enum BlendOperation {
    /// Src + Dst
    #[default]
    Add = 0,
    /// Src - Dst
    Subtract = 1,
    /// Dst - Src
    ReverseSubtract = 2,
    /// min(Src, Dst)
    Min = 3,
    /// max(Src, Dst)
    Max = 4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlendComponent {
    /// Multiplier for the source, which is produced by the fragment shader.
    pub src_factor: BlendFactor,
    /// Multiplier for the destination, which is stored in the target.
    pub dst_factor: BlendFactor,
    /// The binary operation applied to the source and destination,
    /// multiplied by their respective factors.
    pub operation: BlendOperation,
}

impl BlendComponent {
    /// Default blending state that replaces destination with the source.
    pub const REPLACE: Self = Self {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::Zero,
        operation: BlendOperation::Add,
    };

    /// Blend state of (1 * src) + ((1 - src_alpha) * dst)
    pub const OVER: Self = Self {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::OneMinusSrcAlpha,
        operation: BlendOperation::Add,
    };

    /// Returns true if the state relies on the constant color, which is
    /// set independently on a render command encoder.
    pub fn uses_constant(&self) -> bool {
        match (self.src_factor, self.dst_factor) {
            (BlendFactor::Constant, _)
            | (BlendFactor::OneMinusConstant, _)
            | (_, BlendFactor::Constant)
            | (_, BlendFactor::OneMinusConstant) => true,
            (_, _) => false,
        }
    }
}

impl Default for BlendComponent {
    fn default() -> Self {
        Self::REPLACE
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlendState {
    /// Color equation.
    pub color: BlendComponent,
    /// Alpha equation.
    pub alpha: BlendComponent,
}

impl BlendState {
    /// Blend mode that does no color blending, just overwrites the output with the contents of the shader.
    pub const REPLACE: Self = Self {
        color: BlendComponent::REPLACE,
        alpha: BlendComponent::REPLACE,
    };

    /// Blend mode that does standard alpha blending with non-premultiplied alpha.
    pub const ALPHA_BLENDING: Self = Self {
        color: BlendComponent {
            src_factor: BlendFactor::SrcAlpha,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent::OVER,
    };

    /// Blend mode that does standard alpha blending with premultiplied alpha.
    pub const PREMULTIPLIED_ALPHA_BLENDING: Self = Self {
        color: BlendComponent::OVER,
        alpha: BlendComponent::OVER,
    };
}

fn map_blend_factor(factor: BlendFactor) -> vk::BlendFactor {
    match factor {
        BlendFactor::Zero => vk::BlendFactor::ZERO,
        BlendFactor::One => vk::BlendFactor::ONE,
        BlendFactor::Src => vk::BlendFactor::SRC_COLOR,
        BlendFactor::OneMinusSrc => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        BlendFactor::Dst => vk::BlendFactor::DST_COLOR,
        BlendFactor::OneMinusDst => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
        BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        BlendFactor::SrcAlphaSaturated => vk::BlendFactor::SRC_ALPHA_SATURATE,
        BlendFactor::Constant => vk::BlendFactor::CONSTANT_COLOR,
        BlendFactor::OneMinusConstant => vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
    }
}

fn map_blend_op(operation: BlendOperation) -> vk::BlendOp {
    match operation {
        BlendOperation::Add => vk::BlendOp::ADD,
        BlendOperation::Subtract => vk::BlendOp::SUBTRACT,
        BlendOperation::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        BlendOperation::Min => vk::BlendOp::MIN,
        BlendOperation::Max => vk::BlendOp::MAX,
    }
}

pub fn map_blend_component(
    component: &BlendComponent,
) -> (vk::BlendOp, vk::BlendFactor, vk::BlendFactor) {
    let op = map_blend_op(component.operation);
    let src = map_blend_factor(component.src_factor);
    let dst = map_blend_factor(component.dst_factor);
    (op, src, dst)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MultisampleState {
    /// The number of samples calculated per pixel (for MSAA). For non-multisampled textures,
    /// this should be `1`
    pub count: u32,
    /// Bitmask that restricts the samples of a pixel modified by this pipeline. All samples
    /// can be enabled using the value `!0`
    pub mask: u64,
    /// When enabled, produces another sample mask per pixel based on the alpha output value, that
    /// is ANDed with the sample_mask and the primitive coverage to restrict the set of samples
    /// affected by a primitive.
    ///
    /// The implicit mask produced for alpha of zero is guaranteed to be zero, and for alpha of one
    /// is guaranteed to be all 1-s.
    pub alpha_to_coverage_enabled: bool,
}

impl Default for MultisampleState {
    fn default() -> Self {
        MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Extent2d {
    pub width: u32,
    pub height: u32,
}

impl From<vk::Extent2D> for Extent2d {
    fn from(extent: vk::Extent2D) -> Self {
        Extent2d {
            width: extent.width,
            height: extent.height,
        }
    }
}

impl From<Extent2d> for vk::Extent2D {
    fn from(val: Extent2d) -> Self {
        vk::Extent2D {
            width: val.width,
            height: val.height,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Extent3d {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl From<vk::Extent3D> for Extent3d {
    fn from(extent: vk::Extent3D) -> Self {
        Extent3d {
            width: extent.width,
            height: extent.height,
            depth: extent.depth,
        }
    }
}

impl From<Extent3d> for vk::Extent3D {
    fn from(val: Extent3d) -> Self {
        vk::Extent3D {
            width: val.width,
            height: val.height,
            depth: val.depth,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ShaderStages {
    None,
    Vertex,
    Fragment,
    Compute,
    AllGraphics,
}

impl From<ShaderStages> for vk::ShaderStageFlags {
    fn from(val: ShaderStages) -> Self {
        match val {
            ShaderStages::None => vk::ShaderStageFlags::empty(),
            ShaderStages::Vertex => vk::ShaderStageFlags::VERTEX,
            ShaderStages::Fragment => vk::ShaderStageFlags::FRAGMENT,
            ShaderStages::Compute => vk::ShaderStageFlags::COMPUTE,
            ShaderStages::AllGraphics => vk::ShaderStageFlags::ALL_GRAPHICS,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PushConstantRange {
    pub stages: ShaderStages,
    pub range: u32,
    pub offset: u32,
}

impl From<PushConstantRange> for vk::PushConstantRange {
    fn from(val: PushConstantRange) -> Self {
        vk::PushConstantRange {
            stage_flags: val.stages.into(),
            offset: val.offset,
            size: val.range,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Hash, Eq, PartialEq)]
pub enum VertexStepMode {
    /// Vertex data is advanced every vertex.
    #[default]
    Vertex = 0,
    /// Vertex data is advanced every instance.
    Instance = 1,
}

/// [`vertex_attr_array`]: ../wgpu/macro.vertex_attr_array.html
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexAttribute {
    /// Format of the input
    pub format: vk::Format,
    /// Byte offset of the start of the input
    pub offset: u32,
    /// Location for this input. Must match the location in the shader.
    pub shader_location: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum VertexFormat {
    /// Two unsigned bytes (u8). `vec2<u32>` in shaders.
    Uint8x2 = 0,
    /// Four unsigned bytes (u8). `vec4<u32>` in shaders.
    Uint8x4 = 1,
    /// Two signed bytes (i8). `vec2<i32>` in shaders.
    Sint8x2 = 2,
    /// Four signed bytes (i8). `vec4<i32>` in shaders.
    Sint8x4 = 3,
    /// Two unsigned bytes (u8). [0, 255] converted to float [0, 1] `vec2<f32>` in shaders.
    Unorm8x2 = 4,
    /// Four unsigned bytes (u8). [0, 255] converted to float [0, 1] `vec4<f32>` in shaders.
    Unorm8x4 = 5,
    /// Two signed bytes (i8). [-127, 127] converted to float [-1, 1] `vec2<f32>` in shaders.
    Snorm8x2 = 6,
    /// Four signed bytes (i8). [-127, 127] converted to float [-1, 1] `vec4<f32>` in shaders.
    Snorm8x4 = 7,
    /// Two unsigned shorts (u16). `vec2<u32>` in shaders.
    Uint16x2 = 8,
    /// Four unsigned shorts (u16). `vec4<u32>` in shaders.
    Uint16x4 = 9,
    /// Two signed shorts (i16). `vec2<i32>` in shaders.
    Sint16x2 = 10,
    /// Four signed shorts (i16). `vec4<i32>` in shaders.
    Sint16x4 = 11,
    /// Two unsigned shorts (u16). [0, 65535] converted to float [0, 1] `vec2<f32>` in shaders.
    Unorm16x2 = 12,
    /// Four unsigned shorts (u16). [0, 65535] converted to float [0, 1] `vec4<f32>` in shaders.
    Unorm16x4 = 13,
    /// Two signed shorts (i16). [-32767, 32767] converted to float [-1, 1] `vec2<f32>` in shaders.
    Snorm16x2 = 14,
    /// Four signed shorts (i16). [-32767, 32767] converted to float [-1, 1] `vec4<f32>` in shaders.
    Snorm16x4 = 15,
    /// Two half-precision floats (no Rust equiv). `vec2<f32>` in shaders.
    Float16x2 = 16,
    /// Four half-precision floats (no Rust equiv). `vec4<f32>` in shaders.
    Float16x4 = 17,
    /// One single-precision float (f32). `f32` in shaders.
    Float32 = 18,
    /// Two single-precision floats (f32). `vec2<f32>` in shaders.
    Float32x2 = 19,
    /// Three single-precision floats (f32). `vec3<f32>` in shaders.
    Float32x3 = 20,
    /// Four single-precision floats (f32). `vec4<f32>` in shaders.
    Float32x4 = 21,
    /// One unsigned int (u32). `u32` in shaders.
    Uint32 = 22,
    /// Two unsigned ints (u32). `vec2<u32>` in shaders.
    Uint32x2 = 23,
    /// Three unsigned ints (u32). `vec3<u32>` in shaders.
    Uint32x3 = 24,
    /// Four unsigned ints (u32). `vec4<u32>` in shaders.
    Uint32x4 = 25,
    /// One signed int (i32). `i32` in shaders.
    Sint32 = 26,
    /// Two signed ints (i32). `vec2<i32>` in shaders.
    Sint32x2 = 27,
    /// Three signed ints (i32). `vec3<i32>` in shaders.
    Sint32x3 = 28,
    /// Four signed ints (i32). `vec4<i32>` in shaders.
    Sint32x4 = 29,
    /// One double-precision float (f64). `f32` in shaders. Requires [`Features::VERTEX_ATTRIBUTE_64BIT`].
    Float64 = 30,
    /// Two double-precision floats (f64). `vec2<f32>` in shaders. Requires [`Features::VERTEX_ATTRIBUTE_64BIT`].
    Float64x2 = 31,
    /// Three double-precision floats (f64). `vec3<f32>` in shaders. Requires [`Features::VERTEX_ATTRIBUTE_64BIT`].
    Float64x3 = 32,
    /// Four double-precision floats (f64). `vec4<f32>` in shaders. Requires [`Features::VERTEX_ATTRIBUTE_64BIT`].
    Float64x4 = 33,
}

impl From<VertexFormat> for vk::Format {
    fn from(val: VertexFormat) -> Self {
        match val {
            VertexFormat::Uint8x2 => vk::Format::R8G8_UINT,
            VertexFormat::Uint8x4 => vk::Format::R8G8B8A8_UINT,
            VertexFormat::Sint8x2 => vk::Format::R8G8_SINT,
            VertexFormat::Sint8x4 => vk::Format::R8G8B8A8_SINT,
            VertexFormat::Unorm8x2 => vk::Format::R8G8_UNORM,
            VertexFormat::Unorm8x4 => vk::Format::R8G8B8A8_UNORM,
            VertexFormat::Snorm8x2 => vk::Format::R8G8_SNORM,
            VertexFormat::Snorm8x4 => vk::Format::R8G8B8A8_SNORM,
            VertexFormat::Uint16x2 => vk::Format::R16G16_UINT,
            VertexFormat::Uint16x4 => vk::Format::R16G16B16A16_UINT,
            VertexFormat::Sint16x2 => vk::Format::R16G16_SINT,
            VertexFormat::Sint16x4 => vk::Format::R16G16B16A16_SINT,
            VertexFormat::Unorm16x2 => vk::Format::R16G16_UNORM,
            VertexFormat::Unorm16x4 => vk::Format::R16G16B16A16_UNORM,
            VertexFormat::Snorm16x2 => vk::Format::R16G16_SNORM,
            VertexFormat::Snorm16x4 => vk::Format::R16G16B16A16_SNORM,
            VertexFormat::Float16x2 => vk::Format::R16G16_SFLOAT,
            VertexFormat::Float16x4 => vk::Format::R16G16B16A16_SFLOAT,
            VertexFormat::Float32 => vk::Format::R32_SFLOAT,
            VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
            VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
            VertexFormat::Float32x4 => vk::Format::R32G32B32A32_SFLOAT,
            VertexFormat::Uint32 => vk::Format::R32_UINT,
            VertexFormat::Uint32x2 => vk::Format::R32G32_UINT,
            VertexFormat::Uint32x3 => vk::Format::R32G32B32_UINT,
            VertexFormat::Uint32x4 => vk::Format::R32G32B32A32_UINT,
            VertexFormat::Sint32 => vk::Format::R32_SINT,
            VertexFormat::Sint32x2 => vk::Format::R32G32_SINT,
            VertexFormat::Sint32x3 => vk::Format::R32G32B32_SINT,
            VertexFormat::Sint32x4 => vk::Format::R32G32B32A32_SINT,
            VertexFormat::Float64 => vk::Format::R64_SFLOAT,
            VertexFormat::Float64x2 => vk::Format::R64G64_SFLOAT,
            VertexFormat::Float64x3 => vk::Format::R64G64B64_SFLOAT,
            VertexFormat::Float64x4 => vk::Format::R64G64B64A64_SFLOAT,
        }
    }
}

impl VertexFormat {
    /// Returns the byte size of the format.
    pub const fn size(&self) -> u64 {
        match self {
            Self::Uint8x2 | Self::Sint8x2 | Self::Unorm8x2 | Self::Snorm8x2 => 2,
            Self::Uint8x4
            | Self::Sint8x4
            | Self::Unorm8x4
            | Self::Snorm8x4
            | Self::Uint16x2
            | Self::Sint16x2
            | Self::Unorm16x2
            | Self::Snorm16x2
            | Self::Float16x2
            | Self::Float32
            | Self::Uint32
            | Self::Sint32 => 4,
            Self::Uint16x4
            | Self::Sint16x4
            | Self::Unorm16x4
            | Self::Snorm16x4
            | Self::Float16x4
            | Self::Float32x2
            | Self::Uint32x2
            | Self::Sint32x2
            | Self::Float64 => 8,
            Self::Float32x3 | Self::Uint32x3 | Self::Sint32x3 => 12,
            Self::Float32x4 | Self::Uint32x4 | Self::Sint32x4 | Self::Float64x2 => 16,
            Self::Float64x3 => 24,
            Self::Float64x4 => 32,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VertexBufferLayout {
    /// The stride, in bytes, between elements of this buffer.
    pub array_stride: u32,
    /// How often this vertex buffer is "stepped" forward.
    pub step_mode: VertexStepMode,
    /// The list of attributes which comprise a single vertex.
    pub attributes: SmallVec<[VertexAttribute; 10]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VertexState {
    pub shader: ResourceHandle<Shader>,
    pub buffers: SmallVec<[VertexBufferLayout; 2]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FragmentState {
    pub shader: ResourceHandle<Shader>,
    pub color_attachment_formats: SmallVec<[vk::Format; 8]>,
    pub depth_attachment_format: vk::Format,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct GraphicsPipelineDescriptor {
    pub vertex: VertexState,
    pub fragment: FragmentState,

    /// Viewport dimensions. If `None`, the viewport will be the size of the swapchain.
    pub viewport: Option<Extent2d>,
    pub primitive: PrimitiveState,
    pub depth_stencil: Option<DepthStencilState>,
    pub push_constant_range: Option<PushConstantRange>,
    /// Blend state for each color attachment
    pub blend: Vec<BlendState>,
    pub multisample: MultisampleState,
}

#[derive(Debug, Default)]
pub struct GraphicsPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pub descriptor_pool: vk::DescriptorPool,
    pub set_layout_info: Vec<HashMap<u32, DescriptorType>>,
    pub color_attachments: Vec<vk::Format>,
    pub depth_attachment: vk::Format,
    pub viewports: Vec<vk::Viewport>,
    pub scissors: Vec<vk::Rect2D>,
}

impl ResourceHooks for GraphicsPipeline {
    fn cleanup(
        &mut self,
        device: std::sync::Arc<ash::Device>,
        allocator: std::sync::Arc<std::sync::Mutex<gpu_allocator::vulkan::Allocator>>,
    ) {
        drop(allocator);
        self.destroy(&device)
    }

    fn on_swapchain_resize(
        &mut self,
        _ctx: &RenderContext,
        old_surface_resolution: vk::Extent2D,
        new_surface_resolution: vk::Extent2D,
    ) {
        if self.pipeline == Default::default() {
            return;
        }

        if self.viewports.len() != 1 {
            return;
        }

        // If the viewport isnt the old size, then this pipeline doesnt use the swapchain size for its viewport
        if self.viewports[0].width != old_surface_resolution.width as f32
            && self.viewports[0].height != old_surface_resolution.height as f32
        {
            return;
        }

        self.viewports[0] = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: new_surface_resolution.width as f32,
            height: new_surface_resolution.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        self.scissors[0] = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: new_surface_resolution,
        };
    }
}

impl GraphicsPipeline {
    pub fn new(
        device: &Device,
        desc: &GraphicsPipelineDescriptor,
        shader_info: &ImmutableShaderInfo,
        swapchain_size: Extent2d,
        shader_manager: Arc<ResourceManager<Shader>>,
    ) -> Self {
        let vk_sample_mask = [
            desc.multisample.mask as u32,
            (desc.multisample.mask >> 32) as u32,
        ];
        let multisample_state_info: vk::PipelineMultisampleStateCreateInfo<'_> =
            vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::from_raw(desc.multisample.count))
                .alpha_to_coverage_enable(desc.multisample.alpha_to_coverage_enabled)
                .sample_mask(&vk_sample_mask);

        let mut color_blend_attachment_states =
            Vec::with_capacity(desc.fragment.color_attachment_formats.len());
        for _ in desc.fragment.color_attachment_formats.iter() {
            color_blend_attachment_states.push(
                vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::RGBA),
            );
        }

        for (index, blend) in desc.blend.iter().enumerate() {
            let (color_op, color_src, color_dst) = map_blend_component(&blend.color);
            let (alpha_op, alpha_src, alpha_dst) = map_blend_component(&blend.alpha);
            color_blend_attachment_states[index] = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(true)
                .color_blend_op(color_op)
                .src_color_blend_factor(color_src)
                .dst_color_blend_factor(color_dst)
                .alpha_blend_op(alpha_op)
                .src_alpha_blend_factor(alpha_src)
                .dst_alpha_blend_factor(alpha_dst);
        }
        assert!(
            color_blend_attachment_states.len() == desc.fragment.color_attachment_formats.len(),
            "Each color attachment must have a blend state if writing to BlendState"
        );

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&color_blend_attachment_states);

        let dynamic_state = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::BLEND_CONSTANTS,
            vk::DynamicState::STENCIL_REFERENCE,
        ];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_state);

        let fragment_shader = shader_manager.get(&desc.fragment.shader).unwrap();
        let (descriptor_set_layouts, set_layout_info) =
            fragment_shader.create_descriptor_set_layouts(device, shader_info);

        let push_constant_ranges: Vec<vk::PushConstantRange> = desc
            .push_constant_range
            .map_or(vec![], |range| vec![range.into()]);

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&descriptor_set_layouts)
                        .push_constant_ranges(&push_constant_ranges),
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

        let vertex_shader = shader_manager.get(&desc.vertex.shader).unwrap();
        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .name(&vertex_shader.entry_point_cstr)
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_shader.module),
            vk::PipelineShaderStageCreateInfo::default()
                .name(&fragment_shader.entry_point_cstr)
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_shader.module),
        ];

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(desc.primitive.topology)
            .primitive_restart_enable(false);

        let viewport = desc.viewport.unwrap_or(swapchain_size);

        let viewports = vec![vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: viewport.width as f32,
            height: viewport.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        let vk_viewport: vk::Extent2D = viewport.into();
        let scissors = vec![vk_viewport.into()];
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .scissors(&scissors)
            .viewports(&viewports);

        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&desc.fragment.color_attachment_formats)
            .depth_attachment_format(desc.fragment.depth_attachment_format);

        let mut vertex_buffers = Vec::with_capacity(desc.vertex.buffers.len());
        let mut vertex_attributes = Vec::new();

        for (i, vb) in desc.vertex.buffers.iter().enumerate() {
            vertex_buffers.push(vk::VertexInputBindingDescription {
                binding: i as u32,
                stride: vb.array_stride,
                input_rate: match vb.step_mode {
                    VertexStepMode::Vertex => vk::VertexInputRate::VERTEX,
                    VertexStepMode::Instance => vk::VertexInputRate::INSTANCE,
                },
            });
            for at in vb.attributes.iter() {
                vertex_attributes.push(vk::VertexInputAttributeDescription {
                    location: at.shader_location,
                    binding: i as u32,
                    format: at.format,
                    offset: at.offset,
                });
            }
        }

        let vk_vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vertex_buffers)
            .vertex_attribute_descriptions(&vertex_attributes);

        let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vk_vertex_input)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .depth_stencil_state(&depth_stencil)
            .dynamic_state(&dynamic_state_info)
            .layout(pipeline_layout)
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

        let (descriptor_sets, descriptor_pool) = if !set_layout_info.is_empty() {
            fragment_shader.create_descriptor_sets(
                device,
                shader_info,
                &descriptor_set_layouts,
                &set_layout_info,
            )
        } else {
            (vec![], vk::DescriptorPool::null())
        };

        Self {
            pipeline,
            layout: pipeline_layout,
            descriptor_set_layouts,
            set_layout_info,
            descriptor_sets,
            color_attachments: desc.fragment.color_attachment_formats.to_vec(),
            depth_attachment: desc.fragment.depth_attachment_format,
            viewports,
            scissors,
            descriptor_pool,
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
            if !self.descriptor_sets.is_empty() {
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.layout,
                    0,
                    &self.descriptor_sets,
                    &[],
                );
            }
        }
    }

    pub fn destroy(&self, device: &Device) {
        if self.pipeline == Default::default() {
            return;
        }

        unsafe {
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_pipeline(self.pipeline, None);

            if !self.descriptor_sets.is_empty() {
                device
                    .free_descriptor_sets(self.descriptor_pool, &self.descriptor_sets)
                    .unwrap();
            }
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            for layout in self.descriptor_set_layouts.iter() {
                device.destroy_descriptor_set_layout(*layout, None);
            }
        }
    }
}
