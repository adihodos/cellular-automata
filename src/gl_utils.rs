#![allow(dead_code)]

use crate::gl;
use crate::gl::types::{GLbitfield, GLsizei, GLsizeiptr, GLuint};
use std::ffi::CString;

use std::path::Path;
use std::result::Result;

pub trait ResourceDeleter<T>
where
    T: Copy + std::cmp::PartialEq,
{
    fn null() -> T;

    fn is_null(r: T) -> bool {
        r == Self::null()
    }

    fn destroy(&mut self, r: T);
}

pub struct UniqueResource<T, D>
where
    T: Copy + std::cmp::PartialEq,
    D: ResourceDeleter<T> + std::default::Default,
{
    res: T,
    deleter: D,
}

impl<T, D> UniqueResource<T, D>
where
    T: Copy + std::cmp::PartialEq,
    D: ResourceDeleter<T> + std::default::Default,
{
    pub fn new(r: T) -> Option<Self> {
        if D::is_null(r) {
            None
        } else {
            Some(Self {
                res: r,
                deleter: D::default(),
            })
        }
    }

    pub fn new_with_deleter(r: T, d: D) -> Option<Self> {
        if D::is_null(r) {
            None
        } else {
            Some(Self { res: r, deleter: d })
        }
    }

    pub fn is_valid(&self) -> bool {
        !D::is_null(self.res)
    }

    pub fn handle(&self) -> T {
        self.res
    }
}

impl<T, D> std::ops::Deref for UniqueResource<T, D>
where
    T: Copy + std::cmp::PartialEq,
    D: ResourceDeleter<T> + std::default::Default,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        if !self.is_valid() {
            panic!("Derefing a null handle!");
        }

        &self.res
    }
}

impl<T, D> std::ops::DerefMut for UniqueResource<T, D>
where
    T: Copy + std::cmp::PartialEq,
    D: ResourceDeleter<T> + std::default::Default,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        if !self.is_valid() {
            panic!("Derefing a null handle!");
        }

        &mut self.res
    }
}

impl<T, D> std::default::Default for UniqueResource<T, D>
where
    T: Copy + std::cmp::PartialEq,
    D: ResourceDeleter<T> + std::default::Default,
{
    fn default() -> Self {
        Self {
            res: D::null(),
            deleter: D::default(),
        }
    }
}

impl<T, D> std::ops::Drop for UniqueResource<T, D>
where
    T: Copy + std::cmp::PartialEq,
    D: ResourceDeleter<T> + std::default::Default,
{
    fn drop(&mut self) {
        if self.is_valid() {
            self.deleter.destroy(self.res);
        }
    }
}

#[macro_export]
macro_rules! gen_unique_resource_type {
    ($tyname:tt, $delname:tt, $resty:ty, $nullval:expr, $dtor:expr) => {
        #[derive(Default)]
        pub struct $delname {}

        impl ResourceDeleter<$resty> for $delname {
            fn null() -> $resty {
                ($nullval)
            }
            fn destroy(&mut self, h: $resty) {
                ($dtor)(h);
            }
        }

        pub type $tyname = UniqueResource<$resty, $delname>;
    };
}

gen_unique_resource_type!(
    UniqueBuffer,
    GLBufferDeleter,
    gl::types::GLuint,
    0u32,
    |buff: gl::types::GLuint| unsafe {
        gl::DeleteBuffers(1, &buff);
    }
);

gen_unique_resource_type!(
    UniqueVertexArray,
    GLVertexArrayDeleter,
    gl::types::GLuint,
    0u32,
    |vao: gl::types::GLuint| unsafe {
        gl::DeleteVertexArrays(1, &vao);
    }
);

gen_unique_resource_type!(
    UniqueShaderProgram,
    GLProgramDeleter,
    gl::types::GLuint,
    0u32,
    |prg: gl::types::GLuint| unsafe {
        gl::DeleteProgram(prg);
    }
);

gen_unique_resource_type!(
    UniquePipeline,
    GLPipelineDeleter,
    gl::types::GLuint,
    0u32,
    |pp: gl::types::GLuint| unsafe {
        gl::DeleteProgramPipelines(1, &pp);
    }
);

gen_unique_resource_type!(
    UniqueSampler,
    GLSamplerDeleter,
    gl::types::GLuint,
    0u32,
    |s: gl::types::GLuint| unsafe {
        gl::DeleteSamplers(1, &s);
    }
);

gen_unique_resource_type!(
    UniqueTexture,
    GLTextureDeleter,
    gl::types::GLuint,
    0u32,
    |t: gl::types::GLuint| unsafe {
        gl::DeleteTextures(1, &t);
    }
);

pub struct UniqueBufferMapping {
    buffer: gl::types::GLuint,
    mapped_memory: *mut std::os::raw::c_void,
    mapping_size: i64,
}

impl UniqueBufferMapping {
    pub fn new(
        buffer: gl::types::GLuint,
        access: gl::types::GLbitfield,
    ) -> Option<UniqueBufferMapping> {
        let buffer_size = unsafe {
            let mut bsize = 0i64;
            gl::GetNamedBufferParameteri64v(buffer, gl::BUFFER_SIZE, &mut bsize);
            bsize
        };

        if buffer_size == 0 {
            return None;
        }

        let mapped_memory =
            unsafe { gl::MapNamedBufferRange(buffer, 0, buffer_size as isize, access) };
        if mapped_memory.is_null() {
            return None;
        }

        Some(UniqueBufferMapping {
            buffer,
            mapped_memory,
            mapping_size: buffer_size,
        })
    }

    pub fn as_slice<T>(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.as_ptr::<T>(), self.size() / std::mem::size_of::<T>())
        }
    }

    pub fn as_mut_slice<T>(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.as_mut_ptr(),
                self.size() / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn as_ptr<T>(&self) -> *const T {
        self.memory() as *const _
    }

    pub fn as_mut_ptr<T>(&mut self) -> *mut T {
        self.memory() as *mut _
    }

    pub fn memory(&self) -> *mut std::os::raw::c_void {
        self.mapped_memory
    }

    pub fn size(&self) -> usize {
        self.mapping_size as usize
    }
}

impl std::ops::Drop for UniqueBufferMapping {
    fn drop(&mut self) {
        unsafe {
            gl::UnmapNamedBuffer(self.buffer);
        }
    }
}

impl<T> std::convert::AsRef<[T]> for UniqueBufferMapping {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> std::convert::AsMut<[T]> for UniqueBufferMapping {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ShaderType {
    Vertex,
    Geometry,
    Fragment,
    Compute,
}

pub fn create_shader_program_from_file<P: AsRef<std::path::Path>>(
    file_path: P,
    program_type: ShaderType,
) -> Result<UniqueShaderProgram, OpenGLError> {
    let mut file = std::fs::File::open(file_path)
        .map_err(|e| OpenGLError::GenericError(format!("Shader file error: {e}"), 0))?;
    let mut s = String::new();
    use std::io::Read;
    file.read_to_string(&mut s)
        .map_err(|e| e.to_string())
        .map_err(|e| OpenGLError::GenericError(format!("Read error {e}"), 0))?;

    create_shader_program_from_string(&s, program_type)
}

pub fn create_shader_program_from_string(
    s: &str,
    prog_type: ShaderType,
) -> Result<UniqueShaderProgram, OpenGLError> {
    let src_code = std::ffi::CString::new(s)
        .map_err(|e| OpenGLError::ShaderSourceCodeInvalid(e.to_string()))?;

    let x = [src_code.as_ptr()];
    let prog_type = match prog_type {
        ShaderType::Vertex => gl::VERTEX_SHADER,
        ShaderType::Fragment => gl::FRAGMENT_SHADER,
        ShaderType::Geometry => gl::GEOMETRY_SHADER,
        ShaderType::Compute => gl::COMPUTE_SHADER,
    };

    let prg =
        UniqueShaderProgram::new(unsafe { gl::CreateShaderProgramv(prog_type, 1, x.as_ptr()) })
            .ok_or_else(|| OpenGLError::ResourceCreateError(unsafe { gl::GetError() }))?;

    let linked_successfully = (|| {
        let mut link_status = 0i32;
        unsafe {
            gl::GetProgramiv(*prg, gl::LINK_STATUS, &mut link_status);
        }
        link_status == gl::TRUE as i32
    })();

    if linked_successfully {
        return Ok(prg);
    }

    let mut info_log_buff: Vec<u8> = vec![0; 1024];
    let mut info_log_size = 0i32;
    unsafe {
        gl::GetProgramInfoLog(
            *prg,
            info_log_buff.len() as gl::types::GLsizei,
            &mut info_log_size,
            info_log_buff.as_mut_ptr() as *mut i8,
        );
    }

    if info_log_size > 0 {
        info_log_buff[info_log_size as usize] = 0;
        use std::ffi::CStr;

        let err_msg = unsafe { CStr::from_ptr(info_log_buff.as_ptr() as *const i8) };
        let err_str = err_msg
            .to_str()
            .map(|e| {
                e.chars()
                    .filter(|c| *c != '\"' /* && *c != '\n' */)
                    .collect::<String>()
            })
            .unwrap_or_else(|_| "Cant display shader compile error".into());

        return Err(OpenGLError::ShaderCompilationError(err_str));
    }

    Err(OpenGLError::GenericError(
        "Cant get compile error log".into(),
        unsafe { gl::GetError() },
    ))
}

/// Stores a snapshot of the OpenGL state machine at some point in time.
#[derive(Debug)]
pub struct OpenGLStateSnapshot {
    last_blend_src: gl::types::GLint,
    last_blend_dst: gl::types::GLint,
    last_blend_eq_rgb: gl::types::GLint,
    last_blend_eq_alpha: gl::types::GLint,
    blend_enabled: bool,
    cullface_enabled: bool,
    depth_enabled: bool,
    scissors_enabled: bool,
}

impl OpenGLStateSnapshot {
    pub fn new() -> Self {
        unsafe {
            let mut glstate = std::mem::MaybeUninit::<OpenGLStateSnapshot>::zeroed().assume_init();

            gl::GetIntegerv(gl::BLEND_SRC, &mut glstate.last_blend_src);
            gl::GetIntegerv(gl::BLEND_DST, &mut glstate.last_blend_dst);
            gl::GetIntegerv(gl::BLEND_EQUATION_RGB, &mut glstate.last_blend_eq_rgb);
            gl::GetIntegerv(gl::BLEND_EQUATION_ALPHA, &mut glstate.last_blend_eq_alpha);

            glstate.blend_enabled = gl::IsEnabled(gl::BLEND) != gl::FALSE;
            glstate.cullface_enabled = gl::IsEnabled(gl::CULL_FACE) != gl::FALSE;
            glstate.depth_enabled = gl::IsEnabled(gl::DEPTH_TEST) != gl::FALSE;
            glstate.scissors_enabled = gl::IsEnabled(gl::SCISSOR_TEST) != gl::FALSE;

            glstate
        }
    }
}

impl std::ops::Drop for OpenGLStateSnapshot {
    fn drop(&mut self) {
        unsafe {
            gl::BlendEquationSeparate(
                self.last_blend_eq_rgb as u32,
                self.last_blend_eq_alpha as u32,
            );
            gl::BlendFunc(self.last_blend_src as u32, self.last_blend_dst as u32);

            if self.blend_enabled {
                gl::Enable(gl::BLEND);
            } else {
                gl::Disable(gl::BLEND);
            }

            if self.cullface_enabled {
                gl::Enable(gl::CULL_FACE);
            } else {
                gl::Disable(gl::CULL_FACE);
            }

            if self.depth_enabled {
                gl::Enable(gl::DEPTH_TEST);
            } else {
                gl::Disable(gl::DEPTH_TEST);
            }

            if self.scissors_enabled {
                gl::Enable(gl::SCISSOR_TEST);
            } else {
                gl::Disable(gl::SCISSOR_TEST);
            }
        }
    }
}

pub struct PipelineBuilder<'a> {
    vertexshader: Option<&'a UniqueShaderProgram>,
    geometryshader: Option<&'a UniqueShaderProgram>,
    fragmentshader: Option<&'a UniqueShaderProgram>,
}

impl<'a> PipelineBuilder<'a> {
    pub fn new() -> Self {
        PipelineBuilder {
            vertexshader: None,
            geometryshader: None,
            fragmentshader: None,
        }
    }

    pub fn add_vertex_shader(&mut self, vs: &'a UniqueShaderProgram) -> &mut Self {
        self.vertexshader = Some(vs);
        self
    }

    pub fn add_fragment_shader(&mut self, fs: &'a UniqueShaderProgram) -> &mut Self {
        self.fragmentshader = Some(fs);
        self
    }

    pub fn add_geometry_shader(&mut self, gs: &'a UniqueShaderProgram) -> &mut Self {
        self.geometryshader = Some(gs);
        self
    }

    pub fn build(&self) -> Result<UniquePipeline, String> {
        let pp = UniquePipeline::new(unsafe {
            let mut pp = 0u32;
            gl::CreateProgramPipelines(1, &mut pp);
            pp
        })
        .ok_or_else(|| "Failed to create program pipeline object!".to_string())?;

        if let Some(vs) = self.vertexshader {
            unsafe {
                gl::UseProgramStages(*pp, gl::VERTEX_SHADER_BIT, **vs);
            }
        }

        if let Some(fs) = self.fragmentshader {
            unsafe {
                gl::UseProgramStages(*pp, gl::FRAGMENT_SHADER_BIT, **fs);
            }
        }

        if let Some(gs) = self.geometryshader {
            unsafe {
                gl::UseProgramStages(*pp, gl::GEOMETRY_SHADER_BIT, **gs);
            }
        }

        Ok(pp)
    }
}

pub struct SamplerBuilder {
    border_color: Option<(f32, f32, f32, f32)>,
    mag_filter: Option<i32>,
    min_filter: Option<i32>,
    wrap_s: Option<i32>,
    wrap_t: Option<i32>,
}

impl SamplerBuilder {
    pub fn new() -> SamplerBuilder {
        SamplerBuilder {
            border_color: None,
            mag_filter: None,
            min_filter: None,
            wrap_s: None,
            wrap_t: None,
        }
    }

    pub fn set_border_color(&mut self, r: f32, g: f32, b: f32) -> &mut Self {
        self.border_color = Some((r, g, b, 1f32));
        self
    }

    pub fn set_min_filter(&mut self, minfilter: i32) -> &mut Self {
        self.min_filter = Some(minfilter);
        self
    }

    pub fn set_mag_filter(&mut self, magfilter: i32) -> &mut Self {
        self.mag_filter = Some(magfilter);
        self
    }

    pub fn build(&self) -> Result<UniqueSampler, String> {
        let s = UniqueSampler::new(unsafe {
            let mut s = 0u32;
            gl::CreateSamplers(1, &mut s);
            s
        })
        .ok_or_else(|| "Failed to create sampler!".to_string())?;

        if let Some(c) = self.border_color {
            unsafe {
                let border_color = [c.0, c.1, c.2, c.3];
                gl::SamplerParameterfv(*s, gl::TEXTURE_BORDER_COLOR, border_color.as_ptr());
            }
        }

        if let Some(minflt) = self.min_filter {
            unsafe {
                gl::SamplerParameteri(*s, gl::TEXTURE_MIN_FILTER, minflt);
            }
        }

        if let Some(magflt) = self.mag_filter {
            unsafe {
                gl::SamplerParameteri(*s, gl::TEXTURE_MAG_FILTER, magflt);
            }
        }

        if let Some(wraps) = self.wrap_s {
            unsafe {
                gl::SamplerParameteri(*s, gl::TEXTURE_WRAP_S, wraps);
            }
        }

        if let Some(wrapt) = self.wrap_t {
            unsafe {
                gl::SamplerParameteri(*s, gl::TEXTURE_WRAP_T, wrapt);
            }
        }

        Ok(s)
    }
}

enum ShaderBuildingBlock<'a> {
    StringCode(&'a dyn AsRef<str>),
    File(&'a dyn AsRef<Path>),
}

enum ShaderStringSource {
    FfiString(CString),
}

pub struct ShaderProgramBuilder<'a> {
    blocks: Vec<ShaderBuildingBlock<'a>>,
}

impl<'a> ShaderProgramBuilder<'a> {
    pub fn new() -> Self {
        Self { blocks: vec![] }
    }

    pub fn add_string<S: AsRef<str>>(&mut self, s: &'a S) -> &mut Self {
        self.blocks.push(ShaderBuildingBlock::StringCode(s));
        self
    }

    pub fn add_file<P: AsRef<Path>>(&mut self, p: &'a P) -> &mut Self {
        self.blocks.push(ShaderBuildingBlock::File(p));
        self
    }

    pub fn build(&self, shader_type: ShaderType) -> Result<UniqueShaderProgram, String> {
        let string_sources = self
            .blocks
            .iter()
            .filter_map(|block| match block {
                ShaderBuildingBlock::StringCode(s) => {
                    if let Some(txt_code) = CString::new(s.as_ref()).ok() {
                        Some(ShaderStringSource::FfiString(txt_code))
                    } else {
                        None
                    }
                }

                ShaderBuildingBlock::File(path) => std::fs::File::open(path)
                    .ok()
                    .and_then(|mut f| {
                        use std::io::Read;
                        let mut bytes = Vec::<u8>::new();
                        f.read_to_end(&mut bytes).ok().map(|_| bytes)
                    })
                    .and_then(|bytes| CString::from_vec_with_nul(bytes).ok())
                    .map(|s| ShaderStringSource::FfiString(s)),
            })
            .collect::<Vec<_>>();

        if string_sources.len() != self.blocks.len() {
            return Err("Failed to load some shader files!".to_string());
        }

        let c_strings_ptrs = string_sources
            .iter()
            .map(|src| match src {
                ShaderStringSource::FfiString(ref ffi_str) => ffi_str.as_ptr(),
            })
            .collect::<Vec<_>>();

        use gl::types::*;

        let shader_program = UniqueShaderProgram::new(unsafe {
            let gl_shader_type = match shader_type {
                ShaderType::Vertex => gl::VERTEX_SHADER,
                ShaderType::Geometry => gl::GEOMETRY_SHADER,
                ShaderType::Fragment => gl::FRAGMENT_SHADER,
                ShaderType::Compute => gl::COMPUTE_SHADER,
            };

            gl::CreateShaderProgramv(
                gl_shader_type,
                c_strings_ptrs.len() as GLsizei,
                c_strings_ptrs.as_ptr(),
            )
        })
        .ok_or_else(|| "Failed to create shader!".to_string())?;

        let linked_successfully = (|| {
            let mut link_status = 0i32;
            unsafe {
                gl::GetProgramiv(*shader_program, gl::LINK_STATUS, &mut link_status);
            }
            link_status == gl::TRUE as i32
        })();

        if linked_successfully {
            return Ok(shader_program);
        }

        let mut info_log_buff: Vec<u8> = vec![0; 1024];
        let mut info_log_size = 0i32;
        unsafe {
            gl::GetProgramInfoLog(
                *shader_program,
                info_log_buff.len() as gl::types::GLsizei,
                &mut info_log_size,
                info_log_buff.as_mut_ptr() as *mut i8,
            );
        }

        if info_log_size > 0 {
            info_log_buff[info_log_size as usize] = 0;

            use std::ffi::CStr;
            let err_desc = unsafe { CStr::from_ptr(info_log_buff.as_ptr() as *const i8) };

            return Err(err_desc
                .to_str()
                .unwrap_or_else(|_| "unknown shader compiler error")
                .to_string());
        }

        Err("Error but no compile log is available".to_string())
    }
}

pub fn create_buffer(size: usize, access: Option<GLbitfield>) -> Option<UniqueBuffer> {
    unsafe {
        let mut buf: GLuint = 0;
        gl::CreateBuffers(1, &mut buf);
        gl::NamedBufferStorage(
            buf,
            size as GLsizeiptr,
            std::ptr::null(),
            access.unwrap_or_default(),
        );
        UniqueBuffer::new(buf)
    }
}

pub struct OpenGLDebugScopePush {}

impl OpenGLDebugScopePush {
    pub fn new(id: u32, message: &str) -> Self {
        unsafe {
            let c_msg = CString::new(message).unwrap();
            let pmsg = c_msg.as_bytes_with_nul();

            gl::PushDebugGroup(
                gl::DEBUG_SOURCE_APPLICATION,
                id,
                pmsg.len() as GLsizei,
                pmsg.as_ptr() as *const _,
            );
        }

        Self {}
    }
}

impl std::ops::Drop for OpenGLDebugScopePush {
    fn drop(&mut self) {
        unsafe {
            gl::PopDebugGroup();
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum OpenGLError {
    #[error("Failed to compile shader: `{0}`")]
    ShaderCompilationError(String),
    #[error("Shader source code convert error: `{0}`")]
    ShaderSourceCodeInvalid(String),
    #[error("OpenGL resource creation error: `{0}`")]
    ResourceCreateError(gl::types::GLenum),
    #[error("OpenGL geneneric error:\ncode: `{0}`\ndesc: `{1}`")]
    GenericError(String, u32),
}

impl OpenGLError {
    pub fn current_error(s: &str) -> Self {
        let e = unsafe { gl::GetError() };

        Self::GenericError(s.to_string(), e)
    }
}

pub fn label_object(obj_type: gl::types::GLuint, obj: u32, label: &str) {
    let c_msg = CString::new(label).unwrap();
    let pmsg = c_msg.as_bytes_with_nul();

    unsafe {
        gl::ObjectLabel(
            obj_type,
            obj,
            pmsg.len() as GLsizei,
            pmsg.as_ptr() as *const _,
        );
    }
}
