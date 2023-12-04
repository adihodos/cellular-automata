use std::io::Read;

use crate::gl::{self, types::*};
use crate::gl_utils::*;
use nalgebra_glm as glm;

const UI_VERTEX_SHADER: &'static str = r#"
#version 450 core

layout (location = 0) in vec2 vsInPos;
layout (location = 1) in vec2 vsInUV;
layout (location = 2) in vec4 vsInColor;

out gl_PerVertex {
  vec4 gl_Position;
};

out VS_OUT_FS_IN {
  vec2 uv;
  vec4 color;
} vs_out_fs_in;

layout (location = 0) uniform mat4 TransformMatrix;

void main(void) {
  gl_Position = TransformMatrix * vec4(vsInPos, 0.0, 1.0);
  vs_out_fs_in.uv = vsInUV;
  vs_out_fs_in.color = vsInColor;
}
"#;

const UI_FRAGMENT_SHADER: &'static str = r#"
#version 450 core

in VS_OUT_FS_IN {
  vec2 uv;
  vec4 color;
} fs_in;

layout (binding = 0) uniform sampler2D TexAtlas;
out vec4 FinalFragColor;

void main(void) {
  FinalFragColor = fs_in.color * texture(TexAtlas, fs_in.uv);
}
"#;

struct UserInterfaceRenderState {
    vertexcap: usize,
    indexcap: usize,
    mapping_buf_vertex: UniqueBufferMapping,
    mapping_buf_index: UniqueBufferMapping,
    vertexbuffer: UniqueBuffer,
    indexbuffer: UniqueBuffer,
    vertexarray: UniqueVertexArray,
    vertexshader: UniqueShaderProgram,
    _fragmentshader: UniqueShaderProgram,
    pipeline: UniquePipeline,
    sampler: UniqueSampler,
    atlas: UniqueTexture,
}

impl UserInterfaceRenderState {
    pub fn new(font_texture: imgui::FontAtlasTexture) -> Result<UserInterfaceRenderState, String> {
        const INITIAL_BUFFER_CAPACITY: usize = 2048;

        let vertexbuffer = UniqueBuffer::new(unsafe {
            let mut buff: GLuint = 0;
            gl::CreateBuffers(1, &mut buff);
            gl::NamedBufferStorage(
                buff,
                (INITIAL_BUFFER_CAPACITY * std::mem::size_of::<imgui::DrawVert>()) as GLsizeiptr,
                std::ptr::null(),
                gl::MAP_WRITE_BIT | gl::MAP_PERSISTENT_BIT | gl::MAP_COHERENT_BIT,
            );
            buff
        })
        .ok_or_else(|| "UI init error: failed to allocate vertex buffer!".to_string())?;

        let mapping_buf_vertex = UniqueBufferMapping::new(
            *vertexbuffer,
            gl::MAP_WRITE_BIT
                | gl::MAP_COHERENT_BIT
                | gl::MAP_PERSISTENT_BIT
                | gl::MAP_INVALIDATE_BUFFER_BIT,
        )
        .ok_or_else(|| {
            format!(
                "UI init error: failed to map vertex buffer in CPU memory, error {:#x}",
                unsafe { gl::GetError() }
            )
        })?;

        let indexbuffer = UniqueBuffer::new(unsafe {
            let mut buff: GLuint = 0;
            gl::CreateBuffers(1, &mut buff);
            gl::NamedBufferStorage(
                buff,
                (INITIAL_BUFFER_CAPACITY * 2 * std::mem::size_of::<imgui::DrawIdx>()) as GLsizeiptr,
                std::ptr::null(),
                gl::MAP_WRITE_BIT | gl::MAP_PERSISTENT_BIT | gl::MAP_COHERENT_BIT,
            );
            buff
        })
        .ok_or_else(|| "UI init error: failed to allocate elements buffer!".to_string())?;

        let mapping_buf_index = UniqueBufferMapping::new(
            *indexbuffer,
            gl::MAP_WRITE_BIT
                | gl::MAP_PERSISTENT_BIT
                | gl::MAP_COHERENT_BIT
                | gl::MAP_INVALIDATE_BUFFER_BIT,
        )
        .ok_or_else(|| "UI init error: failed to map element buffer in CPU memory!".to_string())?;

        let vertexarray = UniqueVertexArray::new(unsafe {
            let mut vao: GLuint = 0;
            gl::CreateVertexArrays(1, &mut vao);
            gl::VertexArrayVertexBuffer(
                vao,
                0,
                *vertexbuffer,
                0,
                std::mem::size_of::<imgui::DrawVert>() as GLsizei,
            );
            gl::VertexArrayElementBuffer(vao, *indexbuffer);

            gl::VertexArrayAttribFormat(vao, 0, 2, gl::FLOAT, gl::FALSE, 0);
            gl::VertexArrayAttribBinding(vao, 0, 0);
            gl::EnableVertexArrayAttrib(vao, 0);

            gl::VertexArrayAttribFormat(vao, 1, 2, gl::FLOAT, gl::FALSE, 8);
            gl::VertexArrayAttribBinding(vao, 1, 0);
            gl::EnableVertexArrayAttrib(vao, 1);

            gl::VertexArrayAttribFormat(vao, 2, 4, gl::UNSIGNED_BYTE, gl::TRUE, 16);
            gl::VertexArrayAttribBinding(vao, 2, 0);
            gl::EnableVertexArrayAttrib(vao, 2);

            vao
        })
        .ok_or_else(|| "UI init error: failed to create vertex array !".to_string())?;

        let vertexshader = create_shader_program_from_string(UI_VERTEX_SHADER, ShaderType::Vertex)
            .map_err(|e| format!("{e}"))?;

        let fragmentshader =
            create_shader_program_from_string(UI_FRAGMENT_SHADER, ShaderType::Fragment)
                .map_err(|e| format!("{e}"))?;

        let pipeline = PipelineBuilder::new()
            .add_vertex_shader(&vertexshader)
            .add_fragment_shader(&fragmentshader)
            .build()?;

        let sampler = SamplerBuilder::new()
            .set_min_filter(gl::LINEAR as i32)
            .set_mag_filter(gl::LINEAR as i32)
            .build()?;

        let atlas = {
            UniqueTexture::new(unsafe {
                let mut tex: GLuint = 0;
                gl::CreateTextures(gl::TEXTURE_2D, 1, &mut tex);
                gl::TextureStorage2D(
                    tex,
                    1,
                    gl::RGBA8,
                    font_texture.width as i32,
                    font_texture.height as i32,
                );
                gl::TextureSubImage2D(
                    tex,
                    0,
                    0,
                    0,
                    font_texture.width as GLsizei,
                    font_texture.height as GLsizei,
                    gl::RGBA,
                    gl::UNSIGNED_BYTE,
                    font_texture.data.as_ptr() as *const _,
                );
                tex
            })
            .ok_or_else(|| "UI init error: failed to create font atlas!")?
        };

        Ok(UserInterfaceRenderState {
            vertexcap: INITIAL_BUFFER_CAPACITY,
            indexcap: INITIAL_BUFFER_CAPACITY * 2,
            vertexbuffer,
            indexbuffer,
            mapping_buf_vertex,
            mapping_buf_index,
            vertexarray,
            vertexshader,
            _fragmentshader: fragmentshader,
            pipeline,
            sampler,
            atlas,
        })
    }

    fn upload_render_data(&mut self, draw_data: &imgui::DrawData) -> std::io::Result<()> {
        let (vertex_count, index_count) = draw_data.draw_lists().fold(
            (0usize, 0usize),
            |(vertex_count, index_count), draw_list| {
                (
                    vertex_count + draw_list.vtx_buffer().len(),
                    index_count + draw_list.idx_buffer().len(),
                )
            },
        );

        //
        // Ensure vertex data can fit into buffer
        const BUFFER_GROW_FACTOR: f32 = 1.5f32;

        if vertex_count > self.vertexcap {
            let new_vtxcap = (vertex_count as f32 * BUFFER_GROW_FACTOR) as usize;
            println!("Growing vertex buffer {} -> {}", self.vertexcap, new_vtxcap);
            let new_vtxbuff = UniqueBuffer::new(unsafe {
                let mut buff: GLuint = 0;
                gl::CreateBuffers(1, &mut buff);
                gl::NamedBufferStorage(
                    buff,
                    (new_vtxcap * std::mem::size_of::<imgui::DrawVert>()) as GLsizeiptr,
                    std::ptr::null(),
                    gl::MAP_WRITE_BIT | gl::MAP_COHERENT_BIT | gl::MAP_PERSISTENT_BIT,
                );
                buff
            })
            .ok_or_else(|| std::io::Error::from_raw_os_error(unsafe { gl::GetError() as i32 }))?;

            let new_vtxbuffmapping = UniqueBufferMapping::new(
                *new_vtxbuff,
                gl::MAP_WRITE_BIT
                    | gl::MAP_PERSISTENT_BIT
                    | gl::MAP_COHERENT_BIT
                    | gl::MAP_INVALIDATE_BUFFER_BIT,
            )
            .ok_or_else(|| std::io::Error::from_raw_os_error(unsafe { gl::GetError() as i32 }))?;

            unsafe {
                gl::VertexArrayVertexBuffer(
                    *self.vertexarray,
                    0,
                    *new_vtxbuff,
                    0,
                    std::mem::size_of::<imgui::DrawVert>() as GLsizei,
                );
            }
            self.mapping_buf_vertex = new_vtxbuffmapping;
            self.vertexbuffer = new_vtxbuff;
            self.vertexcap = new_vtxcap;
        }

        //
        //  Ensure index data can fit into buffer
        if index_count > self.indexcap {
            let new_idxcap = (index_count as f32 * BUFFER_GROW_FACTOR) as usize;
            println!("Growing index buffer {} -> {}", self.indexcap, new_idxcap);
            let new_idxbuff = UniqueBuffer::new(unsafe {
                let mut buff: GLuint = 0;
                gl::CreateBuffers(1, &mut buff);
                gl::NamedBufferStorage(
                    buff,
                    (new_idxcap * std::mem::size_of::<imgui::DrawIdx>()) as GLsizeiptr,
                    std::ptr::null(),
                    gl::MAP_WRITE_BIT | gl::MAP_COHERENT_BIT | gl::MAP_PERSISTENT_BIT,
                );
                buff
            })
            .ok_or_else(|| std::io::Error::from_raw_os_error(unsafe { gl::GetError() as i32 }))?;

            let new_idxbuffmapping = UniqueBufferMapping::new(
                *new_idxbuff,
                gl::MAP_WRITE_BIT
                    | gl::MAP_PERSISTENT_BIT
                    | gl::MAP_COHERENT_BIT
                    | gl::MAP_INVALIDATE_BUFFER_BIT,
            )
            .ok_or_else(|| std::io::Error::from_raw_os_error(unsafe { gl::GetError() as i32 }))?;

            unsafe {
                gl::VertexArrayElementBuffer(*self.vertexarray, *new_idxbuff);
            }
            self.mapping_buf_index = new_idxbuffmapping;
            self.indexbuffer = new_idxbuff;
            self.indexcap = new_idxcap;
        }

        let (_vertices_count, _indices_count) =
            draw_data
                .draw_lists()
                .fold((0usize, 0usize), |(off_vtx, off_idx), draw_list| {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            draw_list.vtx_buffer().as_ptr(),
                            self.mapping_buf_vertex
                                .as_mut_ptr::<imgui::DrawVert>()
                                .add(off_vtx),
                            draw_list.vtx_buffer().len(),
                        );

                        std::ptr::copy_nonoverlapping(
                            draw_list.idx_buffer().as_ptr(),
                            self.mapping_buf_index
                                .as_mut_ptr::<imgui::DrawIdx>()
                                .add(off_idx),
                            draw_list.idx_buffer().len(),
                        );
                    }

                    (
                        off_vtx + draw_list.vtx_buffer().len(),
                        off_idx + draw_list.idx_buffer().len(),
                    )
                });

        Ok(())
    }

    fn render(&mut self, fb_width: i32, fb_height: i32, draw_data: &imgui::DrawData) {
        assert!(fb_width > 0);
        assert!(fb_height > 0);

        if !self.upload_render_data(draw_data).is_ok() {
            return;
        }

        let _gl_state = OpenGLStateSnapshot::new();

        unsafe {
            gl::Enable(gl::BLEND);
            gl::BlendEquation(gl::FUNC_ADD);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            gl::Disable(gl::CULL_FACE);
            gl::Disable(gl::DEPTH_TEST);
            gl::Enable(gl::SCISSOR_TEST);
        }

        let projection_matrix =
            glm::ortho(0f32, fb_width as f32, fb_height as f32, 0f32, -1f32, 1f32);

        unsafe {
            gl::BindVertexArray(*self.vertexarray);
            gl::BindSampler(0, *self.sampler);
            gl::ProgramUniformMatrix4fv(
                *self.vertexshader,
                0,
                1,
                gl::FALSE,
                projection_matrix.as_ptr(),
            );
            gl::BindProgramPipeline(*self.pipeline);
        }

        let _elements_drawn =
            draw_data
                .draw_lists()
                .fold((0usize, 0usize), |(vertex_off, index_off), draw_list| {
                    for draw_cmd in draw_list.commands() {
                        use imgui::DrawCmd;

                        match draw_cmd {
                            DrawCmd::Elements { count, cmd_params } => {
                                unsafe {
                                    gl::BindTextureUnit(0, cmd_params.texture_id.id() as GLuint);
                                    gl::Scissor(
                                        cmd_params.clip_rect[0] as i32,
                                        fb_height - cmd_params.clip_rect[3] as i32,
                                        (cmd_params.clip_rect[2] - cmd_params.clip_rect[0]) as i32,
                                        (cmd_params.clip_rect[3] - cmd_params.clip_rect[1]) as i32,
                                    );
                                }

                                let index_type = match std::mem::size_of::<imgui::DrawIdx>() {
                                    2 => gl::UNSIGNED_SHORT,
                                    1 => gl::UNSIGNED_BYTE,
                                    _ => gl::UNSIGNED_INT,
                                };

                                unsafe {
                                    gl::DrawElementsBaseVertex(
                                        gl::TRIANGLES,
                                        count as GLsizei,
                                        index_type,
                                        std::ptr::null::<imgui::DrawIdx>()
                                            .offset((index_off + cmd_params.idx_offset) as isize)
                                            as *const _,
                                        (vertex_off + cmd_params.vtx_offset) as GLsizei,
                                    );
                                }
                            }
                            _ => {}
                        }
                    }

                    (
                        vertex_off + draw_list.vtx_buffer().len(),
                        index_off + draw_list.idx_buffer().len(),
                    )
                });
    }
}

pub struct UiBackend {
    context: imgui::Context,
    render_state: UserInterfaceRenderState,
    last_frame: std::time::Instant,
}

impl UiBackend {
    pub fn new() -> std::io::Result<UiBackend> {
        let mut context = imgui::Context::create();
        {
            let io = context.io_mut();
            io.backend_flags
                .insert(imgui::BackendFlags::HAS_SET_MOUSE_POS);
        }

        context.set_platform_name(Some("Cell automaton app".into()));

        let render_state = {
            let font_atlas = context.fonts();
            if let Ok(font) = load_compressed_font("data/fonts/iosevka-nfm.ttf") {
                use imgui::{FontConfig, FontSource};
                font_atlas.add_font(&[FontSource::TtfData {
                    data: &font,
                    size_pixels: 18.0,
                    config: Some(FontConfig {
                        oversample_h: 4,
                        oversample_v: 4,
                        rasterizer_multiply: 1.5f32,
                        ..FontConfig::default()
                    }),
                }]);
            }

            UserInterfaceRenderState::new(font_atlas.build_rgba32_texture())
        }
        .map_err(|e| {
            log::error!("UI rener state creation error {e}");
            std::io::Error::new(std::io::ErrorKind::Other, e)
        })?;

        context.fonts().tex_id = imgui::TextureId::from(render_state.atlas.handle() as usize);

        Ok(UiBackend {
            context,
            render_state,
            last_frame: std::time::Instant::now(),
        })
    }

    pub fn handle_event(&mut self, event: &glfw::WindowEvent) -> bool {
        use glfw::WindowEvent;

        match *event {
            WindowEvent::Scroll(x_offset, y_offset) => {
                self.context
                    .io_mut()
                    .add_mouse_wheel_event([x_offset as f32, y_offset as f32]);
            }

            WindowEvent::MouseButton(btn, action, _) => {
                self.handle_mouse_button(btn, action);
            }

            WindowEvent::Char(c) => {
                self.context.io_mut().add_input_character(c);
            }

            WindowEvent::Key(key, _, action, mods) => {
                handle_key_modifier(self.context.io_mut(), &mods);
                handle_key(self.context.io_mut(), key, action);
            }

            WindowEvent::CursorPos(cx, cy) => {
                self.context.io_mut().mouse_pos = [cx as f32, cy as f32];
            }

            _ => {}
        }

        self.context.io().want_capture_keyboard || self.context.io().want_capture_mouse
    }

    fn handle_mouse_button(&mut self, button: glfw::MouseButton, action: glfw::Action) {
        let pressed = action == glfw::Action::Press;
        let io = self.context.io_mut();

        match button {
            glfw::MouseButtonLeft => io.add_mouse_button_event(imgui::MouseButton::Left, pressed),
            glfw::MouseButtonRight => io.add_mouse_button_event(imgui::MouseButton::Right, pressed),
            glfw::MouseButtonMiddle => {
                io.add_mouse_button_event(imgui::MouseButton::Middle, pressed)
            }

            _ => {}
        }
    }

    pub fn prepare_frame(
        &mut self,
        window: &mut glfw::Window,
        fbw: i32,
        fbh: i32,
        w: i32,
        h: i32,
    ) -> &mut imgui::Ui {
        let io = self.context.io_mut();

        let now = std::time::Instant::now();
        io.update_delta_time(now.duration_since(self.last_frame));
        self.last_frame = now;

        io.display_size = [w as f32, h as f32];
        io.display_framebuffer_scale = [fbw as f32 / w as f32, fbh as f32 / h as f32];

        if io.want_set_mouse_pos {
            window.set_cursor_pos(io.mouse_pos[0] as f64, io.mouse_pos[1] as f64);
        }

        self.context.new_frame()
    }

    pub fn render(&mut self) {
        let draw_data = self.context.render();

        let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];

        if !(fb_width > 0f32) && !(fb_height > 0f32) {
            return;
        }

        self.render_state
            .render(fb_width as i32, fb_height as i32, draw_data);
    }
}

/// Handle changes in the key states.
fn handle_key(io: &mut imgui::Io, key: glfw::Key, action: glfw::Action) {
    let igkey = match key {
        glfw::Key::A => imgui::Key::A,
        glfw::Key::B => imgui::Key::B,
        glfw::Key::C => imgui::Key::C,
        glfw::Key::D => imgui::Key::D,
        glfw::Key::E => imgui::Key::E,
        glfw::Key::F => imgui::Key::F,
        glfw::Key::G => imgui::Key::G,
        glfw::Key::H => imgui::Key::H,
        glfw::Key::I => imgui::Key::I,
        glfw::Key::J => imgui::Key::J,
        glfw::Key::K => imgui::Key::K,
        glfw::Key::L => imgui::Key::L,
        glfw::Key::M => imgui::Key::M,
        glfw::Key::N => imgui::Key::N,
        glfw::Key::O => imgui::Key::O,
        glfw::Key::P => imgui::Key::P,
        glfw::Key::Q => imgui::Key::Q,
        glfw::Key::R => imgui::Key::R,
        glfw::Key::S => imgui::Key::S,
        glfw::Key::T => imgui::Key::T,
        glfw::Key::U => imgui::Key::U,
        glfw::Key::V => imgui::Key::V,
        glfw::Key::W => imgui::Key::W,
        glfw::Key::X => imgui::Key::X,
        glfw::Key::Y => imgui::Key::Y,
        glfw::Key::Z => imgui::Key::Z,
        glfw::Key::Num1 => imgui::Key::Keypad1,
        glfw::Key::Num2 => imgui::Key::Keypad2,
        glfw::Key::Num3 => imgui::Key::Keypad3,
        glfw::Key::Num4 => imgui::Key::Keypad4,
        glfw::Key::Num5 => imgui::Key::Keypad5,
        glfw::Key::Num6 => imgui::Key::Keypad6,
        glfw::Key::Num7 => imgui::Key::Keypad7,
        glfw::Key::Num8 => imgui::Key::Keypad8,
        glfw::Key::Num9 => imgui::Key::Keypad9,
        glfw::Key::Num0 => imgui::Key::Keypad0,
        glfw::Key::Enter => imgui::Key::Enter, // TODO: Should this be treated as alias?
        glfw::Key::Escape => imgui::Key::Escape,
        glfw::Key::Backspace => imgui::Key::Backspace,
        glfw::Key::Tab => imgui::Key::Tab,
        glfw::Key::Space => imgui::Key::Space,
        glfw::Key::Minus => imgui::Key::Minus,
        glfw::Key::Equal => imgui::Key::Equal,
        glfw::Key::LeftBracket => imgui::Key::LeftBracket,
        glfw::Key::RightBracket => imgui::Key::RightBracket,
        glfw::Key::Backslash => imgui::Key::Backslash,
        glfw::Key::Semicolon => imgui::Key::Semicolon,
        glfw::Key::Apostrophe => imgui::Key::Apostrophe,
        glfw::Key::GraveAccent => imgui::Key::GraveAccent,
        glfw::Key::Comma => imgui::Key::Comma,
        glfw::Key::Period => imgui::Key::Period,
        glfw::Key::Slash => imgui::Key::Slash,
        glfw::Key::CapsLock => imgui::Key::CapsLock,
        glfw::Key::F1 => imgui::Key::F1,
        glfw::Key::F2 => imgui::Key::F2,
        glfw::Key::F3 => imgui::Key::F3,
        glfw::Key::F4 => imgui::Key::F4,
        glfw::Key::F5 => imgui::Key::F5,
        glfw::Key::F6 => imgui::Key::F6,
        glfw::Key::F7 => imgui::Key::F7,
        glfw::Key::F8 => imgui::Key::F8,
        glfw::Key::F9 => imgui::Key::F9,
        glfw::Key::F10 => imgui::Key::F10,
        glfw::Key::F11 => imgui::Key::F11,
        glfw::Key::F12 => imgui::Key::F12,
        glfw::Key::PrintScreen => imgui::Key::PrintScreen,
        glfw::Key::ScrollLock => imgui::Key::ScrollLock,
        glfw::Key::Pause => imgui::Key::Pause,
        glfw::Key::Insert => imgui::Key::Insert,
        glfw::Key::Home => imgui::Key::Home,
        glfw::Key::PageUp => imgui::Key::PageUp,
        glfw::Key::Delete => imgui::Key::Delete,
        glfw::Key::End => imgui::Key::End,
        glfw::Key::PageDown => imgui::Key::PageDown,
        glfw::Key::Right => imgui::Key::RightArrow,
        glfw::Key::Left => imgui::Key::LeftArrow,
        glfw::Key::Down => imgui::Key::DownArrow,
        glfw::Key::Up => imgui::Key::UpArrow,
        glfw::Key::KpDivide => imgui::Key::KeypadDivide,
        glfw::Key::KpMultiply => imgui::Key::KeypadMultiply,
        glfw::Key::KpSubtract => imgui::Key::KeypadSubtract,
        glfw::Key::KpAdd => imgui::Key::KeypadAdd,
        glfw::Key::KpEnter => imgui::Key::KeypadEnter,
        glfw::Key::Kp1 => imgui::Key::Keypad1,
        glfw::Key::Kp2 => imgui::Key::Keypad2,
        glfw::Key::Kp3 => imgui::Key::Keypad3,
        glfw::Key::Kp4 => imgui::Key::Keypad4,
        glfw::Key::Kp5 => imgui::Key::Keypad5,
        glfw::Key::Kp6 => imgui::Key::Keypad6,
        glfw::Key::Kp7 => imgui::Key::Keypad7,
        glfw::Key::Kp8 => imgui::Key::Keypad8,
        glfw::Key::Kp9 => imgui::Key::Keypad9,
        glfw::Key::Kp0 => imgui::Key::Keypad0,
        glfw::Key::KpDecimal => imgui::Key::KeypadDecimal,
        glfw::Key::Menu => imgui::Key::Menu,
        glfw::Key::KpEqual => imgui::Key::KeypadEqual,
        glfw::Key::LeftControl => imgui::Key::LeftCtrl,
        glfw::Key::LeftShift => imgui::Key::LeftShift,
        glfw::Key::LeftAlt => imgui::Key::LeftAlt,
        glfw::Key::LeftSuper => imgui::Key::LeftSuper,
        glfw::Key::RightControl => imgui::Key::RightCtrl,
        glfw::Key::RightShift => imgui::Key::RightShift,
        glfw::Key::RightAlt => imgui::Key::RightAlt,
        glfw::Key::RightSuper => imgui::Key::RightSuper,
        _ => {
            // Ignore unknown keys
            return;
        }
    };

    io.add_key_event(igkey, action == glfw::Action::Press);
}

/// Handle changes in the key modifier states.
fn handle_key_modifier(io: &mut imgui::Io, keymod: &glfw::Modifiers) {
    use glfw::Modifiers;

    io.add_key_event(imgui::Key::ModShift, keymod.intersects(Modifiers::Shift));
    io.add_key_event(imgui::Key::ModCtrl, keymod.intersects(Modifiers::Control));
    io.add_key_event(imgui::Key::ModAlt, keymod.intersects(Modifiers::Alt));
    io.add_key_event(imgui::Key::ModSuper, keymod.intersects(Modifiers::Super));
}

fn load_compressed_font<P: AsRef<std::path::Path>>(cff: P) -> std::io::Result<Vec<u8>> {
    let mut f = std::fs::File::open(cff)?;
    let mut buf = Vec::<u8>::with_capacity(1024 * 1024);
    f.read_to_end(&mut buf)?;

    Ok(buf)
}
