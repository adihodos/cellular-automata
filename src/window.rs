use nalgebra_glm as glm;

use crate::{
    gl::{self},
    ui_backend::UiBackend,
};

#[derive(Copy, Clone)]
pub struct FrameRenderContext {
    pub delta_time: std::time::Duration,
    pub framebuffer_size: glm::IVec2,
    pub projection_view: glm::Mat4,
    pub projection_matrix: glm::Mat4,
    pub view_matrix: glm::Mat4,
}

#[derive(Copy, Clone)]
pub struct GlobalProgramOptions {
    pub debug_show_app_msg: bool,
    pub debug_show_glsys_msg: bool,
}

pub static mut G_OPTIONS: GlobalProgramOptions = GlobalProgramOptions {
    debug_show_app_msg: false,
    debug_show_glsys_msg: false,
};

pub struct MainWindow {}

impl MainWindow {
    pub fn run() {
        let _logger = flexi_logger::Logger::with(
            flexi_logger::LogSpecification::builder()
                .default(flexi_logger::LevelFilter::Debug)
                .build(),
        )
        .adaptive_format_for_stderr(flexi_logger::AdaptiveFormat::Detailed)
        .start()
        .unwrap_or_else(|e| {
            panic!("Failed to start the logger {}", e);
        });

        let mut glfw = glfw::init(|err, s| {
            log::error!("GLFW init error {} :: {s}", err);
        })
        .expect("Failed to initialize GLFW");

        let (content_scale, phys_size, pos, work_area, name) = glfw
            .with_primary_monitor(|_, pmon| {
                pmon.map(|m| {
                    (
                        m.get_content_scale(),
                        m.get_physical_size(),
                        m.get_pos(),
                        m.get_workarea(),
                        m.get_name(),
                    )
                })
            })
            .expect("Failed to query primary monitor!");

        log::info!(
            "Primary monitor: {}, physical size {:?}, position {:?}, work area {:?}, content scale {:?}",
            name.unwrap_or_else(|| "unknown".to_string()),
            phys_size,
            pos,
            work_area,
            content_scale
        );

        use glfw::WindowHint;
        glfw.window_hint(WindowHint::ClientApi(glfw::ClientApiHint::OpenGl));
        glfw.window_hint(WindowHint::ContextCreationApi(
            glfw::ContextCreationApi::Native,
        ));
        glfw.window_hint(WindowHint::ContextVersion(4, 6));
        glfw.window_hint(WindowHint::OpenGlDebugContext(true));
        glfw.window_hint(WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));
        glfw.window_hint(WindowHint::OpenGlForwardCompat(true));
        glfw.window_hint(WindowHint::DoubleBuffer(true));
        glfw.window_hint(WindowHint::DepthBits(Some(24)));
        glfw.window_hint(WindowHint::StencilBits(Some(8)));
        glfw.window_hint(WindowHint::Decorated(false));

        let (mut window, event_pump) = glfw
            .create_window(
                work_area.2 as u32,
                work_area.3 as u32,
                "Cellular automata simulation",
                glfw::WindowMode::Windowed,
            )
            .expect("Failed to create main window");

        use glfw::Context;
        window.make_current();
        window.set_all_polling(true);

        gl::load_with(|s| window.get_proc_address(s));

        unsafe {
            gl::DebugMessageCallback(Some(debug_message_callback), std::ptr::null());
        }

        let (mut fbw, mut fbh) = window.get_framebuffer_size();
        let (mut w, mut h) = window.get_size();

        let mut ui_backend = UiBackend::new().expect("Failed to create UI backend");

        let mut cell_sim = crate::cell_sim::CellSimulation::new();
        let mut camera = crate::arcball::ArcballCamera::new(
            [0f32, 0f32, 0f32].into(),
            [0f32, 0f32, -72f32].into(),
            1f32,
            [fbw, fbh].into(),
        );

        let mut projection = glm::perspective_fov(
            glm::radians(&glm::Vec1::new(75f32)).x,
            fbw as f32,
            fbh as f32,
            0.1f32,
            512f32,
        );

        let mut timepoint = std::time::Instant::now();

        while !window.should_close() {
            glfw.poll_events();
            for (_, event) in glfw::flush_messages(&event_pump) {
                match event {
                    glfw::WindowEvent::Key(glfw::Key::Escape, _, glfw::Action::Press, _) => {
                        window.set_should_close(true);
                    }

                    glfw::WindowEvent::FramebufferSize(w, h) => {
                        if w > 0 && h > 0 {
                            fbw = w;
                            fbh = h;

                            projection = glm::perspective_fov(
                                glm::radians(&glm::Vec1::new(75f32)).x,
                                fbw as f32,
                                fbh as f32,
                                0.1f32,
                                512f32,
                            );
                        }
                    }

                    glfw::WindowEvent::Size(wx, wy) => {
                        if wx > 0 && wy > 0 {
                            w = wx;
                            h = wy;
                        }
                    }

                    _ => {}
                }

                if !ui_backend.handle_event(&event) {
                    camera.handle_event(&event);
                    cell_sim.handle_event(&event);
                }
            }

            let time_now = std::time::Instant::now();
            let elapsed = time_now - timepoint;
            timepoint = time_now;

            let frame_context = FrameRenderContext {
                delta_time: elapsed,
                framebuffer_size: [fbw, fbh].into(),
                view_matrix: camera.view_matrix(),
                projection_matrix: projection,
                projection_view: projection * camera.view_matrix(),
            };

            cell_sim.update(&frame_context);

            {
                let ui = ui_backend.prepare_frame(&mut window, fbw, fbh, w, h);
                cell_sim.ui(ui);
            }

            unsafe {
                gl::ViewportIndexedfv(0, [0f32, 0f32, fbw as f32, fbh as f32].as_ptr());
                gl::ClearNamedFramebufferfv(0, gl::COLOR, 0, [0f32, 0f32, 0f32, 1f32].as_ptr());
                gl::ClearNamedFramebufferfi(0, gl::DEPTH_STENCIL, 0, 1f32, 0);
            }

            cell_sim.render(&frame_context);
            ui_backend.render();

            window.swap_buffers();
        }
    }
}

extern "system" fn debug_message_callback(
    source: gl::types::GLenum,
    gltype: gl::types::GLenum,
    id: gl::types::GLuint,
    severity: gl::types::GLenum,
    _length: gl::types::GLsizei,
    message: *const gl::types::GLchar,
    _user_param: *mut std::ffi::c_void,
) {
    unsafe {
        if G_OPTIONS.debug_show_app_msg
            && severity == gl::DEBUG_SEVERITY_NOTIFICATION
            && source == gl::DEBUG_SOURCE_APPLICATION
        {
            let msg = std::ffi::CStr::from_ptr(message as *const i8);

            eprintln!(
                "[OpenGL - custom {id}] :: {}",
                msg.to_str().unwrap_or("error decondig error :)")
            );
            return;
        }

        if G_OPTIONS.debug_show_glsys_msg
            && (severity == gl::DEBUG_SEVERITY_HIGH || severity == gl::DEBUG_SEVERITY_MEDIUM)
        {
            let msg = std::ffi::CStr::from_ptr(message as *const i8);

            log::error!(
            "OpenGL error:\n\tSource: {source}\n\tGLType: {gltype}\n\tseverity {severity}\n\tId: {id}\n\tMessage: {}",
            msg.to_str().unwrap_or("error decondig error :)")
        );
        }
    }
}
