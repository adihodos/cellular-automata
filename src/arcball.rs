use glm::{dot, inverse, normalize, IVec2, Mat4, Quat, Vec2, Vec3, Vec4};
use nalgebra::UnitQuaternion;
use nalgebra_glm as glm;

#[derive(Copy, Clone, Debug)]
pub struct ArcballCamera {
    starting_pos: Vec3,
    translation: Mat4,
    center_translation: Mat4,
    rotation: UnitQuaternion<f32>,
    view_transform: Mat4,
    inverse_view_transform: Mat4,
    zoom_speed: f32,
    inv_screen: Vec2,
    is_rotating: bool,
    is_panning: bool,
    last_mouse: Option<glm::Vec2>,
}

//
// Adapted from this: https://github.com/Twinklebear/arcball/blob/master/src/lib.rs
impl ArcballCamera {
    // pub const INITIAL_FOV: f32 = 30f32;
    pub const TRANSLATION_FACTOR: f32 = 50f32;

    pub fn new(center: Vec3, initial_pos: Vec3, zoom_speed: f32, screen: IVec2) -> Self {
        let mut arcball_cam = ArcballCamera {
            starting_pos: initial_pos,
            translation: Mat4::new_translation(&initial_pos),
            center_translation: inverse(&Mat4::new_translation(&center)),
            rotation: UnitQuaternion::identity(),
            view_transform: Mat4::identity(),
            inverse_view_transform: Mat4::identity(),
            zoom_speed,
            inv_screen: Vec2::new(1f32 / screen.x as f32, 1f32 / screen.y as f32),
            is_rotating: false,
            is_panning: false,
            last_mouse: None,
        };

        arcball_cam.update_camera();
        arcball_cam
    }

    pub fn view_matrix(&self) -> Mat4 {
        self.view_transform
    }

    fn update_camera(&mut self) {
        self.view_transform =
            self.translation * self.rotation.to_homogeneous() * self.center_translation;
        self.inverse_view_transform = inverse(&self.view_transform);
    }

    pub fn update_screen(&mut self, width: i32, height: i32) {
        self.inv_screen = Vec2::new(1f32 / width as f32, 1f32 / height as f32);
    }

    pub fn rotate(&mut self, mouse_pos: Vec2) {
        self.last_mouse.take().map(|prev_mouse| {
            let mouse_cur = Vec2::new(
                (mouse_pos.x * 2f32 * self.inv_screen.x - 1f32).clamp(-1f32, 1f32),
                (1f32 - 2f32 * mouse_pos.y * self.inv_screen.y).clamp(-1f32, 1f32),
            );

            let mouse_prev = Vec2::new(
                (prev_mouse.x * 2f32 * self.inv_screen.x - 1f32).clamp(-1f32, 1f32),
                (1f32 - 2f32 * prev_mouse.y * self.inv_screen.y).clamp(-1f32, 1f32),
            );

            let mouse_cur_ball = Self::screen_to_arcball(mouse_cur);
            let mouse_prev_ball = Self::screen_to_arcball(mouse_prev);
            self.rotation = mouse_cur_ball * mouse_prev_ball * self.rotation;
            self.update_camera();
        });

        self.last_mouse = Some(mouse_pos);
    }

    pub fn pan(&mut self, mouse_cur: Vec2) {
        self.last_mouse.take().map(|mpos| {
            let mouse_delta = (mouse_cur - mpos) * Self::TRANSLATION_FACTOR;

            let zoom_dist = self.translation.m33.abs();
            let delta = Vec4::new(
                mouse_delta.x * self.inv_screen.x,
                -mouse_delta.y * self.inv_screen.y,
                0f32,
                0f32,
            ) * zoom_dist;

            let motion = self.inverse_view_transform * delta;

            self.center_translation =
                Mat4::new_translation(&motion.xyz()) * self.center_translation;
            self.update_camera();
        });
        self.last_mouse = Some(mouse_cur);
    }

    pub fn zoom(&mut self, amount: f32, _elapsed: f32) {
        let motion = Vec3::new(0f32, 0f32, amount);
        self.translation = Mat4::new_translation(&(motion * self.zoom_speed)) * self.translation;
        self.update_camera();
    }

    fn screen_to_arcball(p: Vec2) -> UnitQuaternion<f32> {
        let distance = dot(&p, &p);

        if distance <= 1f32 {
            UnitQuaternion::new_normalize(Quat::new(0f32, p.x, p.y, (1f32 - distance).sqrt()))
        } else {
            let unit_p = normalize(&p);
            UnitQuaternion::new_normalize(Quat::new(0f32, unit_p.x, unit_p.y, 0f32))
        }
    }

    pub fn handle_event(&mut self, event: &glfw::WindowEvent) {
        use glfw::{Action, MouseButtonLeft, MouseButtonRight, WindowEvent};

        match *event {
            WindowEvent::Key(keycode, _, glfw::Action::Press, ..) => {
                if keycode == glfw::Key::Backspace {
                    self.translation = Mat4::new_translation(&self.starting_pos);
                    self.center_translation = Mat4::identity();
                    self.rotation = UnitQuaternion::identity();
                    self.update_camera();
                }
            }

            WindowEvent::MouseButton(button, state, _) => match state {
                Action::Release => {
                    self.is_rotating = false;
                    self.is_panning = false;
                    self.last_mouse = None;
                }

                Action::Press => {
                    if button == MouseButtonLeft {
                        self.is_rotating = true;
                    } else if button == MouseButtonRight {
                        self.is_panning = true;
                    }
                }

                _ => {}
            },

            WindowEvent::CursorPos(x, y) => {
                if self.is_rotating {
                    self.rotate([x as f32, y as f32].into());
                } else if self.is_panning {
                    self.pan([x as f32, y as f32].into());
                }
            }

            WindowEvent::FramebufferSize(width, heigth) => self.update_screen(width, heigth),

            WindowEvent::Scroll(_, y) => {
                self.zoom(y as f32, 0f32);
            }

            _ => (),
        }
    }
}
