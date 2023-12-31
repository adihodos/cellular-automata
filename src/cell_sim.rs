use crate::gl::{self, types::*};
use crate::gl_utils::*;
use crate::window::FrameRenderContext;
use enum_iterator::Sequence;
use nalgebra_glm as glm;
use rand::prelude::Distribution;
use std::io::{Error, ErrorKind, Result};
use std::mem::{size_of, size_of_val};
use std::ops::RangeInclusive;

use enterpolation::linear::ConstEquidistantLinear;
use palette::LinSrgb;

use self::mesh_geometry::{UNIT_CUBE_INDICES, UNIT_CUBE_WIRE_INDICES};

#[derive(Copy, Clone)]
#[repr(C)]
struct VSUniformTransforms {
    projection: glm::Mat4,
    view: glm::Mat4,
    cell_states: u32,
    cell_coloring: u32,
    world_size: u32,
}

#[derive(Copy, Clone)]
#[repr(C)]
struct VSUniformLighting {
    ambient_light: glm::Vec3,
    pad0: u32,
    directional_light: glm::Vec3,
    pad1: u32,
    directional_light_color: glm::Vec3,
}

#[derive(Copy, Clone)]
#[repr(C)]
struct CSUniformEvalRule {
    live_neighbours: [u32; 32],
    birth_neighbours: [u32; 32],
    live_count: u32,
    birth_count: u32,
    states: u32,
    neigbor_rule: u32,
}

#[derive(Copy, Clone)]
#[repr(C)]
struct CellStateGPU {
    state: u32,
}

#[derive(Copy, Clone)]
#[repr(C)]
struct DrawElementsIndirectCommand {
    count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: u32,
    base_instance: u32,
}

#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
enum RandomClusterType {
    Cube,
    Sphere,
}

#[derive(Copy, Clone)]
struct InitialClusterGenOptions {
    /// If true, a different random distribution is used for each axis, inseatd of common one
    independent_axis: bool,
    cluster_size: u32,
    cluster_type: RandomClusterType,
}

impl InitialClusterGenOptions {
    const MIN_CLUSTER_SIZE: u32 = 4;
}

#[rustfmt::skip]
mod mesh_geometry {
    #[derive(Copy, Clone)]
    pub struct VertexPN {
	pub p: [f32; 3],
	pub n: [f32; 3]
    }

    #[rustfmt::skip]
    pub const UNIT_CUBE_VERTICES: [VertexPN; 24] = [
	VertexPN { p:  [-0.5f32, -0.5f32, -0.5f32], n: [0f32, 0f32, -1f32] },
	VertexPN { p:  [-0.5f32, 0.5f32, -0.5f32], n:[0f32, 0f32, -1f32] },
	VertexPN { p:  [0.5f32, 0.5f32, -0.5f32], n:[0f32, 0f32, -1f32] },
	VertexPN { p:  [0.5f32, -0.5f32, -0.5f32], n:[0f32, 0f32, -1f32] },

	VertexPN { p:  [-0.5f32, -0.5f32, 0.5f32], n: [0f32, 0f32, 1f32] },
	VertexPN { p:  [0.5f32, -0.5f32, 0.5f32], n:[0f32, 0f32, 1f32] },
	VertexPN { p:  [0.5f32, 0.5f32, 0.5f32], n:[0f32, 0f32, 1f32] },
	VertexPN { p:  [-0.5f32, 0.5f32, 0.5f32], n:[0f32, 0f32, 1f32] },
	
	VertexPN { p:  [-0.5f32, 0.5f32, -0.5f32], n: [0f32, 1f32, 0f32] },
	VertexPN { p:  [-0.5f32, 0.5f32, 0.5f32], n:[0f32, 1f32, 0f32] },
	VertexPN { p:  [0.5f32, 0.5f32, 0.5f32], n:[0f32, 1f32, 0f32] },
	VertexPN { p:  [0.5f32, 0.5f32, -0.5f32], n:[0f32, 1f32, 0f32] },
	
	VertexPN { p:  [-0.5f32, -0.5f32, -0.5f32], n: [0f32, -1f32, 0f32] },
	VertexPN { p:  [0.5f32, -0.5f32, -0.5f32], n:[0f32, -1f32, 0f32] },
	VertexPN { p:  [0.5f32, -0.5f32, 0.5f32], n:[0f32, -1f32, 0f32] },
	VertexPN { p:  [-0.5f32, -0.5f32, 0.5f32], n:[0f32, -1f32, 0f32] },
	
	VertexPN { p:  [-0.5f32, -0.5f32, 0.5f32], n: [-1f32, 0f32, 0f32] },
	VertexPN { p:  [-0.5f32, 0.5f32, 0.5f32], n:[-1f32, 0f32, 0f32] },
	VertexPN { p:  [-0.5f32, 0.5f32, -0.5f32], n:[-1f32, 0f32, 0f32] },
	VertexPN { p:  [-0.5f32, -0.5f32, -0.5f32], n:[-1f32, 0f32, 0f32] },
	
	VertexPN { p:  [0.5f32, -0.5f32, -0.5f32], n: [1f32, 0f32, 0f32] },
	VertexPN { p:  [0.5f32, 0.5f32, -0.5f32], n:[1f32, 0f32, 0f32] },
	VertexPN { p:  [0.5f32, 0.5f32, 0.5f32], n:[1f32, 0f32, 0f32] },
	VertexPN { p:  [0.5f32, -0.5f32, 0.5f32], n:[1f32, 0f32, 0f32] },
    ];

    pub const UNIT_CUBE_INDICES: [u32; 36] = [
	// front face index data
        0,
        1,
        2,
        0,
        2,
        3,
        
        // back face index data
        4,
        5,
        6,
        4,
        6,
        7,
        
        // top face index data
        8,
        9,
        10,
        8,
        10,
        11,
        
        // bottom face index data
        12,
        13,
        14,
        12,
        14,
        15,
        
        // left face index data
        16,
        17,
        18,
        16,
        18,
        19,
        
        // right face index data
        20,
        21,
        22,
        20,
        22,
        23,        
    ];

    pub const UNIT_CUBE_WIRE_INDICES : [u32; 24] = [
	// front face
	0, 1,
	1, 2,
	2, 3,
	3, 0,
	// back face
	4, 5,
	5, 6,
	6, 7,
	7, 4,
	//
	8, 9,
	10, 11,
	//
	12, 15,
	13, 14
    ];
}

#[allow(dead_code)]
struct CellSimulationRenderState {
    ubo_transform: UniqueBuffer,
    ubo_lighting: UniqueBuffer,
    ubo_rules: UniqueBuffer,
    ssbo_packed_cells: UniqueBuffer,
    draw_indirect_buffer: UniqueBuffer,
    copy_draw_ind_buf: UniqueBuffer,
    instances_current: UniqueBuffer,
    instances_previous: UniqueBuffer,
    vertexbuffer: UniqueBuffer,
    indexbuffer: UniqueBuffer,
    vertexarray: UniqueVertexArray,
    vertexshader: UniqueShaderProgram,
    fragmentshader: UniqueShaderProgram,
    compute_shader: UniqueShaderProgram,
    pipeline: UniquePipeline,
    vertexshader_basic: UniqueShaderProgram,
    pipeline_basic: UniquePipeline,
    gradient_tex: UniqueTexture,
    sampler: UniqueSampler,
    gpu_buf_capacity: usize,
}

impl CellSimulationRenderState {
    fn new(max_instances: usize) -> Result<CellSimulationRenderState> {
        let ubo_transform =
            create_buffer(size_of::<VSUniformTransforms>(), Some(gl::MAP_WRITE_BIT))
                .ok_or_else(|| Error::new(ErrorKind::Other, "Failed to create UBO"))?;

        let ubo_lighting =
            create_buffer(size_of::<VSUniformLighting>(), Some(gl::MAP_WRITE_BIT))
                .ok_or_else(|| Error::new(ErrorKind::Other, "Failed to create lights UBO"))?;

        let ubo_rules = create_buffer(size_of::<CSUniformEvalRule>(), Some(gl::MAP_WRITE_BIT))
            .ok_or_else(|| Error::new(ErrorKind::Other, "Failed to create UBO"))?;

        label_object(gl::BUFFER, *ubo_rules, "UBO cell rules");

        let ssbo_packed_cells = create_buffer(
            size_of::<u32>() * (max_instances + 1),
            Some(gl::MAP_WRITE_BIT),
        )
        .ok_or_else(|| Error::new(ErrorKind::Other, "Failed to create SSBO"))?;

        label_object(gl::BUFFER, *ssbo_packed_cells, "SSBO Packed cells data");

        let draw_indirect_buffer = UniqueBuffer::new(unsafe {
            let mut buf: GLuint = 0;
            gl::CreateBuffers(1, &mut buf);
            gl::NamedBufferStorage(
                buf,
                size_of::<DrawElementsIndirectCommand>() as GLsizeiptr,
                std::ptr::null(),
                gl::MAP_WRITE_BIT,
            );

            buf
        })
        .expect("Failed to create draw indirect buffer");

        label_object(
            gl::BUFFER,
            *draw_indirect_buffer,
            "[[Draw commands indirect buffer]]",
        );

        let instances_current = create_buffer(
            size_of::<CellStateGPU>() * max_instances,
            Some(gl::MAP_WRITE_BIT),
        )
        .expect("Failed to create SSBO");

        let instances_previous = create_buffer(
            size_of::<CellStateGPU>() * max_instances,
            Some(gl::MAP_WRITE_BIT),
        )
        .expect("Failed to create SSBO");

        let vertexbuffer = UniqueBuffer::new(unsafe {
            let mut buf: GLuint = 0;
            gl::CreateBuffers(1, &mut buf);

            gl::NamedBufferStorage(
                buf,
                size_of_val(&mesh_geometry::UNIT_CUBE_VERTICES) as GLsizeiptr,
                mesh_geometry::UNIT_CUBE_VERTICES.as_ptr() as *const _,
                0,
            );

            buf
        })
        .ok_or_else(|| Error::new(ErrorKind::Other, "Failed to create vertex buffer"))?;

        let indexbuffer = UniqueBuffer::new(unsafe {
            let mut buf: GLuint = 0;
            gl::CreateBuffers(1, &mut buf);

            let indices = mesh_geometry::UNIT_CUBE_INDICES
                .iter()
                .chain(mesh_geometry::UNIT_CUBE_WIRE_INDICES.iter())
                .map(|i| *i)
                .collect::<Vec<_>>();

            let bytes = size_of_val(indices.as_slice()) as GLsizeiptr;
            log::info!("Index bytes {bytes}");

            gl::NamedBufferStorage(buf, bytes, indices.as_ptr() as *const _, 0);

            buf
        })
        .ok_or_else(|| Error::new(ErrorKind::Other, "Failed to create index buffer"))?;

        let vertexarray = UniqueVertexArray::new(unsafe {
            let mut vao: GLuint = 0;
            gl::CreateVertexArrays(1, &mut vao);

            gl::VertexArrayAttribFormat(vao, 0, 3, gl::FLOAT, gl::FALSE, 0);
            gl::VertexArrayAttribBinding(vao, 0, 0);
            gl::EnableVertexArrayAttrib(vao, 0);

            gl::VertexArrayAttribFormat(vao, 1, 3, gl::FLOAT, gl::FALSE, 12);
            gl::VertexArrayAttribBinding(vao, 1, 0);
            gl::EnableVertexArrayAttrib(vao, 1);

            gl::VertexArrayVertexBuffer(
                vao,
                0,
                *vertexbuffer,
                0,
                size_of::<mesh_geometry::VertexPN>() as i32,
            );
            gl::VertexArrayElementBuffer(vao, *indexbuffer);

            vao
        })
        .ok_or_else(|| Error::new(ErrorKind::Other, "Failed to create vertex array object"))?;

        let vertexshader = ShaderProgramBuilder::new()
            .add_file(&"data/shaders/instanced.vert")
            .add_macros(&[("LIGHTING_ON", None), ("PACKED_DATA", None)])
            .compile(ShaderType::Vertex)?;

        let fragmentshader =
            create_shader_program_from_file("data/shaders/instanced.frag", ShaderType::Fragment)
                .map_err(|e| Error::new(ErrorKind::Other, e))?;

        let pipeline = UniquePipeline::new(unsafe {
            let mut pipeline: GLuint = 0;
            gl::CreateProgramPipelines(1, &mut pipeline);
            gl::UseProgramStages(pipeline, gl::VERTEX_SHADER_BIT, *vertexshader);
            gl::UseProgramStages(pipeline, gl::FRAGMENT_SHADER_BIT, *fragmentshader);

            pipeline
        })
        .ok_or_else(|| Error::new(ErrorKind::Other, "Failed to create graphics pipeline"))?;

        let compute_shader = ShaderProgramBuilder::new()
            .add_file(&"data/shaders/cellsim.comp")
            .add_macros(&[("LIGHTING_ON", None)])
            .compile(ShaderType::Compute)?;

        let vertexshader_basic =
            create_shader_program_from_file("data/shaders/basic.vert", ShaderType::Vertex)?;
        let pipeline_basic = UniquePipeline::new(unsafe {
            let mut pipeline: GLuint = 0;
            gl::CreateProgramPipelines(1, &mut pipeline);
            gl::UseProgramStages(pipeline, gl::VERTEX_SHADER_BIT, *vertexshader_basic);
            gl::UseProgramStages(pipeline, gl::FRAGMENT_SHADER_BIT, *fragmentshader);

            pipeline
        })
        .ok_or_else(|| Error::new(ErrorKind::Other, "Failed to create graphics pipeline"))?;

        use enterpolation::Generator;
        let color_palette = ConstEquidistantLinear::<f32, _, 3>::equidistant_unchecked([
            LinSrgb::new(0.95, 0.90, 0.30),
            LinSrgb::new(0.70, 0.10, 0.20),
            LinSrgb::new(0.0, 0.05, 0.20),
        ]);

        let gradient_tex = UniqueTexture::new(unsafe {
            let mut tex: GLuint = 0;
            gl::CreateTextures(gl::TEXTURE_1D_ARRAY, 1, &mut tex);
            gl::TextureStorage2D(tex, 1, gl::RGBA8, 256, 2);

            let pixels = (0..256)
                .map(|i| {
                    let c = color_palette.gen(i as f32 / 256f32).into_format();
                    [c.red, c.green, c.blue, 255u8]
                })
                .collect::<Vec<_>>();

            gl::TextureSubImage2D(
                tex,
                0,
                0,
                0,
                256,
                1,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                pixels.as_ptr() as *const u8 as *const _,
            );

            tex
        })
        .expect("Failed to create texture");

        let sampler = UniqueSampler::new(unsafe {
            let mut smp: GLuint = 0;
            gl::CreateSamplers(1, &mut smp);
            gl::SamplerParameteri(smp, gl::TEXTURE_MIN_FILTER, gl::LINEAR as GLint);
            gl::SamplerParameteri(smp, gl::TEXTURE_MAG_FILTER, gl::LINEAR as GLint);

            smp
        })
        .expect("Failed to create sampler");

        let copy_draw_ind_buf = create_buffer(
            size_of::<DrawElementsIndirectCommand>(),
            Some(gl::MAP_READ_BIT),
        )
        .expect("Failed to create buffer");

        label_object(
            gl::BUFFER,
            *copy_draw_ind_buf,
            "[[COPY draw indirect buffer]]",
        );

        Ok(CellSimulationRenderState {
            ubo_rules,
            ubo_lighting,
            ubo_transform,
            ssbo_packed_cells,
            draw_indirect_buffer,
            instances_current,
            instances_previous,
            vertexbuffer,
            indexbuffer,
            vertexarray,
            vertexshader,
            fragmentshader,
            pipeline,
            compute_shader,
            vertexshader_basic,
            pipeline_basic,
            gradient_tex,
            sampler,
            copy_draw_ind_buf,
            gpu_buf_capacity: max_instances,
        })
    }

    fn set_initial_state(&mut self, cells: &[CellStateGPU], world_size: u32) {
        let live_cells = cells.iter().filter(|c| c.state != 0).count() as u32;

        if cells.len() > self.gpu_buf_capacity {
            let _ = OpenGLDebugScopePush::new(0x1, "[[Resizing buffers]]");

            log::info!(
                "Resizing buffers {} -> {}",
                self.gpu_buf_capacity,
                size_of_val(cells)
            );

            self.instances_current = create_buffer(size_of_val(cells), Some(gl::MAP_WRITE_BIT))
                .expect("Failed to create SSBO");
            self.instances_previous = create_buffer(size_of_val(cells), Some(gl::MAP_WRITE_BIT))
                .expect("Failed to create SSBO");

            self.ssbo_packed_cells = create_buffer(
                size_of::<u32>() * (cells.len() + 1),
                Some(gl::MAP_WRITE_BIT),
            )
            .expect("Failed to resize SSBO");

            self.gpu_buf_capacity = cells.len();
        }

        [*self.instances_current, *self.instances_previous]
            .iter()
            .for_each(|&grid_buf| {
                UniqueBufferMapping::new(grid_buf, gl::MAP_WRITE_BIT).map(|mut buf| {
                    let mut dst = buf.as_mut_ptr::<CellStateGPU>();
                    cells.iter().for_each(|c| unsafe {
                        dst.write(CellStateGPU { state: c.state });
                        dst = dst.add(1);
                    });
                });
            });

        UniqueBufferMapping::new(*self.ssbo_packed_cells, gl::MAP_WRITE_BIT).map(|mut buf| {
            let mut dst = buf.as_mut_ptr::<u32>();

            //
            // world box
            let s = world_size / 2;
            unsafe {
                *dst = s | (s << 8) | (s << 16) | (0xFF << 24);
                dst = dst.add(1);
            }

            for z in 0..world_size {
                for y in 0..world_size {
                    for x in 0..world_size {
                        let c = cells[(z * world_size * world_size + y * world_size + x) as usize];
                        if c.state != 0 {
                            unsafe {
                                *dst = x | (y << 8) | (z << 16) | (c.state << 24);
                                dst = dst.add(1);
                            }
                        };
                    }
                }
            }
        });

        UniqueBufferMapping::new(*self.draw_indirect_buffer, gl::MAP_WRITE_BIT).map(
            |mut dbuf| unsafe {
                let cmd = DrawElementsIndirectCommand {
                    count: 36,
                    instance_count: live_cells,
                    first_index: 0,
                    base_vertex: 0,
                    base_instance: 1,
                };
                *dbuf.as_mut_ptr::<DrawElementsIndirectCommand>() = cmd;
            },
        );
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum NeighbourEval {
    Moore,
    VonNeumann,
}

#[derive(Clone)]
struct CellRule {
    name: String,
    survival: Vec<u32>,
    birth: Vec<u32>,
    states: u32,
    eval: NeighbourEval,
}

impl CellRule {
    fn detailed_description(&self) -> String {
        use rustring_builder::StringBuilder;
        let mut sb = StringBuilder::new();
        sb.append_line("\u{f0877} Neighbours to survive: [");
        self.survival.iter().enumerate().for_each(|(i, x)| {
            if i > 0 {
                sb.push(',');
            }
            sb.append(x);
        });

        sb.push(']');

        sb.append_line("\u{f0877} Neighbours to be born: [");
        self.birth.iter().enumerate().for_each(|(i, x)| {
            if i > 0 {
                sb.push(',');
            }
            sb.append(x);
        });
        sb.push(']');
        sb.append_line("\u{f11ef} States: ");
        sb.append(self.states);
        sb.append_line("\u{f0877} Neighbour evaluation method: ");

        match self.eval {
            NeighbourEval::Moore => sb.append("Moore"),
            NeighbourEval::VonNeumann => sb.append("VonNeumann"),
        };

        sb.to_string()
    }
}

struct CellRuleBuilder {
    r: CellRule,
}

impl CellRuleBuilder {
    fn new(name: &str) -> Self {
        Self {
            r: CellRule {
                name: name.into(),
                survival: Vec::new(),
                birth: Vec::new(),
                states: 0,
                eval: NeighbourEval::Moore,
            },
        }
    }

    fn add_survival_rule(mut self, neighbor_count: &[u32]) -> Self {
        self.r.survival.extend(neighbor_count.iter());
        self
    }

    fn add_survival_rule_range(mut self, r: &[RangeInclusive<u32>]) -> Self {
        self.r
            .survival
            .extend(r.iter().map(|a| a.clone().into_iter()).flatten());

        self
    }

    fn add_birth_rule(mut self, ncount: &[u32]) -> Self {
        self.r.birth.extend(ncount.iter());
        self
    }

    fn add_birth_rule_range(mut self, nrcount: &[RangeInclusive<u32>]) -> Self {
        self.r
            .birth
            .extend(nrcount.iter().map(|r| r.clone().into_iter()).flatten());
        self
    }

    fn set_states(mut self, states: u32) -> Self {
        self.r.states = states;
        self
    }

    fn set_eval_func(mut self, eval_func: NeighbourEval) -> Self {
        self.r.eval = eval_func;
        self
    }

    fn build(mut self) -> CellRule {
        assert!(self.r.states != 0, "Cell cannot be born dead");
        assert_ne!(self.r.birth.is_empty(), true, "Birth rules not specified");

        self.r.survival.sort_unstable();
        self.r.survival.dedup();

        self.r.birth.sort_unstable();
        self.r.birth.dedup();

        self.r
    }
}

#[derive(Copy, Clone, Debug, Sequence, Eq, PartialEq)]
#[repr(u8)]
enum ColoringMethod {
    DistanceToCenter,
    CenterToRGB,
    StateLerpGradient,
}

impl ColoringMethod {
    fn description(c: ColoringMethod) -> &'static str {
        match c {
            ColoringMethod::DistanceToCenter => "Distance to center",
            ColoringMethod::CenterToRGB => "Cell position to RGB coords",
            ColoringMethod::StateLerpGradient => "Linear interp of cell states",
        }
    }
}

struct CellSimulationParams {
    rule: usize,
    update_freq: std::time::Duration,
    coloring_method: ColoringMethod,
    live_cells: u32,
    world_size: u32,
    draw_world_box: bool,
    cluster: InitialClusterGenOptions,
}

impl CellSimulationParams {
    const MIN_UPDATE_FREQ: std::time::Duration = std::time::Duration::from_millis(9000);
    const MAX_UPDATE_FREQ: std::time::Duration = std::time::Duration::from_millis(10);
    const WORLD_SIZE_MIN: u32 = 8;
    const WORLS_SIZE_MAX: u32 = 128;
    const WORLD_INITIAL_SIZE: u32 = 64;
}

pub struct CellSimulation {
    render_state: CellSimulationRenderState,
    grid_current: Vec<CellStateGPU>,
    initial_grid: Vec<CellStateGPU>,
    eval_rules: Vec<CellRule>,
    params: CellSimulationParams,
    time_elapsed: std::time::Duration,
    paused: bool,
    lighting: VSUniformLighting,
}

impl CellSimulation {
    pub fn new() -> CellSimulation {
        let cluster = InitialClusterGenOptions {
            independent_axis: true,
            cluster_size: 8,
            cluster_type: RandomClusterType::Cube,
        };

        let grid_current =
            generate_grid(CellSimulationParams::WORLD_INITIAL_SIZE as usize, &cluster);
        let initial_grid = grid_current.clone();
        let mut render_state = CellSimulationRenderState::new(
            (CellSimulationParams::WORLD_INITIAL_SIZE
                * CellSimulationParams::WORLD_INITIAL_SIZE
                * CellSimulationParams::WORLD_INITIAL_SIZE) as usize,
        )
        .expect("Failed to initialize cell rendering state");

        render_state.set_initial_state(&grid_current, CellSimulationParams::WORLD_INITIAL_SIZE);

        CellSimulation {
            grid_current,
            initial_grid,
            render_state,
            eval_rules: build_simulation_rules(),
            params: CellSimulationParams {
                rule: 0,
                update_freq: std::time::Duration::from_millis(60),
                coloring_method: ColoringMethod::CenterToRGB,
                live_cells: 0,
                world_size: CellSimulationParams::WORLD_INITIAL_SIZE,
                draw_world_box: true,
                cluster,
            },
            time_elapsed: std::time::Duration::from_millis(0),
            paused: true,
            lighting: VSUniformLighting {
                ambient_light: [0.1f32, 0.1f32, 0.1f32].into(),
                pad0: 0,
                directional_light: glm::normalize(&[1f32, 1f32, 1f32].into()),
                pad1: 0,
                directional_light_color: glm::make_vec3(&[1f32, 1f32, 1f32]),
            },
        }
    }

    pub fn update(&mut self, ctx: &FrameRenderContext) {
        if self.paused {
            return;
        }

        self.time_elapsed += ctx.delta_time;

        if self.time_elapsed < self.params.update_freq {
            return;
        }

        self.time_elapsed = std::time::Duration::from_millis(0);
        self.step_simulation();
    }

    pub fn render(&mut self, ctx: &FrameRenderContext) {
        unsafe {
            gl::CopyNamedBufferSubData(
                *self.render_state.draw_indirect_buffer,
                *self.render_state.copy_draw_ind_buf,
                0,
                0,
                std::mem::size_of::<DrawElementsIndirectCommand>() as _,
            );
        }

        UniqueBufferMapping::new(
            *self.render_state.ubo_transform,
            gl::MAP_WRITE_BIT | gl::MAP_INVALIDATE_BUFFER_BIT,
        )
        .map(|mut ubo_tf| unsafe {
            *(ubo_tf.as_mut_ptr::<VSUniformTransforms>()) = VSUniformTransforms {
                projection: ctx.projection_view,
                view: ctx.view_matrix,
                cell_states: self.eval_rules[self.params.rule].states,
                cell_coloring: self.params.coloring_method as u32,
                world_size: self.params.world_size,
            };
        });

        UniqueBufferMapping::new(
            *self.render_state.ubo_lighting,
            gl::MAP_WRITE_BIT | gl::MAP_INVALIDATE_BUFFER_BIT,
        )
        .map(|mut buf| unsafe {
            std::ptr::copy_nonoverlapping(
                &self.lighting as *const _,
                buf.as_mut_ptr::<VSUniformLighting>(),
                1,
            );
        });

        unsafe {
            let _ = OpenGLDebugScopePush::new(0x1, "[[Drawing cells]]");
            gl::UseProgram(0);
            gl::Enable(gl::DEPTH_TEST);
            gl::Enable(gl::CULL_FACE);

            gl::BindBuffersBase(
                gl::UNIFORM_BUFFER,
                0,
                2,
                [
                    *self.render_state.ubo_transform,
                    *self.render_state.ubo_lighting,
                ]
                .as_ptr() as *const _,
            );

            gl::BindBuffersBase(
                gl::SHADER_STORAGE_BUFFER,
                0,
                1,
                [*self.render_state.ssbo_packed_cells].as_ptr(),
            );
            gl::BindBuffer(
                gl::DRAW_INDIRECT_BUFFER,
                *self.render_state.draw_indirect_buffer,
            );
            gl::BindVertexArray(*self.render_state.vertexarray);

            gl::BindProgramPipeline(*self.render_state.pipeline);
            gl::BindTextureUnit(0, *self.render_state.gradient_tex);
            gl::BindSampler(0, *self.render_state.sampler);
            gl::DrawElementsIndirect(gl::TRIANGLES, gl::UNSIGNED_INT, std::ptr::null());

            if self.params.draw_world_box {
                //
                // draw world box as wireframe

                gl::BindProgramPipeline(*self.render_state.pipeline_basic);
                gl::ProgramUniform4f(
                    *self.render_state.vertexshader_basic,
                    0,
                    1f32,
                    0f32,
                    0f32,
                    1f32,
                );

                let transform =
                    ctx.projection_view * glm::Mat4::new_scaling(self.params.world_size as f32);
                gl::ProgramUniformMatrix4fv(
                    *self.render_state.vertexshader_basic,
                    1,
                    1,
                    gl::FALSE,
                    transform.as_ptr(),
                );

                gl::DrawElements(
                    gl::LINES,
                    UNIT_CUBE_WIRE_INDICES.len() as _,
                    gl::UNSIGNED_INT,
                    std::ptr::null::<u32>().offset(UNIT_CUBE_INDICES.len() as isize) as *const _,
                );
            }
        }

        unsafe {
            let mut data =
                std::mem::MaybeUninit::<DrawElementsIndirectCommand>::zeroed().assume_init();
            gl::GetNamedBufferSubData(
                *self.render_state.copy_draw_ind_buf,
                0,
                std::mem::size_of_val(&data) as GLsizeiptr,
                &mut data as *mut _ as *mut _,
            );
            self.params.live_cells = data.instance_count;
        }
    }

    fn reset_grid(&mut self) {
        self.grid_current = generate_grid(self.params.world_size as usize, &self.params.cluster);
        self.initial_grid = self.grid_current.clone();
        self.render_state
            .set_initial_state(&self.grid_current, self.params.world_size);
    }

    fn step_simulation(&mut self) {
        // transforms UBO
        {
            let _ = OpenGLDebugScopePush::new(0x1, "[[Updating CS uniforms]]");

            // eval rules UBO
            UniqueBufferMapping::new(
                *self.render_state.ubo_rules,
                gl::MAP_WRITE_BIT | gl::MAP_INVALIDATE_BUFFER_BIT,
            )
            .map(|mut ubo| unsafe {
                let mut x = CSUniformEvalRule {
                    live_neighbours: [0u32; 32],
                    birth_neighbours: [0u32; 32],
                    live_count: self.eval_rules[self.params.rule].survival.len() as u32,
                    birth_count: self.eval_rules[self.params.rule].birth.len() as u32,
                    states: self.eval_rules[self.params.rule].states,
                    neigbor_rule: self.eval_rules[self.params.rule].eval as u32,
                };

                for i in 0..self.eval_rules[self.params.rule].survival.len() {
                    x.live_neighbours[i] = self.eval_rules[self.params.rule].survival[i];
                }

                for i in 0..self.eval_rules[self.params.rule].birth.len() {
                    x.birth_neighbours[i] = self.eval_rules[self.params.rule].birth[i];
                }

                *(ubo.as_mut_ptr::<CSUniformEvalRule>()) = x;
            });
        }

        {
            let _ = OpenGLDebugScopePush::new(0x1, "[[Updating indirect draw cmd buffer]]");
            // draw indirect command SSBO
            UniqueBufferMapping::new(*self.render_state.draw_indirect_buffer, gl::MAP_WRITE_BIT)
                .map(|mut buf| unsafe {
                    *buf.as_mut_ptr::<DrawElementsIndirectCommand>() =
                        DrawElementsIndirectCommand {
                            count: 36,
                            instance_count: 0,
                            first_index: 0,
                            base_vertex: 0,
                            base_instance: 1,
                        };
                });
        }

        unsafe {
            let _ = OpenGLDebugScopePush::new(0x1, "[[Running CS cell simulation]]");
            gl::BindBuffersBase(
                gl::UNIFORM_BUFFER,
                1,
                1,
                [*self.render_state.ubo_rules].as_ptr(),
            );

            gl::BindBuffersBase(
                gl::SHADER_STORAGE_BUFFER,
                0,
                4,
                [
                    *self.render_state.draw_indirect_buffer,
                    *self.render_state.instances_current,
                    *self.render_state.instances_previous,
                    *self.render_state.ssbo_packed_cells,
                ]
                .as_ptr(),
            );

            gl::UseProgram(*self.render_state.compute_shader);
            gl::ProgramUniform1ui(*self.render_state.compute_shader, 0, self.params.world_size);
            gl::DispatchCompute(
                self.params.world_size / 4,
                self.params.world_size / 4,
                self.params.world_size / 4,
            );
            gl::MemoryBarrier(gl::SHADER_STORAGE_BARRIER_BIT | gl::COMMAND_BARRIER_BIT);
        }

        std::mem::swap(
            &mut self.render_state.instances_current,
            &mut self.render_state.instances_previous,
        );
    }

    pub fn ui(&mut self, ui: &mut imgui::Ui) {
        ui.window("Cell automata parameters - \u{e7a8} \u{f033d} \u{f17a}")
            .size([400f32, 800f32], imgui::Condition::FirstUseEver)
            .build(|| {
                let (text, color) = if self.paused {
                    ("State: [PAUSED]", [1f32, 0f32, 0f32, 1f32])
                } else {
                    ("State: [RUNNING]", [0f32, 1f32, 0f32, 1f32])
                };

                ui.text_colored(color, text);
                ui.same_line();
                if ui.button("\u{f0e62} Restart with random seed (g)") {
                    self.reset_grid();
                }
                ui.separator();

                ui.text("\u{f016b} Cell evaluation rules");

                let mut eval_rule = self.params.rule;
                if let Some(_) = ui.begin_combo("##", &self.eval_rules[self.params.rule].name) {
                    for (idx, r) in self.eval_rules.iter().enumerate() {
                        let is_selected = eval_rule == idx;
                        if ui.selectable(&r.name) {
                            eval_rule = idx;
                        }

                        if is_selected {
                            ui.set_item_default_focus();
                        }
                    }
                }

                //
                // reset grid when changing rule otherwise results will not be correct
                if eval_rule != self.params.rule {
                    self.params.rule = eval_rule;
                    self.reset_grid()
                }

                ui.text_wrapped(self.eval_rules[self.params.rule].detailed_description());
                ui.separator();

                ui.text_colored([0f32, 1f32, 0f32, 1f32], "\u{e22e} initialization");

                let mut regen_sim = false;
                let cluster_types = [
                    ("\u{f01a7} Cube", RandomClusterType::Cube),
                    ("\u{f1954} Sphere", RandomClusterType::Sphere),
                ];
                cluster_types.iter().for_each(|(btn_name, btn_val)| {
                    if ui.radio_button_bool(btn_name, *btn_val == self.params.cluster.cluster_type)
                    {
                        self.params.cluster.cluster_type = *btn_val;
                        regen_sim = true;
                    }
                });

                regen_sim |= ui.slider(
                    "\u{f1b3} cluster size",
                    InitialClusterGenOptions::MIN_CLUSTER_SIZE,
                    self.params.world_size,
                    &mut self.params.cluster.cluster_size,
                );

                regen_sim |= ui.checkbox(
                    "\u{f0d49} sample each axis independently",
                    &mut self.params.cluster.independent_axis,
                );

                if regen_sim {
                    self.reset_grid();
                }

                ui.separator();

                let mut update_freq = self.params.update_freq.as_millis() as u32;
                if ui.slider(
                    "\u{f06b0} Update frequency (ms)",
                    CellSimulationParams::MIN_UPDATE_FREQ.as_millis() as u32,
                    CellSimulationParams::MAX_UPDATE_FREQ.as_millis() as u32,
                    &mut update_freq,
                ) {
                    self.params.update_freq = std::time::Duration::from_millis(update_freq as u64);
                }

                ui.separator();

                if let Some(_) = ui.begin_combo(
                    "\u{e22b} Coloring",
                    ColoringMethod::description(self.params.coloring_method),
                ) {
                    enum_iterator::all::<ColoringMethod>().for_each(|e| {
                        let selected = e == self.params.coloring_method;

                        if ui.selectable(ColoringMethod::description(e)) {
                            self.params.coloring_method = e;
                        }

                        if selected {
                            ui.set_item_default_focus();
                        }
                    });
                }

                ui.checkbox("\u{f01a7} Draw world box", &mut self.params.draw_world_box);

                ui.separator();
                ui.text(format!("\u{f0ed5} Live cells: {}", self.params.live_cells));

                ui.separator();
                ui.text("\u{f0a30} \u{e20f} Show OpenGL debug output");
                unsafe {
                    use crate::window::G_OPTIONS;
                    ui.checkbox("application messages", &mut G_OPTIONS.debug_show_app_msg);
                    ui.checkbox("GPU driver", &mut G_OPTIONS.debug_show_glsys_msg);
                }

                ui.separator();
                ui.group(|| {
                    ui.text("\u{e22e} World size");
                    Pow2Iterator::new(
                        CellSimulationParams::WORLD_SIZE_MIN,
                        CellSimulationParams::WORLS_SIZE_MAX,
                    )
                    .for_each(|p| {
                        if ui.radio_button_bool(
                            format!("{p}x{p}x{p} \u{f1b3}"),
                            self.params.world_size == p,
                        ) {
                            self.params.world_size = p;
                            self.reset_grid();
                        }
                    });
                });
            });
    }

    pub fn handle_event(&mut self, event: &glfw::WindowEvent) -> bool {
        use glfw::{Action, Key, WindowEvent};

        match *event {
            WindowEvent::Key(key, _, Action::Press, _) => {
                let mut handled = true;

                match key {
                    Key::R => {
                        self.step_simulation();
                    }

                    Key::G => {
                        self.reset_grid();
                    }

                    Key::Backslash => {
                        self.grid_current = self.initial_grid.clone();
                        self.render_state
                            .set_initial_state(&self.grid_current, self.params.world_size);
                    }

                    Key::Pause => {
                        self.paused = !self.paused;
                    }

                    _ => {
                        handled = false;
                    }
                }
                handled
            }
            _ => false,
        }
    }
}

struct RandomCoordsGenerator {
    dist_x: rand::distributions::Uniform<usize>,
    dist_y: rand::distributions::Uniform<usize>,
    dist_z: rand::distributions::Uniform<usize>,
    independent_axis: bool,
}

impl RandomCoordsGenerator {
    fn new(range: std::ops::RangeInclusive<usize>) -> Self {
        let dist = rand::distributions::Uniform::new_inclusive(range.start(), range.end());
        Self {
            dist_x: dist.clone(),
            dist_y: dist.clone(),
            dist_z: dist.clone(),
            independent_axis: false,
        }
    }

    fn new_independent(
        x: std::ops::RangeInclusive<usize>,
        y: std::ops::RangeInclusive<usize>,
        z: std::ops::RangeInclusive<usize>,
    ) -> Self {
        use rand::distributions::Uniform;
        Self {
            dist_x: Uniform::new_inclusive(x.start(), x.end()),
            dist_y: Uniform::new_inclusive(y.start(), y.end()),
            dist_z: Uniform::new_inclusive(z.start(), z.end()),
            independent_axis: true,
        }
    }

    fn sample(&self, rng: &mut dyn rand::RngCore) -> [usize; 3] {
        if self.independent_axis {
            [
                self.dist_x.sample(rng),
                self.dist_y.sample(rng),
                self.dist_z.sample(rng),
            ]
        } else {
            [
                self.dist_x.sample(rng),
                self.dist_x.sample(rng),
                self.dist_x.sample(rng),
            ]
        }
    }
}

fn generate_grid(tiles: usize, opts: &InitialClusterGenOptions) -> Vec<CellStateGPU> {
    let mut rng = rand::thread_rng();

    let mut grid_current = vec![CellStateGPU { state: 0 }; tiles * tiles * tiles];

    let init_cluster_size: usize = if tiles == opts.cluster_size as usize {
        opts.cluster_size - 1
    } else {
        opts.cluster_size
    } as usize;

    let cluster_center: usize = tiles as usize / 2;
    let center = glm::Vec3::new(
        cluster_center as f32,
        cluster_center as f32,
        cluster_center as f32,
    );

    let coords_range =
        cluster_center - init_cluster_size / 2..=cluster_center + init_cluster_size / 2;

    let cgen = if opts.independent_axis {
        RandomCoordsGenerator::new_independent(
            coords_range.clone(),
            coords_range.clone(),
            coords_range.clone(),
        )
    } else {
        RandomCoordsGenerator::new(coords_range)
    };

    if opts.cluster_type == RandomClusterType::Cube {
        (0..init_cluster_size * init_cluster_size * init_cluster_size).for_each(|_| {
            let [x, y, z] = cgen.sample(&mut rng);
            grid_current[z * tiles * tiles + y * tiles + x].state = 1;
        });
    } else {
        (0..init_cluster_size * init_cluster_size * init_cluster_size).for_each(|_| {
            let [x, y, z] = cgen.sample(&mut rng);

            let pt = glm::make_vec3(&[x as f32, y as f32, z as f32]);
            let dist = glm::length(&(pt - center));

            if dist > (init_cluster_size / 2) as f32 {
                return;
            }

            grid_current[z * tiles * tiles + y * tiles + x].state = 1;
        });
    }

    grid_current
}

fn build_simulation_rules() -> Vec<CellRule> {
    vec![
        CellRuleBuilder::new("4/4/5")
            .add_survival_rule(&[4])
            .add_birth_rule(&[4])
            .set_states(5)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        //
        // Amoeba (Jason Rampe) 9-26/5-7,12-13,15/5/M
        CellRuleBuilder::new("Amoeba")
            .add_survival_rule_range(&[9..=26])
            .add_birth_rule_range(&[5..=7, 12..=13])
            .add_birth_rule(&[15])
            .set_states(5)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        //
        //
        CellRuleBuilder::new("Architecture")
            .add_survival_rule_range(&[4..=6])
            .add_birth_rule(&[3])
            .set_states(2)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        //
        // Builder 1 (Jason Rampe) 2,6,9/4,6,8-9/10/M
        CellRuleBuilder::new("Builder #1")
            .add_survival_rule(&[2, 6, 9])
            .add_birth_rule(&[4, 6])
            .add_birth_rule_range(&[8..=9])
            .set_states(10)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        //
        // Builder 2 (Jason Rampe) 5-7/1/2/M
        CellRuleBuilder::new("Builder #2")
            .add_survival_rule_range(&[5..=7])
            .add_birth_rule(&[1])
            .set_states(2)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        //
        // Clouds 1 (Jason Rampe) 13-26/13-14,17-19/2/M
        CellRuleBuilder::new("Clouds #1")
            .add_survival_rule_range(&[13..=26])
            .add_birth_rule_range(&[13..=14, 17..=19])
            .set_states(2)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        //
        // Clouds 2 (Jason Rampe) 12-26/13-14/2/M
        CellRuleBuilder::new("Clouds #2")
            .add_survival_rule_range(&[12..=26])
            .add_birth_rule_range(&[13..=14])
            .set_states(2)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        //
        // Construction (Jason Rampe) 0-2,4,6-11,13-17,21-26/9-10,16,23-24/2/M
        CellRuleBuilder::new("Construction")
            .add_survival_rule(&[4])
            .add_survival_rule_range(&[0..=2, 6..=11, 13..=17, 21..=26])
            .add_birth_rule(&[16])
            .add_birth_rule_range(&[9..=10, 23..=24])
            .set_states(2)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        //
        // Coral (Jason Rampe) 5-8/6-7,9,12/4/M
        CellRuleBuilder::new("Coral")
            .add_survival_rule_range(&[5..=8])
            .add_birth_rule(&[9, 12])
            .add_birth_rule_range(&[6..=7])
            .set_states(4)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        //
        // Crystal growth 1
        CellRuleBuilder::new("Crystal growth #1")
            .add_survival_rule(&[1])
            .add_survival_rule_range(&[0..=6])
            .add_birth_rule(&[1, 3])
            .set_states(2)
            .set_eval_func(NeighbourEval::VonNeumann)
            .build(),
        //
        // Crystal growth 2
        CellRuleBuilder::new("Crystal growth #2")
            // .add_survival_rule(&[1])
            .add_survival_rule_range(&[1..=2])
            .add_birth_rule(&[1, 3])
            .set_states(5)
            .set_eval_func(NeighbourEval::VonNeumann)
            .build(),
        //
        // Diamond Growth (Jason Rampe) 5-6/1-3/7/N
        CellRuleBuilder::new("Diamond growth")
            // .add_survival_rule(&[1])
            .add_survival_rule_range(&[5..=6])
            .add_birth_rule_range(&[1..=3])
            .set_states(7)
            .set_eval_func(NeighbourEval::VonNeumann)
            .build(),
        //
        // Pulse Waves (Jason Rampe) 3/1-3/10/M
        CellRuleBuilder::new("Pulse wave")
            .add_survival_rule(&[3])
            .add_birth_rule_range(&[1..=3])
            .set_states(10)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        //
        // Pyroclastic
        CellRuleBuilder::new("Pyroclastic")
            .add_survival_rule_range(&[4..=7])
            .add_birth_rule_range(&[6..=8])
            .set_states(10)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        //
        // Sample 1 (Jason Rampe) 10-26/5,8-26/4/M
        CellRuleBuilder::new("Sample #1")
            .add_survival_rule_range(&[10..=26])
            .add_birth_rule_range(&[8..=26])
            .add_birth_rule(&[5])
            .set_states(4)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        //
        // Symmetry (Jason Rampe) /2/10/M
        CellRuleBuilder::new("Symmetry")
            .add_birth_rule(&[2])
            .set_states(10)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        //
        // von Neumann Builder (Jason Rampe) 1-3/1,4-5/5/N
        CellRuleBuilder::new("von Neumann builder")
            .add_birth_rule(&[1])
            .add_birth_rule_range(&[4..=5])
            .add_survival_rule_range(&[1..=3])
            .set_states(5)
            .set_eval_func(NeighbourEval::VonNeumann)
            .build(),
        // Expanding Shell (Jason Rampe) 6,7-9,11,13,15-16,18.6-10,13-14,16,18-19,22-25/5/M
        CellRuleBuilder::new("Expanding shell")
            .add_birth_rule(&[6, 11, 13, 18])
            .add_birth_rule_range(&[7..=9, 15..=16])
            .add_survival_rule(&[16])
            .add_survival_rule_range(&[6..=10, 13..=14, 18..=19, 22..=25])
            .set_states(5)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        // More Structures (Jason Rampe) 7-26/4/4/M
        CellRuleBuilder::new("More structures")
            .add_survival_rule_range(&[7..=26])
            .add_birth_rule(&[4])
            .set_states(4)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        // Pulse Waves (Jason Rampe) 3/1-3/10/M
        CellRuleBuilder::new("Pulse waves")
            .add_survival_rule(&[3])
            .add_birth_rule_range(&[1..=3])
            .set_states(10)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        // Shells (Jason Rampe) 3,5,7,9,11,15,17,19,21,23-24,26/3,6,8-9,11,14-17,19,24/7/M
        CellRuleBuilder::new("Shells")
            .add_survival_rule(&[3, 5, 7, 9, 11, 15, 17, 19, 21, 26])
            .add_survival_rule_range(&[23..=24])
            .add_birth_rule(&[3, 6, 11, 19, 24])
            .add_birth_rule_range(&[8..=9, 14..=17])
            .set_states(7)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        // Slow Decay 1 (Jason Rampe) 13-26/10-26/3/M
        CellRuleBuilder::new("Slow decay #1")
            .add_survival_rule_range(&[13..=26])
            .add_birth_rule_range(&[10..=26])
            .set_states(3)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        // Stable Structures (Evan Wallace) 13-26/14-19/2/M
        CellRuleBuilder::new("Stable structures")
            .add_survival_rule_range(&[13..=26])
            .add_birth_rule_range(&[14..=19])
            .set_states(2)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        // Single Point Replication (Jason Rampe) /1/2/M
        CellRuleBuilder::new("Single point replication")
            .add_birth_rule(&[1])
            .set_states(2)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        // 3D Brain (Jason Rampe) /4/2/M
        CellRuleBuilder::new("3D brain")
            .add_birth_rule(&[4])
            .set_states(2)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
        // Construction (Jason Rampe) 0-2,4,6-11,13-17,21-26/9-10,16,23-24/2/M
        CellRuleBuilder::new("Construction #2")
            .add_survival_rule(&[4])
            .add_survival_rule_range(&[0..=2, 6..=11, 13..=17, 21..=26])
            .add_birth_rule(&[16])
            .add_birth_rule_range(&[9..=10, 23..=24])
            .set_states(2)
            .set_eval_func(NeighbourEval::Moore)
            .build(),
    ]
}

fn roundup_next_power_of2(n: u32) -> u32 {
    if n == 0 {
        0
    } else {
        let mut n = n - 1;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;

        n + 1
    }
}

struct Pow2Iterator {
    start: u32,
    end: u32,
}

impl Pow2Iterator {
    fn new(s: u32, e: u32) -> Self {
        assert!(e >= s);
        Self {
            start: roundup_next_power_of2(s),
            end: roundup_next_power_of2(e),
        }
    }
}

impl std::iter::Iterator for Pow2Iterator {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start > self.end {
            return None;
        }

        if self.start == self.end {
            self.start = 0xffffffffu32;
            return Some(self.end);
        }

        let val = self.start;
        self.start = roundup_next_power_of2(self.start + 1).min(self.end);
        Some(val)
    }
}
