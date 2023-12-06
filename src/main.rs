mod arcball;
mod cell_sim;
mod gl;
mod gl_utils;
mod ui_backend;
mod window;
mod pack;

fn main() -> std::io::Result<()> {
    window::MainWindow::run();
    Ok(())
}
