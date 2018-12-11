#![feature(duration_float)]

use std::fs::File;
use std::io::prelude::*;
use std::time::Instant;

use ocl::ProQue;
use ocl::prm::*;

use argparse::{ArgumentParser, StoreTrue, Store};

use serde_derive::Deserialize;
use ron::de::from_str;

#[repr(C)]
#[derive(Copy, PartialEq, Clone, Default, Debug)]
struct Target {
    axis: [f64; 3],
    dist: f64,
    id: i32,
}
unsafe impl ocl::traits::OclPrm for Target {}

enum Mode {
    SceneDebug,
    Trace(TraceOptions),
}

struct TraceOptions {
    impulse_len: u32,
    channels: u16,
    out_file: String,
    num_runs: i32,
}
struct Options {
    mode: Mode,
    width: u32,
    height: u32,
    subscene: i32,
}
#[derive(Deserialize)]
struct SceneConfig {
    num_targets: u16,
    num_runs: i32,
    num_subscenes: i32,
    impulse_len: u32,
}

fn main() {
    let mut scene_debug_only = false;
    let mut size = 1024;
    {
        let mut ap = ArgumentParser::new();
        ap.set_description("Impulse response path marcher");
        ap.refer(&mut scene_debug_only)
            .add_option(&["-d", "--debug"], StoreTrue,
                        "Only render debug image of scene");
        ap.refer(&mut size)
            .add_option(&["-s", "--size"], Store,
                        "Change dimension of kernel. Default 1024.");
        ap.parse_args_or_exit();
    }

    let mut config_src = String::new();
    let mut config_file = File::open("src/scene_config.ron").expect("Unable to open scene config file");
    config_file.read_to_string(&mut config_src).expect("Unable to read scene config file");
    let scene_config: SceneConfig = from_str(&config_src).expect("Failed to parse scene config");

    println!("{} subscenes to render. {} runs each", scene_config.num_subscenes, scene_config.num_runs);
    for subscene in 0..scene_config.num_subscenes {
        println!("Rendering subscene {}", subscene);
        run(Options {
            mode: Mode::SceneDebug,
            width: size,
            height: size,
            subscene: subscene,
        }).unwrap();
        if !scene_debug_only {
            run(Options {
                mode: Mode::Trace(TraceOptions {
                    impulse_len: scene_config.impulse_len,
                    channels: scene_config.num_targets,
                    out_file: format!("output/impulse{:03}.wav", subscene),
                    num_runs: scene_config.num_runs,
                }),
                width: size,
                height: size,
                subscene: subscene,
            }).unwrap();
        }
    }
}

fn run(options: Options) -> ocl::Result<()> {
    let (width, height) = (options.width, options.height);
    let work_size = [width,height];
    let subscene = options.subscene;

    let mut kernel_file = File::open("src/kernel.cl").unwrap();
    let mut src = String::new();
    kernel_file.read_to_string(&mut src).unwrap();

    let pro_que = ProQue::builder()
        .src(src)
        .dims(width*height)
        .build()?;

    match options.mode {
        Mode::Trace(trace_options) => {
            let impulse_len = trace_options.impulse_len;

            let min_target = pro_que.buffer_builder::<Target>().build()?;
            let total_dist = pro_que.create_buffer::<f64>()?;
            let intersect_pos = pro_que.create_buffer::<Double3>()?;
            let intersect_dir = pro_que.create_buffer::<Double3>()?;
            let closest_bounce = pro_que.create_buffer::<i32>()?;
            let debug_image = pro_que.create_buffer::<Double3>()?;
            let impulse = pro_que.create_buffer::<f64>()?;

            let debug_render_kernel = pro_que.kernel_builder("debug_render")
                .arg(&debug_image)
                .arg(&min_target)
                .arg(&total_dist)
                .arg(&intersect_pos)
                .arg(&intersect_dir)
                .arg(&closest_bounce)
                .global_work_size(work_size)
                .build()?;
            let impulse_kernel = pro_que.kernel_builder("gen_impulse_response")
                .arg(&impulse)
                .arg(impulse_len)
                .arg(&min_target)
                .arg(&total_dist)
                .arg(&intersect_pos)
                .arg(&intersect_dir)
                .arg(&closest_bounce)
                .global_work_size(work_size)
                .build()?;

            for run in 0..trace_options.num_runs {
                // do path tracing
                //
                let timer = Instant::now();
                let trace_kernel = pro_que.kernel_builder("trace")
                    .arg(run)
                    .arg(subscene)
                    .arg(&min_target)
                    .arg(&total_dist)
                    .arg(&intersect_pos)
                    .arg(&intersect_dir)
                    .arg(&closest_bounce)
                    .global_work_size(work_size)
                    .build()?;
                unsafe { trace_kernel.enq()?; }
                pro_que.finish()?;
                println!("Run {} trace time: {}", run, timer.elapsed().as_float_secs());

                // Debug render
                unsafe { debug_render_kernel.enq()?; }

                // Generate impulse response
                unsafe { impulse_kernel.enq()?; }
            }

            // render debug image
            let mut vec = vec![Double3::splat(0.0f64); debug_image.len()];
            debug_image.read(&mut vec).enq()?;
            let mut imgbuf = image::RgbImage::new(width, height);
            let max_v = vec.iter().fold(Double3::splat(std::f64::NEG_INFINITY), |best, x|
                                        Double3::new(
                                            best[0].max(x[0]),
                                            best[1].max(x[1]),
                                            best[2].max(x[2])
                                        ));
            let min_v = vec.iter().fold(Double3::splat(std::f64::INFINITY), |best, x|
                                        Double3::new(
                                            best[0].min(x[0]),
                                            best[1].min(x[1]),
                                            best[2].min(x[2])
                                        ));
            for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
                let val = (vec[(x * width + y) as usize] - min_v) / (max_v - min_v);
                *pixel = image::Rgb([(val[0]*255.0) as u8, (val[1]*255.0) as u8, (val[2]*255.0) as u8]);
            }
            imgbuf.save(&format!("output/debug{:03}.png", subscene)).unwrap();

            let mut impulse_response = vec![0.0f64; impulse_len as usize];
            impulse.read(&mut impulse_response).enq()?;
            let spec = hound::WavSpec {
                channels: trace_options.channels,
                sample_rate: 44100,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&trace_options.out_file, spec).unwrap();
            let max_v = impulse_response.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let min_v = impulse_response.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let norm_v = max_v.max(-min_v);
            for sample in &impulse_response {
                let norm_sample = sample / norm_v;
                let int_sample = (norm_sample * 32767.0) as i16;
                writer.write_sample(int_sample).unwrap();
            }
            drop(writer);
        }
        Mode::SceneDebug => {
            let scene_image = pro_que.create_buffer::<Double3>()?;

            let scene_render_kernel = pro_que.kernel_builder("scene_render")
                .arg(subscene)
                .arg(&scene_image)
                .global_work_size(work_size)
                .build()?;
            let timer = Instant::now();
            unsafe { scene_render_kernel.enq()?; }
            let mut vec = vec![Double3::splat(0.0f64); scene_image.len()];
            scene_image.read(&mut vec).enq()?;
            let mut imgbuf = image::RgbImage::new(width, height);
            let max_v = vec.iter().fold(Double3::splat(std::f64::NEG_INFINITY), |best, x|
                                        Double3::new(
                                            best[0].max(x[0]),
                                            best[1].max(x[1]),
                                            best[2].max(x[2])
                                        ));
            let min_v = vec.iter().fold(Double3::splat(std::f64::INFINITY), |best, x|
                                        Double3::new(
                                            best[0].min(x[0]),
                                            best[1].min(x[1]),
                                            best[2].min(x[2])
                                        ));
            for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
                let val = (vec[(x * width + y) as usize] - min_v) / (max_v - min_v);
                *pixel = image::Rgb([(val[0]*255.0) as u8, (val[1]*255.0) as u8, (val[2]*255.0) as u8]);
            }
            imgbuf.save(&format!("output/scene{:03}.png", subscene)).unwrap();
            println!("Scene render time: {}", timer.elapsed().as_float_secs());
        }
    }

    Ok(())
}
