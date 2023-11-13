use lazy_static::lazy_static;
use notify::{RecursiveMode, Watcher};

use crate::{
    compute_pipeline::ComputePipeline,
    ctx::RenderContext,
    graphics_pipeline::GraphicsPipeline,
    shaders::{Shader, ShaderDescriptor, ShaderSource},
};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

pub struct ShaderHotReload {
    pub watcher: notify::PollWatcher,
}

lazy_static! {
    pub static ref INCLUDED_SHADERS: std::sync::Mutex<HashMap<PathBuf, PathBuf>> =
        std::sync::Mutex::new(HashMap::new());
}

impl ShaderHotReload {
    pub fn new(ctx: Arc<RenderContext>, shader_dirs: &[std::path::PathBuf]) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        // use the PollWatcher and disable automatic polling
        let mut watcher = notify::PollWatcher::new(
            tx,
            notify::Config::default().with_poll_interval(Duration::from_millis(100)),
        )
        .unwrap();

        // Add a path to be watched. All files and directories at that path and
        // below will be monitored for changes.
        for shaders_dir in shader_dirs {
            watcher
                .watch(shaders_dir.as_ref(), RecursiveMode::Recursive)
                .unwrap();
        }

        std::thread::spawn(move || {
            for res in rx {
                match res {
                    Ok(event) => {
                        let path = &event.paths[0];
                        let path = path.to_str().unwrap();

                        let shader_mapping = ctx.shader_mapping.read().unwrap();

                        let Some(shader_desc) = shader_mapping.keys().find(|shader_desc| {
                            if let ShaderSource::File(shader_desc_path) = &shader_desc.source {
                                let shader_desc_path_absolute =
                                    std::fs::canonicalize(shader_desc_path).unwrap();
                                let path_absolute = std::fs::canonicalize(path).unwrap();

                                let lock = INCLUDED_SHADERS.lock().unwrap();
                                let maybe_included_shader = lock.get(&path_absolute);

                                if let Some(source_file) = maybe_included_shader {
                                    let absolute_source_file =
                                        std::fs::canonicalize(source_file).unwrap();

                                    if shader_desc_path_absolute == absolute_source_file {
                                        return true;
                                    }
                                }

                                shader_desc_path_absolute == path_absolute
                            } else {
                                false
                            }
                        }) else {
                            log::error!("No shader found for path: {}", path);
                            continue;
                        };

                        let shader_desc = shader_desc.clone();
                        let ShaderSource::File(shader_desc_path) = &shader_desc.source else {
                            log::error!(
                                "Tried to hot reload a shader that wasn't loaded from a file"
                            );
                            continue;
                        };

                        let shader_handle = shader_mapping.get(&shader_desc).unwrap();

                        match shader_desc.kind {
                            crate::shaders::ShaderKind::Fragment
                            | crate::shaders::ShaderKind::Vertex => {
                                let pipeline_mapping =
                                    ctx.graphics_pipeline_mapping.read().unwrap();
                                let Some(pipeline_desc) = pipeline_mapping.keys().find(|desc| {
                                    desc.vertex.shader == *shader_handle
                                        || desc.fragment.shader == *shader_handle
                                }) else {
                                    log::error!(
                                        "[Hot-reload] No pipeline found for shader: {:?}",
                                        shader_desc
                                    );
                                    continue;
                                };

                                let mut shader_defines = shader_desc.defines.clone();
                                shader_defines.push((
                                    "RELOAD_TIME".to_string(),
                                    Some(
                                        SystemTime::now()
                                            .duration_since(UNIX_EPOCH)
                                            .unwrap()
                                            .as_secs()
                                            .to_string(),
                                    ),
                                ));
                                let new_shader = Shader::from_source_text(
                                    &ctx,
                                    &ShaderDescriptor {
                                        defines: shader_defines,
                                        entry_point: shader_desc.entry_point.clone(),
                                        kind: shader_desc.kind,
                                        label: shader_desc.label,
                                        optimization: shader_desc.optimization,
                                        source: ShaderSource::File(shader_desc_path.clone()),
                                    },
                                );

                                let mut shader = ctx.shader_manager.get_mut(shader_handle).unwrap();
                                shader.inner = new_shader;
                                drop(shader);

                                let pipeline_handle = pipeline_mapping.get(pipeline_desc).unwrap();
                                let mut pipeline =
                                    ctx.graphics_pipelines.get_mut(pipeline_handle).unwrap();
                                let mut new_pipeline = GraphicsPipeline::new(&ctx, pipeline_desc);
                                new_pipeline.descriptor_sets = pipeline.descriptor_sets.clone();
                                new_pipeline.descriptor_pool = pipeline.descriptor_pool;
                                pipeline.inner = new_pipeline;
                            }
                            crate::shaders::ShaderKind::Compute => {
                                let pipeline_mapping = ctx.compute_pipeline_mapping.read().unwrap();
                                let Some(pipeline_desc) = pipeline_mapping
                                    .keys()
                                    .find(|desc| desc.shader == *shader_handle)
                                else {
                                    log::error!(
                                        "[Hot-reload] No pipeline found for shader: {:?}",
                                        shader_desc
                                    );
                                    continue;
                                };

                                let mut shader_defines = shader_desc.defines.clone();
                                shader_defines.push((
                                    "RELOAD_TIME".to_string(),
                                    Some(
                                        SystemTime::now()
                                            .duration_since(UNIX_EPOCH)
                                            .unwrap()
                                            .as_secs()
                                            .to_string(),
                                    ),
                                ));
                                let new_shader = Shader::from_source_text(
                                    &ctx,
                                    &ShaderDescriptor {
                                        defines: shader_defines,
                                        entry_point: shader_desc.entry_point.clone(),
                                        kind: shader_desc.kind,
                                        label: shader_desc.label,
                                        optimization: shader_desc.optimization,
                                        source: ShaderSource::File(shader_desc_path.clone()),
                                    },
                                );

                                let mut shader = ctx.shader_manager.get_mut(shader_handle).unwrap();
                                shader.inner = new_shader;
                                drop(shader);

                                let pipeline_handle = pipeline_mapping.get(pipeline_desc).unwrap();
                                let mut pipeline =
                                    ctx.compute_pipelines.get_mut(pipeline_handle).unwrap();
                                let mut new_pipeline = ComputePipeline::new(&ctx, pipeline_desc);
                                new_pipeline.descriptor_sets = pipeline.descriptor_sets.clone();
                                new_pipeline.descriptor_pool = pipeline.descriptor_pool;
                                pipeline.inner = new_pipeline;
                            }
                        }
                    }
                    Err(e) => println!("watch error: {:?}", e),
                }
            }
        });

        Self { watcher }
    }
}
