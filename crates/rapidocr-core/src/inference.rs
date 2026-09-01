use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use ndarray::{Array4, ArrayD};
use ort::{session::builder::GraphOptimizationLevel, session::Session, value::TensorRef};

use crate::{cancellation::OcrCancellationToken, config::InferenceOptions};

pub(crate) struct OnnxSession {
    session: Session,
}

impl OnnxSession {
    pub(crate) fn new(
        model_path: impl AsRef<Path>,
        options: InferenceOptions,
        dynamic_input_shape: bool,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();
        if !model_path.exists() {
            bail!(
                "ONNX model file not found at {}; check the config model path or prepare the model cache",
                model_path.display()
            );
        }
        if !model_path.is_file() {
            bail!("ONNX model path is not a file: {}", model_path.display());
        }
        options.validate()?;

        // Helper to build a Session::builder with the common options for a given EP.
        let build = |ep_is_directml: bool| -> Result<Session> {
            let mut b = Session::builder()
                .map_err(|e| anyhow!(e.to_string()))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| anyhow!(e.to_string()))?
                .with_intra_threads(options.intra_threads)
                .map_err(|e| anyhow!(e.to_string()))?
                .with_inter_threads(options.inter_threads)
                .map_err(|e| anyhow!(e.to_string()))?
                .with_parallel_execution(options.parallel_execution)
                .map_err(|e| anyhow!(e.to_string()))?
                .with_memory_pattern(!dynamic_input_shape && !ep_is_directml)
                .map_err(|e| anyhow!(e.to_string()))?;
            if ep_is_directml {
                #[cfg(feature = "directml")]
                {
                    b = b
                        .with_execution_providers([
                            ort::ep::DirectML::default().build().error_on_failure(),
                        ])
                        .map_err(|e| anyhow!(e.to_string()))?;
                }
                #[cfg(not(feature = "directml"))]
                {
                    bail!("DirectML execution provider requested but `directml` feature not enabled");
                }
            } else {
                b = b
                    .with_execution_providers([ort::ep::CPU::default()
                        .with_arena_allocator(options.enable_cpu_mem_arena)
                        .build()])
                    .map_err(|e| anyhow!(e.to_string()))?;
            }
            let s = b
                .commit_from_file(model_path)
                .map_err(|e| anyhow!(e.to_string()))
                .with_context(|| format!("failed to load ONNX model {}", model_path.display()))?;
            Ok(s)
        };

        let is_directml = options.execution_provider == crate::config::ExecutionProvider::DirectMl;
        let session = if is_directml {
            match build(true) {
                Ok(s) => {
                    eprintln!(
                        "[rapidocr-core] DirectML EP active for {}",
                        model_path.display()
                    );
                    s
                }
                Err(e) => {
                    eprintln!(
                        "[rapidocr-core] DirectML EP init failed for {}: {e} – falling back to CPU",
                        model_path.display()
                    );
                    build(false)?
                }
            }
        } else {
            build(false)?
        };
        Ok(Self { session })
    }

    pub(crate) fn run_f32(
        &mut self,
        input: &Array4<f32>,
        cancellation: &OcrCancellationToken,
    ) -> Result<ArrayD<f32>> {
        cancellation.checkpoint()?;
        let run_options = cancellation.begin_onnx_run()?;
        let output = {
            let outputs = self
                .session
                .run_with_options(
                    ort::inputs![TensorRef::from_array_view(input)?],
                    run_options.options(),
                )
                .map_err(|e| {
                    if cancellation.is_cancelled() {
                        crate::cancellation::OcrCancelled.into()
                    } else {
                        anyhow!(e.to_string())
                    }
                })?;
            outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| anyhow!(e.to_string()))?
                .to_owned()
        };
        drop(run_options);
        cancellation.checkpoint()?;
        Ok(output)
    }
}
