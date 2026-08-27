//! Streaming detection/recognition pipeline with parallel detection workers.
//!
//! ONNX Runtime sessions are stateful and not thread-safe, so parallel OCR
//! needs one session per concurrent stage. Detection is the slow stage and
//! recognition the fast one, so this pipeline runs `det_workers` detection
//! sessions in parallel (one dedicated thread each) feeding a single
//! recognition session. Images are submitted by run index; results come back
//! in submission order regardless of which worker finished which run first.

use std::{
    collections::BTreeMap,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

use anyhow::{anyhow, Result};
use image::RgbImage;
use thiserror::Error;

use crate::{
    cancellation::{is_cancelled_error, OcrCancellationToken},
    config::{InferenceOptions, RapidOcrConfig},
    detect_and_crop,
    det::TextDetector,
    rec::TextRecognizer,
    types::{OcrLine, Quad},
};

/// How long workers wait for the next job before checking shutdown; keeps a
/// cancelled pipeline from hanging on an empty queue.
const POLL_INTERVAL: Duration = Duration::from_millis(200);

/// Default capacity of the bounded detection job queue.
const DET_QUEUE_CAPACITY: usize = 64;

#[derive(Debug, Error)]
/// Errors produced by the pipeline and its workers.
pub enum PipelineError {
    /// The bounded detection queue is full (submit more than the app's window).
    #[error("OCR pipeline queue is full")]
    QueueFull,
    /// The pipeline is shutting down or the workers have exited.
    #[error("OCR pipeline is not available")]
    WorkerStopped,
    /// The OCR run was cooperatively cancelled.
    #[error("OCR run was cancelled")]
    Cancelled,
    /// OCR execution failed.
    #[error(transparent)]
    Ocr(#[from] anyhow::Error),
}

/// A single submitted run: `run` is the caller's index, `image` the stitched
/// canvas to detect and recognize.
struct DetJob {
    run: usize,
    image: RgbImage,
}

/// Detection output flowing from a detection worker to the recognition
/// worker: boxes in the canvas's space plus the crops to recognize.
struct RecJob {
    run: usize,
    result: Result<(Vec<Quad>, Vec<RgbImage>)>,
}

#[derive(Default)]
struct PipelineState {
    shutdown: AtomicBool,
}

/// Streaming OCR pipeline: `det_workers` parallel detection sessions feeding
/// one recognition session.
///
/// The pipeline owns the ONNX sessions on dedicated worker threads; the
/// handle itself is only channel endpoints, so it is `Send + Sync` and cheap
/// to clone ([`Arc`]). Results are returned strictly in submission order by
/// [`DetRecPipeline::recv`]; use [`DetRecPipeline::recv_unordered`] when the
/// caller reorders itself (e.g. mixed fallback + pipeline results) to avoid
/// stalling on a missing index.
///
/// Dropping the last handle cancels the pipeline and detaches the workers;
/// they observe the closed channels on their next poll and exit. Call
/// [`DetRecPipeline::cancel`] to abort in-flight inference immediately.
pub struct DetRecPipeline {
    det_tx: Vec<SyncSender<DetJob>>,
    next_worker: std::sync::atomic::AtomicUsize,
    results: Mutex<Receiver<(usize, Result<Vec<OcrLine>>)>>,
    ordered_buffer: Mutex<BTreeMap<usize, Result<Vec<OcrLine>>>>,
    next_ordered: std::sync::atomic::AtomicUsize,
    state: Arc<PipelineState>,
    cancellation: OcrCancellationToken,
}

impl DetRecPipeline {
    /// Builds `det_workers` detection sessions plus one recognition session,
    /// each on its own dedicated thread. `det_inference` and `rec_inference`
    /// tune the sessions' ONNX Runtime threads separately (detection is the
    /// bottleneck, recognition is fast). The classifier stage is not
    /// supported: configs with `pipeline.use_cls` are rejected.
    pub fn new(
        cfg: &RapidOcrConfig,
        det_inference: InferenceOptions,
        rec_inference: InferenceOptions,
        det_workers: usize,
    ) -> Result<Self, PipelineError> {
        if det_workers == 0 {
            return Err(anyhow!("det_workers must be greater than zero").into());
        }
        if cfg.pipeline.use_cls {
            return Err(anyhow!("DetRecPipeline does not support the classifier stage").into());
        }
        let det_cfg = cfg
            .det
            .clone()
            .ok_or_else(|| anyhow!("pipeline.use_det is true but [det] config is missing"))?;
        let rec_cfg = cfg
            .rec
            .clone()
            .ok_or_else(|| anyhow!("pipeline.use_rec is true but [rec] config is missing"))?;
        if !cfg.pipeline.use_det || !cfg.pipeline.use_rec {
            return Err(anyhow!("DetRecPipeline requires both detection and recognition").into());
        }
        det_inference.validate()?;
        rec_inference.validate()?;

        let state = Arc::new(PipelineState::default());
        let cancellation = OcrCancellationToken::new();
        let (rec_tx, rec_rx) = mpsc::sync_channel(DET_QUEUE_CAPACITY);
        let (results_tx, results_rx) = mpsc::sync_channel(DET_QUEUE_CAPACITY);

        let mut det_tx = Vec::with_capacity(det_workers);
        for i in 0..det_workers {
            let det_cfg = det_cfg.clone();
            let (rec_tx, results_tx) = (rec_tx.clone(), results_tx.clone());
            let (state, cancellation) = (Arc::clone(&state), cancellation.clone());
            let (worker_tx, det_rx) = mpsc::sync_channel(DET_QUEUE_CAPACITY);
            let pre = crate::DetPreprocess::from(cfg);
            thread::Builder::new()
                .name(format!("rapidocr-det-{i}"))
                .spawn(move || {
                    let mut detector = match TextDetector::new(det_cfg, det_inference) {
                        Ok(detector) => detector,
                        Err(error) => {
                            let _ = results_tx.send((
                                0,
                                Err(anyhow!("failed to initialize detection worker {i}: {error}")),
                            ));
                            return;
                        }
                    };
                    det_worker_main(
                        &mut detector,
                        &pre,
                        det_rx,
                        rec_tx,
                        results_tx,
                        &state,
                        &cancellation,
                    );
                })
                .map_err(|error| PipelineError::Ocr(error.into()))?;
            det_tx.push(worker_tx);
        }

        let worker_state = Arc::clone(&state);
        let worker_cancellation = cancellation.clone();
        let text_score = cfg.text_score;
        thread::Builder::new()
            .name("rapidocr-rec".to_string())
            .spawn(move || {
                let mut recognizer = match TextRecognizer::new(rec_cfg, rec_inference) {
                    Ok(recognizer) => recognizer,
                    Err(error) => {
                        let _ = results_tx.send((
                            0,
                            Err(anyhow!("failed to initialize recognition worker: {error}")),
                        ));
                        return;
                    }
                };
                rec_worker_main(
                    &mut recognizer,
                    text_score,
                    rec_rx,
                    results_tx,
                    &worker_state,
                    &worker_cancellation,
                );
            })
            .map_err(|error| PipelineError::Ocr(error.into()))?;

        Ok(Self {
            det_tx,
            next_worker: std::sync::atomic::AtomicUsize::new(0),
            results: Mutex::new(results_rx),
            ordered_buffer: Mutex::new(BTreeMap::new()),
            next_ordered: std::sync::atomic::AtomicUsize::new(0),
            state,
            cancellation,
        })
    }

    /// Submits one canvas for detection and recognition, tagged with the
    /// caller's run index. Results are received in ascending run order via
    /// [`DetRecPipeline::recv`]. Jobs are round-robined across the detection
    /// workers' per-worker queues.
    pub fn submit(&self, run: usize, image: RgbImage) -> Result<(), PipelineError> {
        if self.state.shutdown.load(Ordering::Acquire) {
            return Err(PipelineError::WorkerStopped);
        }
        let index = self.next_worker.fetch_add(1, Ordering::Relaxed) % self.det_tx.len();
        match self.det_tx[index].try_send(DetJob { run, image }) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(PipelineError::QueueFull),
            Err(TrySendError::Disconnected(_)) => Err(PipelineError::WorkerStopped),
        }
    }

    /// Blocks until the next run's lines are ready. Runs are returned in
    /// submission order (0, 1, 2, ...) regardless of worker completion order.
    /// Ordering is done handle-side on unordered completions so a missing
    /// fallback index does not stall later runs.
    pub fn recv(&self) -> Result<(usize, Vec<OcrLine>), PipelineError> {
        let mut buffer = self
            .ordered_buffer
            .lock()
            .map_err(|_| PipelineError::Ocr(anyhow!("pipeline ordered buffer poisoned")))?;
        loop {
            let next = self.next_ordered.load(Ordering::Acquire);
            if let Some(result) = buffer.remove(&next) {
                self.next_ordered.store(next + 1, Ordering::Release);
                return result
                    .map(|lines| (next, lines))
                    .map_err(|error| {
                        if is_cancelled_error(&error) {
                            PipelineError::Cancelled
                        } else {
                            PipelineError::Ocr(error)
                        }
                    });
            }
            // Need another unordered completion.
            let (run, result) = {
                let receiver = self.results.lock().map_err(|_| {
                    PipelineError::Ocr(anyhow!("pipeline results lock poisoned"))
                })?;
                match receiver.recv() {
                    Ok(v) => v,
                    Err(_) if self.state.shutdown.load(Ordering::Acquire) => {
                        return Err(PipelineError::Cancelled)
                    }
                    Err(_) => return Err(PipelineError::WorkerStopped),
                }
            };
            buffer.insert(run, result);
        }
    }

    /// Blocks until any run's lines are ready, returning its index regardless
    /// of order. The caller is responsible for reordering if needed. This
    /// avoids stalling when a run index is handled outside the pipeline
    /// (e.g. undecodable-page fallback).
    pub fn recv_unordered(&self) -> Result<(usize, Vec<OcrLine>), PipelineError> {
        // Drain ordered buffer first if it already holds unordered completions
        // that arrived via `recv` interleaving — but `recv_unordered` callers
        // never mix with `recv`, so just pull directly.
        let receiver = self.results.lock().map_err(|_| {
            PipelineError::Ocr(anyhow!("pipeline results lock poisoned"))
        })?;
        match receiver.recv() {
            Ok((run, result)) => result
                .map(|lines| (run, lines))
                .map_err(|error| {
                    if is_cancelled_error(&error) {
                        PipelineError::Cancelled
                    } else {
                        PipelineError::Ocr(error)
                    }
                }),
            Err(_) if self.state.shutdown.load(Ordering::Acquire) => {
                Err(PipelineError::Cancelled)
            }
            Err(_) => Err(PipelineError::WorkerStopped),
        }
    }

    /// Cancels in-flight inference. Workers observe the token at their next
    /// checkpoint; queued jobs are discarded once the workers poll shutdown.
    pub fn cancel(&self) {
        self.cancellation.cancel();
        self.state.shutdown.store(true, Ordering::Release);
    }

    /// The pipeline's cancellation token; the caller may also cancel through
    /// its own shared token passed to the workers indirectly.
    pub fn cancellation_token(&self) -> &OcrCancellationToken {
        &self.cancellation
    }
}

impl Drop for DetRecPipeline {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// One detection worker: owns one detection session, detects and crops each
/// queued canvas, and forwards boxes + crops to the recognition worker.
fn det_worker_main(
    detector: &mut TextDetector,
    pre: &crate::DetPreprocess,
    det_rx: Receiver<DetJob>,
    rec_tx: SyncSender<RecJob>,
    results_tx: SyncSender<(usize, Result<Vec<OcrLine>>)>,
    state: &PipelineState,
    cancellation: &OcrCancellationToken,
) {
    loop {
        let job = match det_rx.recv_timeout(POLL_INTERVAL) {
            Ok(job) => job,
            Err(RecvTimeoutError::Timeout) => {
                if state.shutdown.load(Ordering::Acquire) {
                    break;
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => break,
        };
        if state.shutdown.load(Ordering::Acquire) {
            break;
        }
        let run = job.run;
        eprintln!("[pipeline det {run}] starting detect");
        let result = catch_unwind(AssertUnwindSafe(|| {
            detect_and_crop(detector, &job.image, pre, true, cancellation)
        }));
        match result {
            Ok(Ok((boxes, crops))) => {
                eprintln!(
                    "[pipeline det {run}] detected {} box(es), forwarding to rec",
                    boxes.len()
                );
                let start = std::time::Instant::now();
                if rec_tx
                    .send(RecJob {
                        run,
                        result: Ok((boxes, crops)),
                    })
                    .is_err()
                {
                    break;
                }
                eprintln!("[pipeline det {run}] rec_tx accepted in {:?}", start.elapsed());
            }
            Ok(Err(error)) => {
                let _ = rec_tx.send(RecJob {
                    run: job.run,
                    result: Err(error),
                });
            }
            Err(_) => {
                let _ = results_tx.send((
                    job.run,
                    Err(anyhow!("detection worker panicked")),
                ));
                state.shutdown.store(true, Ordering::Release);
                break;
            }
        }
    }
}

/// The recognition worker: owns one recognition session, recognizes each
/// run's crops, and emits results as soon as each run is recognized
/// (unordered). Ordering is reconstructed handle-side by `DetRecPipeline::recv`
/// so a missing fallback index does not stall later runs; callers that
/// need unordered completions use `recv_unordered`.
fn rec_worker_main(
    recognizer: &mut TextRecognizer,
    text_score: f32,
    rec_rx: Receiver<RecJob>,
    results_tx: SyncSender<(usize, Result<Vec<OcrLine>>)>,
    state: &PipelineState,
    cancellation: &OcrCancellationToken,
) {
    loop {
        let job = match rec_rx.recv_timeout(POLL_INTERVAL) {
            Ok(job) => job,
            Err(RecvTimeoutError::Timeout) => {
                if state.shutdown.load(Ordering::Acquire) {
                    break;
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => break,
        };
        if state.shutdown.load(Ordering::Acquire) {
            break;
        }
        let start = std::time::Instant::now();
        let run = job.run;
        let crop_count = job.result.as_ref().map(|(_, c)| c.len()).unwrap_or(0);
        eprintln!("[pipeline rec {run}] received det output, recognizing {crop_count} crop(s)");
        let lines = job.result.and_then(|(boxes, crops)| {
            if crops.is_empty() {
                return Ok(Vec::new());
            }
            let recognized = recognizer.recognize_timed(&crops, cancellation)?;
            Ok(crate::build_lines(
                boxes,
                recognized.texts,
                text_score,
                true,
            ))
        });
        eprintln!("[pipeline rec {run}] recognize done in {:?}", start.elapsed());
        if results_tx.send((run, lines)).is_err() {
            return;
        }
        eprintln!("[pipeline rec {run}] emitted unordered result to stream");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn app_style_config() -> RapidOcrConfig {
        let model_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models");
        let mut cfg = RapidOcrConfig {
            pipeline: crate::config::PipelineConfig::without_cls(),
            inference: InferenceOptions::default(),
            text_score: 0.5,
            min_side_len: 30,
            max_side_len: 2000,
            min_height: 30,
            width_height_ratio: 8.0,
            det: Some(crate::config::DetConfig {
                model_path: model_dir.join("PP-OCRv6_det_tiny.onnx"),
                limit_side_len: 736,
                limit_type: crate::config::LimitType::Min,
                mean: [0.5, 0.5, 0.5],
                std: [0.5, 0.5, 0.5],
                thresh: 0.3,
                box_thresh: 0.5,
                max_candidates: 1000,
                unclip_ratio: 1.6,
                min_size: 3,
                input_limits: Default::default(),
            }),
            cls: None,
            rec: Some(crate::config::RecConfig {
                model_path: model_dir.join("korean_PP-OCRv5_rec_mobile.onnx"),
                dict_path: model_dir.join("korean_dict.txt"),
                image_shape: [3, 48, 320],
                batch_size: 6,
            }),
        };
        cfg.inference = InferenceOptions {
            intra_threads: 1,
            inter_threads: 1,
            parallel_execution: false,
            ..Default::default()
        };
        cfg
    }

    fn blank_image(width: u32, height: u32, value: u8) -> RgbImage {
        RgbImage::from_pixel(width, height, Rgb([value, value, value]))
    }

    #[test]
    #[ignore]
    fn pipeline_returns_runs_in_submission_order_with_multiple_workers() {
        let cfg = app_style_config();
        let pipeline = DetRecPipeline::new(&cfg, cfg.inference, cfg.inference, 3).unwrap();

        for run in 0..5 {
            pipeline
                .submit(run, blank_image(600, 800, 250))
                .expect("submit must succeed within the window");
        }

        let mut got = Vec::new();
        for _ in 0..5 {
            let (run, lines) = pipeline.recv().unwrap();
            got.push((run, lines.len()));
        }
        assert_eq!(got, vec![(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]);
    }

    #[test]
    #[ignore]
    fn pipeline_accepts_submissions_out_of_numeric_order_but_emits_in_order() {
        let cfg = app_style_config();
        let pipeline = DetRecPipeline::new(&cfg, cfg.inference, cfg.inference, 2).unwrap();

        pipeline.submit(2, blank_image(400, 500, 250)).unwrap();
        pipeline.submit(0, blank_image(400, 500, 250)).unwrap();
        pipeline.submit(1, blank_image(400, 500, 250)).unwrap();

        let mut runs = Vec::new();
        for _ in 0..3 {
            runs.push(pipeline.recv().unwrap().0);
        }
        assert_eq!(runs, vec![0, 1, 2]);
    }

    #[test]
    #[ignore]
    fn pipeline_cancellation_reports_cancelled() {
        let cfg = app_style_config();
        let pipeline = DetRecPipeline::new(&cfg, cfg.inference, cfg.inference, 1).unwrap();

        pipeline
            .submit(0, blank_image(1200, 1600, 250))
            .expect("submit must succeed");
        pipeline.cancel();

        let error = pipeline.recv().unwrap_err();
        assert!(matches!(error, PipelineError::Cancelled));
    }

    #[test]
    fn new_rejects_zero_workers_and_classifier() {
        let cfg = app_style_config();
        assert!(matches!(
            DetRecPipeline::new(&cfg, cfg.inference, cfg.inference, 0),
            Err(PipelineError::Ocr(_))
        ));

        let mut with_cls = cfg.clone();
        with_cls.pipeline.use_cls = true;
        assert!(matches!(
            DetRecPipeline::new(&with_cls, cfg.inference, cfg.inference, 1),
            Err(PipelineError::Ocr(_))
        ));
    }

    #[test]
    fn new_requires_det_and_rec_stages() {
        let cfg = app_style_config();
        let mut det_only = cfg.clone();
        det_only.pipeline.use_rec = false;
        assert!(matches!(
            DetRecPipeline::new(&det_only, cfg.inference, cfg.inference, 1),
            Err(PipelineError::Ocr(_))
        ));

        let mut rec_only = cfg.clone();
        rec_only.pipeline.use_det = false;
        assert!(matches!(
            DetRecPipeline::new(&rec_only, cfg.inference, cfg.inference, 1),
            Err(PipelineError::Ocr(_))
        ));
    }
}