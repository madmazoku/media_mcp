# Changelog

All notable changes to the `media_mcp` project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Each release version `0.<N>.0` directly corresponds to task **AI-<N>**.

## [0.14.0] - 2026-10-28 (AI-14)

### Added
- **CLI Version Flag (`--version`, `-v`)**:
  - Added standalone version inquiry flags `bin/cli.cmd --version` and `-v` operating without configuration requirements.
  - Introduced central `__version__ = "0.14.0"` in `src/core/__init__.py` and synchronized `pyproject.toml`.
  - Added comprehensive version parity test asserting exact equality across CLI output, `core.__version__`, `pyproject.toml`, and `CHANGELOG.md`.
- **First-Last Frame Conditioning (`end_image`)**:
  - Implemented bidirectional conditioning for video scenario blocks across Wan 2.2 (`WanFirstLastFrameToVideo` in 1-frame mode and `WanLatentPrefixToVideo` with 4-channel binary subframe mask in 5D prefix mode) and LTX-2.3 (`LTXVAddGuide` with `euler_ancestral`).
  - Added support for dynamic runtime references:
    - `"image_path": "$start"`: performs img2img relative to the block's initial keyframe (`anchor_image` for Block 1, or previous block's `-tail.png`).
    - `"image_path": "$anchor"`: performs img2img relative to the scene's base `anchor_image`.
    - `"keyframe_path": "$start"` / `"$anchor"`: directly reuses the existing frame without invoking image generation.
  - Spec literals preservation: preserved literal strings `"$start"` and `"$anchor"` in serialized `*-spec.json` files to ensure idempotent re-runs without stale transient file paths.
  - Tail image extraction contract: `-tail.png` is now extracted directly from `vae_decode` (`ImageFromBatch` + `SaveImage`) across all non-final blocks (`save_tail_image = not is_last`) regardless of `overlap` value.
- **Documentation & Knowledge Base**:
  - Added Section 14 to `MEDIA_MCP_KNOWLEDGE_BASE.md`: in-depth mathematical analysis of phase cancellation and dissolve noise in LTX-2.3 token guides vs Wan 2.2 36-channel concat inpainting, complete with decision matrix.
  - Updated `video-scenario` skill with runtime references and First-Last Frame best practices.

### Changed
- Switched default LTX-Video sampler from deterministic `euler` to stochastic `euler_ancestral` to allow smooth guidance token adaptation without rigid freeze.

## [0.13.0] - 2026-10-27 (AI-13)

### Added
- **Hierarchical Breadcrumb Error Formatting**:
  - Implemented nested try/except blocks across scenario execution layers (`scenario`, `scene`, `anchor_image`, `block`, FFmpeg `assembly`).
  - Standardized error schema: `failed scenario '<scenario>': scene #N ('<scene>'): block #M ('<block>'): failed video/anchor_image: <cause>`.
- **Pre-Flight Anchor & Aspect Ratio Validation**:
  - Added pre-flight check requiring `prompt` when neither `keyframe_path` nor `image_path` is specified for anchor images.
  - Enforced disk file existence verification for `image_path` even when `fit='crop'` or other fit modes are requested.
- **Optional Prompt for Img2Img & Passive Crop Adaptation**:
  - Allowed `prompt` to be omitted in `schemas/requests/image_generate.json` when `image_path` is supplied.
  - When `image_path` with `fit='crop'` lacks prompt and mask, executed passive cropping via Lanczos and spec saving without empty-prompt diffusion.

### Fixed
- Removed duplicate `"Failed: "` prefix concatenation in `JobManager.fail_job`.

## [0.12.0] - 2026-10-23 (AI-12)

### Added
- **Asynchronous Token Pipeline**:
  - `video_generate` performs fast synchronous pre-flight validation and spec serialization, registers a job in `JobManager`, and returns a queued status token immediately.
  - Introduced dedicated single daemon worker thread with `queue.Queue` executing jobs in strict FIFO order under `_GPU_LOCK`.
- **Job Lifecycle & Management Tools**:
  - Added `media_get_job` and `media_remove_job` tools with canonical lifecycle statuses (`queued`, `running`, `completed`, `failed`, `cancelled`).
  - Added terminal status protection, skipping of cancelled queued jobs upon acquiring GPU lock, and prioritized eviction (`cancelled` -> `failed` -> `completed`).
- **Dry-Run Validation**:
  - Added optional `dry_run` flag to `video_generate` for instant full scenario and prompt validation without queuing GPU tasks.

### Changed
- **CLI Architecture**:
  - Replaced blocking CLI execution with `PollWaitJobStrategy` featuring live progress polling on stderr and clean Ctrl+C GPU interruption.
  - Removed deprecated `video_validate` and obsolete `REPO_DIR` references.

## [0.11.0] - 2026-10-17 (AI-11)

### Added
- **Latent AdaIN Normalization & Color Stabilization**:
  - Implemented custom node `LTXVAdainLatent` in `ComfyUI-MediaMCP` supporting 4D and 5D latent tensors.
  - Implemented two-stage graph AdaIN normalization (`stage1_adain_core` and `stage2_adain_core`) for Wan generators (`wan-fast`, `wan-full`).
  - Added Latent AdaIN color normalization to `WanLatentPrefixRestore`.
- **Native LTX-2.3 Latent Guides**:
  - Configured 5D latent prefix conditioning via `LTXVAddLatentGuide` + `LTXVCropGuides` + `LTXVAdainLatent`.
  - Added automatic Distilled LoRA strength scaling: `0.7` with `euler_ancestral` in distilled mode (`cfg <= 1.5`), `0.2` with `euler` in full guidance mode (`cfg > 1.5`).
  - Implemented LTX 1-frame mode using `LTXVPreprocess(img_compression=33)` + `LTXVImgToVideoInplace(strength=0.9)`.
- **Immutable Per-Engine Duration Limits**:
  - Enforced strict immutable limits: `6.0s` for Wan (`wan-fast`/`wan-full`) and `16.0s` for LTX (`ltx`).

## [0.10.0] - 2026-10-17 (AI-10)

### Added
- **Modular Test Suites**:
  - Replaced monolithic test script with targeted suites: `test_discovery.py`, `test_image_pipeline.py`, `test_mcp_and_cli.py`, `test_schemas.py`, and `test_video_pipeline.py`.
- **Comprehensive Documentation Suite (`docs/`)**:
  - Added `ARCHITECTURE.md`, `ARTIFACTS_AND_NAMING.md`, `CLIENT_INTEGRATION.md`, `IMAGE_PIPELINE.md`, `MODELS_AND_ENGINES.md`, `NETWORK_AND_QUEUE.md`, `SCENARIO_DESIGN.md`, and `RULES.md`.
- **Packaging & Core Hardening**:
  - Added `pyproject.toml`.
  - Hardened LoRA inspection, trigger discovery, and binary safetensors header parsing without torch dependency.

## [0.9.0] - 2026-10-17 (AI-9)

### Added
- **HTTP Keep-Alive Connection Pooling**:
  - Replaced unpooled `urllib.request.urlopen` with persistent `requests.Session()` with connection pooling for ComfyUI and Automatic1111 backends, eliminating Windows localhost socket churn.
  - Added mandatory `s.trust_env = False` to bypass environment/sandbox proxy resolution for `127.0.0.1`.
- **Active ComfyUI Queue Tracking**:
  - Implemented lifecycle tracking (`queue_pending` -> `queue_running` -> `/history`) in `wait_for_prompt()`, preventing timeout exhaustion during long backend queues.
- **GPU Mutex Lock**:
  - Added global `_GPU_LOCK` in `src/core/tools.py` ensuring serialization between A1111 and ComfyUI generation calls to prevent concurrent VRAM contention.

## [0.8.0] - 2026-10-16 (AI-8)

### Added
- **Strict Configuration Enforcement**:
  - Eliminated heuristic auto-discovery of configuration in CWD; configuration is strictly required via `--config` or `MEDIA_CONFIG`.
- **POSIX Path Unification**:
  - Standardized all internal path handling and JSON serialization on `pathlib.Path.resolve().as_posix()`, eliminating escaped double backslashes on Windows.
- **Interactive CLI `--help`**:
  - Added dynamic schema formatter generating parameter tables, types, and constraints for CLI users.
- **Cross-Platform Launchers**:
  - Added standard launchers in `bin/` (`cli.cmd`, `cli.sh`, `run_media_stdio.*`, `run_media_sse.*`).

## [0.7.0] - 2026-10-16 (AI-7)

### Added
- **Inpainting Masks & Auto-Dimensions**:
  - Added continuous grayscale mask support across SDXL, FLUX.1, and FLUX.2 Klein (via `SplitSigmasDenoise`).
  - Added native dimension calculation targeting ~1.0 MP optimal diffusion size for img2img.
- **Fit Modes (`crop` | `outpaint`)**:
  - Added pure PIL crop and outpaint strategies with automatic feathered gradient masks.
- **Keyframe Path Decoupling**:
  - Decoupled pre-existing `keyframe_path` from scenario root overrides; normalized denoise to 1.0 for pure txt2img.

## [0.6.0] - 2026-10-16 (AI-6)

### Added
- **Img2Img Support in Core Tools**:
  - Added optional `image_path` and `denoise` parameters to `image_generate` across A1111 and ComfyUI FLUX.1 / FLUX.2.
  - Streamlined scenario `anchor_image` schema to shot-level overrides (`prompt`, `negative_prompt`, `image_path`, `denoise`, `seed`, `loras`).
  - Implemented union merge for LoRAs between global scenario image and shot anchor.

## [0.5.0] - 2026-10-16 (AI-5)

### Changed
- **Unified Engine Architecture**:
  - Streamlined architecture into a single unified engine with transport-agnostic tools dispatch (CLI, Stdio MCP, SSE).
  - Enforced canonical JSON Schema contracts, strict `scenario_path` resolution, and standard run prefix naming (`-final.png`, `-final.mp4`, `-spec.json`).
  - Added regex pattern filtering in `media_get_loras`.

## [0.4.0] - 2026-10-16 (AI-4)

### Added
- **Centralized JSON Schema Validation**:
  - Consolidated schemas under `schemas/` with strict Draft 2020-12 validation and conditional engine rules (`sdxl`, `flux1`, `flux2`, `wan-fast`, `wan`, `ltx`).
- **Dynamic Timing Validation**:
  - Enforced `overlap_duration < clip_duration` and `prefix_duration + clip_duration < max_duration` inside montage scenes.

### Removed
- **Zero Legacy Code Cleanup**:
  - Removed obsolete video slicing routines, VHS_LoadVideoPath, dead config fields, and unused client helper methods.

## [0.3.0] - 2026-10-16 (AI-3)

### Added
- **Generator Factory Architecture**:
  - Introduced polymorphic `BaseGenerator`, `BaseImageGenerator`, `BaseVideoGenerator`, and `GeneratorFactory`.
  - Added implementations for SDXL, FLUX.1, FLUX.2 Klein, Wan 2.2 Fast, Wan 2.2 Full, and LTX-2.3 22B.
- **True Forced Prefix Conditioning & Latent-Chaining**:
  - Added custom nodes `WanLatentPrefixToVideo`, `WanLatentPrefixLock`, and `WanLatentPrefixRestore`.
  - Supported 1-frame tail keyframe chaining (`overlap: 0.0`) and 5D latent chaining (`overlap > 0`).
- **Lossless Stream-Copy Assembly**:
  - Added ComfyUI `ImageFromBatch` slicing and instant FFmpeg `-c copy` concatenation.
  - Enforced canonical hyphenated naming standard: `YYYYMMDD-HHMMSS-{hash6}-{suffix}.ext`.

## [0.2.0] - 2026-10-14 (AI-2)

### Added
- **Config Portability & Launchers**:
  - Replaced hardcoded paths with portable hierarchical configuration resolution.
- **Unified Status & Emergency Interrupt**:
  - Implemented `media_get_status` and `media_interrupt` circuit breaker halting Python scenario loops and signaling backend cancellation.
- **Temporal Deflicker & Chroma Stabilization**:
  - Added FFmpeg `deflicker` and `tmidequalizer` filters for intra-scene prefix block concatenation.
- **Anti-Timeout Heartbeat Streaming**:
  - Added periodic progress heartbeat notifications during long video diffusion runs.

## [0.1.0] - 2026-10-14 (AI-1)

### Added
- **Initial Production Pipeline Alignment**:
  - Aligned FLUX.2 Klein modular graph (`KSamplerSelect` + `Flux2Scheduler` + `BasicGuider` + `SamplerCustomAdvanced`) with verified ComfyUI nodes.
  - Aligned Wan 2.2 dual-noise distilled pipeline (high/low noise models + LoRAs + prefix conditioning).
