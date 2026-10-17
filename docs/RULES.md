# media_mcp Repository Engineering Rules

These rules govern the development, testing, and operation of the `media_mcp` pipeline.

---

## Quick Reference & Documentation Map
- **Server Architecture & Contracts:** [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)
- **Scenario Design & Chaining (Deep Merge, Templates, Continuity):** [`docs/SCENARIO_DESIGN.md`](SCENARIO_DESIGN.md)
- **Models, Engines & LoRA Guide (SDXL, FLUX, Wan, LTX):** [`docs/MODELS_AND_ENGINES.md`](MODELS_AND_ENGINES.md)
- **Image Pipeline (Optimal ~1.0 MP, Img2Img, Inpaint, Feathering, Guidance):** [`docs/IMAGE_PIPELINE.md`](IMAGE_PIPELINE.md)
- **Network, Queue & Concurrency (Keep-Alive, GPU Lock):** [`docs/NETWORK_AND_QUEUE.md`](NETWORK_AND_QUEUE.md)
- **Artifacts & Naming Standard (POSIX Paths, Hyphenated Prefix):** [`docs/ARTIFACTS_AND_NAMING.md`](ARTIFACTS_AND_NAMING.md)
- **Client Integration & Philosophy (Multi-Client, Zero Legacy):** [`docs/CLIENT_INTEGRATION.md`](CLIENT_INTEGRATION.md)
- **Latent Math & Bug Knowledge Base:** [`docs/MEDIA_MCP_KNOWLEDGE_BASE.md`](MEDIA_MCP_KNOWLEDGE_BASE.md)
- **Video Chaining Research & Algorithms:** [`docs/VIDEO_BLOCK_CHAINING_RESEARCH.md`](VIDEO_BLOCK_CHAINING_RESEARCH.md)
- **ComfyUI Pipeline Quickstart:** [`docs/COMFYUI_GUIDE.md`](COMFYUI_GUIDE.md)

---

## 1. Zero Hardcoding & Strict Configuration
- No hardcoded local drive letters, IP addresses, or ports in code.
- No heuristic auto-discovery or guessing of configuration paths in CWD.
- All executions (CLI, Stdio MCP, SSE) strictly require explicit configuration via `--config <path>` or the `MEDIA_CONFIG` environment variable.
- Configuration is strictly validated against `schemas/config.json`.
- Directories are derived hierarchically:
  - `output_dir` (root output folder) $\to$ `images/`, `videos/`, `workflows/`, `temp/`.
  - `comfy_dir` (ComfyUI root folder) $\to$ `input/`, `output/`.

---

## 2. Model & LoRA Selection Rule
- **Both `engine` and `model` are strictly required** in all image and video generation specifications.
- **The active checkpoint/model is primary**: never change or adapt the checkpoint model to match a LoRA.
- LoRAs must be selected strictly to match the architecture of the current/chosen model (using `media_get_loras` with binary safetensors header architecture inspection).
- Never switch models just to fit an incompatible LoRA unless explicitly requested.
- **FLUX Compatibility**: Never attach FLUX.1 LoRAs to FLUX.2 Klein (tensor shape mismatch: 4096 vs 3072).

---

## 3. Video Pipeline & Latent Chaining
- Follow the hierarchy: `Root (.image, .video) -> Scene (anchor_image) -> Blocks (motion_prompt)`:
  - Technical parameters (`engine`, `model`, `width`, `height`, `fps`, `steps`, `cfg`, `guidance`, `sampler`, `seed`, `max_duration`, `ramp_latents`, `min_cfg`, `weight_dtype`) belong strictly in root `.image` and `.video` sections.
  - `anchor_image` and `blocks[]` contain only narrative/framing parameters (`prompt`, `motion_prompt`, `negative_prompt`, `duration`, `overlap`, `loras`, `denoise`, `fit`, `image_path`, `keyframe_path`, `mask_path`).
- Default to 1-frame I2V chaining (`overlap: 0.0s`) for clean production video without Causal VAE deconvolution artifacts.
- Use **True Forced Prefix Conditioning** (`WanLatentPrefixLock` + `WanLatentPrefixRestore` for Wan 2.2, `LTXVAddLatentGuide` + `LTXVCropGuides` + `WanLatentPrefixRestore` for LTX) with Latent AdaIN color stabilization in 5D latent space between sequential blocks when `overlap > 0.0s`. Tail latents are always cut from `latent_for_decode` post-normalization.
- Lossless Stream Copy (`-c copy`) for final multi-block FFmpeg assembly.
- Before making changes to latent processing, padding, masking, or stride math in video pipelines, **ALWAYS** consult [`docs/MEDIA_MCP_KNOWLEDGE_BASE.md`](MEDIA_MCP_KNOWLEDGE_BASE.md) and [`docs/VIDEO_BLOCK_CHAINING_RESEARCH.md`](VIDEO_BLOCK_CHAINING_RESEARCH.md) to avoid repeating solved bugs.

---

## 4. Pure Structured Output (No Base64 Bloat)
- All tool outputs are strictly structured JSON objects validated against `schemas/responses/<tool>.json`.
- Artifacts are stored on disk (`image_path`, `video_path`, `poster_path`, `spec_path`); multi-megabyte base64 payloads are never emitted in CLI or core responses.
- In SSE mode, media files are streamed directly over HTTP Range endpoints (`/images/`, `/videos/`).

---

## 5. Canonical Artifact Naming & POSIX Path Serialization
- All files across `images/`, `videos/`, `workflows/`, `temp/`, and `ComfyUI/input/` must strictly adhere to the canonical hyphenated prefix: `YYYYMMDD-HHMMSS-{hash6}-{suffix}.ext`.
  - Never emit ad-hoc filenames, un-prefixed helper files, underscores in artifact delimiters, or inject scene names into the output filenames.
- Intermediate scene anchors belong in `temp/` as `{run_prefix}-s001-anchor.png`. The Scene 1 anchor is promoted to `images/{run_prefix}-poster.png` upon video assembly.
- **All paths in JSON responses and spec files MUST be normalized via `Path(...).resolve().as_posix()`**: never serialize raw Windows backslashes (`\`) into JSON strings.

---

## 6. Network Pooling & Loopback Isolation
- All HTTP communication with local backends (ComfyUI, A1111) MUST use persistent `requests.Session()` with connection pooling (`HTTPAdapter`) and Keep-Alive.
- **Mandatory `s.trust_env = False`**: always disable proxy resolution for loopback `127.0.0.1` traffic to eliminate network latency and socket access errors.
- Never use unpooled `urllib.request.urlopen` in polling loops.

---

## 7. Queue Preservation & GPU Mutex Lock
- Generations must strictly queue up. **NEVER** implicitly abort or interrupt an active generation when a new generation is requested.
- All generation calls (`execute_image_generate`, `execute_video_generate`) must be guarded by the server-wide `_GPU_LOCK` mutex to prevent concurrent VRAM allocation between ComfyUI and A1111.
- Interruption is strictly manual and explicit via `media_interrupt`. Any interrupted pipeline or generator MUST reset the local interrupt flag in its `finally` block to allow queued jobs to proceed.

---

## 8. Zero Legacy Code Principle
- This is a clean, modern, personal repository. Do NOT retain deprecated backwards-compatibility shims, legacy fallback modes, or obsolete code paths. Aggressively remove dead code.

---

## 9. No Self-Verification
- Do NOT call `view_file` or any analysis/reading tools on generated images or videos for self-checking unless explicitly requested by the user. Immediately return generation results.
