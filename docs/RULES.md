# media_mcp Repository Engineering Rules

These rules govern the development, testing, and operation of the `media_mcp` pipeline.

---

## 1. Zero Hardcoding & Strict Configuration
- No hardcoded local drive letters, IP addresses, or ports in code.
- No heuristic auto-discovery or guessing of configuration paths in CWD.
- All executions (CLI, Stdio MCP, SSE) strictly require explicit configuration via `--config <path>` or the `MEDIA_CONFIG` environment variable.
- Configuration is strictly validated against `schemas/config.json`.
- Directories are derived hierarchically:
  - `output_dir` (root output folder) -> `images/`, `videos/`, `workflows/`, `temp/`.
  - `comfy_dir` (ComfyUI root folder) -> `input/`, `output/`.

---

## 2. Model & LoRA Selection Rule
- **The active checkpoint/model is primary**: never change or adapt the checkpoint model to match a LoRA.
- LoRAs must be selected strictly to match the architecture of the current/chosen model (using `media_get_loras` with architecture compatibility checks).
- Never switch models just to fit an incompatible LoRA unless explicitly requested.
- **FLUX Compatibility**: Never attach FLUX.1 LoRAs to FLUX.2 Klein (tensor shape mismatch: 4096 vs 3072).

---

## 3. Video Pipeline & Latent Chaining
- Follow the hierarchy: `Scene -> Blocks`:
  - `anchor_image`: visual appearance prompt or ready `image_path` / `keyframe_path`.
  - `blocks[].motion_prompt`: action and camera motion.
- Use **True Forced Prefix Conditioning** between sequential blocks in a multi-block scene.
- Lossless Stream Copy (`-c copy`) for final multi-block FFmpeg assembly.
- Before making changes to latent processing, padding, masking, or stride math in video pipelines, **ALWAYS** consult [`docs/MEDIA_MCP_KNOWLEDGE_BASE.md`](MEDIA_MCP_KNOWLEDGE_BASE.md) and [`docs/VIDEO_BLOCK_CHAINING_RESEARCH.md`](VIDEO_BLOCK_CHAINING_RESEARCH.md) to avoid repeating solved bugs.

---

## 4. Pure Structured Output (No Base64 Bloat)
- All tool outputs are strictly structured JSON objects validated against `schemas/responses/response_<tool>.json`.
- Artifacts are stored on disk (`image_path`, `video_path`, `spec_path`); multi-megabyte base64 payloads are never emitted in CLI or core responses.
- In SSE mode, media files are streamed directly over HTTP Range endpoints (`/images/`, `/videos/`).

---

## 5. No Self-Verification
- Do NOT call `view_file` or any analysis/reading tools on generated images or videos for self-checking unless explicitly requested by the user. Immediately return generation results.
