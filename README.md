# media_mcp

Универсальный фреймворк и сервер **Model Context Protocol (MCP)** для генерации изображений и последовательного многоблочного видео.

Разработан для работы с **ComfyUI** (FLUX.1 Dev, FLUX.2 Klein, Wan 2.2 I2V, LTX-2.3 22B) и **Automatic1111 WebUI** (SD 1.5, SDXL, Pony, Illustrious).

Поддерживает интеграцию с **Google Antigravity**, **Open-WebUI**, **Cursor**, **ChatGPT Agent**, **Claude Desktop** и **Grok**.

---

## 🌟 Ключевые возможности

1. **Разделение ответственности (Separation of Concerns):**
   - `src/mcp/image_mcp.py` — тонкий MCP-интерфейс для генерации изображений (A1111 / FLUX).
   - `src/mcp/video_mcp.py` — тонкий MCP-интерфейс для многосценарных видеороликов (Wan 2.2 / LTX-2.3 / FFmpeg).
   - `src/core/` — общее ядро с переиспользуемыми клиентами, генераторами, валидатором JSON Schema и пайплайнами.
   - `schemas/` — единый каталог строгих схем **JSON Schema (Draft 2020-12)** для `config.json`, сценариев и всех MCP-запросов с условной (`if/then`) проверкой параметров в зависимости от `engine`.
2. **True Forced Prefix Conditioning (Latent-Chaining):**
   - Непрерывное сохранение динамики сцены внутри монтажа: при `overlap > 0` хвост предыдущего блока (`-tail.latent`) отрезается через `LatentCut` и передаётся в следующий блок как префикс (`prefix_duration`). Следующий блок диффундирует окно `prefix_duration + clip_duration`, после чего `ImageFromBatch` отсекает префикс и сохраняет полную длительность `clip_duration` в `{block_prefix}-core.mp4`. При `overlap == 0.0` (дефолт) используется бесшовная 1-кадровая сцепка по последнему кадру (`-tail.png`).
3. **Чистовая основа блоков и Stream Copy (-c copy):**
   - Чистовая основа каждого блока (`-core.mp4`) сохраняется напрямую в H.264. Финальная сборка всех блоков в FFmpeg происходит через мгновенный **Stream Copy (`-c copy`) за 0.05 сек без перекодирования**.
4. **Полиморфные генераторы (Strategy & Factory):**
   - Видеодвижки: **`wan-fast`** (Wan 2.2 14B 4-step LoRA, дефолт), **`wan`** (Wan 2.2 14B Full 20-step), **`ltx`** (LTX-2.3 22B).
   - Фотодвижки: **`sdxl`** (Automatic1111 SDXL/Pony, дефолт для опорных кадров), **`flux1`** (FLUX.1 Dev), **`flux2`** (FLUX.2 Klein).
5. **Строгая валидация без легаси:**
   - Каждый JSON (`config.json`, `scenario.json`, аргументы MCP-вызовов) проверяется через `jsonschema` (`additionalProperties: false`). Опечатки в именах полей и несовместимые с выбранным `engine` параметры отвергаются до обращения к GPU.
   - Имена всех выходных файлов формируются исключительно сервером по дефисному шаблону `{timestamp}-{random}-...`.

---

## 📁 Структура проекта

```
media_mcp/
├── schemas/                    # Централизованные JSON Schema (Draft 2020-12)
│   ├── config.json
│   ├── scenario.json
│   ├── mcp_video_run_scenario.json
│   ├── mcp_video_validate_scenario.json
│   ├── mcp_video_generate_block.json
│   ├── mcp_comfy_flux_txt2img.json
│   ├── mcp_sd_txt2img.json
│   ├── mcp_sd_set_model.json
│   └── mcp_empty.json
│
├── src/
│   ├── core/                   # Единое ядро генерации (Core Engine)
│   │   ├── __init__.py
│   │   ├── config.py           # Конфигурация + валидация config.json по схеме
│   │   ├── schema_validator.py # Загрузчик, инлайнер $ref и валидатор JSON Schema
│   │   ├── comfy_client.py     # HTTP-клиент ComfyUI (очередь, опрос, прерывание)
│   │   ├── a1111_client.py     # HTTP-клиент Automatic1111 (модели, сэмплеры, LoRA)
│   │   ├── ffmpeg_engine.py    # Монтаж Stream Copy (-c copy), извлечение постеров и HUD
│   │   ├── image_engine.py     # Генерация опорных кадров и изображений
│   │   ├── video_pipeline.py   # Иерархический пайплайн видео (Scene -> Blocks)
│   │   └── generators/         # Полиморфные стратегии генераторов (SDXL, FLUX, Wan, LTX)
│   │
│   └── mcp/                    # Тонкие MCP-врапперы (JSON-RPC 2.0)
│       ├── __init__.py
│       ├── image_mcp.py        # Сервер картинок (Stdio + SSE :7865)
│       └── video_mcp.py        # Сервер видео (Stdio + SSE :7866)
│
├── .agents/skills/             # Скиллы для AI-агентов
│   ├── add-to-master/          # Процесс публикации правок в master через PR
│   ├── image-generation/       # Скилл генерации изображений
│   └── video-scenario/         # Скилл генерации видео по сценариям
│
├── config.example.json         # Шаблон конфигурационного файла
├── requirements.txt            # Зависимости (jsonschema>=4.20.0)
└── README.md                   # Главная документация
```

---

## ⚙️ Системные требования и зависимости

### 1. Системные компоненты
- **Python:** `>= 3.10`, пакет `jsonschema>=4.20.0` (`pip install -r requirements.txt`).
- **FFmpeg:** доступен в системе через `PATH` или указан в `ffmpeg_bin` (рекомендуется сборка с поддержкой `h264_nvenc`).
- **ComfyUI:** запущен локально (по умолчанию `http://127.0.0.1:8188`).
- **Automatic1111 WebUI (опционально):** запущен с флагом `--api` (по умолчанию `http://127.0.0.1:7860`).

---

### 2. Необходимые ноды и пакеты ComfyUI

| Категория | Используемые ноды | Где находится / Пакет |
| :--- | :--- | :--- |
| **Базовые ноды** | `LoadImage`, `SaveImage`, `ImageFromBatch`, `VAELoader`, `VAEDecode`, `CreateVideo`, `SaveVideo`, `CLIPTextEncode`, `UNETLoader`, `LoraLoaderModelOnly` | ComfyUI Core |
| **Latent-Chaining** | `SaveLatent`, `LoadLatent`, `LatentCut` | ComfyUI Core |
| **Wan 2.2** | `WanImageToVideo`, `WanLatentPrefixToVideo`, `WanLatentPrefixLock`, `WanLatentPrefixRestore`, `ModelSamplingSD3`, `CLIPLoader` (type: `"wan"`) | `comfy_extras/nodes_wanvideo.py` |
| **LTX-2.3 (22B)** | `EmptyLTXVLatentVideo`, `LTXVImgToVideoInplace`, `LTXVConditioning`, `LTXVScheduler`, `DualCLIPLoader` (type: `"ltxv"`) | `comfy_extras/nodes_ltx.py` или `ComfyUI-LTXVideo` |
| **Продвинутый сэмплинг** | `SamplerCustomAdvanced`, `KSamplerSelect`, `RandomNoise`, `CFGGuider`, `BasicGuider` | `comfy_extras/nodes_custom_sampler.py` |
| **FLUX / FLUX.2** | `FluxGuidance`, `Flux2Scheduler`, `EmptySD3LatentImage`, `CLIPLoader` (type: `"flux2"`) | `comfy_extras/nodes_sd3.py`, `nodes_flux.py` |

---

## 🔧 Конфигурация (`config.json`)

Скопируйте `config.example.json` в `config.json` (валидируется по `schemas/config.json`):

```json
{
  "comfy_url": "http://127.0.0.1:8188",
  "sd_webui_url": "http://127.0.0.1:7860",
  "ffmpeg_bin": "ffmpeg",
  "ffprobe_bin": "ffprobe",
  "output_dir": "./output",
  "output_img_dir": "./output/images",
  "output_vid_dir": "./output/videos",
  "temp_dir": "./output/temp",
  "workflows_dir": "./workflows",
  "comfy_input_dir": "",
  "comfy_output_dir": "",
  "image_mcp_port": 7865,
  "video_mcp_port": 7866,
  "max_video_duration": 10.0
}
```

---

## 🛠️ Справочник инструментов (MCP Tools)

Все инструменты загружают `inputSchema` из `schemas/mcp_*.json` и проходят строгую валидацию перед выполнением.

### 1. Сервер изображений (`src/mcp/image_mcp.py`)

- **`comfy_flux_txt2img`** (`schemas/mcp_comfy_flux_txt2img.json`):
  - `prompt` *(string, обязателен)*
  - `model_type` *(`"flux1_dev"` | `"flux2"`, по умолчанию `"flux1_dev"`)*
  - `width`, `height` *(int, кратны 16, по умолчанию `1024`)*
  - `steps` *(int)*, `guidance` *(number, по умолчанию `3.5`)*, `seed` *(int)*
  - `lora_name` *(string, запрещён при `model_type: "flux2"`)*, `lora_strength` *(number)*
- **`sd_txt2img`** (`schemas/mcp_sd_txt2img.json`):
  - `prompt` *(string, обязателен)*, `negative_prompt`, `model_name`, `steps`, `cfg_scale`, `width`, `height`, `seed`, `sampler_name`.
- **`sd_get_models`**, **`sd_set_model`**, **`sd_get_loras`**.

### 2. Сервер видео (`src/mcp/video_mcp.py`)

- **`video_run_scenario`** (`schemas/mcp_video_run_scenario.json`):
  - Требуется ровно один из параметров (`oneOf`): `scenario_path` *(string)* или `scenario_data` *(object по `scenario.json`)*.
  - `image` *(object, опционально)*: глобальные настройки опорного кадра (`engine`: `"sdxl"` | `"flux1"` | `"flux2"`, `model`, `checkpoint`, `prompt`, `negative_prompt`, `width`, `height`, `steps`, `cfg`, `guidance`, `sampler`, `seed`, `lora`, `lora_strength`).
  - `video` *(object, опционально)*: глобальные настройки видео (`engine`: `"wan-fast"` | `"wan"` | `"ltx"`, `model`, `checkpoint`, `checkpoint_low`, `prompt`, `negative_prompt`, `width`, `height`, `fps`, `max_duration`, `steps`, `cfg`, `seed`, `lora_name`, `lora_strength`, `weight_dtype`, `ramp_latents`, `min_cfg`).
  - `enable_hud` *(boolean, по умолчанию `false`)*: дополнительно создаёт `{run_prefix}-hud.mp4`.
- **`video_validate_scenario`** (`schemas/mcp_video_validate_scenario.json`):
  - Проверяет соответствие `schemas/scenario.json`, наличие файлов `anchor_image.image_path` и динамические ограничения слияния блоков внутри монтажа:
    1. `overlap_duration < clip_duration`
    2. `prefix_duration + clip_duration < max_duration` (где `prefix_duration` — `overlap` предыдущего блока сцены, а `max_duration` по умолчанию `10.0` сек).
- **`video_generate_block`** (`schemas/mcp_video_generate_block.json`):
  - Одиночная генерация блока (`prompt`, `engine`: `"wan-fast"` | `"wan"` | `"ltx"`, `start_image` или `prefix_latent`, `duration`, `width`, `height`, `fps`, `steps`, `cfg`, `seed`).
- **`media_get_status`**, **`media_interrupt`**.

---

## 🎬 Формат сценария (`schemas/scenario.json`)

```json
{
  "name": "Hermione Train Compartment",
  "image": {
    "engine": "sdxl",
    "cfg": 5.5,
    "steps": 30
  },
  "video": {
    "engine": "wan-fast",
    "width": 832,
    "height": 480,
    "fps": 16,
    "max_duration": 10.0,
    "negative_prompt": "blurry, low quality, jitter, static, deformed, extra limbs, bad anatomy, text, watermark"
  },
  "scenes": [
    {
      "scene_name": "Scene 1: Train Compartment",
      "anchor_image": {
        "prompt": "score_9, score_8_up,score_7_up, 1girl, Hermione Granger, bushy brown hair, Hogwarts uniform, inside cozy vintage train compartment, golden afternoon sunlight"
      },
      "blocks": [
        {
          "name": "Block 1: Opening Door",
          "duration": 5.0,
          "overlap": 1.0,
          "motion_prompt": "She gently slides the compartment door open with her hand and steps inside, looking around curiously"
        },
        {
          "name": "Block 2: Greeting",
          "duration": 5.0,
          "overlap": 0.0,
          "motion_prompt": "She notices the passenger across from her and smiles warmly while sitting down"
        }
      ]
    }
  ]
}
```

### Законы таймингов и слияния блоков внутри одной сцены (монтажа):
В текущем пайплайне блок $i$ получает хвост предыдущего блока $\text{prefix\_duration} = \text{overlap}_{i-1}$ (для первого блока сцены $\text{prefix\_duration} = 0$), диффундирует суммарное окно $\text{prefix\_duration} + \text{clip\_duration}$, отсекает префикс через `ImageFromBatch` и отдаёт в финальную склейку ровно $\text{clip\_duration}$ чистого видео:
1. **Хвост меньше самого клипа:**
   $$\text{overlap\_duration} < \text{clip\_duration}$$
2. **Суммарное окно генерации меньше лимита модели (`max_duration`, по умолчанию 10.0 сек):**
   $$\text{prefix\_duration} + \text{clip\_duration} < \text{max\_duration}$$
3. **Итоговая длительность сцены:** равна чистой сумме $\sum \text{clip\_duration}$ всех блоков сцены.

---

## 🤖 Правило для AI-агентов: Процесс публикации изменений (PR Workflow)

При работе с данным репозиторием AI-агенты **обязаны строго следовать скиллу [`.agents/skills/add-to-master/SKILL.md`](.agents/skills/add-to-master/SKILL.md)**:

1. **Никаких прямых коммитов в `master`:** все изменения оформляются через ветку `AI-<n>-<slug>`, созданную от актуального `master`.
2. **Обязательная актуализация `README.md` и `schemas/`:** любые изменения контрактов обязаны быть отражены в `schemas/*.json` и `README.md` до создания коммита.
3. **Публикация и слияние:** ветка пушится в `origin`, открывается Pull Request через `gh pr create`, слияние производится через `gh pr merge <PR_NUMBER> --merge`, локальный `master` синхронизируется через `git pull origin master`.

---

## Лицензия

MIT License.
