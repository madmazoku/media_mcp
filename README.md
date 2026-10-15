# media_mcp

Универсальный фреймворк и сервер **Model Context Protocol (MCP)** для генерации изображений и последовательного многоблочного видео.

Разработан для работы с **ComfyUI** (FLUX.1 Dev, FLUX.2 Klein, Wan 2.2 I2V) и **Automatic1111 WebUI** (SD 1.5, SDXL, Pony, Illustrious).

Поддерживает интеграцию с **Google Antigravity**, **Open-WebUI**, **Cursor**, **ChatGPT Agent**, **Claude Desktop** и **Grok**.

---

## 🌟 Ключевые возможности

1. **Разделение ответственности (Separation of Concerns):**
   - `src/mcp/image_mcp.py` — тонкий MCP-интерфейс для генерации и редактирования изображений (A1111 / FLUX).
   - `src/mcp/video_mcp.py` — тонкий MCP-интерфейс для многосценарных видеороликов (Wan 2.2 / LTX-2.3 / FFmpeg).
   - `src/core/` — общее ядро с переиспользуемыми клиентами и пайплайнами.
2. **True Forced Prefix Conditioning:**
   - Непрерывное сохранение динамики сцены без рывков, затуханий и деградации цветов: хвосты блоков передаются в следующий как многокадровый префикс (17 кадров при 16 fps = 1.0 сек) через нативный `WanImageToVideo` (`start_image`), маскирующий 100% префиксных кадров как conditioned (`mask = 0.0`). Никаких поломок масок, размазывания контуров и паразитных «червяков»!
3. **Чистовая основа блоков и Stream Copy (-c copy):**
   - Чистовая основа каждого блока (`block_core`) сохраняется напрямую в H.264. Финальная сборка сцен и фильма в FFmpeg происходит через мгновенный **Stream Copy (`-c copy`) за 0.05 сек без перекодирования**.
4. **Двойной видеодвижок (Dual Engine):**
   - Поддержка **Wan 2.2 (14B)** для кинематографичной плавности, аниме и лиц, и **LTX-2.3 (22B)** с CFG 3.0 и дистиллированной LoRA 1.1 для масштабной динамики и экшена.
5. **Negative Prompt первого класса:**
   - Полная поддержка `negative_prompt` на уровне сценария, сцены, блока и тулов. Полное удаление китайского хардкода.
   - Отказ от искусственного Color-Lock: полная свобода для смены освещения, магии (*Lumos*) и ночных сцен.
6. **Асинхронный Stdio MCP Server:**
   - Вызовы генерации выполняются в фоновом worker-потоке: `media_get_status` и `media_interrupt` отвечают за < 50 мс без блокировки stdio.
7. **Кинематографическая иерархия «Сцена (Scene) -> Блоки движения (Blocks)»:**
   - Чёткое разделение промптов: **Appearance Prompt** (статичная эстетика опорного кадра для FLUX/SDXL) и **Motion Prompt** (векторы действий и движения камеры).
8. **Гибридный опорный кадр (Hybrid Anchor):**
   - Мгновенное оживление уже существующего файла (`image_path`), либо автоматическая фоновая генерация по промпту (`prompt`).
9. **Автоматические постеры в чате (No Context Bloat):**
   - Вместо перегрузки контекста десятками мегабайт base64-видео сервер возвращает легкий постер первого кадра через нативный тип MCP `image`, а под ним формирует кликабельную ссылку на воспроизведение.
10. **Zero External Dependencies:**
    - Написан исключительно на стандартной библиотеке Python 3.10+ (без внешних pip-пакетов, со встроенной поддержкой `safetensors`).


---

## 📁 Структура проекта

```
media_mcp/
├── src/
│   ├── core/                   # Единое ядро генерации (Core Engine)
│   │   ├── __init__.py
│   │   ├── config.py           # Конфигурация, пути, переменные окружения
│   │   ├── comfy_client.py     # HTTP-клиент ComfyUI (очередь, опрос, прерывание)
│   │   ├── a1111_client.py     # HTTP-клиент Automatic1111 (модели, сэмплеры, LoRA)
│   │   ├── ffmpeg_engine.py    # Монтаж, префиксная нарезка Wan 2.2, извлечение постеров
│   │   ├── image_engine.py     # Генерация FLUX.1 / FLUX.2 / SDXL + авто-LoRA
│   │   └── video_pipeline.py   # Иерархический пайплайн Wan 2.2 (Scene -> Blocks)
│   │
│   └── mcp/                    # Тонкие MCP-врапперы (JSON-RPC 2.0)
│       ├── __init__.py
│       ├── image_mcp.py        # Сервер картинок (Stdio + SSE :7865 для Open-WebUI)
│       └── video_mcp.py        # Сервер видео (Stdio + SSE :7866 для Open-WebUI)
│
├── workflows/                  # Библиотека сценариев (visual_novel_template.json)
├── .agents/skills/             # Скиллы для AI-агентов
│   ├── add-to-master/          # Процесс публикации правок в master через PR
│   ├── image-generation/       # Скилл генерации изображений
│   └── video-scenario/         # Скилл генерации видео
│
├── .env.example                # Шаблон переменных окружения
├── requirements.txt            # Описание зависимостей (чистый Python 3.10+)
└── README.md                   # Главная документация
```

---

## ⚙️ Системные требования и зависимости

### 1. Системные компоненты
- **Python:** `>= 3.10` (чистая стандартная библиотека Python, встроенный модуль `safetensors.torch`).
- **FFmpeg:** доступен в системе через `PATH` или указан в `FFMPEG_BIN` (рекомендуется сборка с поддержкой `h264_nvenc`).
- **ComfyUI:** запущен локально (по умолчанию `http://127.0.0.1:8188`).
- **Automatic1111 WebUI (опционально):** запущен с флагом `--api` (по умолчанию `http://127.0.0.1:7860`).

---

### 2. Необходимые ноды и пакеты ComfyUI

Большинство нод входят в стандартную поставку современного ComfyUI (`comfy_extras`), но для работы требуются следующие группы:

| Категория | Используемые ноды | Где находится / Пакет |
| :--- | :--- | :--- |
| **Базовые ноды** | `LoadImage`, `VAELoader`, `VAEDecode`, `CreateVideo`, `SaveVideo`, `CLIPTextEncode`, `UNETLoader`, `LoraLoaderModelOnly` | ComfyUI Core |
| **Latent-Chaining** | `SaveLatent`, `LoadLatent` | ComfyUI Core |
| **Wan 2.2** | `WanImageToVideo`, `WanFirstLastFrameToVideo`, `ModelSamplingSD3`, `CLIPLoader` (type: `"wan"`) | `comfy_extras/nodes_wanvideo.py` |
| **LTX-2.3 (22B)** | `EmptyLTXVLatentVideo`, `LTXVImgToVideoInplace`, `LTXVConditioning`, `LTXVScheduler`, `DualCLIPLoader` (type: `"ltxv"`) | `comfy_extras/nodes_ltx.py` или `ComfyUI-LTXVideo` |
| **Продвинутый сэмплинг** | `SamplerCustomAdvanced`, `KSamplerSelect`, `RandomNoise`, `CFGGuider`, `BasicGuider` | `comfy_extras/nodes_custom_sampler.py` |
| **FLUX / FLUX.2** | `FluxGuidance`, `Flux2Scheduler`, `EmptySD3LatentImage`, `CLIPLoader` (type: `"flux2"`) | `comfy_extras/nodes_sd3.py`, `nodes_flux.py` |
| **Legacy Video Prefix** | `VHS_LoadVideoPath` | Кастомный пакет `ComfyUI-VideoHelperSuite` |

---

### 3. Комбинации моделей и пути размещения в ComfyUI

Для каждого из поддерживаемых движков требуется строго согласованная комбинация весов:

#### А. Видеодвижок Wan 2.2 (14B I2V — Двухстадийная дистилляция 4 шага)
* **UNET (High Noise):** `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` $\rightarrow$ `ComfyUI/models/diffusion_models/` (или `unet/`)
* **UNET (Low Noise):** `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` $\rightarrow$ `ComfyUI/models/diffusion_models/` (или `unet/`)
* **LoRA (High Noise 4-step):** `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` $\rightarrow$ `ComfyUI/models/loras/`
* **LoRA (Low Noise 4-step):** `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` $\rightarrow$ `ComfyUI/models/loras/`
* **Text Encoder (CLIP):** `umt5_xxl_fp8_e4m3fn_scaled.safetensors` $\rightarrow$ `ComfyUI/models/text_encoders/` (или `clip/`)
* **VAE:** `wan_2.1_vae.safetensors` $\rightarrow$ `ComfyUI/models/vae/`

#### Б. Видеодвижок LTX-2.3 (22B — Дистиллированный CFG 3.0)
* **UNET:** `ltx-2.3-22b-dev-fp8.safetensors` $\rightarrow$ `ComfyUI/models/diffusion_models/` (или `unet/`)
* **LoRA (Distilled 1.1):** `ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors` $\rightarrow$ `ComfyUI/models/loras/`
* **Text Encoder (DualCLIP):** `ltx-2-3-22b-text_encoder.safetensors` $\rightarrow$ `ComfyUI/models/text_encoders/` (или `clip/`)
* **VAE:** `ltx-2-3-22b-VAE.safetensors` $\rightarrow$ `ComfyUI/models/vae/`

#### В. Генерация опорных кадров через FLUX.1 Dev
* **UNET:** `persephoneFluxNSFWSFW_20FP16.safetensors` $\rightarrow$ `ComfyUI/models/unet/`
* **CLIP:** `t5xxl_fp16.safetensors` + `clip_l.safetensors` $\rightarrow$ `ComfyUI/models/clip/`
* **VAE:** `ae.safetensors` $\rightarrow$ `ComfyUI/models/vae/`
* **LoRA (опционально):** `aidmaNSFWunlock-FLUX-V0.2.safetensors` $\rightarrow$ `ComfyUI/models/loras/`

#### Г. Генерация опорных кадров через FLUX.2 Klein
* **UNET:** `aishaNSFWFlux2Klein_betaV88.safetensors` $\rightarrow$ `ComfyUI/models/unet/`
* **CLIP:** `qwen_3_8b_fp8mixed.safetensors` $\rightarrow$ `ComfyUI/models/clip/`
* **VAE:** `flux2-vae.safetensors` $\rightarrow$ `ComfyUI/models/vae/`
* *(Внимание: LoRA от FLUX.1 несовместимы с FLUX.2 Klein из-за несовпадения формы тензоров).*

#### Д. Опорные кадры через Automatic1111 (SDXL / Pony / Illustrious)
* **Чекпоинты:** `models/Stable-diffusion/`
* **LoRA:** `models/Lora/`

---


## 🔧 Конфигурация (`.env`)

Скопируйте `.env.example` в `.env` при необходимости переопределить стандартные пути:

```ini
# Бэкенды генерации
COMFY_URL=http://127.0.0.1:8188
SD_WEBUI_URL=http://127.0.0.1:7860

# Бинарники FFmpeg
FFMPEG_BIN=E:/Git/ffmpeg/bin/ffmpeg.exe
FFPROBE_BIN=E:/Git/ffmpeg/bin/ffprobe.exe

# Папки вывода
MEDIA_OUTPUT_DIR=G:/Git/Gemini/output
MEDIA_TEMP_DIR=G:/Git/media_mcp/temp

# Порты для режима SSE (Open-WebUI)
IMAGE_MCP_PORT=7865
VIDEO_MCP_PORT=7866
```

---

## 🛠️ Справочник инструментов (MCP Tools)

### 1. Сервер изображений (`src/mcp/image_mcp.py`)

- **`comfy_flux_txt2img`**: Генерация через FLUX.1 Dev или FLUX.2 Klein в ComfyUI.
  - `prompt` *(string, обязателен)*: Визуальное описание сцены.
  - `model_type` *(string)*: `"flux1_dev"` (по умолчанию) или `"flux2"`.
  - `width`, `height` *(int)*: Размеры в пикселях (автоматически округляются до кратности 16, по умолчанию `1024x1024`).
  - `steps` *(int)*: Число шагов сэмплирования.
  - `guidance` *(float)*: Guidance scale (по умолчанию `3.5`).
  - `seed` *(int)*: Сид (-1 для случайного).
  - `lora_name` *(string)*: Имя LoRA (автоматически отключает FLUX.1 LoRA при выборе модели FLUX.2 Klein во избежание конфликта тензоров).
  - `lora_strength` *(float)*: Сила LoRA (по умолчанию `0.8`).
- **`sd_txt2img`**: Генерация через Automatic1111 WebUI (SDXL, Pony, Illustrious, SD 1.5).
  - `prompt`, `negative_prompt`, `steps`, `cfg_scale`, `width`, `height`, `seed`, `sampler_name`.
- **`sd_get_models`**: Получение списка доступных моделей и активного чекпоинта A1111.
- **`sd_set_model`**: Переключение чекпоинта в A1111.
- **`sd_get_loras`**: Список зарегистрированных LoRA в A1111.

### 2. Сервер видео (`src/mcp/video_mcp.py`)

- **`video_run_scenario`**: Запуск выполнения сценария видео.
  - `scenario_path` *(string)*: Имя файла в папке `workflows/` или абсолютный путь.
  - `scenario_data` *(array/object)*: Иерархическая структура сценария (альтернатива файлу, для генерации «на лету»).
  - `start_image` *(string)*: Опциональный путь к стартовому изображению для Сцены 1.
  - `output_name` *(string)*: Имя итогового MP4-файла в `output/video/`.
  - `width`, `height` *(int)*: Разрешение (по умолчанию `832x480`).
  - `fps` *(int)*: Частота кадров (по умолчанию `16`).
  - `enable_hud` *(boolean)*: Если `true`, дополнительно создает отладочную версию видео `_hud.mp4` с динамическими таймерами обратного отсчета для каждого блока и нахлёста префикса.
- **`video_validate_scenario`**: Предварительная валидация таймингов, перекрытий (`overlap < duration`) и путей к опорным кадрам перед запуском.
- **`video_generate_block`**: Одиночная генерация блока в Wan 2.2 с поддержкой опорного кадра (`start_image`), граничного кадра (`end_image`) или префикса движения (`prefix_video`).
- **`media_get_status`**: Сквозной мониторинг статуса выполнения (активный сценарий/блок, очередь ComfyUI Wan/FLUX, прогресс сэмплера Automatic1111).
- **`media_interrupt`**: Экстренная остановка активной генерации во всех компонентах (прерывает сценарий, ComfyUI Wan/FLUX и Automatic1111).

---

## 🎬 Кинематографический формат сценария (`workflows/*.json`)

Сценарий строится по иерархическому принципу:

```json
[
  {
    "scene_name": "Сцена 1: Купе поезда",
    "anchor_image": {
      "prompt": "Cinematic visual novel CG, first-person POV shot from inside cozy vintage train compartment, golden afternoon sunlight. Hermione Granger unicorn mare standing in open doorway, looking inside inquisitively, foreground shows pony hooves holding glowing ancient spellbook. Makoto Shinkai lighting, clean detailed line art",
      "model": "flux1_dev",
      "image_path": null
    },
    "blocks": [
      {
        "name": "Блок 1: Поиск жабы",
        "duration": 5.0,
        "overlap": 1.5,
        "motion_prompt": "Her horn sparkles with magic telekinesis aura, sliding the compartment door open smoothly, trotting forward slightly, looking around the floor and under seats searching for Neville's lost pet toad, cute expressive animated equine expressions"
      },
      {
        "name": "Блок 2: Реакция на книгу",
        "duration": 5.0,
        "overlap": 0.0,
        "motion_prompt": "Her eyes suddenly spot the glowing magical tome held in the viewer's hooves, her big cartoon eyes widening with intense scholarly curiosity and wonder, taking an eager little step forward, tilting her head with an excited friendly smile to inspect the glowing ancient runes on the open pages"
      }
    ]
  }
]
```

### Принцип разделения промптов:
- **`anchor_image.prompt` (Appearance):** Описывает **КАК ЭТО ВЫГЛЯДИТ** (персонаж, свет, объектив, одежда, детализация). Не используйте длинные глаголы движения.
- **`blocks[].motion_prompt` (Motion):** Описывает **ЧТО ПРОИСХОДИТ** (действия, жесты, поворот головы, траектория камеры). Не дублируйте описание внешности.
- **`anchor_image.image_path`:** Если указан путь к существующему файлу, генерация пропускается, и видео сразу начинает строиться из этого кадра.

---

## 🔌 Инструкция по подключению клиентов

### Google Antigravity & Cursor (Stdio Transport)

В конфигурационном файле `mcp_config.json`:

```json
{
  "mcpServers": {
    "media-image": {
      "command": "python",
      "args": ["G:/Git/media_mcp/src/mcp/image_mcp.py"],
      "env": {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1"
      }
    },
    "media-video": {
      "command": "python",
      "args": ["G:/Git/media_mcp/src/mcp/video_mcp.py"],
      "env": {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

### Open-WebUI (SSE HTTP Transport)

1. Запустите серверы в фоновом режиме:
   ```bash
   python G:/Git/media_mcp/src/mcp/image_mcp.py --sse --port 7865
   python G:/Git/media_mcp/src/mcp/video_mcp.py --sse --port 7866
   ```
2. В интерфейсе Open-WebUI (**Admin Panel -> Settings -> Tools / External Connections**):
   - Добавьте Image Server: `http://host.docker.internal:7865/sse`
   - Добавьте Video Server: `http://host.docker.internal:7866/sse`
3. Сервер автоматически отдает сгенерированные видео через встроенный HTTP-стриминг: `http://host.docker.internal:7866/video/<filename>.mp4`.

---

## 🤖 Правило для AI-агентов: Процесс публикации изменений (PR Workflow)

При работе с данным репозиторием AI-агенты **обязаны строго следовать скиллу [`.agents/skills/add-to-master/SKILL.md`](.agents/skills/add-to-master/SKILL.md)**:

1. **Никаких прямых коммитов в `master`:**
   - Все изменения оформляются через ветку `AI-<n>-<slug>`, созданную от актуального `master`.
2. **Обязательная актуализация `README.md`:**
   - Если добавлены, изменены или удалены инструменты, параметры конфига, форматы сценариев или зависимости, агент **обязан обновить `README.md`** до создания коммита. `README.md` должен оставаться единым и исчерпывающим источником правды.
3. **Публикация и слияние:**
   - Ветка пушится в `origin`, открывается Pull Request через `gh pr create`.
   - Слияние производится строго через GitHub CLI: `gh pr merge <PR_NUMBER> --merge`.
   - Локальный `master` синхронизируется через `git pull origin master`.

---

## Лицензия

MIT License.
