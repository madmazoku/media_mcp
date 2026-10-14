# media_mcp

Универсальный фреймворк и сервер **Model Context Protocol (MCP)** для генерации изображений и последовательного многоблочного видео.

Разработан для работы с **ComfyUI** (FLUX.1 Dev, FLUX.2 Klein, Wan 2.2 I2V) и **Automatic1111 WebUI** (SD 1.5, SDXL, Pony, Illustrious).

Поддерживает интеграцию с **Google Antigravity**, **Open-WebUI**, **Cursor**, **ChatGPT Agent**, **Claude Desktop** и **Grok**.

---

## 🌟 Ключевые возможности

1. **Разделение ответственности (Separation of Concerns):**
   - `src/mcp/image_mcp.py` — тонкий MCP-интерфейс для генерации и редактирования изображений (A1111 / FLUX).
   - `src/mcp/video_mcp.py` — тонкий MCP-интерфейс для многосценарных видеороликов (Wan 2.2 / FFmpeg).
   - `src/core/` — общее ядро с переиспользуемыми клиентами и пайплайнами.
2. **Истинное префиксное кондиционирование (True Forced Prefix):**
   - Непрерывное сохранение динамики сцены без рывков и затуханий (crossfade) в латентном пространстве Wan 2.2 через ноду `VHS_LoadVideoPath`.
3. **Кинематографическая иерархия «Сцена (Scene) -> Блоки движения (Blocks)»:**
   - Чёткое разделение промптов: **Appearance Prompt** (статичная эстетика опорного кадра для FLUX/SDXL) и **Motion Prompt** (векторы действий и движения камеры для Wan 2.2).
4. **Гибридный опорный кадр (Hybrid Anchor):**
   - Мгновенное оживление уже существующего файла (`image_path`), либо автоматическая фоновая генерация по промпту (`prompt`).
5. **Автоматические постеры в чате (No Context Bloat):**
   - Вместо перегрузки контекста десятками мегабайт base64-видео сервер возвращает легкий постер первого кадра через нативный тип MCP `image`, а под ним формирует кликабельную ссылку на воспроизведение.
6. **Zero External Dependencies:**
   - Написан исключительно на стандартной библиотеке Python 3.10+ (без внешних pip-пакетов).

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

- **Python:** `>= 3.10` (системный, pip-зависимости не требуются).
- **FFmpeg:** доступен в системе через `PATH` или явно указан в переменной `FFMPEG_BIN` (рекомендуется сборка с поддержкой аппаратного кодировщика `h264_nvenc`).
- **ComfyUI:** запущен локально (по умолчанию `http://127.0.0.1:8188`).
  - Модели для картинок: `persephoneFluxNSFWSFW_20FP16.safetensors` (FLUX.1 Dev) или `aishaNSFWFlux2Klein_betaV88.safetensors` (FLUX.2 Klein).
  - Модели для видео: `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`, `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`, `wan2.2_vae.safetensors`, `umt5_xxl_fp8_e4m3fn_scaled.safetensors`.
  - Ноды ComfyUI: `ComfyUI-VideoHelperSuite` (`VHS_LoadVideoPath`, `VHS_VideoCombine`).
- **Automatic1111 WebUI (опционально):** запущен с флагом `--api` (по умолчанию `http://127.0.0.1:7860`) для моделей SDXL, Pony или Illustrious.

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
- **`comfy_get_models`**: Проверка статуса очереди ComfyUI.
- **`sd_interrupt`**: Прерывание активной генерации картинок в A1111 и ComfyUI.

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
- **`video_create_scenario`**: Создание и сохранение нового JSON-файла сценария в `workflows/<name>.json`.
- **`video_list_scenarios`**: Сканирование сценариев в `workflows/` с подсчетом длительности и количества сцен.
- **`video_get_status`**: Статус очереди задач ComfyUI (активная задача, число ожидающих).
- **`video_interrupt`**: Мгновенная остановка очереди рендера в ComfyUI.

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
