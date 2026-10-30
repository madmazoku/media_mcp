# media_mcp

Универсальный фреймворк и единый сервер **Model Context Protocol (MCP)** для генерации изображений и последовательного многоблочного видео.

Разработан для работы с **ComfyUI** (FLUX.1 Dev, FLUX.2 Klein, Wan 2.2 I2V, LTX-Video) и **Automatic1111 WebUI** (SD 1.5, SDXL, Pony, Illustrious).

Поддерживает интеграцию с **Google Antigravity**, **Open-WebUI**, **Cursor**, **ChatGPT Agent**, **Claude Desktop** и **Grok**.

---

## 🌟 Ключевые возможности

1. **Единый сервис и чистая архитектура (Unified Server):**
   - `src/mcp/media_mcp.py` — единый сервер MCP (Stdio + SSE `:7865`) с поддержкой Range-запросов (HTTP 206) для мгновенной перемотки видео в веб-интерфейсах.
   - `src/core/` — общее ядро с переиспользуемыми клиентами, генераторами, инспектором safetensors, валидатором JSON Schema и пайплайнами.
   - `schemas/` — единый каталог строгих схем **JSON Schema (Draft 2020-12)** для `config.json`, сценариев, спецификаций изображений и всех MCP-запросов и ответов (`additionalProperties: false`).
2. **Канонический набор из 10 инструментов (9 пользовательских + 1 для разработчиков):**
   - Единые инвариантные контракты входа и выхода для CLI, Stdio MCP и HTTP SSE.
   - Выделенный инструмент `media_reload_modules` для горячей перезагрузки модулей ядра в топологическом порядке зависимостей без перезапуска процесса сервера.
3. **Единый пайплайн оптимального разрешения (~1.0 MP) для всех движков изображений:**
   - Все генераторы (`sdxl`, `flux1`, `flux2`) во всех режимах (`txt2img`, `img2img`, `inpaint`, `outpaint`) всегда выполняют диффузию в оптимальном нативном разрешении ~1.0 MP с сохранением пропорций (`native_width` x `native_height`, шаг квантования `64px` для SDXL UNet и `16px` для FLUX DiT), после чего результат масштабируется (Lanczos) к запрошенному размеру `width` x `height`.
   - При запросе апскейла свыше 1.1 MP или экстремального соотношения сторон (`> 2.4:1`) возвращается информативное поле `warnings`.
4. **True Forced Prefix Conditioning (5D Latent-Chaining):**
   - Непрерывное сохранение динамики сцены внутри монтажа для `wan-fast`, `wan-full` и `ltx`: при `overlap > 0` хвост предыдущего блока (`-tail.latent`) отрезается через `LatentCut` и передаётся в следующий блок через 5D-ноды `WanLatentPrefixLock` / `WanLatentPrefixRestore` (для Wan) и in-canvas ноду `LTXLatentPrefixToVideo` (для LTX) без промежуточного декодирования в пиксели. При `overlap == 0.0` (дефолт) используется бесшовная 1-кадровая сцепка по последнему кадру (`-tail.png`).
5. **Чистовая основа блоков и Stream Copy (-c copy):**
   - Чистовая основа каждого блока (`-core.mp4`) сохраняется напрямую в H.264. Финальная сборка всех блоков в FFmpeg происходит через мгновенный **Stream Copy (`-c copy`) без перекодирования**.
6. **Zero-VRAM Discovery, бандлы Wan 2.2 и парсинг триггеров LoRA:**
   - Автоматическая классификация архитектур моделей и LoRA (`sdxl`, `flux1`, `flux2`, `wan-fast`, `wan-full`, `ltx`) по бинарным заголовкам `.safetensors` (`<0.5ms`, без зависимости от `torch`).
   - Автоматическая группировка парных моделей `high_noise` и `low_noise` для `wan-fast` и `wan-full` в единый канонический идентификатор бандла (например, `wan2.2_i2v_14B_fp8_scaled.safetensors`).
   - Извлечение триггерных слов LoRA из заголовков `.safetensors` и sidecar-файлов (`.json`, `.civitai.info`) для всех движков.

---

## 📁 Структура проекта

```
media_mcp/
├── docs/                       # Инженерные правила, исследования и база знаний
│   ├── RULES.md                # Инженерные правила репозитория
│   ├── ARCHITECTURE.md         # Архитектура сервера и контракты
│   ├── SCENARIO_DESIGN.md      # Иерархия сценариев и правила слияния
│   ├── MODELS_AND_ENGINES.md   # Справочник моделей, бандлов и LoRA
│   ├── IMAGE_PIPELINE.md       # Конвейер генерации изображений (~1.0 MP)
│   ├── ARTIFACTS_AND_NAMING.md # Единый дефисный стандарт именования артефактов
│   ├── NETWORK_AND_QUEUE.md    # Сетевой пулинг, трекинг очереди и GPU Lock
│   ├── CLIENT_INTEGRATION.md   # Подключение к клиентам
│   ├── COMFYUI_GUIDE.md        # Справочник видеопайплайна ComfyUI
│   ├── MEDIA_MCP_KNOWLEDGE_BASE.md # База знаний латентной математики
│   └── VIDEO_BLOCK_CHAINING_RESEARCH.md # Исследование латентного чейнинга
│
├── schemas/                    # Централизованные JSON Schema (Draft 2020-12)
│   ├── config.json             # Схема файла конфигурации
│   ├── scenario.json           # Каноническая спецификация видеосценария
│   ├── requests/               # Схемы валидации входящих аргументов инструментов
│   │   ├── image_generate.json
│   │   ├── video_generate.json
│   │   ├── media_get_job.json
│   │   ├── media_remove_job.json
│   │   ├── media_get_models.json
│   │   ├── media_get_samplers.json
│   │   ├── media_get_loras.json
│   │   ├── media_get_status.json
│   │   ├── media_interrupt.json
│   │   └── media_reload_modules.json
│   └── responses/              # Строгие схемы ответов инструментов (Response Schemas)
│       ├── image_generate.json
│       ├── video_generate.json
│       ├── media_get_job.json
│       ├── media_remove_job.json
│       ├── media_get_models.json
│       ├── media_get_samplers.json
│       ├── media_get_loras.json
│       ├── media_get_status.json
│       ├── media_interrupt.json
│       └── media_reload_modules.json
│
├── src/
│   ├── core/                   # Единое ядро генерации (Core Engine)
│   │   ├── __init__.py
│   │   ├── config.py           # Строгая конфигурация без CWD-автопоиска
│   │   ├── schema_validator.py # Загрузчик, инлайнер $ref, валидатор входа и выхода
│   │   ├── schema_formatter.py # Движок терминальной справки CLI и валидации примеров
│   │   ├── comfy_client.py     # HTTP-клиент ComfyUI (очередь, опрос, прерывание)
│   │   ├── a1111_client.py     # HTTP-клиент Automatic1111 (модели, сэмплеры, LoRA)
│   │   ├── discovery.py        # Инвариантная инспекция моделей, бандлов, сэмплеров и LoRA
│   │   ├── lora_inspector.py   # Быстрый бинарный парсер safetensors заголовков и латентов
│   │   ├── ffmpeg_engine.py    # Монтаж Stream Copy (-c copy) и извлечение кадров
│   │   ├── image_engine.py     # Унифицированная генерация изображений (~1.0 MP + post-resize)
│   │   ├── video_pipeline.py   # Иерархический пайплайн видео (Scene -> Blocks)
│   │   ├── tools.py            # Единый движок исполнения и реестр всех инструментов
│   │   └── generators/         # Полиморфные стратегии генераторов (SDXL, FLUX, Wan, LTX)
│   │
│   ├── cli.py                  # Автономный CLI runner с чистым JSON stdout и справкой
│   └── mcp/                    # Единый MCP-сервер (JSON-RPC 2.0)
│       ├── __init__.py
│       └── media_mcp.py        # Единый сервер (Stdio + SSE :7865)
│
├── examples/                   # Эталонные примеры конфигурации, сценария и запросов
│   ├── config.json             # Минималистичный шаблон конфигурации
│   ├── scenario.json           # Эталонный многосценарный видеоролик
│   ├── mcp_image_generate.json
│   ├── mcp_image_generate_inpaint.json
│   ├── mcp_image_generate_outpaint.json
│   ├── mcp_video_generate.json
│   ├── mcp_media_get_job.json
│   ├── mcp_media_remove_job.json
│   ├── mcp_media_get_models.json
│   ├── mcp_media_get_samplers.json
│   └── mcp_media_get_loras.json
│
├── bin/                        # Кросс-платформенные лаунчеры (Windows .cmd + Unix .sh)
│   ├── cli.cmd / cli.sh        # Автономный CLI runner для всех инструментов
│   ├── run_media_sse.cmd / .sh # Запуск единого MCP SSE сервера (:7865)
│   └── run_media_stdio.cmd/.sh # Запуск MCP Stdio сервера для Antigravity / Claude
│
├── tests/                      # Модульный тестовый набор (5 сьютов)
│   ├── test_schemas.py
│   ├── test_discovery.py
│   ├── test_image_pipeline.py
│   ├── test_video_pipeline.py
│   └── test_mcp_and_cli.py
│
├── pyproject.toml              # Конфигурация пакета и pytest
├── CHANGELOG.md                # История изменений по стандарту Keep a Changelog и SemVer
├── requirements.txt            # Зависимости (jsonschema, requests, urllib3, Pillow)
└── README.md                   # Главная документация
```

---

## 🛠️ Справочник инструментов (MCP Tools & CLI)

Все инструменты работают через единый диспетчер `src/core/tools.py`. Доступны через MCP (Stdio), SSE HTTP (`:7865`) и командную строку (`bin\cli.cmd` / `bin/cli.sh`).
Флаги CLI: `bin\cli.cmd --help` (справка по инструментам и схемам) и `bin\cli.cmd --version` / `-v` (инспекция версии сборки без необходимости передачи `--config`).
Все параметры строго валидируются по `schemas/requests/<tool>.json` перед выполнением, а ответы — по `schemas/responses/<tool>.json`.

### 1. Генерация медиа
- **`image_generate`** (`schemas/requests/image_generate.json`):
  - Принимает параметры генерации изображения:
    - `engine` *(`"sdxl"` | `"flux1"` | `"flux2"`, **обязателен**)*.
    - `model` *(string, **обязателен**)*: имя файла модели/чекпоинта из `media_get_models`.
    - `prompt` *(string, **обязателен**)*: текстовый промпт (передаётся в модель без скрытых авто-префиксов).
    - `width`, `height` *(int, по умолчанию 1024x1024)*. Диффузия всегда выполняется в оптимальном разрешении ~1.0 MP (`native_width` x `native_height`) с тем же соотношением сторон, затем изображение приводится к целевому `(width, height)`.
    - `steps` *(int)*, `seed` *(int)*.
    - `cfg` *(number, для SDXL)*, `guidance` *(number, для FLUX)*.
    - `sampler` *(string, опционально)*: сэмплер.
    - `prompt` *(string, опционально при наличии `image_path`)*: текстовый позитивный промпт. Если `image_path` указан без `prompt` и без `mask_path` при `fit="crop"`, выполняется пассивная подгонка разрешения без запуска диффузии.
    - `loras` *(array of `{"name": str, "strength": float}`, опционально)*.
    - `negative_prompt` *(string, для SDXL)*.
    - `image_path` *(string, опционально)*: путь к исходному изображению для img2img / inpaint / outpaint / crop.
    - `mask_path` *(string, опционально)*: путь к полутоновой маске инпейнтинга (1.0/белый = перегенерация, 0.0/чёрный = сохранение исходника).
    - `fit` *(`"crop"` | `"outpaint"`, опционально)*: стратегия подгонки при изменении пропорций исходника.
    - `denoise` *(number, 0.05..1.0, по умолчанию 0.65, для outpaint 0.85)*.
  - Сохраняет разрешённую спецификацию в `workflows/{run_prefix}-spec.json` и возвращает `run_prefix`, `image_path`, `spec_path`, `width`, `height`, `native_width`, `native_height` и опциональный массив `warnings`.
- **`video_generate`** (`schemas/requests/video_generate.json`):
  - Асинхронный запуск генерации: выполняет предварительную синхронную валидацию, сохраняет сценарий в `workflows/{run_prefix}-spec.json`, регистрирует задачу в очереди сервера и мгновенно возвращает `{"run_prefix": "...", "status": "queued", "spec_path": "..."}`.
  - Поддерживает `scenario_path` *(string)* и/или инлайн объект сценария (`video`, `image`, `scenes`).
  - В блоках сценария (`scenes[].blocks[]`) поддерживается опциональный `end_image` (first-last frame conditioning для Wan 2.2 и LTX-2.3).
  - Все технические параметры (`engine`, `model`, `width`, `height`, `fps`, `steps`, `cfg`, `guidance`, `sampler`, `seed`, `ramp_latents`, `min_cfg`, `weight_dtype`) задаются строго в корневых секциях `.image` и `.video`.
  - Опциональный флаг `dry_run` *(boolean, по умолчанию `false`)*: при `true` выполняет мгновенную pre-flight валидацию, расчет хронометража и сохранение spec без постановки задачи на GPU (возвращает `status: "validated"`, `is_valid: true`, `total_duration`, `scenes_count`, `blocks_count`).
- **`media_get_job`** (`schemas/requests/media_get_job.json`):
  - Опрос статуса, прогресса и получение результатов задачи по `run_prefix`. Поддерживает блокирующее ожидание `wait` (0-55 сек). По завершении возвращает структурированные контейнеры `video` и `image` (постер).
- **`media_remove_job`** (`schemas/requests/media_remove_job.json`):
  - Отмена задачи в очереди или на GPU, либо удаление метаданных завершённой задачи из серверного реестра.

### 2. Инспекция и обнаружение моделей (Discovery)
- **`media_get_models`** (`schemas/requests/media_get_models.json`):
  - Опциональные фильтры: `type` (`"image"` | `"video"`), `engine` (`"sdxl"`, `"flux1"`, `"flux2"`, `"wan-fast"`, `"wan-full"`, `"ltx"`).
  - Возвращает инвариантное дерево моделей (для `wan-fast` и `wan-full` парные файлы `high_noise` + `low_noise` группируются в канонический идентификатор бандла):
    ```json
    {
      "image": {
        "sdxl": ["ponyDiffusionV6XL_v6StartWithThis.safetensors"],
        "flux1": ["persephoneFlux1Dev_v10FP8.safetensors"],
        "flux2": ["aishaNSFWFlux2Klein9B_v11Fp8.safetensors"]
      },
      "video": {
        "wan-fast": ["wan2.2_i2v_14B_fp8_scaled.safetensors"],
        "wan-full": ["wan2.2_i2v_14B_fp8_scaled.safetensors"],
        "ltx": ["ltx-video-2b-v0.9.5.safetensors"]
      }
    }
    ```
- **`media_get_samplers`** (`schemas/requests/media_get_samplers.json`):
  - Обязательные аргументы: `engine` *(string)* и `model` *(string)*.
  - Возвращает список совместимых сэмплеров, где элемент `[0]` — рекомендуемый по умолчанию.
- **`media_get_loras`** (`schemas/requests/media_get_loras.json`):
  - Обязательные аргументы: `engine` *(string)* и `model` *(string)*; опциональный фильтр `pattern` (регистронезависимый Python regex).
  - Возвращает список совместимых по архитектуре LoRA с извлечёнными триггерными словами из `.safetensors` и sidecar-файлов.

### 3. Управление пайплайном и разработка
- **`media_get_status`** (`schemas/requests/media_get_status.json`):
  - Возвращает статус `_GPU_LOCK`, очередей ComfyUI (`comfy_online`), Automatic1111 (`a1111_available`) и текущую фазу видеопайплайна.
- **`media_interrupt`** (`schemas/requests/media_interrupt.json`):
  - Немедленно прерывает генерацию во всех активных движках.
- **`media_reload_modules`** (`schemas/requests/media_reload_modules.json`):
  - Инструмент разработчика: выполняет горячую перезагрузку модулей пакета `core` в топологическом порядке зависимостей с сохранением `_GPU_LOCK` и активной конфигурации.

---

## 🎬 Формат сценария (`schemas/scenario.json`)

```json
{
  "name": "Hermione Train Compartment",
  "image": {
    "engine": "sdxl",
    "model": "ponyDiffusionV6XL_v6StartWithThis.safetensors",
    "cfg": 5.5,
    "steps": 30
  },
  "video": {
    "engine": "wan-fast",
    "model": "wan2.2_i2v_14B_fp8_scaled.safetensors",
    "width": 832,
    "height": 480,
    "fps": 16,
    "negative_prompt": "blurry, low quality, jitter, static, deformed, extra limbs, bad anatomy, text, watermark"
  },
  "scenes": [
    {
      "name": "Scene 1: Train Compartment",
      "anchor_image": {
        "prompt": "score_9, score_8_up, score_7_up, 1girl, Hermione Granger, bushy brown hair, Hogwarts uniform, inside cozy vintage train compartment, golden afternoon sunlight"
      },
      "blocks": [
        {
          "name": "Block 1: Opening Door",
          "duration": 4.5,
          "overlap": 1.0,
          "prompt": "She gently slides the compartment door open with her hand and steps inside, looking around curiously"
        },
        {
          "name": "Block 2: Greeting",
          "duration": 4.5,
          "overlap": 0.0,
          "prompt": "She notices the passenger across from her and smiles warmly while sitting down"
        }
      ]
    }
  ]
}
```

### Законы таймингов и слияния блоков внутри монтажа:
Блок $i$ получает хвост предыдущего блока $\text{prefix\_duration} = \text{overlap}_{i-1}$ (для первого блока сцены $\text{prefix\_duration} = 0$), диффундирует суммарное окно $\text{prefix\_duration} + \text{clip\_duration}$, отсекает префикс через `ImageFromBatch` и отдаёт в финальную склейку ровно $\text{clip\_duration}$ чистого видео:
1. **Хвост меньше самого клипа:**
   $$\text{overlap\_duration} < \text{clip\_duration}$$
2. **Суммарное окно генерации строго меньше лимита движка (6.0 сек для `wan-fast`/`wan-full` и 16.0 сек для `ltx`):**
   $$\text{prefix\_duration} + \text{clip\_duration} < \text{engine\_max\_duration}$$
3. **Итоговая длительность сцены:** равна чистой сумме $\sum \text{clip\_duration}$ всех блоков сцены.

---

## Лицензия

MIT License.
