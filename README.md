# media_mcp

Универсальный фреймворк и единый сервер **Model Context Protocol (MCP)** для генерации изображений и последовательного многоблочного видео.

Разработан для работы с **ComfyUI** (FLUX.1 Dev, FLUX.2 Klein, Wan 2.2 I2V, LTX-2.3 22B) и **Automatic1111 WebUI** (SD 1.5, SDXL, Pony, Illustrious).

Поддерживает интеграцию с **Google Antigravity**, **Open-WebUI**, **Cursor**, **ChatGPT Agent**, **Claude Desktop** и **Grok**.

---

## 🌟 Ключевые возможности

1. **Единый сервис и чистая архитектура (Unified Server):**
   - `src/mcp/media_mcp.py` — единый сервер MCP (Stdio + SSE `:7865`) с поддержкой Range-запросов (HTTP 206) для мгновенной перемотки видео в веб-интерфейсах.
   - `src/core/` — общее ядро с переиспользуемыми клиентами, генераторами, инспектором LoRA, валидатором JSON Schema и пайплайнами.
   - `schemas/` — единый каталог строгих схем **JSON Schema (Draft 2020-12)** для `config.json`, сценариев, спецификаций изображений и всех MCP-запросов (`additionalProperties: false`).
2. **Канонический набор из 8 инструментов (7 пользовательских + 1 для разработчиков):**
   - Никаких разрозненных или дублирующих тулов — единые инвариантные контракты входа и выхода.
   - Выделенный инструмент `media_reload_modules` для горячей перезагрузки модулей ядра на лету без перезапуска процесса сервера.
3. **True Forced Prefix Conditioning (Latent-Chaining):**
   - Непрерывное сохранение динамики сцены внутри монтажа: при `overlap > 0` хвост предыдущего блока (`-tail.latent`) отрезается через `LatentCut` и передаётся в следующий блок как префикс (`prefix_duration`). Следующий блок диффундирует окно `prefix_duration + clip_duration`, после чего `ImageFromBatch` отсекает префикс и сохраняет полную длительность `clip_duration` в `{block_prefix}-core.mp4`. При `overlap == 0.0` (дефолт) используется бесшовная 1-кадровая сцепка по последнему кадру (`-tail.png`).
4. **Чистовая основа блоков и Stream Copy (-c copy):**
   - Чистовая основа каждого блока (`-core.mp4`) сохраняется напрямую в H.264. Финальная сборка всех блоков в FFmpeg происходит через мгновенный **Stream Copy (`-c copy`) за 0.05 сек без перекодирования**.
5. **Мульти-LoRA и автоматический парсинг триггеров:**
   - Поддержка нескольких LoRA в запросе: `"loras": [{"name": "...", "strength": 0.8}]`.
   - Мгновенное чтение метаданных safetensors заголовка (<1ms) для извлечения триггерных слов (`media_get_loras`).
   - Динамический чейнинг нод `LoraLoaderModelOnly` в FLUX.1 и добавление тегов в SDXL.

---

## 📁 Структура проекта

```
media_mcp/
├── schemas/                    # Централизованные JSON Schema (Draft 2020-12)
│   ├── config.json             # Схема файла конфигурации
│   ├── image_spec.json         # Каноническая спецификация параметров изображения
│   ├── scenario.json           # Каноническая спецификация видеосценария
│   ├── mcp_image_generate.json # Схема вызова image_generate
│   ├── mcp_video_generate.json # Схема вызова video_generate
│   ├── mcp_video_validate.json # Схема вызова video_validate
│   ├── mcp_media_get_models.json
│   ├── mcp_media_get_samplers.json
│   ├── mcp_media_get_loras.json
│   └── mcp_empty.json
│
├── src/
│   ├── core/                   # Единое ядро генерации (Core Engine)
│   │   ├── __init__.py
│   │   ├── config.py           # Конфигурация + валидация config.json по схеме
│   │   ├── schema_validator.py # Загрузчик, инлайнер $ref и валидатор JSON Schema
│   │   ├── comfy_client.py     # HTTP-клиент ComfyUI (очередь, опрос, прерывание)
│   │   ├── a1111_client.py     # HTTP-клиент Automatic1111 (модели, сэмплеры, LoRA)
│   │   ├── discovery.py        # Инвариантная инспекция моделей, сэмплеров и LoRA
│   │   ├── lora_inspector.py   # Быстрый парсер safetensors метаданных и триггеров
│   │   ├── ffmpeg_engine.py    # Монтаж Stream Copy (-c copy), извлечение постеров и HUD
│   │   ├── image_engine.py     # Унифицированная генерация опорных кадров и изображений
│   │   ├── video_pipeline.py   # Иерархический пайплайн видео (Scene -> Blocks)
│   │   ├── tools.py            # Единый движок исполнения и реестр всех инструментов
│   │   └── generators/         # Полиморфные стратегии генераторов (SDXL, FLUX, Wan, LTX)
│   │
│   ├── cli.py                  # Автономный CLI runner
│   └── mcp/                    # Единый MCP-сервер (JSON-RPC 2.0)
│       ├── __init__.py
│       └── media_mcp.py        # Единый сервер (Stdio + SSE :7865)
│
├── examples/                   # Эталонные примеры конфигурации, сценария и запросов
│   ├── config.json             # Эталонный шаблон конфигурации
│   ├── scenario.json           # Эталонный многосценарный видеоролик
│   ├── mcp_image_generate.json
│   ├── mcp_video_generate.json
│   ├── mcp_video_validate.json
│   ├── mcp_media_get_models.json
│   ├── mcp_media_get_samplers.json
│   └── mcp_media_get_loras.json
│
├── cmd/                        # Командные файлы запуска
│   ├── cli.cmd                 # Автономный CLI runner для всех инструментов
│   └── run_media_sse.cmd       # Запуск единого MCP SSE сервера (:7865)
│
├── tests/                      # Автоматизированный тестовый набор
│   └── test_streamline.py
│
├── requirements.txt            # Зависимости (jsonschema>=4.20.0)
└── README.md                   # Главная документация
```

---

## 🛠️ Справочник инструментов (MCP Tools & CLI)

Все инструменты работают через единый диспетчер `src/core/tools.py`. Доступны через MCP (Stdio), SSE HTTP (`:7865`) и командную строку (`cmd\cli.cmd`).
Все параметры строго валидируются по `schemas/mcp_*.json` перед выполнением.

### 1. Генерация медиа
- **`image_generate`** (`schemas/mcp_image_generate.json`):
  - Принимает спецификацию изображения (`image_spec.json`):
    - `prompt` *(string, обязателен)*: текстовый промпт.
    - `engine` *(`"sdxl"` | `"flux1"` | `"flux2"`, по умолчанию `"sdxl"`)*.
    - `model` *(string, опционально)*: имя файла модели/чекпоинта.
    - `width`, `height` *(int, кратны 16, по умолчанию 832x480 для SDXL, 1024x1024 для FLUX)*.
    - `steps` *(int)*, `seed` *(int)*.
    - `cfg` *(number, для SDXL)*, `guidance` *(number, для FLUX)*.
    - `sampler` *(string, опционально)*: сэмплер.
    - `loras` *(array of `{"name": str, "strength": float}`, опционально; запрещён для FLUX.2)*.
    - `negative_prompt` *(string, SDXL only)*.
  - Сохраняет разрешённую чистую спецификацию в `workflows/{run_prefix}-spec.json` и возвращает `run_prefix` и `spec_path`.
- **`video_generate`** (`schemas/mcp_video_generate.json`):
  - Поддерживает `scenario_path` *(string)* и/или инлайн объект сценария (`scenes`).
  - При указании базового файла читает его с диска и применяет оверлей переданных полей (глубокий мердж для словарей `image`/`video`, атомарная замена для списков `scenes`/`blocks`).
  - Ключ `scenario_path` немедленно удаляется из результирующего словаря, предотвращая циклы.
  - Сохраняет чистый итоговый сценарий в `workflows/{run_prefix}-spec.json` и возвращает `run_prefix`, `spec_path`, `video_path`, `poster_path`.
  - Поддерживает `enable_hud` *(boolean, по умолчанию `false`)* для вывода диагностического оверлея.
- **`video_validate`** (`schemas/mcp_video_validate.json`):
  - Dry-run валидация сценария по схеме и законам таймингов без запуска рендеринга на GPU. Поддерживает `scenario_path` и инлайн-структуры.

### 2. Инспекция и обнаружение моделей (Discovery)
- **`media_get_models`** (`schemas/mcp_media_get_models.json`):
  - Опциональные фильтры: `type` (`"image"` | `"video"`), `engine` (`"sdxl"`, `"flux1"`, `"wan-fast"` и др.).
  - Всегда возвращает инвариантное дерево:
    ```json
    {
      "image": {
        "sdxl": ["ponyDiffusionV6XL...safetensors"],
        "flux1": ["persephoneFlux...safetensors"],
        "flux2": ["aishaNSFWFlux2...safetensors"]
      },
      "video": {
        "wan-fast": ["wan2.2_i2v_high_noise...safetensors"],
        "wan-full": ["wan2.2_i2v_high_noise...safetensors"],
        "ltx": ["ltx-video-2b...safetensors"]
      }
    }
    ```
- **`media_get_samplers`** (`schemas/mcp_media_get_samplers.json`):
  - Обязательные аргументы: `engine` *(string)* и `model` *(string)*.
  - Возвращает компактный список совместимых сэмплеров, где элемент с индексом `[0]` — рекомендуемый по умолчанию:
    `["Euler a", "Euler", "DPM++ 2M Karras", ...]`
- **`media_get_loras`** (`schemas/mcp_media_get_loras.json`):
  - Обязательные аргументы: `engine` *(string)* и `model` *(string)*.
  - Возвращает список совместимых LoRA с извлечёнными триггерными словами:
    ```json
    [
      {
        "name": "detailed_hair_pony",
        "triggers": ["detailed hair", "wavy strands", "highlights"]
      }
    ]
    ```

### 3. Управление пайплайном и разработка
- **`media_get_status`** (`schemas/mcp_empty.json`):
  - Возвращает статус очередей ComfyUI, Automatic1111 и текущую сцену/блок видеопайплайна.
- **`media_interrupt`** (`schemas/mcp_empty.json`):
  - Немедленно прерывает генерацию во всех активных движках.
- **`media_reload_modules`** (`schemas/mcp_empty.json`):
  - Инструмент разработчика: выполняет горячую перезагрузку всех модулей пакета `core` в памяти в обратном порядке зависимостей.

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
      "name": "Scene 1: Train Compartment",
      "anchor_image": {
        "prompt": "score_9, score_8_up, score_7_up, 1girl, Hermione Granger, bushy brown hair, Hogwarts uniform, inside cozy vintage train compartment, golden afternoon sunlight"
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

### Законы таймингов и слияния блоков внутри монтажа:
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
