# media_mcp

Универсальный сервер **Model Context Protocol (MCP)** и фреймворк для локальной генерации изображений и последовательного многоблочного видео.

Работает с **ComfyUI** (FLUX.1 Dev, FLUX.2 Klein, Wan 2.2 I2V) и **Automatic1111 WebUI** (SD 1.5, SDXL, Pony, Illustrious).

Поддерживает работу с **Google Antigravity**, **Open-WebUI**, **Cursor**, **ChatGPT Agent**, **Claude Desktop** и **Grok**.

---

## Возможности

- **Разделение ответственности:**
  - `image_mcp.py` — тонкий интерфейс для генерации и редактирования изображений (A1111 / FLUX).
  - `video_mcp.py` — тонкий интерфейс для многосценарных видеороликов (Wan 2.2 / FFmpeg).
- **Истинное префиксное кондиционирование (True Forced Prefix):**
  - Непрерывное сохранение динамики сцены без рывков и затуханий (crossfade) в латентном пространстве Wan 2.2 через ноду `VHS_LoadVideoPath`.
- **Иерархия «Сцена (Scene) -> Блоки движения (Blocks)»:**
  - Разделение промптов: **Appearance Prompt** (статичная эстетика опорного кадра для FLUX/SDXL) и **Motion Prompt** (векторы действий и движения камеры для Wan 2.2).
- **Гибридный опорный кадр:**
  - Поддерживает как мгновенное оживление уже существующего файла (`image_path`), так и автоматическую фоновую генерацию по промпту (`prompt`).
- **Автоматические постеры в чате:**
  - Никакого переполнения контекста тяжелым base64 видео — сервер возвращает легкий постер первого кадра через нативный тип MCP `image`, а под ним формирует кликабельную ссылку на просмотр.
- **Zero-Dependency:**
  - Работает на стандартной библиотеке Python 3.10+ без необходимости в `pip`-пакетах.

---

## Архитектура проекта

```
media_mcp/
├── src/
│   ├── core/                   # Единое ядро генерации (Core Engine)
│   │   ├── config.py           # Конфигурация, пути и окружение
│   │   ├── comfy_client.py     # HTTP-клиент ComfyUI
│   │   ├── a1111_client.py     # HTTP-клиент Automatic1111
│   │   ├── ffmpeg_engine.py    # Монтаж, префиксная нарезка, постеры и HUD
│   │   ├── image_engine.py     # Генерация FLUX.1 / FLUX.2 / SDXL + авто-LoRA
│   │   └── video_pipeline.py   # Иерархический пайплайн Wan 2.2 Scene -> Blocks
│   │
│   └── mcp/                    # Тонкие MCP-адаптеры (JSON-RPC 2.0)
│       ├── image_mcp.py        # MCP-сервер картинок (Stdio / SSE :7865)
│       └── video_mcp.py        # MCP-сервер видео (Stdio / SSE :7866)
│
├── workflows/                  # Библиотека сценариев JSON
├── .agents/skills/             # Наборы скиллов для AI-агентов
│   ├── add-to-master/          # Процесс публикации правок через PR
│   ├── image-generation/       # Скилл генерации изображений
│   └── video-scenario/         # Скилл генерации видео
│
├── .env.example                # Шаблон переменных окружения
├── requirements.txt            # Описание зависимостей
└── README.md
```

---

## Быстрый старт

### 1. Проверка окружения
Убедитесь, что запущены локальные бэкенды:
- **ComfyUI:** `http://127.0.0.1:8188` (с моделями FLUX и Wan 2.2)
- **Automatic1111:** `http://127.0.0.1:7860` (опционально, для SDXL/Pony)
- **FFmpeg:** доступен в системе или прописан в `.env`

### 2. Запуск тестов серверов
```bash
# Тест подключения сервера картинок
python src/mcp/image_mcp.py --test

# Тест подключения сервера видео
python src/mcp/video_mcp.py --test
```

---

## Подключение клиентов MCP

### Google Antigravity & Cursor (Stdio Transport)

В файле конфигурации MCP (`~/.gemini/config/mcp_config.json` или в настройках проекта):

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

1. Запустите серверы в фоновом режиме SSE:
```bash
python src/mcp/image_mcp.py --sse --port 7865
python src/mcp/video_mcp.py --sse --port 7866
```

2. В панели управления Open-WebUI (**Admin Panel -> Settings -> Tools -> External Connections**):
- Image MCP: `http://host.docker.internal:7865/sse`
- Video MCP: `http://host.docker.internal:7866/sse`

---

## Список инструментов (MCP Tools)

### `image_mcp.py`
| Инструмент | Описание |
|---|---|
| `comfy_flux_txt2img` | Генерация через FLUX.1 Dev или FLUX.2 Klein в ComfyUI. |
| `sd_txt2img` | Генерация через Automatic1111 (SDXL, Pony, Illustrious, SD 1.5). |
| `sd_get_models` | Список чекпоинтов A1111. |
| `sd_set_model` | Переключение активного чекпоинта A1111. |
| `sd_get_loras` | Список LoRA моделей. |
| `sd_interrupt` | Прерывание текущей генерации картинок. |

### `video_mcp.py`
| Инструмент | Описание |
|---|---|
| `video_run_scenario` | Запуск полного рендера сценария (поддерживает файлы и встроенный JSON, авто-постер). |
| `video_validate_scenario` | Валидация и расчет хронометража структуры «Сцена -> Блоки». |
| `video_generate_block` | Одиночная генерация блока в Wan 2.2 с поддержкой префикса. |
| `video_create_scenario` | Сохранение структуры сценария в `workflows/<name>.json`. |
| `video_list_scenarios` | Список и хронометраж сценариев в папке `workflows/`. |
| `video_get_status` | Статус очереди задач ComfyUI. |
| `video_interrupt` | Немедленная остановка генерации видео. |

---

## Лицензия
MIT
