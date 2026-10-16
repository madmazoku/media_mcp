---
name: video-scenario
description: >-
  Руководство и алгоритм создания видеороликов через единый media_mcp (Wan 2.2 / LTX-2.3 22B / ComfyUI / FFmpeg).
  Поддерживает иерархическую структуру "Сцена -> Блоки", гибридный опорный кадр (готовый файл или промпт),
  истинное латентное кондиционирование (Latent-Chaining), строгую валидацию JSON Schema (schemas/*.json) и мгновенную сборку Stream Copy (-c copy).
---

# Скилл генерации видео по сценариям (media_mcp)

## 1. Архитектура: Сцены, Блоки и Latent-Chaining

Видео строится по кинематографической структуре **Монтажная сцена (`scenes`) -> Непрерывные блоки движения (`blocks`)** и строго валидируется по [`schemas/scenario.json`](file:///schemas/scenario.json):

- **Сцена (`scene`):** непрерывный съёмочный план в одной локации/ракурсе.
  - Начинается с **Опорного кадра (`anchor_image`)**.
  - Опорный кадр задаёт визуальную эстетику, анатомию, одежду, свет и композицию сцены.
  - **Правило разрешения опорного кадра:** генерация опорника всегда должна выполняться в **нативном разрешении фото-модели (~1.0 MP)** с **тем же аспектным соотношением**, что и целевое видео. Генерация опорного кадра напрямую в низком разрешении видео (например, 832x480) категорически запрещена, так как вызывает голодание механизма внимания (Attention Starvation) фото-модели и деформацию лиц/анатомии, которые видео-модель затем неизбежно анимирует:
    - Видео **16:9** (`832x480`) $\longrightarrow$ Опорный кадр **16:9** (`1280x720` в `image`/`anchor_image`)
    - Видео **9:16** (`480x832`) $\longrightarrow$ Опорный кадр **9:16** (`720x1280` в `image`/`anchor_image`)
    - Видео **1:1** (`640x640`) $\longrightarrow$ Опорный кадр **1:1** (`1024x1024` в `image`/`anchor_image`)
    - Видео **4:3** (`832x624`) $\longrightarrow$ Опорный кадр **4:3** (`1152x864` в `image`/`anchor_image`)
    - Видео **21:9** (`832x352`) $\longrightarrow$ Опорный кадр **21:9** (`1536x640` в `image`/`anchor_image`)
- **Блоки сцены (`blocks`):**
  - Описывают динамику, физику и траекторию камеры (`motion_prompt`).
  - **1-Frame I2V Chaining (`overlap: 0.0`, по умолчанию):** последний кадр предыдущего блока (`-tail.png`) передаётся стартовым кадром в следующий блок, а дублирующий первый кадр отрезается через `ImageFromBatch`.
  - **True Latent-Chaining (`overlap > 0`):** хвост предыдущего блока (`-tail.latent`) длительностью `overlap` вырезается через `LatentCut` и передаётся в следующий блок как префикс (`prefix_duration`). Следующий блок диффундирует суммарное окно `prefix_duration + clip_duration`, после чего `ImageFromBatch` отсекает префикс и сохраняет полную длительность `clip_duration` в `{block_prefix}-core.mp4`.
  - **Мгновенная сборка фильма:** все блоки всех сцен соединяются напрямую через `ffmpeg -f concat -c copy` с **нулевым перекодированием (0% потерь качества)**.
  - Одиночный ролик — это просто сцена из одного блока.

---

## 2. Строгие идентификаторы движков (`engine`) и имена параметров

Все схемы находятся в `schemas/` (`additionalProperties: false`).
Все файлы: строго `<entity>_path` (`scenario_path`, `image_path`, `video_path`, `poster_path`).
Модели: строго `model`.
Имена уровней: строго `name` (на уровне сценария, сцены и блока).

### Видео-движки (`video.engine`, `scenes[].engine`, `blocks[].engine`):
- **`"wan-fast"` (Wan 2.2 14B Fast, дефолт)** — 4-step Lightx2v LoRA, CFG 1.0, шаг VAE: 4 кадра ($4k+1$).
- **`"wan-full"` (Wan 2.2 14B Full)** — полный 20-шаговый сэмплинг UNet High/Low Noise, CFG 3.5–5.0, шаг VAE: 4 кадра ($4k+1$).
- **`"ltx"` (LTX-2.3 22B)** — 20 шагов, CFG 3.0, Distilled LoRA 1.1, шаг VAE: 8 кадров ($8k+1$), `width` и `height` строго кратны **32**.

### Фото-движки опорного кадра (`image.engine`):
Параметры среды рендеринга (`engine`, `model`, `width`, `height`, `steps`, `cfg`, `guidance`, `sampler`, `denoise`) задаются глобально в секции `image` для всего фильма:
- **`"sdxl"` (Automatic1111 WebUI, дефолт)** — SDXL / Pony / Illustrious (`cfg`, `sampler`, `negative_prompt`, `loras`).
- **`"flux1"` (ComfyUI FLUX.1 Dev)** — поддерживает `guidance`, `loras`. Запрещены `negative_prompt`, `cfg`.
- **`"flux2"` (ComfyUI FLUX.2 Klein)** — поддерживает `guidance`. Запрещены `loras` (несовместимость размерностей тензоров с FLUX.1), `negative_prompt`, `cfg`.

### Логика опорного кадра сцены (`scenes[].anchor_image`):
Опорный кадр задаёт только визуальную специфику конкретного плана и поддерживает 3 режима:
1. **Готовый кадр (Ready keyframe):** `{"keyframe_path": "path/to/frame.png"}` — моментально берётся с диска без запуска генератора, все остальные параметры генерации запрещены.
2. **img2img перерисовка / адаптация кадра:** `{"image_path": "path/to/source.png", "prompt": "...", "denoise": 0.55}` — трансформирует исходный кадр через img2img (сохраняет анатомию/лицо персонажа, адаптируя ракурс, одежду или окружение под новую сцену).
3. **txt2img генерация:** `{"prompt": "..."}` (без `image_path`) — диффузия опорного кадра с нуля.
- **Объединение LoRA:** массив `scenes[].anchor_image.loras` конкатенируется с глобальным `image.loras` (локальный `strength` имеет приоритет при совпадении имён).

---

## 3. Законы таймингов и слияния блоков внутри одной сцены (монтажа)

При слиянии блоков внутри одной сцены для каждого блока $i$ действуют два строгих математических условия:
- Пусть $\text{clip\_duration} = \text{blocks}[i].\text{duration}$.
- Пусть $\text{prefix\_duration} = \text{blocks}[i-1].\text{overlap}$ при $i > 0$ (и $0.0$ для первого блока сцены $i = 0$).
- Пусть $\text{overlap\_duration} = \text{blocks}[i].\text{overlap}$ при $i < N - 1$ (и $0.0$ для последнего блока сцены).
- Пусть $\text{max\_duration} = \text{video.max\_duration}$ (по умолчанию `10.0` сек из `config.json`).

Обязательные условия валидации:
1. **Хвост перекрытия строго меньше длительности клипа:**
   $$\text{overlap\_duration} < \text{clip\_duration}$$
2. **Суммарное окно диффузии блока строго меньше максимальной длительности (`max_duration`):**
   $$\text{prefix\_duration} + \text{clip\_duration} < \text{max\_duration}$$

Поскольку префикс $\text{prefix\_duration}$ добавляется к генерации и затем отсекается через `ImageFromBatch`, каждый блок отдаёт в итоговый монтаж полную длительность $\text{clip\_duration}$, а суммарная длительность сцены равна $\sum \text{clip\_duration}$.

---

## 4. Спецификация MCP-инструментов (API Reference)

Все MCP-запросы валидируются по схемам в `schemas/mcp_*.json`.

### `video_generate` (`mcp_video_generate.json`)
Запуск генерации видео по сценарию.
Принимает `scenario_path` *(string)* и/или инлайн объект сценария (`scenes`, `image`, `video`, `name`).
Если передан `scenario_path`, сервер загружает базовый JSON-файл и накладывает оверлей переданных параметров (глубокий мердж для объектов, замена для списков). Ключ `scenario_path` немедленно извлекается и удаляется из результирующего словаря, предотвращая утечки и рекурсию.
Каждый запуск сохраняет итоговый разрешённый JSON-сценарий в `workflows/{run_prefix}-spec.json` и возвращает `run_prefix` и `spec_path` в ответе.

| Аргумент | Тип | По умолчанию | Описание |
|---|---|---|---|
| `scenario_path` | `string` | — | Путь к базовому JSON-файлу сценария на диске. |
| `scenes` | `array` | — | Список сцен сценария. |
| `image` | `object` | — | Глобальные настройки опорного кадра (`engine`, `model`, `prompt`, `negative_prompt`, `width`, `height`, `steps`, `cfg`, `guidance`, `sampler`, `seed`, `loras`). |
| `video` | `object` | — | Глобальные настройки видео (`engine`, `model`, `prompt`, `negative_prompt`, `width`, `height`, `fps`, `max_duration`, `loras`). |
| `enable_hud` | `boolean` | `false` | Сгенерировать отладочное видео `{run_prefix}-hud.mp4` с таймлайном блоков. |

### `video_validate` (`mcp_video_validate.json`)
Предварительная сухая валидация сценария по схеме, проверка наличия файлов `image_path` и проверка законов таймингов. Принимает те же `scenario_path` и инлайн-структуры.

### Автономный CLI запуск (Standalone CLI)
Все инструменты поддерживают прямой вызов через единый CLI runner с идентичной логикой обработки:
```cmd
cmd\cli.cmd [--config config.json] <tool_name> [request_path | -]
```
Примеры:
```cmd
cmd\cli.cmd video_validate examples/scenario.json
cmd\cli.cmd --config config.json video_generate workflows/hermione_orbit.json
echo {"scenario_path":"examples/scenario.json"} | cmd\cli.cmd video_validate -
```

### `media_get_models`, `media_get_samplers`, `media_get_loras`
Инспекция чекпоинтов, сэмплеров и LoRA с триггерными словами. `media_get_loras` поддерживает опциональный `pattern` (регистронезависимый Python regex для поиска по имени LoRA или триггерам).

### `media_get_status`, `media_interrupt`, `media_reload_modules`
Управление статусом, прерывание и горячая перезагрузка модулей ядра без рестарта сервера.

---

## 5. Правила именования файлов и вывода в чат

1. **Строгий серверный шаблон имён файлов:**
   - Опорный кадр сцены: `{run_prefix}-s{scene_idx:03d}-anchor.png`
   - Блок видео: `{run_prefix}-s{scene_idx:03d}-b{block_idx:03d}-core.mp4`
   - Хвост блока: `-tail.latent` (при `overlap > 0`) или `-tail.png` (при `overlap == 0`)
   - Финальное видео сценария: `{run_prefix}-final.mp4`, постер: `{run_prefix}-final.png`, HUD: `{run_prefix}-hud.mp4`
2. **Оформление в чате:**
   ```markdown
   [![Нажмите для просмотра](file:///path/to/media_0.png)](file:///path/to/videos/filename.mp4)

   🎬 **[Смотреть видеоролик](file:///path/to/videos/filename.mp4)**
   ```
3. **Запрет на самопроверку медиа:** категорически запрещено вызывать `view_file` на сгенерированных изображениях или видео без явного запроса пользователя.
