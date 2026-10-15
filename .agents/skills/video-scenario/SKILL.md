---
name: video-scenario
description: >-
  Руководство и алгоритм создания многосценарных видеороликов через media_mcp (Wan 2.2 / LTX-2.3 22B / ComfyUI / FFmpeg).
  Поддерживает иерархическую структуру "Сцена -> Блоки", гибридный опорный кадр (готовый файл или промпт),
  истинное латентное кондиционирование (Latent-Chaining), первоклассный negative_prompt и мгновенную сборку Stream Copy (-c copy).
---

# Скилл генерации видео по сценариям (media_mcp 2.1)

## 1. Архитектура: Сцены, Блоки и Latent-Chaining

Видео строится по кинематографической структуре **Монтажная сцена (Scene) -> Непрерывные блоки движения (Blocks)**:

- **Сцена (Scene):** непрерывный съёмочный план в одной локации/ракурсе.
  - Начинается с **Опорного кадра (`anchor_image`)**.
  - Опорный кадр задаёт визуальную эстетику, анатомию, одежду, свет и композицию сцены.
- **Блоки сцены (`blocks`):**
  - Описывают динамику, физику и траекторию камеры (`motion_prompt`).
  - **True Forced Prefix Conditioning:** хвост предыдущего блока передаётся в следующий как многокадровый префикс (17 кадров при 16 fps = 1.0 сек) через нативный `WanImageToVideo` (`start_image`), маскирующий 100% префиксных кадров как conditioned (`mask = 0.0`). Никаких поломок масок, размазывания контуров и паразитных «червяков»!
  - **Чистовая основа (`block_core`):** для промежуточных блоков ComfyUI сразу сохраняет чистый кусок `[0 .. core_frames]` через нативную ноду `ImageFromBatch`. Никакого резания через FFmpeg!
  - **Мгновенная сборка фильма:** все блоки и сцены соединяются через `ffmpeg -f concat -c copy` за 0.05 сек с **нулевым перекодированием (0% потерь качества)**.
- **Смена сцены:**
  - Переход к новой сцене создаёт классическую **монтажную склейку (Hard Cut)** с новым опорным кадром.

---

## 2. Разделение движков: `video_engine` и `image_engine`

Движки разделены строго по назначению:

### Видео-движок (`video_engine`):
- **`wan` (Wan 2.2 14B)** — для плавной кинематографической эстетики, аниме, деталей лица и волос (шаг VAE: 4 кадра, $4k+1$).
- **`ltx` (LTX-2.3 22B)** — для масштабных динамичных действий, экшена и магии (шаг VAE: 8 кадров, $8k+1$).

### Графический движок опорного кадра (`image_engine`):
- **`comfy_flux` (ComfyUI FLUX)** — FLUX 1 Dev / FLUX 2 Klein для кинематографического CG и аниме Visual Novel с тонким лайн-артом.
- **`a1111` (Automatic1111 WebUI)** — единый транспорт для всех моделей SDXL / SD 1.5. Модели вроде Pony, Animagine, Juggernaut — это обычные чекпоинты SDXL, работающие через единый A1111 API без искусственного разделения архитектур.

---

## 3. Negative Prompt первого класса

- Параметр `"negative_prompt"` доступен на всех уровнях:
  - В корне сценария (`"default_negative_prompt"`)
  - В сцене (`"negative_prompt"`)
  - В отдельном блоке (`"negative_prompt"`)
  - В туле `video_generate_block`
- Любой хардкод китайских строк затухания полностью удалён.
- Искусственный Anchor Color-Lock отменён: свет и цвет свободно подчиняются промпту (каст заклинаний, *Lumos*, вспышки, закаты, вход в темные туннели).

---

## 4. Гибридный опорный кадр (`anchor_image`)

Поддерживаются два режима:

1. **Режим готового файла (Human-in-the-loop / Арт-директор):**
   ```json
   "anchor_image": {
     "image_path": "G:/Git/Gemini/media_mcp/images/character_frame.png"
   }
   ```
   Если вы или пользователь уже сгенерировали изображение через `image_mcp` (или загрузили файл), пайплайн сразу берёт его за основу.

2. **Режим автоматической генерации (Autonomous):**
   ```json
   "anchor_image": {
     "prompt": "Cinematic visual novel CG, 1girl, young Hermione Granger, bushy wavy brown hair, warm brown eyes, Hogwarts school uniform with pleated skirt and Gryffindor tie, inside cozy vintage train compartment, soft golden afternoon sunlight streaming in, Makoto Shinkai and Kyoto Animation aesthetic",
     "model": "flux1_dev",
     "guidance": 3.5,
     "steps": 28
   }
   ```
   - **SDXL / Pony High-Res Fix:** Если опорный кадр генерируется через A1111, движок автоматически применяет High-Res Fix с качественным даунскейлом Lanczos.
   - **FLUX 1 Dev / ComfyUI:** Обеспечивает эталонную аниме visual novel стилистику с чистым лайн-артом.

---

## 5. Требование к непересекающимся нахлёстам ($2 \times \text{overlap} < \text{duration}$)

Для любого промежуточного блока $k$, имеющего одновременно входящий стык от блока $k-1$ ($\text{overlap}_{in}$) и исходящий стык к блоку $k+1$ ($\text{overlap}_{out}$), строго обязательно:
$$\text{overlap}_{in} + \text{overlap}_{out} < \text{duration} \quad (\text{при равных нахлёстах: } 2 \times \text{overlap} < \text{duration})$$
- Если это условие нарушено, зоны переходов перекрывают друг друга, «съедая» уникальное тело блока и вызывая временные петли и скачки кадров.
- Валидатор `video_validate_scenario` и пайплайн автоматически блокируют запуск с ошибкой, если на промежуточном блоке $2 \times \text{overlap} \ge \text{duration}$.

---

## 5. Разделение промптов: Внешность vs Движение

- **В `anchor_image.prompt`** пишите, **КАК ЭТО ВЫГЛЯДИТ**:
  - Стиль, внешность персонажа, одежда, черты лица, освещение, объектив, окружение.
  - Избегайте глаголов длинного движения.
- **В `blocks[].motion_prompt`** пишите, **ЧТО ПРОИСХОДИТ**:
  - Глаголы действий, жесты, поворот головы, эмоции, траектория камеры (pan, orbit, zoom in).
  - Не повторяйте описание внешности, модель уже видит её в опорном кадре.

---

## 6. Пример структуры сценария

```json
{
  "engine": "wan",
  "default_negative_prompt": "blurry, low quality, jitter, static, deformed, extra limbs, bad anatomy, text, watermark",
  "scenes": [
    {
      "scene_name": "Сцена 1: Купе поезда",
      "anchor_image": {
        "prompt": "Cinematic visual novel CG, 1girl, young Hermione Granger, bushy wavy brown hair, warm brown eyes, Hogwarts school uniform with pleated skirt and Gryffindor tie, inside cozy vintage train compartment, soft golden afternoon sunlight streaming in, Makoto Shinkai and Kyoto Animation aesthetic",
        "guidance": 3.5,
        "steps": 28
      },
      "blocks": [
        {
          "name": "Блок 1: Открытие двери",
          "duration": 4.5,
          "overlap": 1.0,
          "motion_prompt": "She gently slides the compartment door open with her hand, stepping inside gracefully, looking around with a polite, curious expression"
        },
        {
          "name": "Блок 2: Осмотр купе",
          "duration": 4.5,
          "overlap": 0.0,
          "motion_prompt": "Looking curiously at the passenger across from her, tilting her head with an inquisitive polite smile, soft breeze moving strands of her hair"
        }
      ]
    }
  ]
}
```

---

## 7. Спецификация MCP-инструментов (API Reference)

Агент обязан использовать данное описание как **единственный источник правды** для вызова MCP-инструментов. Категорически запрещено читать исходный код сервера для выяснения аргументов или схем.

### `video_run_scenario`
Запуск генерации видео по сценарию (JSON-файлу или объекту данных) с поддержкой True Latent-Chaining и сборки Stream Copy.

| Аргумент | Тип | По умолчанию | Описание |
|---|---|---|---|
| `scenario_path` | `string` | `null` | Путь к JSON-файлу сценария (ищется в `workflows/`, `temp/` или по абсолютному пути). |
| `scenario_data` / `scenes` | `array` \| `object` | `null` | Данные сценария напрямую в вызове (массив сцен). |
| `video_engine` | `string` (`"wan"`, `"ltx"`) | `"wan"` | Видео-движок: `"wan"` (Wan 2.2 14B) или `"ltx"` (LTX-2.3 22B). |
| `image_engine` | `string` (`"sdxl"`, `"flux1"`, `"flux2"`, `"comfy_flux"`, `"a1111"`) | `"auto"` | Графический движок опорного кадра: `"sdxl"` (Automatic1111 SDXL/Pony), `"flux1"` (ComfyUI FLUX 1 Dev), `"flux2"` (ComfyUI FLUX 2 Klein). |
| `image_model` | `string` | `null` | Опциональное переопределение конкретного чекпоинта (например, если в A1111 нужен специфический `.safetensors`). При обычном использовании не требуется — `image_engine` автоматически выбирает активную/дефолтную модель. |
| `negative_prompt` | `string` | `null` | Глобальный дефолтный негативный промпт для всех блоков сценария. |
| `start_image` | `string` | `null` | Опциональный путь к готовому опорному кадру для первой сцены. |
| `output_name` | `string` | `null` | Имя итогового видео (например, `"hermione_clip"` -> `{timestamp}_{hash}_hermione_clip.mp4`). Если указано, суффикс `_final` НЕ добавляется! |
| `width` | `integer` | `832` | Ширина видео (кратно 16 для Wan, 32 для LTX). |
| `height` | `integer` | `480` | Высота видео. |
| `fps` | `integer` | `16` (wan) / `24` (ltx) | Частота кадров. |
| `enable_hud` | `boolean` | `false` | Сгенерировать отладочную версию видео с HUD-таймлайном блоков. |

**Результат вызова:**
* Текстовая сводка: путь к итоговому файлу (`{timestamp}_{hash}_{output_name}.mp4`), хронометраж, разрешение.
* MCP Image: постер первого кадра в base64 (автоматически кэшируется клиентом).

---

### `video_validate_scenario`
Валидация сценария перед запуском. Проверяет корректность структуры, наличие опорных кадров, промптов и строгое соблюдение условия `prefix_length + tail_length < clip_duration`.

| Аргумент | Тип | По умолчанию | Описание |
|---|---|---|---|
| `scenario_path` | `string` | `null` | Путь к JSON-файлу сценария. |
| `scenario_data` | `array` \| `object` | `null` | Данные сценария напрямую. |

**Результат вызова:**
Отчёт о валидации со статусом `VALID [OK]` или `INVALID [ERROR]`, расчётом суммарной длительности и списком сцен.

---

### `video_generate_block`
Генерация одиночного блока движения (одного видеосегмента).

| Аргумент | Тип | Обязательный | Описание |
|---|---|---|---|
| `prompt` | `string` | **Да** | Промпт движения и камеры. |
| `engine` | `string` (`"wan"`, `"ltx"`) | Нет (`"wan"`) | Движок видео. |
| `negative_prompt` | `string` | Нет | Негативный промпт. |
| `duration` | `number` | Нет (`5.0`) | Длительность блока в секундах. |
| `start_image` | `string` | Нет | Путь к начальному кадру (изображению). |
| `end_image` | `string` | Нет | Путь к конечному кадру (только для Wan 2.2). |
| `prefix_latent` | `string` | Нет | Путь к `.latent` safetensors для True Latent-Chaining. |
| `width` | `integer` | Нет (`832`) | Ширина. |
| `height` | `integer` | Нет (`480`) | Высота. |
| `fps` | `integer` | Нет (`16`/`24`) | FPS. |
| `output_name` | `string` | Нет | Имя выходного файла. |

---

### `media_get_status`
Опрос текущего статуса пайплайна в реальном времени. Не требует аргументов (`{}`).
Возвращает:
* Статус Video Pipeline: активная сцена, блок, фаза генерации, время выполнения.
* Статус ComfyUI: очередь задач, выполняемый prompt_id и node_id.
* Статус Automatic1111: прогресс генерации в %, текущий шаг сэмплинга.

---

### `media_interrupt`
Аварийная остановка текущей генерации. Не требует аргументов (`{}`).
Прерывает выполнение пайплайна, шлёт сигналы отмены в ComfyUI и Automatic1111.

---

## 8. Вывод результатов в чат (Постер + Ссылка)

При вызове `video_run_scenario`:
1. Сервер стримит heartbeat-уведомления каждые 5 секунд (`notifications/message` и `notifications/progress`).
2. По завершении возвращает постер первого кадра в формате MCP `{"type": "image", "data": "..."}`.
3. Antigravity автоматически кэширует его в локальный offload (`media_0.png`).
4. **Всегда оформлять результат в чате как кликабельный постер:**
   ```markdown
   [![Нажмите для просмотра](file:///C:/Users/.../media_0.png)](file:///G:/Git/Gemini/media_mcp/videos/filename.mp4)

   🎬 **[Смотреть видеоролик](file:///G:/Git/Gemini/media_mcp/videos/filename.mp4)**
   ```
5. Если создавалась HUD-версия с таймерами (`enable_hud: true`), также указать ссылку на `_hud.mp4`.

---

## 9. Правила для агентов (Обязательно к исполнению)

1. **Запрет на самопроверку медиа:**
   - Категорически запрещено вызывать `view_file` или любые средства анализа сгенерированных изображений или видео для самопроверки без явного запроса пользователя. Сразу отдавать результат в чат.
2. **Строгий контракт сохранения файлов артефактов в `temp/`:**
   - В ходе генерации сохраняются ТОЛЬКО: `{block_prefix}_core.mp4` и временный `{block_prefix}_tail.mp4` (только для промежуточных блоков, передающих префикс следующему блоку; удаляется сразу после потребления).
   - Никаких промежуточных `_raw.mp4`, `_tmp.mp4`, `.latent` файлов на диске не сохраняется!
   - Для последнего блока сохраняется только `{block_prefix}_core.mp4`.
   - Все блоки всех сцен склеиваются напрямую в финальное видео в один шаг Stream Copy (`-c copy`) без промежуточных монтажей сцен (`_s001.mp4`).
   - Имя итогового видео: если передан `output_name`, формируется `{timestamp}_{hash}_{output_name}.mp4` БЕЗ суффикса `_final`.
3. **Условие перекрытия:**
   - Строго соблюдать условие `prefix_length + tail_length < clip_duration`.
4. **Бесшовный монтаж:**
   - Все блоки внутри сцены и между сценами склеиваются мгновенно через FFmpeg Stream Copy (`-c copy`) без перекодирования.
