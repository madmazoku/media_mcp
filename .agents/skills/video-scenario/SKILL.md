---
name: video-scenario
description: >-
  Руководство и алгоритм создания многосценарных видеороликов через media_mcp (Wan 2.2 / LTX-2.3 22B / ComfyUI / FFmpeg).
  Поддерживает иерархическую структуру "Сцена -> Блоки", гибридный опорный кадр (готовый файл или промпт),
  истинное латентное кондиционирование (Latent-Chaining), строгую валидацию JSON Schema (schemas/*.json) и мгновенную сборку Stream Copy (-c copy).
---

# Скилл генерации видео по сценариям (media_mcp)

## 1. Архитектура: Сцены, Блоки и Latent-Chaining

Видео строится по кинематографической структуре **Монтажная сцена (`scenes`) -> Непрерывные блоки движения (`blocks`)** и строго валидируется по [`G:/Git/media_mcp/schemas/scenario.json`](file:///G:/Git/media_mcp/schemas/scenario.json):

- **Сцена (`scene`):** непрерывный съёмочный план в одной локации/ракурсе.
  - Начинается с **Опорного кадра (`anchor_image`)**.
  - Опорный кадр задаёт визуальную эстетику, анатомию, одежду, свет и композицию сцены.
- **Блоки сцены (`blocks`):**
  - Описывают динамику, физику и траекторию камеры (`motion_prompt`).
  - **1-Frame I2V Chaining (`overlap: 0.0`, по умолчанию):** последний кадр предыдущего блока (`-tail.png`) передаётся стартовым кадром в следующий блок, а дублирующий первый кадр отрезается через `ImageFromBatch`.
  - **True Latent-Chaining (`overlap > 0`):** хвост предыдущего блока (`-tail.latent`) длительностью `overlap` вырезается через `LatentCut` и передаётся в следующий блок как префикс (`prefix_duration`). Следующий блок диффундирует суммарное окно `prefix_duration + clip_duration`, после чего `ImageFromBatch` отсекает префикс и сохраняет полную длительность `clip_duration` в `{block_prefix}-core.mp4`.
  - **Мгновенная сборка фильма:** все блоки всех сцен соединяются напрямую через `ffmpeg -f concat -c copy` с **нулевым перекодированием (0% потерь качества)**.

---

## 2. Строгие идентификаторы движков (`engine`) и условная JSON Schema

Все схемы находятся в `G:/Git/media_mcp/schemas/` (`additionalProperties: false`). Любые легаси-алиасы (`image_engine`, `video_engine`, `cfg_scale`, `sampler_name`, `output_name`, `start_image` внутри `anchor_image`, `prompt` вместо `motion_prompt` внутри `block`) запрещены и отклоняются валидатором.

### Видео-движки (`video.engine`, `scenes[].engine`, `blocks[].engine`):
- **`"wan-fast"` (Wan 2.2 14B Fast, дефолт)** — 4-step Lightx2v LoRA, CFG 1.0, шаг VAE: 4 кадра ($4k+1$).
- **`"wan"` (Wan 2.2 14B Full)** — полный 20-шаговый сэмплинг UNet High/Low Noise, CFG 3.5–5.0, шаг VAE: 4 кадра ($4k+1$).
  - *Условные ограничения схемы для `"wan-fast"` / `"wan"`:* запрещены LTX-специфичные поля `lora_name`, `lora_strength`, `weight_dtype`.
- **`"ltx"` (LTX-2.3 22B)** — 20 шагов, CFG 3.0, Distilled LoRA 1.1, шаг VAE: 8 кадров ($8k+1$), `width` и `height` строго кратны **32**.
  - *Условные ограничения схемы для `"ltx"`:* запрещены Wan-специфичные поля `checkpoint_low`, `ramp_latents`, `min_cfg`.

### Фото-движки опорного кадра (`image.engine`, `anchor_image.engine`):
- **`"sdxl"` (Automatic1111 WebUI, дефолт)** — SDXL / Pony / Illustrious (`cfg`, `sampler`, `negative_prompt`).
- **`"flux1"` (ComfyUI FLUX.1 Dev)** — поддерживает `guidance`, `lora`, `lora_strength`. Запрещены `negative_prompt`, `sampler`.
- **`"flux2"` (ComfyUI FLUX.2 Klein)** — поддерживает `guidance`. Запрещены `lora` (несовместимость размерностей тензоров с FLUX.1), `negative_prompt`, `sampler`.

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

## 4. Пример валидного сценария (`scenario.json`)

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
      "scene_name": "Сцена 1: Купе поезда",
      "anchor_image": {
        "prompt": "score_9, score_8_up, score_7_up, 1girl, young Hermione Granger, bushy wavy brown hair, warm brown eyes, Hogwarts school uniform, inside cozy vintage train compartment, soft golden afternoon sunlight"
      },
      "blocks": [
        {
          "name": "Блок 1: Открытие двери",
          "duration": 4.5,
          "overlap": 0.0,
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

## 5. Спецификация MCP-инструментов (API Reference)

Все MCP-запросы валидируются по схемам в `G:/Git/media_mcp/schemas/mcp_*.json`.

### `video_run_scenario` (`mcp_video_run_scenario.json`)
Запуск генерации видео по сценарию. Требуется ровно один из параметров: `scenario_path` или `scenario_data`.

| Аргумент | Тип | По умолчанию | Описание |
|---|---|---|---|
| `scenario_path` | `string` | — | Путь к JSON-файлу сценария (взаимоисключающий с `scenario_data`). |
| `scenario_data` | `object` | — | Объект сценария по `scenario.json` (взаимоисключающий с `scenario_path`). |
| `image` | `object` | — | Глобальные переопределения секции `image` (`engine`, `model`, `checkpoint`, `prompt`, `negative_prompt`, `width`, `height`, `steps`, `cfg`, `guidance`, `sampler`, `seed`, `lora`, `lora_strength`). |
| `video` | `object` | — | Глобальные переопределения секции `video` (`engine`, `model`, `checkpoint`, `checkpoint_low`, `prompt`, `negative_prompt`, `width`, `height`, `fps`, `max_duration`, `steps`, `cfg`, `seed`, `lora_name`, `lora_strength`, `weight_dtype`, `ramp_latents`, `min_cfg`). |
| `enable_hud` | `boolean` | `false` | Сгенерировать отладочное видео `{run_prefix}-hud.mp4` с таймлайном блоков. |

### `video_validate_scenario` (`mcp_video_validate_scenario.json`)
Предварительная валидация сценария по `scenario.json`, проверка существования файлов `anchor_image.image_path` и проверка условий `overlap_duration < clip_duration` и `prefix_duration + clip_duration < max_duration`.

| Аргумент | Тип | Описание |
|---|---|---|
| `scenario_path` | `string` | Путь к JSON-файлу сценария. |
| `scenario_data` | `object` | Объект сценария по `scenario.json`. |

### `video_generate_block` (`mcp_video_generate_block.json`)
Генерация одиночного видеосегмента. Обязательны `prompt` и хотя бы один из `start_image` / `prefix_latent`.

| Аргумент | Тип | Обязательный | Описание |
|---|---|---|---|
| `prompt` | `string` | **Да** | Промпт движения и камеры. |
| `engine` | `"wan-fast"` \| `"wan"` \| `"ltx"` | Нет (`"wan-fast"`) | Видео-движок. |
| `start_image` | `string` | Да (или `prefix_latent`) | Путь к стартовому изображению. |
| `prefix_latent` | `string` | Да (или `start_image`) | Путь к `.latent` safetensors файлу. |
| `negative_prompt` | `string` | Нет | Негативный промпт. |
| `duration` | `number` | Нет (`5.0`) | Длительность блока в секундах (`0 < duration <= 10.0`). |
| `width` | `integer` | Нет (`832`) | Ширина (кратно 16 для Wan, кратно 32 для LTX). |
| `height` | `integer` | Нет (`480`) | Высота (кратно 16 для Wan, кратно 32 для LTX). |
| `fps` | `integer` | Нет (`16` / `24`) | Частота кадров. |
| `steps` | `integer` | Нет | Число шагов сэмплирования. |
| `cfg` | `number` | Нет | CFG scale. |
| `seed` | `integer` | Нет (`-1`) | Сид генерации. |

### `media_get_status` и `media_interrupt` (`mcp_empty.json`)
Не принимают аргументов (`{}`).

---

## 6. Правила именования файлов и вывода в чат

1. **Строгий серверный шаблон имён файлов (без пользовательских имён):**
   - Опорный кадр сцены: `{run_prefix}-s{scene_idx:03d}-anchor.png`
   - Блок видео: `{run_prefix}-s{scene_idx:03d}-b{block_idx:03d}-core.mp4`
   - Хвост блока: `-tail.latent` (при `overlap > 0`) или `-tail.png` (при `overlap == 0`)
   - Финальное видео сценария: `{run_prefix}-final.mp4`, постер: `{run_prefix}-final.png`, HUD: `{run_prefix}-hud.mp4`
   - Одиночный блок (`video_generate_block`): `{run_prefix}-{engine}.mp4`, постер: `{run_prefix}-{engine}.png`
2. **Оформление в чате:**
   ```markdown
   [![Нажмите для просмотра](file:///C:/Users/.../media_0.png)](file:///G:/Git/Gemini/media_mcp/videos/filename.mp4)

   🎬 **[Смотреть видеоролик](file:///G:/Git/Gemini/media_mcp/videos/filename.mp4)**
   ```
3. **Запрет на самопроверку медиа:** категорически запрещено вызывать `view_file` на сгенерированных изображениях или видео без явного запроса пользователя.
