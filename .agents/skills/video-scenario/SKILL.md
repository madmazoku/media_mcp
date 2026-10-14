---
name: video-scenario
description: >-
  Руководство и алгоритм создания многосценарных видеороликов через media_mcp (Wan 2.2 / ComfyUI / FFmpeg).
  Поддерживает иерархическую структуру "Сцена -> Блоки", гибридный опорный кадр (готовый файл или промпт),
  истинное префиксное кондиционирование (True Forced Prefix) и автоматический возврат кликабельных постеров.
---

# Скилл генерации видео по сценариям (media_mcp)

## 1. Архитектура: Сцены и Блоки

Видео строится по классической кинематографической структуре **Монтажная сцена (Scene) -> Непрерывные блоки движения (Blocks)**:

- **Сцена (Scene):** непрерывный съёмочный план в одной локации/ракурсе.
  - Начинается с **Опорного кадра (`anchor_image`)**.
  - Опорный кадр задаёт визуальную эстетику, анатомию, одежду, свет и композицию сцены.
- **Блоки сцены (`blocks`):**
  - Описывают динамику, физику и траекторию камеры (`motion_prompt`).
  - Между блоками одной сцены работает **True Forced Prefix Conditioning:** хвостовой кусок Блока 1 (длительностью `overlap`, обычно 1.5–5.0 сек) загружается в скрытое пространство Wan 2.2 и становится стартом движения для Блока 2.
  - FFmpeg срезает дублирующийся кусок нахлёста (`[0 .. dur - overlap]`), обеспечивая **100% бесшовную склейку без затуханий (crossfade) и рывков**.
- **Смена сцены:**
  - Переход к новой сцене создаёт классическую **монтажную склейку (Hard Cut)** с новым опорным кадром.

---

## 2. Гибридный опорный кадр (`anchor_image`)

Поддерживаются два режима:

1. **Режим готового файла (Human-in-the-loop / Арт-директор):**
   ```json
   "anchor_image": {
     "image_path": "G:/Git/Gemini/media_mcp/images/character_frame.png"
   }
   ```
   Если вы или пользователь уже сгенерировали идеальную картинку через `image_mcp` (или загрузили файл), пайплайн сразу берёт её за основу без лишних генераций.

2. **Режим автоматической генерации (Autonomous):**
   ```json
   "anchor_image": {
     "prompt": "Cinematic visual novel CG, 1girl, cute unicorn mare Hermione Granger, amber eyes, bushy brown mane, Gryffindor scarf, inside wooden train compartment, warm afternoon sunlight, Makoto Shinkai style",
     "model": "pony",
     "cfg_scale": 5.5,
     "sampler_name": "Euler a"
   }
   ```
   Пайплайн сам сгенерирует опорный кадр через общее ядро `image_engine`.
   - **SDXL / Pony High-Res Fix:** Если опорный кадр генерируется для Wan 2.2 в разрешении 832x480, движок автоматически генерирует его в 1.5x разрешении (1248x720) и делает качественный даунскейл Lanczos, устраняя искажения анатомии, лишние конечности и артефакты.
   - **Pony CFG Scale:** Для моделей Pony автоматически применяется оптимальный CFG 5.0–5.5 (вместо 7.0+), предотвращая пережарку (deep-fried overbaking) и сохраняя чистые аниме/мульт-цвета.

---

## 3. Разделение промптов: Внешность vs Движение

- **В `anchor_image.prompt`** пишите, **КАК ЭТО ВЫГЛЯДИТ**:
  - Стиль, внешность персонажа, одежда, черты лица, освещение, объектив, окружение.
  - Избегайте глаголов длинного движения.
- **В `blocks[].motion_prompt`** пишите, **ЧТО ПРОИСХОДИТ**:
  - Глаголы действий, жесты, поворот головы, эмоции, траектория камеры (pan, orbit, zoom in).
  - Не повторяйте описание внешности, модель уже видит её в опорном кадре.

---

## 4. Пример структуры сценария

Сценарии передаются напрямую в аргумент `scenes` / `scenario_data` инструмента `video_run_scenario`:

```json
[
  {
    "scene_name": "Сцена 1: Купе поезда",
    "anchor_image": {
      "prompt": "Cinematic visual novel CG, first-person POV shot inside cozy vintage train compartment. Cute unicorn pony Hermione Granger standing in open doorway, Makoto Shinkai lighting, detailed, score_9, score_8_up",
      "model": "pony",
      "cfg_scale": 5.5
    },
    "blocks": [
      {
        "name": "Блок 1: Открытие двери",
        "duration": 5.0,
        "overlap": 1.5,
        "motion_prompt": "Her horn sparkles with magic telekinesis aura, sliding the compartment door open, stepping forward into the compartment on all four hooves and looking around searching, cute expressive face animation"
      },
      {
        "name": "Блок 2: Осмотр купе",
        "duration": 5.0,
        "overlap": 0.0,
        "motion_prompt": "Looking curiously at the passenger, tilting her head with an inquisitive polite expression, soft ear twitch"
      }
    ]
  }
]
```

---

## 5. Вывод результатов в чат (Постер + Ссылка)

При вызове `video_run_scenario`:
1. Сервер стримит heartbeat-уведомления каждые 5 секунд (`notifications/message` и `notifications/progress`), предотвращая таймаут клиента.
2. По завершении возвращает постер первого кадра в формате MCP `{"type": "image", "data": "..."}`.
3. Antigravity автоматически кэширует его в локальный offload (`media_0.png`).
4. **Всегда оформлять результат в чате как кликабельный постер:**
   ```markdown
   [![Нажмите для просмотра](file:///C:/Users/.../media_0.png)](file:///G:/Git/Gemini/media_mcp/videos/filename.mp4)

   🎬 **[Смотреть видеоролик](file:///G:/Git/Gemini/media_mcp/videos/filename.mp4)**
   ```
5. Если создавалась HUD-версия с таймерами (`enable_hud: true`), также указать ссылку на `_hud.mp4`.

## Запрет на самопроверку
- **Категорически запрещено** вызывать `view_file` или анализировать сгенерированные видео-файлы для самопроверки без явного запроса пользователя. Сразу отдавать результат генерации в ответ.
