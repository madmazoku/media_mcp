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
     "image_path": "G:/Git/Gemini/output/images/character_frame.png"
   }
   ```
   Если вы или пользователь уже сгенерировали идеальную картинку через `image_mcp` (или загрузили файл), пайплайн сразу берёт её за основу без лишних генераций.

2. **Режим автоматической генерации (Autonomous):**
   ```json
   "anchor_image": {
     "prompt": "Cinematic visual novel CG, Hermione Granger as a cute unicorn mare, amber eyes, bushy mane, Gryffindor scarf, inside wooden train compartment, warm afternoon sunlight, Makoto Shinkai style",
     "model": "flux1_dev",
     "lora": null
   }
   ```
   Пайплайн сам сгенерирует опорный кадр через общее ядро `image_engine`.

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

```json
[
  {
    "scene_name": "Сцена 1: Купе поезда",
    "anchor_image": {
      "prompt": "Cinematic visual novel CG, first-person POV shot inside cozy vintage train compartment. Hermione Granger unicorn mare standing in open doorway, foreground shows pony hooves holding glowing ancient spellbook. Makoto Shinkai lighting, detailed",
      "model": "flux1_dev"
    },
    "blocks": [
      {
        "name": "Блок 1: Поиск жабы",
        "duration": 5.0,
        "overlap": 1.5,
        "motion_prompt": "Her horn sparkles with magic telekinesis aura, sliding the compartment door open, stepping forward into the compartment and looking around the floor searching for a toad, cute expressive face animation"
      },
      {
        "name": "Блок 2: Реакция на книгу",
        "duration": 5.0,
        "overlap": 0.0,
        "motion_prompt": "Her eyes suddenly spot the glowing spellbook, widening in fascinated academic surprise, taking an excited step closer, tilting her head with a joyful smile to examine the runes"
      }
    ]
  }
]
```

---

## 5. Вывод результатов в чат (Постер + Ссылка)

При вызове `video_run_scenario`:
1. Сервер автоматически извлекает первый кадр ролика и возвращает его в формате MCP `{"type": "image", "data": "..."}`.
2. Antigravity автоматически кэширует его в локальный offload (`media_0.png`).
3. **Всегда оформлять результат в чате как кликабельный постер:**
   ```markdown
   [![Нажмите для просмотра](file:///C:/Users/.../media_0.png)](file:///G:/Git/Gemini/output/video/filename.mp4)

   🎬 **[Смотреть видеоролик (8.5 сек)](file:///G:/Git/Gemini/output/video/filename.mp4)**
   ```
4. Если создавалась HUD-версия с таймерами (`enable_hud: true`), также указать ссылку на `_hud.mp4`.

## Запрет на самопроверку
- **Категорически запрещено** вызывать `view_file` или анализировать сгенерированные видео-файлы для самопроверки без явного запроса пользователя.
