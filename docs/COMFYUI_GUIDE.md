# Руководство по генерации видео и цепочек в проекте Gemini

Вся инфраструктура настроена локально в проектной папке `G:\Git\Gemini`.

---

## 1. Структура проекта

```
G:\Git\Gemini\
├── 📂 scripts\                    # Исполняемые скрипты
│   ├── comfyui_generator.py       # Txt2Img генератор (FLUX uncensored)
│   ├── video_extender.py          # Базовые утилиты продления видео и xfade
│   └── scenario_video_pipeline.py # Сценарный конвейер с префиксным перекрытием 20+5
├── 📂 workflows\                  # Схемы и шаблоны сценариев
│   ├── sample_scenario.json       # Пример 3-блочного сценария (20с+5с, 20с+5с, 15с = 45с)
│   └── ...
├── 📂 output\
│   ├── 📂 images\                # Сгенерированные мастер-кадры
│   └── 📂 video\                 # Финальные смонтированные фильмы
└── 📂 temp\                       # Буфер префиксных видеоклипов и кадров стыковки
```

---

## 2. Сценарный генератор (`scenario_video_pipeline.py`)

### Концепция работы:
* Для каждого блока задается: **Длительность (`duration`)** + **Время пересечения (`overlap`)** + **Промпт содержимого**.
* По умолчанию для блоков принято: **`duration = 20.0s`**, **`overlap = 5.0s`**.
* **Префиксное продолжение:** Ровно последние 5 секунд Блока $N$ становятся начальными 5 секундами Блока $N+1$.
* **Поддержка затравочной картинки (`--start-image`):**
  * Если задана картинка (например `--start-image G:\Git\Gemini\gadget_manga.jpg`), Блок 1 стартует с нее с сохранением персонажа и стиля.
  * Если картинка не указана, конвейер автоматически генерирует затравочный мастер-кадр через нецензурированный **FLUX.1 Dev** (`persephone`) в нативном 16:9 разрешении **832×480**.

---

### Примеры использования

#### А. Запуск готового сценария из JSON (например, на 45 секунд):
```powershell
python G:\Git\Gemini\scripts\scenario_video_pipeline.py --scenario G:\Git\Gemini\workflows\sample_scenario.json --start-image G:\Git\Gemini\gadget_manga.jpg --output "final_movie_45s.mp4"
```

#### Б. Быстрый запуск из консоли (2 блока по 20 сек с перекрытием 5 сек = 35 сек):
```powershell
python G:\Git\Gemini\scripts\scenario_video_pipeline.py --prompt "Cyberpunk female operative navigating neon rainy alleys, cinematic moody lighting" --duration 20.0 --overlap 5.0 --output "cyberpunk_35s.mp4"
```

#### В. Формат файла сценария (`sample_scenario.json`):
```json
[
  {
    "duration": 20.0,
    "overlap": 5.0,
    "prompt": "Lord Captain stands on voidship bridge looking through star viewport, slow camera pan"
  },
  {
    "duration": 20.0,
    "overlap": 5.0,
    "prompt": "Lord Captain slowly turns towards holographic tactical console, dramatic amber shadows"
  },
  {
    "duration": 15.0,
    "overlap": 0.0,
    "prompt": "Close-up of Lord Captain operating glowing holographic controls, confident expression"
  }
]
```
*Итоговый хронометраж фильма:* $(20 - 5) + (20 - 5) + 15 = \mathbf{45.0 \text{ секунд}}$ без единого визуального стыка.
