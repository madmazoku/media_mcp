# Руководство по сценарной генерации видео (Scenario Video Pipeline)

Вся инфраструктура видеогенерации и сценарного конвейера инкапсулирована в `media_mcp`. Под капотом используются модели **Wan 2.2** (`wan-fast`, `wan-full`) и **LTX-Video** (`ltx`), управляемые через ComfyUI API с 5D латентным префиксным чейнингом и бесшовным монтажом FFmpeg.

---

## 1. Сценарная иерархия: Корневые `.image`/`.video` $\to$ Сцена $\to$ Блоки

Видеопайплайн оперирует трехуровневой структурой (`schemas/scenario.json`):
1. **Корневые технические секции (`.image` и `.video`):**
   - Задают обязательные `engine` и `model`, а также все технические настройки (`width`, `height`, `fps`, `steps`, `cfg`, `guidance`, `sampler`, `seed`, `max_duration`, `ramp_latents`, `min_cfg`, `weight_dtype`) и глобальные стилевые `prompt`, `negative_prompt`, `loras`.
2. **Сцена (`scenes[]`):** единая локация и состояние персонажа:
   - **`anchor_image`**: опорное изображение (`prompt`, либо `image_path` для img2img/inpaint/outpaint, либо готовый `keyframe_path`).
3. **Блок (`scenes[].blocks[]`):** непрерывный отрезок видео (обычно 3–5 секунд):
   - **`motion_prompt`**: описание действия персонажа и движения камеры.
   - **`duration`**: чистая продолжительность генерируемого клипа (в секундах).
   - **`overlap`**: время латентного перекрытия со следующим блоком (`0.0` по умолчанию для чистого 1-кадрового I2V чейнинга; `> 0.0` для 5D латентного префикса).

---

## 2. Формат спецификации сценария (`scenario.json`)

```json
{
  "name": "Starship Bridge Inspection",
  "image": {
    "engine": "flux2",
    "model": "aishaNSFWFlux2Klein9B_v11Fp8.safetensors",
    "guidance": 3.5,
    "steps": 20
  },
  "video": {
    "engine": "wan-fast",
    "model": "wan2.2_i2v_14B_fp8_scaled.safetensors",
    "width": 832,
    "height": 480,
    "fps": 16,
    "max_duration": 10.0
  },
  "scenes": [
    {
      "name": "Scene 1: Starship Bridge",
      "anchor_image": {
        "prompt": "Lord Captain stands on voidship bridge looking through star viewport, cinematic lighting, 8k"
      },
      "blocks": [
        {
          "duration": 5.0,
          "overlap": 0.0,
          "motion_prompt": "Lord Captain looking through viewport, stars outside drifting slowly, subtle breathing"
        },
        {
          "duration": 5.0,
          "overlap": 0.0,
          "motion_prompt": "Lord Captain slowly turns head towards holographic console, dramatic amber light reflections"
        }
      ]
    }
  ]
}
```

---

## 3. Способы запуска

### А. Через Stdio MCP (AI-ассистент):
Ассистент вызывает инструмент `video_generate` с JSON-телом сценария (или `scenario_path`).
- Перед запуском рекомендуется вызывать `video_validate` для сухой проверки всех полей, таймингов и пропорций.
- Результат возвращает: `video_path` (смонтированный MP4), `poster_path` (постер сцены 1), `spec_path` (зафиксированный JSON запуска).

### Б. Через командную строку (CLI):
```bash
bin/cli.cmd --config path/to/config.json video_generate --payload examples/scenario.json
```
Или передача JSON через стандартный поток ввода (stdin):
```bash
cat examples/scenario.json | bin/cli.cmd --config path/to/config.json video_generate --stdin
```

### В. Сухая валидация через CLI:
```bash
bin/cli.cmd --config path/to/config.json video_validate --payload examples/scenario.json
```

---

## 4. Математика таймингов и склейки кадров

1. **Квантование кадров VAE:**
   - Для **Wan 2.2** VAE temporal stride равен 4: $T_{\text{pixel}} = 1 + 4 \times k_{\text{lat}}$ (17, 33, 49, 65, 81 кадров).
   - Для **LTX-Video** VAE temporal stride равен 8: $T_{\text{pixel}} = 1 + 8 \times k_{\text{lat}}$ (17, 25, 33, 41, 49, 65, 81 кадров).
2. **Бесшовный Stream Copy (FFmpeg `-c copy`):**
   - Все блоки кодируются с идентичными параметрами H.264 и собираются через `ffmpeg -f concat -safe 0 -c copy` без пережатия видеопотока.
3. **Исключение дублирования стыковочного кадра:**
   - При 1-кадровом I2V чейнинге (`overlap = 0.0`) кадр 0 нового блока совпадает с последним кадром предыдущего. Пайплайн автоматически сдвигает нарезку ядра нового блока на 1 кадр (`batch_index = 1`), исключая микро-зависание видео на стыке.
4. **5D латентный префиксный чейнинг (`overlap > 0.0s`):**
   - **Wan 2.2 (`wan-fast`, `wan-full`):** использует `WanLatentPrefixLock` (инъекция префикса в сэмплер) + `WanLatentPrefixRestore` (принудительное восстановление префикса и **Latent AdaIN** выравнивание среднего $\mu$ и дисперсии $\sigma$ каналов латентов ядра по префиксу для исключения выцветания/раздувания контраста). Отрезка хвостового латента (`cut_tail_latent`) производится строго из нормализованного `latent_for_decode`.
   - **LTX-Video (`ltx`):** использует связку нативных нод `LTXVAddLatentGuide` (инъекция направляющего кадра с `noise_mask = 0.0` и RoPE-координатами) + `LTXVCropGuides` (отсечение guide-токенов) + `WanLatentPrefixRestore` (Latent AdaIN), полностью устраняя рестарт генерации каждого блока с белого шума.

---

## 5. Дополнительные исследовательские материалы
- Подробный математический анализ латентов, double-scaling ловушки и фазовых искажений Causal VAE: [`docs/MEDIA_MCP_KNOWLEDGE_BASE.md`](MEDIA_MCP_KNOWLEDGE_BASE.md).
- Анализ алгоритмов перекрытия (FreeNoise, AdaIN, Latent Fusing): [`docs/VIDEO_BLOCK_CHAINING_RESEARCH.md`](VIDEO_BLOCK_CHAINING_RESEARCH.md).
