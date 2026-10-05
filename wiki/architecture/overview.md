# Архитектура

## Общее

Два независимых приложения, общая инфраструктура: Python 3.10–3.12, `.venv`, CPU-only PyTorch, Gradio GUI.
Пользовательская целевая машина — Windows с AMD GPU (без CUDA), поэтому всё считается на процессоре.

## podcast-tts (порт 7860)

```text
Сценарий "Имя: реплика"
  → parse_script (имя → голос Silero; длинные реплики режутся ≤350 симв.)
  → Silero v5_ru model.apply_tts (по одному куску на реплику)
  → concat с паузами (numpy) → WAV 48 кГц → gr.Audio
```

## voice-assistant (порт 7861)

```text
Микрофон (браузер, gr.Audio) ─┐
Текстовый ввод ───────────────┼→ user_text
                              ↓
        Vosk transcribe (48 кГц → resample_poly → 16 кГц mono int16)
                              ↓
        LLM: локальная GGUF (llama-cpp-python) ИЛИ LiteLLM (OpenAI-compatible POST)
        system-промпт: коротко, без markdown/эмодзи — ответ озвучивается
                              ↓
        Silero v5_ru → WAV → gr.Audio (autoplay)
        История: gr.State, последние 6 пар реплик
```

## Кэши моделей (вне репозитория)

| Что | Куда качается |
|---|---|
| Silero v5_ru (~140 МБ) | `~/.cache/torch/hub` |
| Vosk small-ru (~45 МБ) | `~/.cache/vosk/vosk-model-small-ru-0.22` |
| GGUF Qwen3-4B / Gemma-3-12B | `~/.cache/huggingface` |

## Конфликты портов
7860 (подкасты) и 7861 (ассистент) — можно держать запущенными одновременно.