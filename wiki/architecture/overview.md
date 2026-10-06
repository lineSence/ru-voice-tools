# Архитектура

## Общее

Два независимых приложения, общая инфраструктура: Python 3.10–3.12, `.venv`, CPU-сборка PyTorch (только для Silero),
Gradio GUI. Целевая машина — Windows с AMD GPU без CUDA (у владельца RX 580 8 ГБ): Whisper и LLM ассистента считает
KoboldCpp на видеокарте через Vulkan, при любой проблеме с Vulkan — автоматически на CPU ([CORE-005], [KCPP-007]).

## podcast-tts (порт 7860)

```text
Сценарий "Имя: реплика"
  → parse_script (имя → голос Silero; длинные реплики режутся ≤350 симв.)
  → Silero v5_ru model.apply_tts (по одному куску на реплику)
  → concat с паузами (numpy) → WAV 48 кГц → gr.Audio
```

## voice-assistant (порт 7861)

```text
Микрофон компьютера (sounddevice) ── Vosk small-ru: только поиск wake word, кольцевой буфер 45 с
   └─ сработало → вырезка [начало слова −0,3 с … конец фразы +0,4 с] → Whisper → убрать wake word из текста
Микрофон браузера (gr.Audio) ──→ Whisper (или Vosk, если выбран / Whisper недоступен)
Текстовый ввод ────────────────→ user_text
                              ↓
        think_answer: system = промпт пользователя + TIME_RULE + правило поиска ([LLM-002], [WEB-001]);
        вопрос = "[Сейчас …]\n" + user_text ([LLM-007]); fit_messages под n_ctx ([LLM-005])
                              ↓
        LLM потоком (SSE): KoboldCpp /v1/chat/completions ИЛИ LiteLLM; <think> режется на лету ([LLM-008])
          └─ ответ начался с "ПОИСК: q" (или «найди…») → поток прерван (/api/extra/abort)
             → ddgs: все поисковики параллельно + текст 3 страниц ([WEB-003], [WEB-004])
             → второй проход LLM с результатами (тот же префикс — кэш KoboldCpp)
                              ↓ (куски текста)
        LIVE → «Диалог» в GUI (gr.Timer 0,5 с)      SpeechStream: предложения → clean_for_tts
                                                     (латиница/числа → слова) → Silero (фоновый поток)
                                                     → колонки сразу (wake word) / один WAV (браузер)
        История: HISTORY (последние 6 пар; вопрос с датой + shown + заметка о поиске/лимите)
```

### Движок KoboldCpp (порт 5011, только 127.0.0.1)
- Один процесс держит и LLM (`--model`), и Whisper (`--whispermodel`); запускается в фоне при старте приложения
  и перезапускается при смене модели/устройства/контекста ([KCPP-006]).
- `--usevulkan` (видеокарта) или `--usecpu`; Vulkan упал → лог в `engine/koboldcpp-vulkan-fail.log`, повтор на CPU.
- Режим LiteLLM + Vosk движок не запускает (не нужен).
- Состояние (`ENGINE_STATE`) показывается в GUI строкой «⚙️ Движок KoboldCpp: …» — устройство, модели, время.

## Кэши моделей (вне репозитория)

| Что | Куда качается |
|---|---|
| Silero v5_ru (~140 МБ) | `~/.cache/torch/hub` |
| Vosk small-ru (~45 МБ) | `~/.cache/vosk/vosk-model-small-ru-0.22` |
| GGUF Qwen3-4B / Gemma-3-12B | `~/.cache/huggingface` |
| Whisper ggml (large-v3-turbo q5_0, ~574 МБ) | `~/.cache/huggingface` (репозиторий `ggerganov/whisper.cpp`) |
| KoboldCpp 1.122.1 (~120 МБ + распаковка) | `voice-assistant/engine/` рядом с app.py, лог `engine/koboldcpp.log` |

## Конфликты портов
7860 (подкасты), 7861 (ассистент) и 5011 (KoboldCpp ассистента) — можно держать запущенными одновременно.
Осиротевший KoboldCpp на 5011 приложение гасит само через `/api/extra/shutdown` ([KCPP-005]).
