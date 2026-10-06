# Правила: локальные LLM и LiteLLM

## [LLM-001] Модели не хардкодим; GGUF — с зеркал bartowski
UI принимает любую GGUF: HF repo + filename или локальный путь (см. `load_llm(cfg)`).
Дефолт в настройках: `bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF` (`...-Q4_K_M.gguf`).
Официальный `Qwen/...-GGUF` отдаёт 401 на прямое скачивание — зеркала bartowski надёжнее.
12B-класс: `bartowski/google_gemma-3-12b-it-GGUF` (проверено HEAD 200).

## [LLM-002] Системный промпт — «без markdown»
Ответ озвучивается TTS: списки, эмодзи, `**`, ссылки читаются вслух как мусор.
Промпт требует 1–3 предложения разговорным языком; `clean_for_tts()` дополнительно
вычищает разметку перед синтезом.

## [LLM-003] Локальная GGUF — через KoboldCpp, на видеокарте (Vulkan)
llama-cpp-python удалён (на Windows не собрать с Vulkan). GGUF запускает KoboldCpp, см. `engine-kobold.md`:
`--gpulayers -1` (autofit по VRAM), `--contextsize` = `n_ctx` (4096). Qwen3-4B Q4_K_M + Whisper turbo
≈ 5 ГБ VRAM — на RX 580 8 ГБ влезают целиком; Gemma-3-12B — частично (autofit сам оставит слои на CPU).
Запрос — `POST /v1/chat/completions` (`max_tokens` 220, temperature 0.6). Модель по-прежнему качается
`hf_hub_download` в кэш HF — уже скачанные пользователем GGUF переиспользуются.
CPU-режим («Только процессор» → `--usecpu`) остаётся рабочим фолбэком: Qwen3-4B ~20–25 с на слабой VPS.

## [LLM-004] LiteLLM-режим
Любой OpenAI-совместимый endpoint: `POST {base}/chat/completions`,
Bearer-ключ опционален, таймаут 180 сек. Тестировать мок-сервером на `http.server`.

## [LLM-005] История диалога
Хранить последние 6 пар реплик (`HISTORY_KEEP`) — иначе контекст 4096 переполняется,
а CPU-инференс деградирует по скорости.

## [LLM-006] Рассуждения «думающих» моделей
`strip_think()` вырезает `<think>…</think>` (и хвост до `</think>`, если открывающий тег съел шаблон) —
иначе Silero зачитывает рассуждения вслух. Применяется и к LiteLLM.
