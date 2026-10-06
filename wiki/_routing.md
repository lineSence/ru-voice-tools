# _routing.md — L1: таблица маршрутизации

Агент читает этот файл, если задача не покрыта триггерами из `AGENTS.md`.

```yaml
triggers:
  - id: stt
    keywords: [vosk, stt, распознавание, микрофон, аудио-вход, 16000, wake word, sounddevice]
    load:
      - wiki/rules/stt-vosk.md
      - wiki/references/models.md
  - id: whisper
    keywords: [whisper, английские слова, ggml, turbo, transcribe, промпт распознавания, галлюцинации]
    load:
      - wiki/rules/stt-whisper.md
      - wiki/rules/engine-kobold.md
  - id: engine
    keywords: [koboldcpp, kobold, vulkan, gpu, видеокарта, rx 580, 5011, движок, gpulayers]
    load:
      - wiki/rules/engine-kobold.md
      - wiki/references/models.md
  - id: tts
    keywords: [silero, tts, синтез, озвучка, голос, aidar, kseniya, ударение, speechstream, по предложениям]
    load:
      - wiki/rules/tts-silero.md
      - wiki/references/models.md
  - id: llm
    keywords: [llm, gguf, qwen, gemma, litellm, llama-cpp, ollama, модель, промпт, системный промпт,
               think, длина ответа, max_tokens, контекст, n_ctx, stream, поток, дата, время]
    load:
      - wiki/rules/llm-local.md
      - wiki/references/models.md
  - id: web
    keywords: [поиск, интернет, ddgs, duckduckgo, yahoo, google, поисковик, "ПОИСК:", новости, погода,
               курс, страницы, выдержки, прокси поиска]
    load:
      - wiki/rules/web-search.md
      - wiki/rules/llm-local.md
  - id: ui
    keywords: [gradio, gui, интерфейс, 7860, 7861, 502, прокси, vpn, localhost]
    load:
      - wiki/rules/ui-gradio.md
  - id: packaging
    keywords: [launch, bat, sh, venv, pip, torch, релиз, zip, установка, winget]
    load:
      - wiki/rules/packaging.md
      - wiki/workflows/release.md
  - id: arch
    keywords: [архитектура, пайплайн, структура, порты, кэш, engine/]
    load:
      - wiki/architecture/overview.md
  - id: sek
    keywords: [inbox, promote, правило, урок, memory]
    load:
      - memory/inbox.md
```
