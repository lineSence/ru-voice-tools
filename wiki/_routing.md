# _routing.md — L1: таблица маршрутизации

Агент читает этот файл, если задача не покрыта триггерами из `AGENTS.md`.

```yaml
triggers:
  - id: stt
    keywords: [vosk, stt, распознавание, микрофон, whisper, аудио-вход, 16000]
    load:
      - wiki/rules/stt-vosk.md
      - wiki/references/models.md
  - id: tts
    keywords: [silero, tts, синтез, озвучка, голос, aidar, kseniya, ударение]
    load:
      - wiki/rules/tts-silero.md
      - wiki/references/models.md
  - id: llm
    keywords: [llm, gguf, qwen, gemma, litellm, llama-cpp, ollama, модель, промпт]
    load:
      - wiki/rules/llm-local.md
      - wiki/references/models.md
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
    keywords: [архитектура, пайплайн, структура, порты, кэш]
    load:
      - wiki/architecture/overview.md
  - id: sek
    keywords: [inbox, promote, правило, урок, memory]
    load:
      - memory/inbox.md
```