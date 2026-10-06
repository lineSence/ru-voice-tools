# AGENTS.md — L0 Bootstrap (SEK)

Ты — агент-разработчик репозитория `ru-voice-tools`. Отвечай и пиши документацию по-русски.

## Проект в двух строках
Два локальных голосовых приложения на Python: `podcast-tts/` (Silero v5, CPU, порт 7860) и `voice-assistant/`
(Vosk wake word → Whisper/Vosk → LLM → Silero, порт 7861; Whisper и LLM считает KoboldCpp на видеокарте через Vulkan,
фолбэк — CPU). GUI — Gradio, лаунчеры — bat/sh с автоустановкой в `.venv`.

## Триггеры загрузки (что читать)
| Задача про... | Загрузи |
|---|---|
| Vosk, микрофон, распознавание, STT | `wiki/rules/stt-vosk.md` |
| Whisper, английские слова в речи, промпт распознавания | `wiki/rules/stt-whisper.md` |
| KoboldCpp, Vulkan, видеокарта/GPU, движок, порт 5011 | `wiki/rules/engine-kobold.md` |
| Silero, TTS, голоса, озвучка | `wiki/rules/tts-silero.md` |
| LLM, GGUF, Qwen, Gemma, LiteLLM | `wiki/rules/llm-local.md` |
| Gradio, GUI, ошибка 502, порты, прокси | `wiki/rules/ui-gradio.md` |
| Лаунчеры, установка, релиз, zip | `wiki/rules/packaging.md` + `wiki/workflows/release.md` |
| Архитектура, пайплайн | `wiki/architecture/overview.md` |
| Модели, размеры, лицензии, ссылки | `wiki/references/models.md` |
| Ничего не подошло | `wiki/_routing.md`, затем `wiki/_index.md` |

## Critical Rules (core, не нарушать)
- [CORE-001] `AGENTS.md` не редактируется автономно — только с явного разрешения владельца.
- [CORE-002] Новое знание → сначала `memory/inbox.md` (draft, ≤3 строк). Promote в `wiki/` — после повтора/подтверждения.
- [CORE-003] Перед добавлением правила — `grep` по `wiki/rules/`; дубли запрещены. Каждому правилу — ID-тег `[XXX-NNN]`.
- [CORE-004] `memory/inbox.md`: лимит 20 записей; записи старше 30 дней без подтверждения удаляются.
- [CORE-005] Пользовательские машины — Windows с AMD (без CUDA). Видеокарта — только через Vulkan (KoboldCpp nocuda);
  у всего, что шипится, обязан быть рабочий CPU-фолбэк. (Изменено по явному решению владельца 2026-10-06:
  «забудь про CPU-only» — см. CORE-001.)