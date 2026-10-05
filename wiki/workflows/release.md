# Workflow: тестирование и релиз

## Чек-лист перед отдачей версии

1. **Чистый venv** в песочнице: `pip install torch (cpu index)` + `-r requirements.txt`.
2. **Свежий $HOME** — проверить, что все модели скачиваются кодом, а не найдены в кэше.
3. **E2E podcast-tts**: 3 реплики × 2 голоса → WAV собрался, длительность адекватна.
4. **E2E voice-assistant**: Silero-вопрос → Vosk транскрипция (точное совпадение) →
   локальная LLM → Silero-ответ. Плюс мок-сервер для LiteLLM-режима.
5. **GUI**: поднять app.py, `curl` → HTTP 200, погасить.
6. **Zip** без `.venv`, `__pycache__`, `.installed`.

## Релиз в GitHub
- Файлы пушатся одним коммитом через GitHub MCP `push_files`.
- Бинарники (demo.wav и т.п.) в git не кладём — только в Releases при необходимости.

## Известные открытые пункты
См. `memory/inbox.md` (Gemma-12B end-to-end, wake word, большая Vosk-модель).