# Workflow: тестирование и релиз

## Чек-лист перед отдачей версии

1. **Чистый venv** в песочнице: `pip install torch (cpu index)` + `-r requirements.txt`.
2. **Свежий $HOME** — проверить, что все модели скачиваются кодом, а не найдены в кэше.
3. **E2E podcast-tts**: 3 реплики × 2 голоса → WAV собрался, длительность адекватна.
4. **E2E voice-assistant**: Silero-вопрос → Vosk транскрипция (точное совпадение) →
   локальная LLM → Silero-ответ. Плюс мок-сервер для LiteLLM-режима.
5. **Движок KoboldCpp** (`test_engine.py`): реальное скачивание с GitHub + sha256 + распаковка, LLM и Whisper отвечают,
   Vulkan без видеокарты → CPU с предупреждением, перезапуск при смене настроек, осиротевший порт 5011, kill -9.
6. **Whisper**: фразы с английскими словами («MIDI-синтезатор», «Python на Windows», «USB микрофон») распознаются
   латиницей; wake word режим: Vosk ловит слово, Whisper — вопрос; Whisper упал → Vosk + предупреждение.
7. **Озвучка текста** (`test_tts_text.py`): латиница и числа превращаются в русские слова.
8. **GUI**: поднять app.py, `curl` → HTTP 200 / Playwright-прогон, погасить; проверить, что не осталось процессов koboldcpp.
9. **Zip** без `.venv`, `__pycache__`, `.installed*`, `engine/`, `settings.json`.

## Релиз в GitHub
- Файлы пушатся одним коммитом через GitHub MCP `push_files`.
- Бинарники (demo.wav и т.п.) в git не кладём — только в Releases при необходимости.

## Известные открытые пункты
См. `memory/inbox.md` (Vulkan на реальной RX 580, Windows-код движка, Gemma-12B end-to-end, wake word).