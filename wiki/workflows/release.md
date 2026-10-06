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
9. **Zip** без `.venv`, `__pycache__`, `.installed*`, `engine/`, `settings.json`, `tests/`.

## Тесты v4 (`voice-assistant/tests/`)
Нужны: venv с requirements + `playwright`, `pyflakes`; KoboldCpp, распакованный в `/tmp/engine`
(`TEST_ENGINE_DIR`); тестовая `qwen05.gguf` (Qwen2.5-0.5B-Instruct Q4_K_M) и для маршрутизации
Qwen3-4B-Instruct-2507 Q4_K_M в `/tmp/kcpp` (`TEST_MODELS_DIR`). Путь к app.py — аргумент или папка выше.

| Файл | Что проверяет | Время (2 vCPU) |
|---|---|---|
| `test_units_v4.py` | 57 юнит-проверок: промпты, дата, fit_messages, sniffer, think, поиск с поддельным ddgs, страницы, озвучка | 1 мин |
| `test_integration_v4.py` | LiteLLM-заглушка (протокол поиска, JSON, 500, лимит), настоящий KoboldCpp: поток, abort, принудительный поиск, Silero, перезапуск | 4 мин |
| `test_wake_stream_v4.py` | поддельный sounddevice: wake word → вопрос → первый звук до конца генерации | 1,5 мин |
| `test_search_live.py` | живой поиск (сеть): погода, версия Python, курс, новости | 20 с |
| `test_routing_qwen3_4b.py`, `test_ask_mode_qwen3_4b.py` | решения «искать/отвечать» и длина ответа на Qwen3-4B | 10 мин |
| `test_gui_e2e_v4.py` (+ `gui_server.py`) | Playwright: элементы, пресеты, автосохранение, ответ вживую, поиск, озвучка | 3 мин |

## Релиз в GitHub
- Файлы пушатся одним коммитом через GitHub MCP `push_files`.
- Бинарники (demo.wav и т.п.) в git не кладём — только в Releases при необходимости.

## Известные открытые пункты
См. `memory/inbox.md` (Vulkan на реальной RX 580, Windows-код движка, Gemma-12B end-to-end, wake word).
