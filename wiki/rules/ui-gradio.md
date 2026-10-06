# Правила: Gradio GUI

## [GRADIO-001] Ошибка 502 за прокси/VPN — фикс no_proxy
На Windows с системным прокси `launch()` падает:
`http://127.0.0.1:7860/gradio_api/startup-events failed (code 502)` —
запрос к localhost уходит в прокси. Фикс в самом верху app.py, до импорта gradio:
```python
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
```

## [GRADIO-002] Порты
7860 — podcast-tts, 7861 — voice-assistant. Не переиспользовать чужой порт:
приложения могут работать одновременно.

## [GRADIO-003] Микрофон в браузере
`http://127.0.0.1` — secure context, микрофон работает без HTTPS.
`gr.Audio(sources=["microphone"], type="filepath")` + событие `stop_recording`.

## [GRADIO-004] Долгие операции
`gr.Progress()` в сигнатуре обработчика + текст «первый раз скачается N МБ» —
иначе пользователь думает, что всё зависло. Исключение — вопрос ассистенту (`respond`): прогресс
только на плеере (`show_progress_on=[reply]`, в старых Gradio — `show_progress="minimal"`), а ход
работы виден в «Диалоге» ([GRADIO-008]).

## [GRADIO-005] Автосохранение настроек
`settings.json` рядом с app.py: `load_settings()` мёржит поверх DEFAULT_SETTINGS,
`.change(autosave, inputs=cfg_inputs)` на каждом поле, значения при старте — из файла.
Файл в .gitignore (может содержать API-ключ LiteLLM — предупреждать в README).
При добавлении полей — не забыть `SETTING_FIELDS` (порядок = порядок inputs; в v4 добавлены
`system_prompt`, `max_tokens`, `web_search`). Миграции — по `settings_rev` в `load_settings()`.
Баг 2026-10-05: галочка wake word не входила в `SETTING_FIELDS` и не сохранялась —
теперь там же `wake_enabled` и `wake_device`; при старте с галочкой слушатель запускается сам.

## [GRADIO-006] Серверные события в UI — gr.Timer
Фоновые потоки (wake word) не могут сами обновить браузер. Состояние лежит в общем dict
(`WAKE_STATE`), диалог — со счётчиком версии `HISTORY_VER`; `gr.Timer(0.5).tick(poll_ui,
inputs=[gr.State], outputs=[...], show_progress="hidden")` отдаёт только изменившееся,
остальное — `gr.skip()`. Без `show_progress="hidden"` компоненты мигают каждые полсекунды.
Ответ wake word НЕ отдавать в `gr.Audio(autoplay=True)`: он уже звучит из динамиков через
sounddevice — будет двойное воспроизведение. И наоборот: пока звучит ответ из браузера,
wake word заглушён (`MUTE_UNTIL`), иначе ассистент услышит себя.

## [GRADIO-007] Playwright-тесты GUI
`page.goto(..., wait_until="load")` + ожидание нужного элемента. `networkidle` не наступает
никогда: таймер опрашивает сервер каждые 0,5 с.

## [GRADIO-008] Ответ «вживую» в «Диалоге»
`respond()` блокирует обработчик до конца ответа, поэтому текст по мере генерации идёт не через
yield, а через общий `LIVE` (вопрос, частичный ответ, статус «🌐 Ищу в интернете…») — его рисует
`render_log()` по таймеру `poll_ui`, как и ответы wake word. Ошибка дописывается в лог с ⚠️,
а `gr.skip()` в поле вопроса сохраняет набранный текст. Пресеты промпта — `button.click(lambda p=preset: p,
outputs=[system_prompt])`, сохранение — обычным `.change(autosave)`.
Playwright: `tests/test_gui_e2e_v4.py` + `tests/gui_server.py` (17 проверок, скриншоты).
