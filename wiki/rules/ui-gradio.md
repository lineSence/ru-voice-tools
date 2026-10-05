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
иначе пользователь думает, что всё зависло.