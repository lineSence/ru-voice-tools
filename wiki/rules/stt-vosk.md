# Правила: Vosk STT

## [VOSK-001] Не использовать vosk.Model(lang="ru") — свой загрузчик
Штатный загрузчик Vosk имеет read timeout 10 секунд и обрывается на медленных
соединениях (реальный кейс: `Read timed out (read timeout=10)` у пользователя за прокси).
Реализация — `vosk_model_path()` в `voice-assistant/app.py`: таймауты (15, 180),
3 попытки, докачка через Range, восстановление после битого zip (BadZipFile → с нуля).

Ручной fallback: скачать в браузере
`https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip`
и распаковать в `~/.cache/vosk/vosk-model-small-ru-0.22/`.

## [VOSK-002] Вход: 16 кГц mono int16
Браузерный микрофон отдаёт 44.1/48 кГц — обязателен ресемплинг:
`scipy.signal.resample_poly(data, 16000//g, sr//g)`, затем `* 32767 → int16`.
Проверено: русская речь, синтезированная Silero 48 кГц, распознаётся без ошибок.

## [VOSK-003] Точность
Малая модель (45 МБ) достаточна для коротких команд и вопросов.
Запасной вариант — `vosk-model-ru-0.42` (~1,5 ГБ): поменять `VOSK_NAME`/`VOSK_URL`.