# Правила: движок KoboldCpp (LLM + Whisper на видеокарте через Vulkan)

## [KCPP-001] Почему KoboldCpp, а не llama-cpp-python / whisper.cpp
- Владелец: Windows + AMD RX 580, CUDA нет → видеокарта доступна только через Vulkan.
- llama-cpp-python: на Windows нет готовых колёс с Vulkan (сборка — MSVC + Vulkan SDK), пользователю не собрать.
- whisper.cpp: в официальных релизах нет Windows-сборки с Vulkan (только CPU/BLAS/cuBLAS), сторонние — непроверенные.
- KoboldCpp (`LostRuins/koboldcpp`, AGPL-3.0): официальная `koboldcpp-nocuda.exe` с Vulkan внутри; один процесс
  держит и LLM, и Whisper; OpenAI API + `/api/extra/transcribe`. Pip-зависимостей не добавляет
  (запускается отдельной программой, не линкуется).

## [KCPP-002] Версия и целостность
Версия зашита: `KOBOLD_VERSION`, файлы и sha256 — `KOBOLD_ASSETS` (sha из GitHub API релиза, поле `digest`).
Качать только с `github.com/LostRuins/koboldcpp/releases` — у проекта есть фишинговый сайт-клон.
Обновляя версию — обновить sha всех трёх платформ и прогнать `test_engine.py`.

## [KCPP-003] Распаковка `--unpack` один раз
Onefile-PyInstaller при каждом старте распаковывает ~310 МБ во временную папку, а при kill оставляет её мусором.
`koboldcpp-nocuda.exe --unpack engine/koboldcpp-<ver>.tmp` → `koboldcpp-launcher(.exe)` + `_internal/`, затем
переименование в `engine/koboldcpp-<ver>/` и удаление onefile. Старт ~3 с, kill без мусора.
Распаковка не удалась → запускаем onefile как есть.

## [KCPP-004] Флаги запуска
`--port 5011 --host 127.0.0.1 --skiplauncher --quiet --singleinstance`
+ `--model X --contextsize N` и/или `--whispermodel Y` + `--usevulkan [--gpulayers N]` либо `--usecpu`.
- `--gpulayers` −1 (по умолчанию) = autofit, Whisper учитывается. Без `vulkaninfo` VRAM не определяется,
  но autofit всё равно включается.
- `--usevulkan` без видеокарты НЕ падает — молча считает на CPU (проверено в песочнице).
- Whisper в KoboldCpp: `use_gpu=true` при Vulkan-бэкенде, `n_threads=4` зашито.
- Готовность — `GET /api/extra/version` → 200 (HTTP поднимается после загрузки всех моделей).

## [KCPP-005] Завершение и «сироты»
- Штатно — `POST /api/extra/shutdown`: работает только с `--singleinstance` и только с localhost.
- Перед запуском `_free_port()`: на 5011 может висеть KoboldCpp от аварийно закрытого запуска; без этого
  ожидание готовности «видит» старый процесс.
- Windows: Job Object с `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` — KoboldCpp умирает вместе с python.exe, даже
  если консоль закрыли крестиком (иначе висит и держит видеопамять). `CREATE_NO_WINDOW` — без своего окна;
  в общей консоли он переименовал бы её через `title`.
- atexit: shutdown → wait 6 с → kill. Linux/macOS: SIGTERM/SIGHUP → `SystemExit` → atexit (иначе kill или
  закрытый терминал оставляют сироту; проверено: движок гаснет за 5 с). `PR_SET_PDEATHSIG` не годится —
  он привязан к потоку-родителю, а движок стартует из фонового `engine_bg()`.

## [KCPP-006] Когда перезапускать
`engine_light_key(cfg)` — только то, что влияет на запуск: источник/путь LLM, n_ctx, модель Whisper,
GPU/CPU, слои. Подсказка Whisper, голос, wake word — без перезапуска. Радио и выпадашки → фоновый
`engine_bg()`; текстовые поля → лениво при следующем вопросе (иначе перезапуск на каждый символ) +
предупреждение в строке движка. LiteLLM + Vosk → движок гасится (освобождает видеопамять).

## [KCPP-007] Что реально считает — из лога
`using device Vulkan0 (AMD Radeon RX 580 …)` или `ggml_vulkan: 0 = …` → имя видеокарты.
`offloaded N/M layers to GPU` печатается и на CPU — сам по себе не признак видеокарты!
LLM загружена, а Vulkan-строк нет → «Vulkan не нашёл видеокарту — обновите драйвер».
Упал при старте с Vulkan → лог копируется в `koboldcpp-vulkan-fail.log`, перезапуск с `--usecpu` и предупреждение.
Скорость генерации — `GET /api/extra/perf` → `last_eval_speed` (без времени чтения промпта).

## [KCPP-008] Пути с кириллицей (Windows)
C++-часть открывает файлы «узкими» строками → пути к моделям и launcher передаются через `GetShortPathNameW`
(имена 8.3 — латиница). Если 8.3 на диске отключены — путь как есть (может не открыться).

## [KCPP-009] Сеть
Запросы к 127.0.0.1 — через `requests.Session()` с `trust_env=False`: прокси/VPN пользователя не должны
перехватывать localhost. Скачивания (GitHub, HF) — обычный requests, прокси учитывается.

## [KCPP-010] Тесты без видеокарты
`koboldcpp-linux-x64-nocuda` в песочнице: реальное скачивание + sha + unpack, LLM Qwen2.5-0.5B + Whisper,
перезапуск при смене настроек, сирота на порту, kill -9 посреди работы, обёртка, падающая на `--usevulkan`
(фолбэк на CPU), «не запускается вообще». GUI-прогон: OOM-killer дважды убил движок посреди ответа —
перезапуск и повтор запроса прошли сами, пользователь получил ответ. Vulkan на реальном RX 580 и Windows-код (Job Object,
8.3-пути) в песочнице не исполнялись — см. `memory/inbox.md`.
