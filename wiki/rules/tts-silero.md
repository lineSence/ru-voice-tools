# Правила: Silero TTS

## [SILERO-001] API v5 — методы модели, а не hub-утилиты
`torch.hub.load(..., speaker="v5_ru")` возвращает **2 значения** `(model, example_text)`,
а не 5, как у старых моделей. Синтез — методом упакованной модели:
```python
model, _ = torch.hub.load(repo_or_dir="snakers4/silero-models", model="silero_tts",
                          language="ru", speaker="v5_ru", verbose=False, trust_repo=True)
audio = model.apply_tts(text=..., speaker="aidar", sample_rate=48000)
```

## [SILERO-002] scipy — обязательная зависимость
torch.package модели v5 импортирует `scipy.signal` при загрузке. Без scipy —
`ModuleNotFoundError` на этапе `load_pickle`.

## [SILERO-003] torch ≥ 2.6: trust_repo=True
Свежий torch интерактивно спрашивает доверие к репозиторию — это вешает bat-лаунчер
(EOFError/ожидание ввода). Передавать `trust_repo=True`, но только если параметр есть
(совместимость — через `inspect.signature(torch.hub.load)`).

## [SILERO-004] Практика синтеза
- Голоса v5_ru: `aidar`, `eugene` (м), `baya`, `kseniya`, `xenia` (ж), `random`.
- Ударение вручную: `+` перед гласной (`молок+о`). Автоударение включено по умолчанию.
- Длинные тексты резать по предложениям (≤350 символов) — иначе артефакты.
- Скорость на CPU: RTF ≈ 0.05–0.15, GPU не нужен.