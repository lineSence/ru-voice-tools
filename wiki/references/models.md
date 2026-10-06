# Справочник моделей

| Компонент | Модель | Источник | Размер | Лицензия | Заметки |
|---|---|---|---|---|---|
| TTS | Silero v5_ru | `github.com/snakers4/silero-models` | ~140 МБ | MIT | 6 голосов, автоударения, SSML |
| TTS (альт.) | Silero v5_cis_base | форк `Dimitrius174/silero-v5-tts-ru` | ~87 МБ | MIT | 29 русских голосов, в roadmap |
| STT | vosk-model-small-ru-0.22 | `alphacephei.com/vosk/models` | 45 МБ | Apache-2.0 | команды/вопросы |
| STT (альт.) | vosk-model-ru-0.42 | там же | ~1,5 ГБ | Apache-2.0 | точнее, медленнее |
| STT (вопрос) | Whisper large-v3-turbo q5_0 | `ggerganov/whisper.cpp` (HF) | 574 МБ | MIT | дефолт: понимает английские слова в русской речи |
| STT (альт.) | Whisper large-v3-turbo q8_0 / f16 | там же | 874 МБ / 1,6 ГБ | MIT | чуть точнее, тяжелее |
| STT (альт.) | Whisper small q8_0 | там же | 264 МБ | MIT | быстро на CPU, но «MIDI» → «меди» |
| LLM | Qwen3-4B-Instruct-2507 Q4_K_M | `bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF` | ~2,5 ГБ | Apache-2.0 | дефолт ассистента |
| LLM | Gemma-3-12B-it Q4_K_M | `bartowski/google_gemma-3-12b-it-GGUF` | ~7,5 ГБ | Gemma Terms of Use | умнее; в 8 ГБ VRAM целиком не влезает (часть слоёв на CPU) |
| LLM (облако) | любая за LiteLLM | endpoint пользователя | — | — | OpenAI-совместимый API |
| Движок | KoboldCpp 1.122.1 (nocuda: Vulkan + CPU) | `github.com/LostRuins/koboldcpp` releases | ~120 МБ | AGPL-3.0 | отдельный процесс, не линкуется с кодом; sha256 зашит в app.py |
| TTS-текст | num2words | PyPI | <1 МБ | LGPL-2.1 | числа → слова перед Silero |

## Проверка ссылок перед релизом
`curl -sI -L <url>` → 200. Официальный репозиторий Qwen GGUF отдаёт 401 — см. [LLM-001].