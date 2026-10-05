# Справочник моделей

| Компонент | Модель | Источник | Размер | Лицензия | Заметки |
|---|---|---|---|---|---|
| TTS | Silero v5_ru | `github.com/snakers4/silero-models` | ~140 МБ | MIT | 6 голосов, автоударения, SSML |
| TTS (альт.) | Silero v5_cis_base | форк `Dimitrius174/silero-v5-tts-ru` | ~87 МБ | MIT | 29 русских голосов, в roadmap |
| STT | vosk-model-small-ru-0.22 | `alphacephei.com/vosk/models` | 45 МБ | Apache-2.0 | команды/вопросы |
| STT (альт.) | vosk-model-ru-0.42 | там же | ~1,5 ГБ | Apache-2.0 | точнее, медленнее |
| LLM | Qwen3-4B-Instruct-2507 Q4_K_M | `bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF` | ~2,5 ГБ | Apache-2.0 | дефолт ассистента |
| LLM | Gemma-3-12B-it Q4_K_M | `bartowski/google_gemma-3-12b-it-GGUF` | ~7,5 ГБ | Gemma Terms of Use | умнее, медленно на CPU |
| LLM (облако) | любая за LiteLLM | endpoint пользователя | — | — | OpenAI-совместимый API |

## Проверка ссылок перед релизом
`curl -sI -L <url>` → 200. Официальный репозиторий Qwen GGUF отдаёт 401 — см. [LLM-001].