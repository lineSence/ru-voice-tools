"""GUI для E2E-теста: настоящий app.build_ui() на 127.0.0.1:7861, модель qwen05 на CPU,
настройки во временном файле (рабочая папка программы не трогается)."""
import json, os
from _paths import QWEN05, WORK, use_test_engine  # noqa: E402
import app  # noqa: E402

use_test_engine(app)
app.SETTINGS_PATH = os.path.join(WORK, "gui_settings.json")
if os.path.exists(app.SETTINGS_PATH):
    os.remove(app.SETTINGS_PATH)
json.dump(dict(app.DEFAULT_SETTINGS, llm_source=app.SRC_LOCAL, local_gguf_path=QWEN05,
               stt_engine=app.STT_VOSK, compute=app.COMPUTE_CPU, n_ctx=4096, max_tokens=320,
               voice="baya"), open(app.SETTINGS_PATH, "w", encoding="utf-8"), ensure_ascii=False)
app._wake["cfg"] = app.load_settings()
import atexit, signal  # noqa: E402  как в app.py: kill -> KoboldCpp гасится вместе с сервером


def _exit(*_):
    raise SystemExit(0)


atexit.register(app._stop_engine_at_exit)
signal.signal(signal.SIGTERM, _exit)
demo = app.build_ui()
app.engine_bg()
demo.launch(server_name="127.0.0.1", server_port=7861, inbrowser=False)
