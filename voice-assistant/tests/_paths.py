"""Общие пути тестов: где app.py, движок KoboldCpp и тестовые модели (переопределяются
переменными окружения). Импортировать до `import app`."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_up = os.path.dirname(HERE)
_arg = sys.argv[1] if len(sys.argv) > 1 and os.path.isfile(os.path.join(sys.argv[1], "app.py")) else None
APP_DIR = (os.environ.get("APP_DIR") or _arg
           or (_up if os.path.isfile(os.path.join(_up, "app.py")) else "/data/gh-repo/voice-assistant"))
ENGINE_DIR = os.environ.get("TEST_ENGINE_DIR", "/tmp/engine")       # распакованный KoboldCpp
MODELS_DIR = os.environ.get("TEST_MODELS_DIR", "/tmp/kcpp")
QWEN05 = os.path.join(MODELS_DIR, "qwen05.gguf")                   # Qwen2.5-0.5B-Instruct Q4_K_M
QWEN4B = os.path.join(MODELS_DIR, "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
WORK = os.environ.get("TEST_WORK_DIR", "/tmp/wk")                  # настройки GUI, скриншоты
os.makedirs(WORK, exist_ok=True)
sys.path.insert(0, APP_DIR)


def use_test_engine(app):
    """Движок — из общей папки тестов, а не engine/ рядом с app.py."""
    app.ENGINE_DIR = ENGINE_DIR
    app.ENGINE_LOG = os.path.join(ENGINE_DIR, "koboldcpp.log")
