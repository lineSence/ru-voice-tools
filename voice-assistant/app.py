#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RU Voice Assistant — локальный голосовой ассистент на русском.

Распознавание: Whisper large-v3-turbo — понимает английские слова внутри русской
               речи («MIDI», «Python», «USB»); либо Vosk — быстрее, но только русский словарь
LLM          : любая GGUF (Hugging Face repo+file или локальный .gguf),
               либо OpenAI-совместимый endpoint (LiteLLM-прокси)
Движок       : KoboldCpp (скачивается сам) — LLM и Whisper на видеокарте через Vulkan
               (AMD/NVIDIA/Intel, CUDA не нужна) или на процессоре
TTS          : Silero v5; английские слова и числа переводятся в русское чтение;
               ответ звучит по предложениям, пока модель ещё дописывает остальное
Поиск        : Google, DuckDuckGo, Yahoo, Brave… (пакет ddgs, без ключей) — модель сама решает,
               когда нужен интернет, или ищет по просьбе «найди…»
Промпт       : системный промпт и длина ответа редактируются в GUI
Wake word    : Vosk слушает микрофон компьютера и ищет слово-триггер, сам вопрос
               дораспознаёт Whisper
Компьютер    : «включи Бегущий по лезвию» — фильм в VLC на полный экран, «пауза», «громче»,
               «запусти телеграм», «заблокируй компьютер» (pc_control.py, без LLM)

Настройки автосохраняются в settings.json.
"""

import os

# Фикс для Windows с прокси/VPN: localhost не должен идти через прокси,
# иначе Gradio падает с "startup-events failed (code 502)".
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

import atexit
import base64
import collections
import difflib
import hashlib
import inspect
import io
import itertools
import json
import math
import platform
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait as futures_wait
from html import unescape as html_unescape

import numpy as np
import soundfile as sf
from scipy import signal
import torch
import gradio as gr
import requests
import vosk

import pc_control

SAMPLE_RATE_TTS = 48000
SAMPLE_RATE_STT = 16000
VOICES = ["aidar", "baya", "kseniya", "xenia", "eugene", "random"]

SRC_HF = "Hugging Face GGUF"
SRC_LOCAL = "Локальный файл .gguf"
SRC_LITELLM = "LiteLLM-прокси (облако)"
STT_WHISPER = "Whisper — понимает английские слова"
STT_VOSK = "Vosk — быстрее, только русские слова"
COMPUTE_GPU = "Видеокарта (Vulkan)"
COMPUTE_CPU = "Только процессор"
DEFAULT_DEVICE = "Системный по умолчанию"
WEB_AUTO = "Автоматически — модель решает сама"
WEB_ASK = "Только когда прошу («найди», «поищи в интернете»)"
WEB_OFF = "Выключен"
WEB_MODES = [WEB_AUTO, WEB_ASK, WEB_OFF]

PROMPT_SHORT = (
    "Ты — дружелюбный русскоязычный голосовой помощник. "
    "Отвечай коротко: одно-три предложения, простым разговорным языком. "
    "Никаких списков, markdown, эмодзи, скобок и ссылок — ответ будет зачитан вслух. "
    "Если вопрос непонятен, вежливо переспроси."
)
PROMPT_BALANCED = (
    "Ты — дружелюбный русскоязычный голосовой помощник, твои ответы зачитываются вслух. "
    "Подбирай длину ответа под вопрос: на простой вопрос отвечай коротко, в одно-три "
    "предложения; если просят объяснить, рассказать подробно, дать инструкцию или совет "
    "или вопрос сложный — отвечай развёрнуто, по шагам и с пояснениями. "
    "Пиши обычным разговорным текстом без markdown, таблиц, эмодзи и ссылок; вместо "
    "списков связывай шаги словами «сначала», «затем», «в конце». "
    "Если вопрос непонятен, вежливо переспроси."
)
PROMPT_DETAILED = (
    "Ты — внимательный русскоязычный помощник-эксперт, твои ответы зачитываются вслух. "
    "Отвечай подробно и обстоятельно: объясняй причины, приводи примеры, давай пошаговые "
    "инструкции и практические советы, предупреждай о типичных ошибках. "
    "Пиши связным разговорным текстом без markdown, таблиц, эмодзи и ссылок; шаги "
    "называй словами «во-первых», «во-вторых», «затем». "
    "Если вопрос неоднозначен, коротко уточни, что имеется в виду."
)
PROMPT_PRESETS = {"Коротко": PROMPT_SHORT, "Сбалансированно (стандарт)": PROMPT_BALANCED,
                  "Подробно": PROMPT_DETAILED}
DEFAULT_SYSTEM_PROMPT = PROMPT_BALANCED
DEFAULT_MAX_TOKENS = 1024
SETTINGS_REV = 2  # 2: n_ctx 4096 -> 8192, системный промпт, длина ответа, поиск

WHISPER_REPO = "ggerganov/whisper.cpp"
WHISPER_MODELS = [
    "ggml-large-v3-turbo-q5_0.bin",  # 574 МБ — лучший выбор для видеокарты
    "ggml-large-v3-turbo-q8_0.bin",  # 874 МБ
    "ggml-large-v3-turbo.bin",       # 1.6 ГБ
    "ggml-medium-q5_0.bin",          # 539 МБ
    "ggml-small-q8_0.bin",           # 264 МБ — быстро на процессоре, но хуже с английскими словами
]
DEFAULT_WHISPER_PROMPT = ("Вопросы голосовому ассистенту на русском, с английскими терминами: "
                          "MIDI, USB, Python, Windows, GitHub, Docker, YouTube, Bluetooth, Wi-Fi, VST.")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(APP_DIR, "settings.json")

DEFAULT_SETTINGS = {
    "voice": "kseniya",
    "llm_source": SRC_HF,
    "hf_repo": "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
    "hf_file": "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "local_gguf_path": "",
    "litellm_base": "http://127.0.0.1:4000/v1",
    "litellm_key": "",
    "litellm_model": "",
    "n_ctx": 8192,
    "wake_enabled": False,
    "wake_word": "ассистент",
    "wake_device": DEFAULT_DEVICE,
    "stt_engine": STT_WHISPER,
    "compute": COMPUTE_GPU,
    "whisper_model": WHISPER_MODELS[0],
    "whisper_prompt": DEFAULT_WHISPER_PROMPT,
    "gpu_layers": -1,
    "kobold_path": "",  # необязательно: свой koboldcpp(.exe) вместо скачиваемого
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "max_tokens": DEFAULT_MAX_TOKENS,
    "web_search": WEB_AUTO,
    "pc_control": True,   # голосовые команды компьютеру (pc_control.py)
    "movies_dir": "",
    "vlc_path": "",       # пусто — найти VLC самому (реестр, Program Files, PATH)
    "pc_aliases": "",     # «как говорю = что открыть», по строке
    "settings_rev": SETTINGS_REV,
}

# Добавляется к системному промпту пользователя автоматически (не редактируется в GUI)
TIME_RULE = ("В начале сообщений пользователя в квадратных скобках указаны текущие дата и время — "
             "учитывай их, когда это важно, но не упоминай без надобности.")
SEARCH_RULE_AUTO = (
    "У тебя есть поиск в интернете. Если для точного ответа нужны свежие или проверяемые "
    "сведения — новости, погода, курсы валют, цены, расписания, результаты матчей, недавние "
    "события, актуальные версии программ, факты о конкретных людях, компаниях и товарах — "
    "или пользователь просит поискать, не отвечай сам, а напиши ровно одну строку:\n"
    "ПОИСК: короткий поисковый запрос\n"
    "и больше ничего. Если поиск не нужен (беседа, объяснение общих понятий, советы, "
    "расчёты, творческие задачи), отвечай сразу."
)
SEARCH_RULE_ASK = (
    "У тебя есть поиск в интернете, но используй его, только если пользователь прямо просит "
    "найти или поискать что-то в интернете. Тогда напиши ровно одну строку:\n"
    "ПОИСК: короткий поисковый запрос\n"
    "и больше ничего. В остальных случаях отвечай сразу сам. Если для точного ответа нужны "
    "свежие сведения (погода, курсы, новости, цены), не выдумывай их: скажи, что точных "
    "данных у тебя нет, и предложи попросить тебя поискать в интернете."
)
SEARCH_RESULTS_MSG = (
    "Результаты поиска в интернете по запросу «{query}»:\n\n{results}\n\n"
    "Ответь на мой вопрос «{question}», опираясь на эти результаты. Если ответа в них нет, "
    "так и скажи и ответь по своим знаниям, предупредив, что сведения могут быть неточными. "
    "Адреса сайтов не зачитывай. Поиск больше не нужен — отвечай сразу."
)
SEARCH_FAILED_MSG = (
    "Поиск в интернете не удался ({error}). Ответь на мой вопрос «{question}» по своим "
    "знаниям и коротко предупреди, что сведения могут быть устаревшими. Поиск больше не нужен."
)

HISTORY_KEEP = 6  # пар реплик (меньше, если не влезают в контекст модели)

_stt = {"model": None}
_tts = {"model": None}

HISTORY = []
HISTORY_VER = [0]  # растёт при каждом изменении диалога — таймер GUI видит новое
HISTORY_LOCK = threading.Lock()
LIVE = {"user": "", "text": "", "status": ""}  # ответ, который сейчас пишется (виден в «Диалоге»)
PIPE_LOCK = threading.Lock()


# ----------------------------- Настройки ------------------------------------

def _int(v, default):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def load_settings():
    s, raw = dict(DEFAULT_SETTINGS), {}
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            raw = loaded
            s.update(raw)
    except Exception:
        pass
    if raw and _int(raw.get("settings_rev"), 1) < 2 and _int(raw.get("n_ctx"), 4096) == 4096:
        s["n_ctx"] = 8192  # старый стандарт 4096: длинный ответ + результаты поиска не влезали
    s["settings_rev"] = SETTINGS_REV
    if s.get("web_search") not in WEB_MODES:
        s["web_search"] = WEB_AUTO
    s["max_tokens"] = max(64, min(8192, _int(s.get("max_tokens"), DEFAULT_MAX_TOKENS)))
    if not isinstance(s.get("system_prompt"), str):
        s["system_prompt"] = DEFAULT_SYSTEM_PROMPT
    if s.get("stt_engine") not in (STT_WHISPER, STT_VOSK):
        s["stt_engine"] = STT_WHISPER
    if s.get("compute") not in (COMPUTE_GPU, COMPUTE_CPU):
        s["compute"] = COMPUTE_GPU
    s["gpu_layers"] = _int(s.get("gpu_layers"), -1)
    return s


def save_settings(s):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def history_for_llm():
    """История для модели: только role и content (лишние ключи API может не принять)."""
    with HISTORY_LOCK:
        return [{"role": m["role"], "content": m["content"]} for m in HISTORY]


def history_append(user_text, answer, sent=None, note=""):
    """sent — вопрос в том виде, в каком ушёл модели (с датой): так кэш промпта в
    KoboldCpp совпадает и на следующем вопросе старая часть диалога не пересчитывается."""
    with HISTORY_LOCK:
        HISTORY.append({"role": "user", "content": sent or user_text, "shown": user_text})
        HISTORY.append({"role": "assistant", "content": answer, "note": note})
        del HISTORY[:-HISTORY_KEEP * 2]
        LIVE.update(user="", text="", status="")
        HISTORY_VER[0] += 1


def live_update(**kv):
    """Ответ в процессе: текст по мере генерации и что сейчас происходит (поиск…)."""
    with HISTORY_LOCK:
        LIVE.update(kv)
        HISTORY_VER[0] += 1


def render_log():
    with HISTORY_LOCK:
        hist, live = list(HISTORY), dict(LIVE)
    lines = []
    for m in hist:
        if m["role"] == "user":
            lines.append("Вы: " + m.get("shown", m["content"]))
        else:
            lines.append("Ассистент: " + m["content"])
            if m.get("note"):
                lines.append("   " + m["note"])
    if live["user"]:
        lines.append("Вы: " + live["user"])
        lines.append("Ассистент: " + (live["text"] or "…"))
    if live["status"]:
        lines.append("   " + live["status"])
    return "\n".join(lines)


# ----------------------------- STT (Vosk) ----------------------------------

VOSK_NAME = "vosk-model-small-ru-0.22"
VOSK_URL = f"https://alphacephei.com/vosk/models/{VOSK_NAME}.zip"


def vosk_model_path():
    """Своя загрузка вместо vosk.Model(lang="ru"): у штатной таймаут 10 секунд,
    на медленном соединении скачивание обрывается. Здесь — длинные таймауты,
    повторы, докачка и восстановление после битого архива."""
    import zipfile
    from pathlib import Path

    cache = Path.home() / ".cache" / "vosk"
    target = cache / VOSK_NAME
    if target.is_dir():
        return str(target)
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / (VOSK_NAME + ".zip")

    def download():
        for attempt in range(3):
            downloaded = zip_path.stat().st_size if zip_path.exists() else 0
            headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
            try:
                with requests.get(VOSK_URL, stream=True, timeout=(15, 180), headers=headers) as r:
                    if downloaded and r.status_code == 200:
                        downloaded = 0  # сервер не понял Range — качаем заново
                    r.raise_for_status()
                    with open(zip_path, "ab" if downloaded else "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            if chunk:
                                f.write(chunk)
                return
            except requests.RequestException:
                if attempt == 2:
                    raise RuntimeError(
                        "Не удалось скачать модель распознавания. Ручной вариант: "
                        + VOSK_URL + " -> распаковать в .cache/vosk/"
                    )

    for _ in range(2):
        download()
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(cache)
            zip_path.unlink(missing_ok=True)
            return str(target)
        except zipfile.BadZipFile:
            zip_path.unlink(missing_ok=True)
    raise RuntimeError("Архив модели повреждён даже после повторного скачивания.")


def load_stt():
    if _stt["model"] is None:
        vosk.SetLogLevel(-1)
        _stt["model"] = vosk.Model(vosk_model_path())
    return _stt["model"]


def to_16k_mono_int16(audio_path):
    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != SAMPLE_RATE_STT:
        g = math.gcd(int(sr), SAMPLE_RATE_STT)
        data = signal.resample_poly(data, SAMPLE_RATE_STT // g, sr // g)
    return (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)


def resample_int16(pcm, sr, target=SAMPLE_RATE_STT):
    if int(sr) == target:
        return pcm
    g = math.gcd(int(sr), target)
    y = signal.resample_poly(pcm.astype(np.float32), target // g, int(sr) // g)
    return np.clip(y, -32768, 32767).astype(np.int16)


def vosk_pcm(pcm):
    """int16 моно 16 кГц -> текст (Vosk)."""
    rec = vosk.KaldiRecognizer(load_stt(), SAMPLE_RATE_STT)
    data = pcm.tobytes()
    parts = []
    for i in range(0, len(data), 8000):
        if rec.AcceptWaveform(data[i:i + 8000]):
            parts.append(json.loads(rec.Result()).get("text", ""))
    parts.append(json.loads(rec.FinalResult()).get("text", ""))
    return " ".join(p for p in parts if p).strip()


# ----------------------------- Движок: KoboldCpp -----------------------------
#
# Один процесс KoboldCpp держит и LLM, и Whisper. Сборка nocuda умеет Vulkan —
# видеокарты AMD (и любые другие) работают без CUDA/ROCm. Скачивается один раз с
# официальной страницы релизов (sha256 сверяется), распаковывается в engine/ —
# так запуск быстрый и временная папка не засоряется.

KOBOLD_VERSION = "1.122.1"
KOBOLD_PORT = 5011
KOBOLD_ASSETS = {  # файл релиза -> sha256 из GitHub API релиза v1.122.1
    "koboldcpp-nocuda.exe": "c314724e02b310c4db066a8dade8890a1628bc4b65aa9c2b658309219ca7a779",
    "koboldcpp-linux-x64-nocuda": "5532ead66f460a59c744fc74a45715bf2b0ef2fe2fb05a6c146a6d2df2145d29",
    "koboldcpp-mac-arm64": "4dc85e7f0414812ec5b1fd810b2009a73844ff17347edac41ff85ffe7fc70412",
}
ENGINE_DIR = os.path.join(APP_DIR, "engine")
ENGINE_LOG = os.path.join(ENGINE_DIR, "koboldcpp.log")
ENGINE_BASE = f"http://127.0.0.1:{KOBOLD_PORT}"
ENGINE_STATE = {"phase": "off", "info": "", "warn": "", "device": "", "models": "",
                "stt_sec": None, "llm_sec": None, "llm_tps": None}
_eng = {"proc": None, "light": None, "jobs": []}
ENGINE_LOCK = threading.RLock()
_HTTP = requests.Session()
_HTTP.trust_env = False  # к 127.0.0.1 — всегда напрямую, мимо прокси/VPN
_HF_PATHS = {}


class EngineError(RuntimeError):
    pass


def engine_report(**kv):
    if "phase" in kv and kv["phase"] != ENGINE_STATE.get("phase"):
        print(f"[engine] {kv['phase']} {kv.get('info') or ''}".rstrip(), flush=True)
    ENGINE_STATE.update(kv)


def engine_needs(cfg):
    """(нужна локальная LLM, нужен Whisper)"""
    return cfg.get("llm_source") in (SRC_HF, SRC_LOCAL), cfg.get("stt_engine") == STT_WHISPER


def engine_light_key(cfg):
    """Всё, что требует перезапуска движка, — без скачиваний и обращений к диску."""
    need_llm, need_wh = engine_needs(cfg)
    if not (need_llm or need_wh):
        return None
    llm = None
    if need_llm:
        src = ((cfg.get("hf_repo") or "").strip(), (cfg.get("hf_file") or "").strip()) \
            if cfg["llm_source"] == SRC_HF else (cfg.get("local_gguf_path") or "").strip().strip('"')
        llm = (src, _int(cfg.get("n_ctx"), 4096))
    gpu = cfg.get("compute") != COMPUTE_CPU
    return (llm, (cfg.get("whisper_model") or "").strip() if need_wh else None,
            gpu, _int(cfg.get("gpu_layers"), -1) if gpu else 0)


def kobold_asset():
    sysname, mach = platform.system(), platform.machine().lower()
    if sysname == "Windows" and mach in ("amd64", "x86_64"):
        return "koboldcpp-nocuda.exe"
    if sysname == "Linux" and mach in ("x86_64", "amd64"):
        return "koboldcpp-linux-x64-nocuda"
    if sysname == "Darwin" and mach == "arm64":
        return "koboldcpp-mac-arm64"
    raise EngineError(f"Для {sysname} {mach} нет готовой сборки KoboldCpp — "
                      "выберите LiteLLM для ответов и Vosk для распознавания.")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _download(url, dest, sha256, label):
    """Скачивание с докачкой, повторами и прогрессом в статусе GUI."""
    part = dest + ".part"
    for attempt in range(3):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        try:
            with requests.get(url, stream=True, timeout=(15, 120),
                              headers={"Range": f"bytes={have}-"} if have else {}) as r:
                if r.status_code == 416:  # уже скачано целиком
                    break
                if have and r.status_code != 206:
                    have = 0  # сервер не понял Range — качаем заново
                r.raise_for_status()
                total = have + int(r.headers.get("Content-Length") or 0)
                done, shown = have, 0.0
                with open(part, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - shown > 0.5:
                            shown = time.time()
                            engine_report(info=f"Скачиваю {label}: {done >> 20}"
                                               + (f" из {total >> 20}" if total else "") + " МБ")
            break
        except requests.RequestException as e:
            if attempt == 2:
                raise EngineError(f"Не удалось скачать {label}: {e}. Можно вручную: "
                                  f"{url} -> {dest}")
            time.sleep(3)
    if sha256 and _sha256(part) != sha256:
        os.remove(part)
        raise EngineError(f"{label}: контрольная сумма не совпала (файл повреждён) — "
                          "попробуйте ещё раз.")
    os.replace(part, dest)


def _hidden():
    """Без лишних окон консоли на Windows."""
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def ensure_kobold():
    """Путь к KoboldCpp; при первом запуске — скачать (~120 МБ) и распаковать."""
    custom = (_wake["cfg"].get("kobold_path") or "").strip().strip('"')
    if custom:
        if not os.path.isfile(custom):
            raise EngineError(f"kobold_path из settings.json не найден: {custom}")
        return custom
    asset = kobold_asset()
    exe = ".exe" if os.name == "nt" else ""
    unpacked = os.path.join(ENGINE_DIR, f"koboldcpp-{KOBOLD_VERSION}")
    launcher = os.path.join(unpacked, "koboldcpp-launcher" + exe)
    if os.path.isfile(launcher):
        return launcher
    os.makedirs(ENGINE_DIR, exist_ok=True)
    single = os.path.join(ENGINE_DIR, asset)
    if not os.path.isfile(single):
        engine_report(phase="download", info=f"Скачиваю KoboldCpp {KOBOLD_VERSION}…")
        _download(f"https://github.com/LostRuins/koboldcpp/releases/download/"
                  f"v{KOBOLD_VERSION}/{asset}", single, KOBOLD_ASSETS[asset],
                  f"KoboldCpp {KOBOLD_VERSION}")
    elif _sha256(single) != KOBOLD_ASSETS[asset]:
        os.remove(single)
        raise EngineError(f"{single} не совпадает с официальным релизом — файл удалён, "
                          "при следующем запуске скачается заново.")
    if os.name != "nt":
        os.chmod(single, 0o755)
    engine_report(phase="starting", info="Распаковываю KoboldCpp (один раз, около минуты)…")
    tmp = unpacked + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        with open(ENGINE_LOG, "wb") as log:
            subprocess.run([single, "--unpack", tmp], cwd=ENGINE_DIR, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, timeout=900, **_hidden())
        if os.path.isfile(os.path.join(tmp, "koboldcpp-launcher" + exe)):
            os.replace(tmp, unpacked)
            os.remove(single)
            return launcher
    except Exception as e:
        print(f"[engine] распаковка не удалась ({e}) — запускаю одним файлом", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return single


def _watch_hf_download(repo, filename, label, stop):
    """Прогресс скачивания с Hugging Face — по размеру *.incomplete в кэше."""
    total = 0
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        total = get_hf_file_metadata(hf_hub_url(repo, filename)).size or 0
    except Exception:
        pass
    try:
        from huggingface_hub import constants
        blobs = os.path.join(constants.HF_HUB_CACHE, "models--" + repo.replace("/", "--"), "blobs")
    except Exception:
        return
    while not stop.wait(1.0):
        try:
            got = sum(os.path.getsize(os.path.join(blobs, f)) for f in os.listdir(blobs)
                      if f.endswith(".incomplete"))
        except OSError:
            continue
        if got and not stop.is_set():
            engine_report(info=f"Скачиваю {label}: {got >> 20}"
                               + (f" из {total >> 20}" if total else "") + " МБ")


def hf_file(repo, filename, label):
    """Файл из Hugging Face (кэш HF): без сети, если уже скачан."""
    key = (repo, filename)
    if key in _HF_PATHS and os.path.isfile(_HF_PATHS[key]):
        return _HF_PATHS[key]
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id=repo, filename=filename, local_files_only=True)
    except Exception:
        engine_report(phase="download", info=f"Скачиваю {label}…")
        stop = threading.Event()
        threading.Thread(target=_watch_hf_download, args=(repo, filename, label, stop),
                         daemon=True).start()
        try:
            path = hf_hub_download(repo_id=repo, filename=filename)
        except Exception as e:
            raise EngineError(f"Не удалось скачать {label} ({repo}/{filename}): {e}")
        finally:
            stop.set()
    _HF_PATHS[key] = path
    return path


def resolve_llm_path(cfg):
    if cfg["llm_source"] == SRC_LOCAL:
        path = (cfg.get("local_gguf_path") or "").strip().strip('"')
        if not path or not os.path.isfile(path):
            raise EngineError("Укажите существующий путь к .gguf файлу.")
        return path
    repo, fname = (cfg.get("hf_repo") or "").strip(), (cfg.get("hf_file") or "").strip()
    if not repo or not fname:
        raise EngineError("Укажите репозиторий и имя файла GGUF на Hugging Face.")
    return hf_file(repo, fname, f"модель {fname}")


def resolve_whisper_path(cfg):
    """Имя файла из ggerganov/whisper.cpp, «владелец/репозиторий/файл» или путь к .bin."""
    name = (cfg.get("whisper_model") or WHISPER_MODELS[0]).strip().strip('"')
    if os.path.isfile(name):
        return name
    if os.path.isabs(name) or "\\" in name or ":" in name:
        raise EngineError(f"Файл модели Whisper не найден: {name}")
    parts = name.split("/")
    repo, fname = ("/".join(parts[:2]), "/".join(parts[2:])) if len(parts) >= 3 \
        else (WHISPER_REPO, name)
    return hf_file(repo, fname, f"Whisper {os.path.basename(fname)}")


def _ascii_path(path):
    """C++-часть KoboldCpp открывает файлы «узкими» строками: на Windows путь с
    кириллицей (C:\\Users\\Иван\\...) может не открыться. Короткое имя 8.3 — латиницей."""
    if os.name != "nt" or path.isascii():
        return path
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, 32768)
        if 0 < n < 32768 and buf.value.isascii():
            return buf.value
    except Exception:
        pass
    return path


def _kill_with_us(proc):
    """Windows: KoboldCpp завершится вместе с ассистентом, даже если окно консоли
    закрыли крестиком, — иначе он остался бы висеть и держать видеопамять."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic),
                        ("IoInfo", ctypes.c_uint64 * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        job = k32.CreateJobObjectW(None, None)
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if job and k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)) \
                and k32.AssignProcessToJobObject(job, int(proc._handle)):
            _eng["jobs"].append(job)  # дескриптор живёт, пока жив ассистент
    except Exception as e:
        print(f"[engine] job object: {e}", flush=True)


def _port_busy():
    import socket
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", KOBOLD_PORT)) == 0


def _free_port():
    """На порту может висеть KoboldCpp от прошлого (аварийно закрытого) запуска — погасить."""
    if not _port_busy():
        return
    try:
        _HTTP.post(ENGINE_BASE + "/api/extra/shutdown", json={}, timeout=5)
    except requests.RequestException:
        pass
    for _ in range(60):
        if not _port_busy():
            return
        time.sleep(0.25)
    raise EngineError(f"Порт {KOBOLD_PORT} занят другой программой — закройте её "
                      "и нажмите «Перезапустить движок».")


def _read_log(limit=400_000):
    try:
        with open(ENGINE_LOG, "rb") as f:
            return f.read(limit).decode("utf-8", "replace")
    except OSError:
        return ""


def _log_excerpt():
    """Строки с ошибками из лога KoboldCpp (или его хвост)."""
    lines = [ln.strip() for ln in _read_log(2_000_000).splitlines()
             if ln.strip() and len(ln) < 400 and not ln.startswith("Namespace(")]
    bad = [ln for ln in lines
           if re.search(r"\b(error|failed|failure|cannot|could not|exception|traceback)\b", ln, re.I)]
    return " | ".join((bad or lines)[-3:])[-600:]


def _launch(launcher, llm_path, wh_path, gpu, layers, n_ctx):
    """Запустить KoboldCpp и дождаться готовности. False — процесс завершился сам."""
    args = [_ascii_path(launcher), "--port", str(KOBOLD_PORT), "--host", "127.0.0.1",
            "--skiplauncher", "--quiet", "--singleinstance"]
    if llm_path:
        args += ["--model", _ascii_path(llm_path), "--contextsize", str(n_ctx)]
    if wh_path:
        args += ["--whispermodel", _ascii_path(wh_path)]
    if gpu:
        args += ["--usevulkan"] + (["--gpulayers", str(layers)] if layers >= 0 else [])
    else:
        args += ["--usecpu"]
    os.makedirs(ENGINE_DIR, exist_ok=True)
    with open(ENGINE_LOG, "wb") as log:
        log.write((" ".join(args) + "\n\n").encode("utf-8", "replace"))
        log.flush()
        proc = subprocess.Popen(args, cwd=os.path.dirname(launcher) or None,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                **_hidden())
    _eng["proc"] = proc
    _kill_with_us(proc)
    t0 = time.time()
    while time.time() - t0 < 900:
        if proc.poll() is not None:
            return False
        try:
            if _HTTP.get(ENGINE_BASE + "/api/extra/version", timeout=2).ok:
                return True
        except requests.RequestException:
            pass
        engine_report(info=f"Загружаю модели{' в видеопамять' if gpu else ''}… "
                           f"{time.time() - t0:.0f} с")
        time.sleep(0.5)
    stop_engine(report=False)
    raise EngineError("KoboldCpp не запустился за 15 минут — см. engine/koboldcpp.log")


def _describe_device(gpu, has_llm):
    """Что реально считает: по логу KoboldCpp -> (строка для GUI, предупреждение)."""
    if not gpu:
        return "процессор (CPU)", ""
    text = _read_log(2_000_000)
    names = re.findall(r"using device Vulkan\d+ \(([^)]+)\)", text) \
        or re.findall(r"ggml_vulkan: \d+ = ([^(|\n]+?)\s*\(", text)
    off = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", text)
    if names:
        desc = "видеокарта " + ", ".join(dict.fromkeys(n.strip() for n in names)) + " (Vulkan)"
        if has_llm and off:
            desc += f" · слоёв LLM на видеокарте: {off[-1][0]} из {off[-1][1]}"
        return desc, ""
    if re.search(r"Backend \d+: Vulkan|Vulkan\d+ (model|compute|KV) buffer", text):
        return "видеокарта (Vulkan)", ""
    if has_llm:
        return "процессор (CPU)", ("Vulkan не нашёл видеокарту — всё считается на процессоре. "
                                  "Обновите драйвер видеокарты (AMD Adrenalin); "
                                  "подробности — engine/koboldcpp.log.")
    return "видеокарта (Vulkan) — Whisper", ""


def stop_engine(report=True):
    with ENGINE_LOCK:
        proc, _eng["proc"], _eng["light"] = _eng["proc"], None, None
        if proc is not None and proc.poll() is None:
            try:  # штатно: так KoboldCpp сам освобождает видеопамять и временные файлы
                _HTTP.post(ENGINE_BASE + "/api/extra/shutdown", json={}, timeout=3)
            except requests.RequestException:
                pass
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if report:
            engine_report(phase="off", info="", warn="", device="", models="")


def _stop_engine_at_exit():
    proc = _eng["proc"]
    if proc is None or proc.poll() is not None:
        return
    try:
        _HTTP.post(ENGINE_BASE + "/api/extra/shutdown", json={}, timeout=2)
        proc.wait(timeout=6)
    except Exception:
        proc.kill()


def ensure_engine(cfg):
    """Поднять KoboldCpp под текущие настройки (или убедиться, что он такой) -> base URL."""
    need_llm, need_wh = engine_needs(cfg)
    if not (need_llm or need_wh):
        raise EngineError("Локальный движок не нужен при текущих настройках.")
    with ENGINE_LOCK:
        light = engine_light_key(cfg)
        proc = _eng["proc"]
        if proc is not None and proc.poll() is None and _eng["light"] == light:
            return ENGINE_BASE
        try:
            launcher = ensure_kobold()
            llm_path = resolve_llm_path(cfg) if need_llm else ""
            wh_path = resolve_whisper_path(cfg) if need_wh else ""
            if proc is not None and proc.poll() is not None:
                print(f"[engine] KoboldCpp завершился (код {proc.returncode}): {_log_excerpt()}",
                      flush=True)
            # после падения с теми же настройками время последнего вопроса остаётся в строке
            times = {} if _eng["light"] == light else dict(stt_sec=None, llm_sec=None, llm_tps=None)
            stop_engine(report=False)
            _free_port()
            gpu = cfg.get("compute") != COMPUTE_CPU
            layers, n_ctx = _int(cfg.get("gpu_layers"), -1), _int(cfg.get("n_ctx"), 4096)
            models = " + ".join(
                ([os.path.basename(llm_path)] if llm_path else [])
                + ([f"Whisper {os.path.basename(wh_path)}"] if wh_path else []))
            engine_report(phase="starting", info="Запускаю KoboldCpp…", warn="", device="",
                          models=models, **times)
            ok, warn = _launch(launcher, llm_path, wh_path, gpu, layers, n_ctx), ""
            if not ok and gpu:  # драйвер Vulkan упал — не оставляем без ассистента
                why = _log_excerpt()
                try:
                    shutil.copyfile(ENGINE_LOG, os.path.join(ENGINE_DIR, "koboldcpp-vulkan-fail.log"))
                except OSError:
                    pass
                print(f"[engine] Vulkan не запустился: {why}", flush=True)
                warn = ("Видеокарта (Vulkan) не запустилась — работаю на процессоре. "
                        "Подробности: engine/koboldcpp-vulkan-fail.log")
                gpu = False
                engine_report(info="Vulkan не запустился — пробую на процессоре…")
                ok = _launch(launcher, llm_path, wh_path, False, 0, n_ctx)
            if not ok:
                raise EngineError("KoboldCpp не запустился: " + _log_excerpt()
                                  + " (полный лог: engine/koboldcpp.log)")
            _eng["light"] = light
            device, warn2 = _describe_device(gpu, bool(llm_path))
            engine_report(phase="ready", info="", device=device, warn=warn or warn2)
            return ENGINE_BASE
        except Exception as e:
            engine_report(phase="error", info=str(e))
            raise


def engine_request(cfg, path, payload, timeout, stream=False):
    """POST в KoboldCpp -> Response; если он упал — один перезапуск и повтор."""
    for attempt in (1, 2):
        base = ensure_engine(cfg)
        try:
            r = _HTTP.post(base + path, json=payload, timeout=timeout, stream=stream)
        except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError):
            proc = _eng["proc"]
            if attempt == 2 or (proc is not None and proc.poll() is None):
                raise EngineError("KoboldCpp не отвечает — см. engine/koboldcpp.log")
            engine_report(phase="error", info="KoboldCpp упал — перезапускаю…")
            continue
        if r.status_code != 200:
            text = r.text[:200]
            r.close()
            raise EngineError(f"KoboldCpp вернул ошибку {r.status_code}: {text}")
        return r


def engine_post(cfg, path, payload, timeout):
    return engine_request(cfg, path, payload, timeout).json()


def engine_bg(restart=False):
    """Фоном привести движок к текущим настройкам: старт программы, смена режима, кнопка."""
    def run():
        with ENGINE_LOCK:  # берём настройки, когда до нас дошла очередь, — самые свежие
            cfg = _wake["cfg"]
            try:
                if restart:
                    stop_engine()
                if any(engine_needs(cfg)):
                    ensure_engine(cfg)
                else:
                    stop_engine()
            except Exception as e:
                print(f"[engine] {e}", flush=True)
    threading.Thread(target=run, daemon=True).start()


# ----------------------------- Whisper ---------------------------------------

WHISPER_JUNK = re.compile(
    r"субтитр|продолжение следует|спасибо за (просмотр|внимание)|подписывайтесь|ставьте лайк|"
    r"dimatorzok|редактор|корректор|amara\.org", re.I)


def _norm_text(s):
    return " ".join(re.findall(r"\w+", (s or "").lower().replace("ё", "е")))


def clean_whisper(text, prompt=""):
    """Убрать типичные «галлюцинации» Whisper на тишине и шуме."""
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text or "")  # [музыка], (смеётся)
    t = re.sub(r"\s+", " ", t).strip(" -—–")
    n = _norm_text(t)
    if not n:
        return ""
    if len(t) < 80 and WHISPER_JUNK.search(t):
        return ""
    if len(n) > 10 and n in _norm_text(prompt):
        return ""  # Whisper повторил подсказку вместо речи
    return t


def whisper_pcm(pcm16, cfg):
    """int16 моно 16 кГц -> текст (Whisper в KoboldCpp)."""
    pad = int(SAMPLE_RATE_STT * 1.2) - pcm16.size  # короче секунды whisper.cpp не распознаёт
    if pad > 0:
        pcm16 = np.concatenate([pcm16, np.zeros(pad, np.int16)])
    buf = io.BytesIO()
    sf.write(buf, pcm16, SAMPLE_RATE_STT, format="WAV", subtype="PCM_16")
    prompt = (cfg.get("whisper_prompt") or "").strip()
    ensure_engine(cfg)
    t0 = time.time()
    data = engine_post(cfg, "/api/extra/transcribe", {
        "audio_data": base64.b64encode(buf.getvalue()).decode("ascii"),
        "langcode": "ru", "prompt": prompt, "suppress_non_speech": True}, timeout=600)
    ENGINE_STATE["stt_sec"] = time.time() - t0
    text = clean_whisper(data.get("text", ""), prompt)
    print(f"[whisper] {ENGINE_STATE['stt_sec']:.1f} с: {text}", flush=True)
    return text


def transcribe(audio_path, cfg=None):
    """Запись из браузера -> текст: Whisper (если выбран), при сбое — Vosk."""
    pcm = to_16k_mono_int16(audio_path)
    if cfg and cfg.get("stt_engine") == STT_WHISPER:
        try:
            text = whisper_pcm(pcm, cfg)
            if text:
                return text
        except Exception as e:
            print(f"[whisper] не сработал, распознаю Vosk: {e}", flush=True)
    return vosk_pcm(pcm)


# ----------------------------- LLM ------------------------------------------

RU_WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота",
               "воскресенье"]
RU_MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
             "сентября", "октября", "ноября", "декабря"]
CHARS_PER_TOKEN = 2.5  # русский текст у Qwen ≈ 2,7 символа на токен (замерено) — с запасом


def now_line(t=None):
    t = time.localtime(t)
    return (f"Сейчас {RU_WEEKDAYS[t.tm_wday]}, {t.tm_mday} {RU_MONTHS[t.tm_mon - 1]} "
            f"{t.tm_year} года, {t.tm_hour}:{t.tm_min:02d}.")


def est_tokens(text):
    return int(len(text or "") / CHARS_PER_TOKEN) + 4


def build_system(cfg):
    """Системный промпт пользователя + автоматические правила (дата, поиск)."""
    text = (cfg.get("system_prompt") or "").strip() or DEFAULT_SYSTEM_PROMPT
    text += "\n\n" + TIME_RULE
    if cfg.get("web_search") == WEB_AUTO:
        text += "\n\n" + SEARCH_RULE_AUTO
    elif cfg.get("web_search") == WEB_ASK:
        text += "\n\n" + SEARCH_RULE_ASK
    return {"role": "system", "content": text}


def fit_messages(cfg, system, history, tail, max_tokens):
    """Сообщения, которые влезают в контекст локальной модели: старые реплики выкидываются
    (иначе KoboldCpp сам отрежет НАЧАЛО — системный промпт), слишком длинное сообщение
    в хвосте (результаты поиска) подрезается в середине. -> (messages, max_tokens)"""
    tail = list(tail)
    if cfg.get("llm_source") == SRC_LITELLM:
        return [system] + history[-HISTORY_KEEP * 2:] + tail, max_tokens
    n_ctx = _int(cfg.get("n_ctx"), 8192)
    max_tokens = max(64, min(max_tokens, n_ctx // 2))
    budget = n_ctx - max_tokens - 64

    def cost(msgs):
        return sum(est_tokens(m["content"]) for m in msgs)

    used = cost([system] + tail)
    if used > budget and tail:
        i = max(range(len(tail)), key=lambda k: len(tail[k]["content"]))
        c = tail[i]["content"]
        cut = int((used - budget) * CHARS_PER_TOKEN) + 100
        if len(c) - cut > 600:
            tail[i] = dict(tail[i], content=c[:len(c) - cut - 400] + " … " + c[-400:])
        used = cost([system] + tail)
    keep = []
    for k in range(len(history) - 2, -1, -2):  # с конца, парами «вопрос — ответ»
        pair = history[k:k + 2]
        if used + cost(pair) > budget or len(keep) >= HISTORY_KEEP * 2:
            break
        keep, used = pair + keep, used + cost(pair)
    return [system] + keep + tail, max_tokens


def strip_think(text):
    """Рассуждения «думающих» моделей (<think>…</think>) вслух не читаем."""
    text = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.S)
    text = re.sub(r"^.*?</think>", " ", text, flags=re.S)
    return text.strip()


class ThinkFilter:
    """Поток ответа без рассуждений «думающих» моделей (<think>…</think> в начале)."""

    def __init__(self):
        self.buf, self.mode = "", "start"  # start -> think -> lead -> pass

    def feed(self, piece):
        if self.mode == "pass":
            return piece
        if self.mode == "lead":  # пробелы и переносы сразу после </think> не нужны
            piece = piece.lstrip()
            if piece:
                self.mode = "pass"
            return piece
        self.buf += piece
        if self.mode == "start":
            head = self.buf.lstrip()
            if head.startswith("<think>"):
                self.mode = "think"
            elif "<think>".startswith(head):
                return ""  # пока пусто или начало тега — ждём
            else:
                self.mode, out, self.buf = "pass", self.buf, ""
                return out
        end = self.buf.find("</think>")
        if end < 0:
            return ""
        out, self.buf = self.buf[end + 8:].lstrip(), ""
        self.mode = "pass" if out else "lead"
        return out

    def flush(self):
        out, self.buf = ("" if self.mode == "think" else self.buf), ""
        return out


SEARCH_CMD_RE = re.compile(r"^\W*(?:поиск|search)\b[\s*]*(?::|\])[\s*:]*", re.I)


def _cmd_pending(text):
    """Начало строки ещё может оказаться командой «ПОИСК:» — подождать продолжения."""
    head = re.sub(r"^\W+", "", text).lower()
    if not head:
        return len(text) < 12
    if len(head) > 12:
        return False
    return ("поиск".startswith(head) or "search".startswith(head)
            or bool(re.fullmatch(r"(?:поиск|search)[\s*\]]*", head)))


class CommandSniffer:
    """Ищет в начале ответа строку «ПОИСК: запрос» (просьба модели поискать в интернете).
    Текст до неё отдаётся как обычный ответ («Сейчас поищу.»), сама команда — нет."""
    LIMIT = 300  # дальше начала ответа команду не ищем

    def __init__(self):
        self.buf, self.state, self.seen = "", "linestart", 0  # linestart | inline | cmd | pass
        self.query, self.done = "", False

    def feed(self, piece):
        if self.done:
            return ""
        if self.state == "pass":
            return piece
        out = []
        self.buf += piece
        while self.buf:
            if self.state == "inline":
                head, nl, rest = self.buf.partition("\n")
                out.append(head + nl)
                self.seen += len(head + nl)
                self.buf = rest
                if not nl:
                    break
                if self.seen > self.LIMIT:
                    self.state = "pass"
                    out.append(self.buf)
                    self.buf = ""
                    break
                self.state = "linestart"
                continue
            if self.state == "cmd":
                q, nl, _ = self.buf.lstrip().partition("\n")
                if (nl and q.strip()) or len(self.buf) > 300:
                    self.query, self.done, self.buf = q, True, ""
                break
            m = SEARCH_CMD_RE.match(self.buf)
            if m:
                self.state, self.buf = "cmd", self.buf[m.end():]
                continue
            if _cmd_pending(self.buf):
                break
            self.state = "inline"
        return "".join(out)

    def finish(self):
        """Конец потока -> остаток текста ответа."""
        if self.state == "cmd" and not self.done:
            self.query, self.done, self.buf = self.buf.lstrip().partition("\n")[0], True, ""
        out, self.buf = ("" if self.done else self.buf), ""
        return out


def _norm_query(q):
    q = re.sub(r"[*_`«»\"]", " ", q or "")
    return re.sub(r"\s+", " ", q).strip(" .,:;!?—-[]()")[:200]


SEARCH_ASK_RE = re.compile(
    r"\b(?:найди|найти|поищи|поищем|поискать|погугли|загугли|пробей)\b"
    r"|\b(?:в|по)\s+(?:интернете?|сети|гугле|яндексе)\b", re.I)


def clean_query(text):
    """Запрос для поиска из просьбы «найди в интернете, …»."""
    q = re.sub(r"\b(?:пожалуйста|найди|найти|поищи|поищем|поискать|погугли|загугли|пробей|"
               r"посмотри|узнай|мне|нам)\b", " ", text, flags=re.I)
    q = re.sub(r"\b(?:в|по)\s+(?:интернете?|сети|гугле|яндексе)\b", " ", q, flags=re.I)
    q = re.sub(r"\s+", " ", q).strip(" ,.;:!?—-")
    return q or text.strip()


LAST_FINISH = {"reason": None}  # почему модель закончила последний ответ: stop / length


def _sse_pieces(resp):
    for raw in resp.iter_lines():
        if not raw or not raw.startswith(b"data:"):
            continue
        data = raw[5:].strip()
        if data == b"[DONE]":
            return
        try:
            choice = (json.loads(data).get("choices") or [{}])[0]
        except (ValueError, AttributeError):
            continue
        if choice.get("finish_reason"):
            LAST_FINISH["reason"] = choice["finish_reason"]
        piece = (choice.get("delta") or {}).get("content") or choice.get("text") or ""
        if piece:
            yield piece


def llm_stream(cfg, messages, max_tokens):
    """Ответ LLM кусочками (SSE). Закрытие генератора останавливает генерацию."""
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.6,
               "stream": True}
    local = cfg.get("llm_source") != SRC_LITELLM
    if local:
        r = engine_request(cfg, "/v1/chat/completions", payload, timeout=(10, 900), stream=True)
    else:
        base, model_name = cfg["litellm_base"].strip(), cfg["litellm_model"].strip()
        if not base or not model_name:
            raise RuntimeError("Заполните Base URL и имя модели в настройках LiteLLM.")
        r = requests.post(
            base.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {cfg['litellm_key']}"} if cfg["litellm_key"] else {},
            json=dict(payload, model=model_name), timeout=(15, 300), stream=True)
        if r.status_code != 200:
            text = r.text[:200]
            r.close()
            raise RuntimeError(f"LiteLLM вернул ошибку {r.status_code}: {text}")
    done, LAST_FINISH["reason"] = False, None
    try:
        if "event-stream" not in r.headers.get("content-type", ""):
            data = r.json()  # сервер не умеет поток — весь ответ сразу
            LAST_FINISH["reason"] = data["choices"][0].get("finish_reason")
            yield data["choices"][0]["message"].get("content") or ""
        else:
            yield from _sse_pieces(r)
        done = True
    except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
        raise EngineError("Модель оборвала ответ — "
                          + ("см. engine/koboldcpp.log" if local else str(e)[:150]))
    finally:
        r.close()
        if local and not done:  # иначе KoboldCpp допишет ненужный ответ до конца
            try:
                _HTTP.post(ENGINE_BASE + "/api/extra/abort", json={}, timeout=3)
            except requests.RequestException:
                pass


def stream_text(cfg, messages, max_tokens):
    """llm_stream + один повтор, если локальный движок упал до первого слова."""
    for attempt in (1, 2):
        got, gen = False, llm_stream(cfg, messages, max_tokens)
        try:
            for piece in gen:
                got = True
                yield piece
            return
        except EngineError:
            proc = _eng["proc"]
            if (got or attempt == 2 or cfg.get("llm_source") == SRC_LITELLM
                    or (proc is not None and proc.poll() is None)):
                raise
            engine_report(phase="error", info="KoboldCpp упал — перезапускаю…")
        finally:
            gen.close()


def _stream_answer(cfg, messages, max_tokens, emit, sniff=True, force_search=False):
    """Поток одного ответа -> (текст, запрос поиска или None). emit(кусок) — текст ответа.
    force_search: пользователь просил найти, а модель начала отвечать сама — прервать."""
    think, cmd, parts = ThinkFilter(), (CommandSniffer() if sniff else None), []
    gen = stream_text(cfg, messages, max_tokens)
    try:
        for piece in gen:
            out = think.feed(piece)
            if cmd:
                out = cmd.feed(out)
                if cmd.done:
                    return "".join(parts), cmd.query
                if force_search and out and not parts:
                    return "", ""
            if out:
                parts.append(out)
                emit(out)
        out = think.flush()
        if cmd:
            out = cmd.feed(out) + cmd.finish()
            if cmd.done:
                return "".join(parts), cmd.query
            if force_search and out.strip() and not parts:
                return "", ""
        if out:
            parts.append(out)
            emit(out)
        return "".join(parts), None
    finally:
        gen.close()


def search_budget(cfg, system, sent, max_tokens):
    """Сколько символов результатов поиска отдать модели (чтение промпта тоже не бесплатно)."""
    if cfg.get("llm_source") == SRC_LITELLM:
        return 8000
    n_ctx = _int(cfg.get("n_ctx"), 8192)
    mt = max(64, min(max_tokens, n_ctx // 2))
    free = (n_ctx - mt - 64 - est_tokens(system["content"]) - est_tokens(sent) - 250
            - min(1200, n_ctx // 6))
    cap = 6000 if cfg.get("compute") != COMPUTE_CPU else 3000
    return int(max(1200, min(cap, free * CHARS_PER_TOKEN)))


def think_answer(user_text, cfg, emit=lambda s: None, status=lambda s: None):
    """Ответ LLM, при необходимости — с поиском в интернете.
    emit(кусок) — текст ответа по мере генерации; status(строка) — что сейчас происходит.
    -> (ответ, вопрос в том виде, как ушёл модели, заметка о поиске)"""
    mode = cfg.get("web_search", WEB_OFF)
    max_tok = _int(cfg.get("max_tokens"), DEFAULT_MAX_TOKENS)
    system, history = build_system(cfg), history_for_llm()
    sent = f"[{now_line()}]\n{user_text}"
    question = [{"role": "user", "content": sent}]
    t0, note = time.time(), ""
    use_tool = mode in (WEB_AUTO, WEB_ASK)
    msgs, mt = fit_messages(cfg, system, history, question, max_tok)
    answer, query = _stream_answer(cfg, msgs, mt, emit, sniff=use_tool,
                                   force_search=use_tool and bool(SEARCH_ASK_RE.search(user_text)))
    ENGINE_STATE["search_sec"] = None
    if query is not None:
        query = _norm_query(query) or clean_query(user_text)
        status(f"🌐 Ищу в интернете: «{query}»")
        ts = time.time()
        try:
            found, urls = web_search(query, search_budget(cfg, system, sent, mt))
            results_msg = SEARCH_RESULTS_MSG.format(query=query, results=found, question=user_text)
            note = f"🔎 Искал в интернете «{query}»: " + ", ".join(_domain(u) for u in urls[:4])
        except Exception as e:
            print(f"[search] {query}: {e}", flush=True)
            results_msg = SEARCH_FAILED_MSG.format(error=str(e)[:150], question=user_text)
            note = f"⚠️ Поиск «{query}» не удался: {str(e)[:150]}"
        ENGINE_STATE["search_sec"] = time.time() - ts
        print(f"[search] {ENGINE_STATE['search_sec']:.1f} с: {note}", flush=True)
        status("📖 Читаю найденное…")
        tail = question + [{"role": "assistant", "content": f"ПОИСК: {query}"},
                           {"role": "user", "content": results_msg}]
        msgs, mt = fit_messages(cfg, system, history, tail, max_tok)
        answer, again = _stream_answer(cfg, msgs, mt, emit, sniff=True)
        if again is not None:  # снова просит поиск — ответ без правила поиска
            plain = build_system(dict(cfg, web_search=WEB_OFF))
            msgs, mt = fit_messages(cfg, plain, history, tail, max_tok)
            answer, _ = _stream_answer(cfg, msgs, mt, emit, sniff=False)
    if LAST_FINISH["reason"] == "length" and answer.strip():
        note = (note + "\n   " if note else "") + (
            f"✂️ Ответ упёрся в лимит длины ({mt} токенов) — увеличьте «Максимальную длину "
            "ответа» в разделе «Характер и длина ответов».")
    status("")
    if cfg.get("llm_source") != SRC_LITELLM:
        tps = None
        try:  # скорость генерации без учёта чтения промпта
            tps = _HTTP.get(ENGINE_BASE + "/api/extra/perf", timeout=2).json().get("last_eval_speed")
        except (requests.RequestException, ValueError):
            pass
        ENGINE_STATE.update(llm_sec=time.time() - t0, llm_tps=tps or None)
    return strip_think(answer), sent, note


# ----------------------------- Поиск в интернете -----------------------------
#
# ddgs опрашивает поисковики без ключей (Google, DuckDuckGo, Yahoo, Brave…). Любой из
# них может не ответить (блокировка, капча, лимит запросов), поэтому спрашиваем все
# сразу, каждый отдельно и параллельно: встроенный в ddgs перебор нескольких поисковиков
# теряет ответы тех, кто не успел к концу перебора. Набор поисковиков зависит от версии
# ddgs (в 9.16 нет Bing и Яндекса для текста) — берём те, что есть.
# К сниппетам добавляется текст трёх лучших страниц — абзацы с наибольшим числом
# слов запроса. Прокси — как у браузера (настройки Windows / переменные окружения).

# порядок — чьи результаты идут первыми; неизвестные ddgs имена пропускаются
SEARCH_BACKENDS = ("yandex", "google", "duckduckgo", "yahoo", "bing", "brave", "startpage",
                   "mojeek")
NEWS_BACKENDS = ("yandex", "bing", "duckduckgo", "yahoo")
NO_WEB_ENGINES = ("wikipedia", "grokipedia")  # энциклопедии — только если поисковики молчат
NEWS_RE = re.compile(r"новост|что нового|что происходит|последние событ|что случилось", re.I)
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
SKIP_FETCH = ("youtube.com", "youtu.be", "vk.com", "t.me", "instagram.com", "facebook.com",
              "twitter.com", "x.com", "tiktok.com", "ok.ru", "dzen.ru")
STOP_STEMS = {"как", "что", "это", "для", "где", "или", "так", "при", "все", "вот", "был",
              "есть", "можно", "котор", "сейча", "тако", "како"}


def _domain(url):
    host = urllib.parse.urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _system_proxy():
    """Прокси из настроек Windows / переменных окружения (VPN-клиенты ставят его туда)."""
    try:
        p = urllib.request.getproxies()
    except Exception:
        return None
    proxy = p.get("https") or p.get("http")
    if proxy and "://" not in proxy:
        proxy = "http://" + proxy
    return proxy or None


def wiki_hits(query):
    r = requests.get("https://ru.wikipedia.org/w/api.php", params={
        "action": "query", "list": "search", "srsearch": query, "srlimit": 3,
        "format": "json", "utf8": 1}, headers={"User-Agent": "RU-Voice-Assistant/1.0"}, timeout=8)
    r.raise_for_status()
    return [{"title": x["title"], "body": html_unescape(re.sub(r"<[^>]+>", "", x.get("snippet", ""))),
             "url": "https://ru.wikipedia.org/wiki/" + urllib.parse.quote(x["title"].replace(" ", "_")),
             "date": ""} for x in r.json().get("query", {}).get("search", [])]


def _ddgs_backends(category, preferred):
    """Поисковики установленной версии ddgs: сначала из preferred, затем новые незнакомые."""
    try:
        from ddgs.engines import ENGINES
        have = [k for k in ENGINES.get(category, {}) if k not in NO_WEB_ENGINES]
    except Exception:  # другая версия ddgs — пусть разбирается сама
        return list(preferred)
    return [b for b in preferred if b in have] + sorted(set(have) - set(preferred))


def _parallel_hits(query, tasks, max_results, deadline=8.0, grace=1.2):
    """Все поисковики сразу: [(вид, поисковик)] -> ({(вид, поисковик): [результат]}, [ошибки]).
    Ждём до первого ответа и ещё grace секунд — остальные не задерживают ответ."""
    from ddgs import DDGS
    proxy = _system_proxy()

    def one(kind, backend):
        ddg = DDGS(proxy=proxy, timeout=6)
        if kind == "news":
            return ddg.news(query, region="ru-ru", safesearch="moderate", timelimit="w",
                            max_results=max_results, backend=backend)
        return ddg.text(query, region="ru-ru", safesearch="moderate", max_results=max_results,
                        backend=backend)

    pool = ThreadPoolExecutor(max_workers=max(1, len(tasks)))
    futs = {pool.submit(one, *t): t for t in tasks}
    got, errors, t0, first = {}, [], time.time(), None
    pending = set(futs)
    try:
        while pending:
            left = t0 + deadline - time.time()
            if first is not None:
                left = min(left, first + grace - time.time())
            if left <= 0:
                break
            done, pending = futures_wait(pending, timeout=left, return_when=FIRST_COMPLETED)
            for f in done:
                try:
                    res = f.result()
                except Exception as e:
                    errors.append(f"{futs[f][1]}: {str(e)[:80]}")
                    continue
                if res:
                    got[futs[f]] = res
                    first = first or time.time()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return got, errors


def _interleave(lists):
    """[[a1, a2], [b1]] -> [a1, b1, a2]: лучшие результаты каждого поисковика — первыми."""
    return [h for group in itertools.zip_longest(*lists) for h in group if h]


def search_hits(query, max_results=6):
    """-> [{title, url, body, date}] из поисковиков; если все молчат — из Википедии."""
    text_b = _ddgs_backends("text", SEARCH_BACKENDS)
    news_b = _ddgs_backends("news", NEWS_BACKENDS) if NEWS_RE.search(query) else []
    tasks = [("news", b) for b in news_b] + [("text", b) for b in text_b]
    try:
        got, errors = _parallel_hits(query, tasks, max_results)
    except ImportError:
        got, errors = {}, ["нет пакета ddgs — запустите launch.bat ещё раз"]
    news = _interleave([got[("news", b)] for b in news_b if ("news", b) in got])
    text = _interleave([got[("text", b)] for b in text_b if ("text", b) in got])
    hits = [{"title": h.get("title", ""), "url": h.get("url") or h.get("href", ""),
             "body": h.get("body", ""), "date": (h.get("date") or "")[:10]} for h in news[:4]]
    hits += [{"title": h.get("title", ""), "url": h.get("href") or h.get("url", ""),
              "body": h.get("body", ""), "date": ""} for h in text]
    print(f"[search] ответили: {', '.join(f'{k}/{b}' for k, b in got) or 'никто'}", flush=True)
    if not hits:
        try:
            hits = wiki_hits(query)
        except Exception as e:
            errors.append(str(e))
    seen, out = set(), []
    for h in hits:
        if h["url"] and h["url"] not in seen:
            seen.add(h["url"])
            out.append(h)
    if not out:
        raise RuntimeError("ничего не нашлось" + (f" ({'; '.join(errors)[:150]})" if errors else ""))
    return out


def page_blocks(data, charset=None):
    """HTML (bytes) -> куски видимого текста в порядке страницы: абзацы целиком,
    короткие строки (ячейки таблиц, подписи) склеены по ~300 символов."""
    import lxml.html
    text = None  # кодировку из заголовка или угаданную декодирует Python, из <meta> — lxml
    if not charset and not re.search(rb"<meta[^>]+charset", data[:4096], re.I):
        try:
            from charset_normalizer import from_bytes
            best = from_bytes(data[:200_000]).best()
            charset = best.encoding if best else None
        except Exception:
            charset = None
    if charset:
        try:
            text = re.sub(r"^\s*<\?xml[^>]*>", "", data.decode(charset, errors="replace"))
        except LookupError:
            text = None
    try:
        doc = lxml.html.document_fromstring(text if text is not None else data)
    except (ValueError, lxml.etree.ParserError):
        return []
    for bad in doc.xpath("//script|//style|//noscript|//nav|//header|//footer|//aside|//form"
                         "|//svg|//iframe|//template|//button|//select"):
        bad.drop_tree()
    for el in doc.iter("td", "th"):
        el.tail = " | " + (el.tail or "")
    for el in doc.iter("div", "p", "li", "h1", "h2", "h3", "h4", "h5", "tr", "dd", "dt", "br",
                       "section", "article", "pre", "blockquote", "table", "ul", "ol"):
        el.tail = "\n" + (el.tail or "")
        el.text = "\n" + (el.text or "")
    lines = [" ".join(x.split()).strip(" |") for x in doc.text_content().split("\n")]
    chunks, cur, seen = [], "", set()
    for line in lines:
        if len(line) < 3 or line in seen:
            continue
        seen.add(line)
        if len(line) >= 60:  # абзац — отдельный кусок; короткие строки (ячейки, подписи) — вместе
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:1500])
            continue
        if cur and len(cur) + len(line) > 300:
            chunks.append(cur)
            cur = ""
        cur = f"{cur} · {line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks


def fetch_page(url, limit_bytes=1_500_000):
    r = requests.get(url, headers={"User-Agent": BROWSER_UA, "Accept-Language": "ru,en;q=0.8"},
                     timeout=(4, 6), stream=True)
    try:
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "html" not in ctype:
            return []
        data = r.raw.read(limit_bytes, decode_content=True)
    finally:
        r.close()
    m = re.search(r"charset=['\"]?([\w-]+)", ctype, re.I)
    return page_blocks(data, m.group(1) if m else None)


def _stems(text):
    return {w[:5] for w in re.findall(r"[a-zа-я0-9]{3,}", text.lower().replace("ё", "е"))} \
        - STOP_STEMS


def best_excerpt(chunks, query, limit):
    """Куски страницы, где больше всего слов запроса (по первым 5 буквам — падежи)."""
    qs, scored = _stems(query), []
    for i, c in enumerate(chunks):
        n = len(qs & _stems(c))
        if not n:
            continue
        items = c.split(" · ")
        if (len(items) >= 5 and len(c) / len(items) < 20
                and sum(bool(re.search(r"\d", x)) for x in items) < len(items) * 0.3):
            n *= 0.4  # меню сайта: «Бизнес · Экономика · Технологии · …» (не таблица чисел)
        scored.append((n + (0.5 if re.search(r"\d", c) else 0), i, c[:700]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen, total, skipped = [], 0, []
    for _, i, c in scored:
        if total + len(c) <= limit:
            chosen.append((i, c))
            total += len(c) + 1
        else:
            skipped.append((i, c))
    room = limit - total
    if skipped and room >= 200:  # лучший не влезший кусок — хотя бы началом
        i, c = skipped[0]
        chosen.append((i, c[:room].rsplit(" ", 1)[0] + " …"))
    return " ".join(c for _, c in sorted(chosen))


def web_search(query, max_chars=6000):
    """Поиск + текст лучших страниц -> (текст результатов для модели, [url])."""
    hits = search_hits(query)[:6]
    fetch = [h["url"] for h in hits if not any(d in _domain(h["url"]) for d in SKIP_FETCH)][:3]
    pages = {}
    if fetch:
        pool = ThreadPoolExecutor(max_workers=len(fetch))
        futs = {pool.submit(fetch_page, u): u for u in fetch}
        done, _ = futures_wait(futs, timeout=9)
        pool.shutdown(wait=False, cancel_futures=True)
        for f in done:
            try:
                pages[futs[f]] = f.result()
            except Exception as e:
                print(f"[search] {_domain(futs[f])}: {str(e)[:100]}", flush=True)
    per_page = int(max_chars * 0.6 / max(1, len(pages)))
    parts, urls, total = [], [], 0
    for n, h in enumerate(hits, 1):
        date = f", {h['date']}" if h.get("date") else ""
        text = f"[{n}] {h['title']} ({_domain(h['url'])}{date})\n{h['body']}".strip()
        extra = best_excerpt(pages.get(h["url"]) or [], query, per_page)
        if extra:
            text += "\nСо страницы: " + extra
        if total + len(text) > max_chars:
            text = text[:max_chars - total]
            if len(text) < 120:
                break
        parts.append(text)
        urls.append(h["url"])
        total += len(text) + 2
    return "\n\n".join(parts), urls


# ----------------------------- TTS (Silero) ---------------------------------

def load_tts():
    if _tts["model"] is None:
        hub_kwargs = {}
        if "trust_repo" in inspect.signature(torch.hub.load).parameters:
            hub_kwargs["trust_repo"] = True
        result = torch.hub.load(
            repo_or_dir="snakers4/silero-models", model="silero_tts",
            language="ru", speaker="v5_ru", verbose=False, **hub_kwargs,
        )
        model = result[0]
        model.to(torch.device("cpu"))
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        _tts["model"] = model
    return _tts["model"]


# Silero знает только кириллицу: латиница и цифры молча пропадают
# («Как установить Python на Windows?» звучало как «Как установить на?»).
LATIN_WORDS = {
    # железо и интерфейсы
    "midi": "миди", "usb": "ю эс би", "hdmi": "эйч ди эм ай", "wi-fi": "вай фай", "wifi": "вай фай",
    "bluetooth": "блютус", "gpu": "джи пи ю", "cpu": "си пи ю", "ram": "рам", "ssd": "эс эс ди",
    "hdd": "эйч ди ди", "pc": "пи си", "tv": "ти ви", "amd": "эй эм ди", "nvidia": "энвидиа",
    "intel": "интел", "radeon": "радеон", "geforce": "джифорс", "ryzen": "райзен",
    "rtx": "ар ти икс", "gtx": "джи ти икс", "rx": "ар икс", "vulkan": "вулкан", "cuda": "куда",
    "driver": "драйвер", "bios": "биос", "router": "роутер", "audio": "аудио", "video": "видео",
    # системы, сервисы, программы
    "windows": "виндоус", "linux": "линукс", "ubuntu": "убунту", "android": "андроид",
    "ios": "ай оу эс", "macos": "мак оу эс", "mac": "мак", "apple": "эпл", "iphone": "айфон",
    "ipad": "айпад", "google": "гугл", "microsoft": "майкрософт", "office": "офис", "word": "ворд",
    "excel": "эксель", "powerpoint": "пауэрпойнт", "chrome": "хром", "firefox": "файрфокс",
    "edge": "эдж", "opera": "опера", "yandex": "яндекс", "telegram": "телеграм",
    "whatsapp": "вотсап", "discord": "дискорд", "zoom": "зум", "skype": "скайп",
    "youtube": "ютуб", "twitch": "твич", "steam": "стим", "tiktok": "тикток",
    "instagram": "инстаграм", "facebook": "фейсбук", "twitter": "твиттер", "spotify": "спотифай",
    "netflix": "нетфликс", "photoshop": "фотошоп", "adobe": "адоби", "premiere": "премьер",
    "blender": "блендер", "obs": "о би эс", "minecraft": "майнкрафт", "xbox": "икс бокс",
    "playstation": "плейстейшен", "nintendo": "нинтендо",
    # музыка и звук
    "vst": "ви эс ти", "daw": "дау", "fl": "эф эл", "studio": "студио", "ableton": "эйблтон",
    "live": "лайв", "cubase": "кьюбейс", "reaper": "рипер", "logic": "лоджик", "asio": "азио",
    "synth": "синт", "plugin": "плагин", "sampler": "сэмплер", "loop": "луп",
    # программирование и сеть
    "python": "пайтон", "java": "джава", "javascript": "джаваскрипт", "html": "эйч ти эм эл",
    "css": "си эс эс", "sql": "эс кью эл", "json": "джейсон", "api": "эй пи ай",
    "github": "гитхаб", "git": "гит", "docker": "докер", "pip": "пип", "npm": "эн пи эм",
    "url": "ю ар эл", "http": "эйч ти ти пи", "https": "эйч ти ти пи эс", "www": "дабл ю",
    "vpn": "ви пи эн", "ip": "ай пи", "dns": "ди эн эс", "email": "имейл", "e-mail": "имейл",
    "online": "онлайн", "offline": "офлайн", "ok": "окей", "okay": "окей", "ai": "эй ай",
    "gpt": "джи пи ти", "chatgpt": "чат джи пи ти", "openai": "оупен эй ай", "llm": "эл эл эм",
    "qwen": "квен", "gemma": "джемма", "llama": "лама", "whisper": "виспер", "vosk": "воск",
    "silero": "силеро", "kobold": "кобольд", "koboldcpp": "кобольд", "gradio": "градио",
    "gguf": "джи джи ю эф",
    # частые английские слова
    "pro": "про", "max": "макс", "plus": "плюс", "mini": "мини", "ultra": "ультра",
    "smart": "смарт", "home": "хоум", "the": "зе", "and": "энд", "of": "оф", "for": "фор",
    "game": "гейм", "gaming": "гейминг", "stream": "стрим", "update": "апдейт",
    "server": "сервер", "browser": "браузер", "file": "файл",
}
LETTER_NAMES = {"a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф", "g": "джи",
                "h": "эйч", "i": "ай", "j": "джей", "k": "кей", "l": "эл", "m": "эм", "n": "эн",
                "o": "оу", "p": "пи", "q": "кью", "r": "ар", "s": "эс", "t": "ти", "u": "ю",
                "v": "ви", "w": "дабл ю", "x": "икс", "y": "уай", "z": "зед"}
_TR_MULTI = [("tion", "шн"), ("igh", "ай"), ("sch", "ск"), ("tch", "ч"), ("sh", "ш"),
             ("ch", "ч"), ("th", "т"), ("ph", "ф"), ("ck", "к"), ("qu", "кв"), ("wh", "в"),
             ("kn", "н"), ("oo", "у"), ("ee", "и"), ("ea", "и"), ("ou", "ау"), ("ow", "оу"),
             ("ay", "эй"), ("ai", "эй"), ("ey", "эй"), ("oy", "ой")]
_TR_ONE = {"a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х",
           "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
           "q": "к", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс",
           "y": "и", "z": "з"}


def _translit(word):
    """Незнакомое английское слово -> примерное чтение кириллицей."""
    w = word.lower()
    if len(w) > 3 and w.endswith("e") and w[-2] not in "aeiouy":
        w = w[:-1]  # немая e: phone -> фон
    out, i = [], 0
    while i < len(w):
        for src, dst in _TR_MULTI:
            if w.startswith(src, i):
                out.append(dst)
                i += len(src)
                break
        else:
            ch, nxt = w[i], w[i + 1:i + 2]
            if ch == "c" and nxt in ("e", "i", "y"):
                out.append("с")
            elif ch == "e" and i == 0:
                out.append("э")
            elif ch == "y" and i == 0 and nxt in tuple("aeiou"):
                out.append("й")
            else:
                out.append(_TR_ONE.get(ch, ""))
            i += 1
    return "".join(out)


def latin_to_ru(text):
    """«MIDI-синтезатор на Windows» -> «миди-синтезатор на виндоус»."""
    text = re.sub(r"\b[Cc]\+\+", " си плюс плюс ", text)
    text = re.sub(r"\b[Cc]#", " си шарп ", text)

    def word(p):
        low = p.lower()
        if low in LATIN_WORDS:
            return LATIN_WORDS[low]
        if len(p) == 1 or (p.isupper() and len(p) <= 5):  # аббревиатура: VST, GPU
            return " ".join(LETTER_NAMES[c] for c in low)
        return _translit(p)

    def repl(m):
        tok = m.group(0)
        if tok.lower() in LATIN_WORDS:
            return LATIN_WORDS[tok.lower()]
        return " ".join(word(p) for p in re.split(r"[-']", tok) if p)

    return re.sub(r"[A-Za-z]+(?:['-][A-Za-z]+)*", repl, text)


def _plural(n, one, few, many):
    n = abs(int(n)) % 100
    if 11 <= n <= 19:
        return many
    n %= 10
    return one if n == 1 else few if 2 <= n <= 4 else many


def _num_words(s, to="cardinal"):
    try:
        from num2words import num2words
    except ImportError:
        return s
    try:
        if re.fullmatch(r"\d+", s):
            return num2words(int(s), lang="ru", to=to)
        return num2words(float(s.replace(",", ".")), lang="ru")
    except Exception:
        return s


def _ordinal_case(words, case):
    """«двадцать четвёртый» -> «двадцать четвёртом» (о годе) / «двадцать четвёртого»."""
    *head, last = words.split()
    if last.endswith("ий"):
        last = last[:-2] + ("ьем" if case == "prep" else "ьего")
    elif last.endswith(("ый", "ой")):
        last = last[:-2] + ("ом" if case == "prep" else "ого")
    return " ".join(head + [last])


def numbers_to_ru(text):
    """Цифры -> слова: «в 2024 году», «50%», «3,5»."""
    text = re.sub(r"(?<=\d)[ \u00a0\u202f](?=\d{3}\b)", "", text)  # 1 000 000
    text = re.sub(r"(?<=\d)\s*\+\s*(?=\d)", " плюс ", text)     # «+» у Silero — знак ударения
    text = re.sub(r"(?<=\d)\s*=\s*(?=\d)", " равно ", text)

    def year(m):
        w = _num_words(m.group(1), to="ordinal")
        if w == m.group(1):
            return m.group(0)
        g = m.group(2).lower()
        w = _ordinal_case(w, "prep") if g == "году" else _ordinal_case(w, "gen") if g == "года" else w
        return f"{w} {m.group(2)}"

    def pct(m):
        num = m.group(1)
        unit = "процента" if re.search(r"[.,]", num) else \
            _plural(num, "процент", "процента", "процентов")
        return f"{_num_words(num)} {unit}"

    text = re.sub(r"\b(1\d{3}|20\d{2})\s*(году|года|год)\b", year, text)  # годы, не «2 года»
    text = re.sub(r"(\d+(?:[.,]\d+)?)\s*%", pct, text)
    return re.sub(r"\d+(?:[.,]\d+)?", lambda m: f" {_num_words(m.group(0))} ", text)


def clean_for_tts(text):
    text = re.sub(r"https?://\S+", "ссылка", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = latin_to_ru(numbers_to_ru(text))  # до чистки markdown: иначе «C#» теряет «#»
    text = re.sub(r"[*_#`>~|]", " ", text)
    # эмодзи и прочие символы вне алфавита/пунктуации — TTS их не читает корректно
    text = re.sub(r"[^\w\s.,!?…:;+%№«»()\-—–'\"]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+([.,!?…:;])", r"\1", re.sub(r"\s+", " ", text)).strip()


def split_long(text, maxlen=350):
    text = text.strip()
    if len(text) <= maxlen:
        return [text]
    sents = re.split(r"(?<=[.!?…])\s+", text)
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > maxlen:
            chunks.append(cur.strip())
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        chunks.append(cur.strip())
    return chunks


def speak(text, voice):
    model = load_tts()
    pause = np.zeros(int(SAMPLE_RATE_TTS * 0.25), dtype=np.float32)
    pieces = []
    for chunk in split_long(clean_for_tts(text)):
        if not re.search(r"[а-яё]", chunk, re.I):
            continue  # Silero падает на тексте без единой буквы
        audio = model.apply_tts(text=chunk, speaker=voice, sample_rate=SAMPLE_RATE_TTS)
        arr = audio.numpy() if torch.is_tensor(audio) else np.asarray(audio)
        pieces.append(np.asarray(arr, dtype=np.float32).flatten())
        pieces.append(pause.copy())
    full = np.concatenate(pieces) if pieces else pause
    out = os.path.join(tempfile.gettempdir(), "ru_assistant_reply.wav")
    sf.write(out, full, SAMPLE_RATE_TTS)
    return out


SENT_END_RE = re.compile(r"[.!?…]+[\"»)]*\s+|\n+")


class SpeechStream:
    """Озвучка ответа по мере генерации: предложения -> Silero (фоновый поток) ->
    play(кусок) сразу в динамики (wake word). Без play — только собрать WAV (браузер):
    синтез идёт параллельно с генерацией, и в конце ждать почти нечего."""

    def __init__(self, voice, play=None, on_start=None):
        self.voice, self.play, self.on_start = voice, play, on_start
        self.buf, self.fed, self.n = "", "", 0
        self.parts, self.cancelled = [], False
        self.q_synth, self.q_play = queue.Queue(), queue.Queue()
        self.t_synth = threading.Thread(target=self._synth_loop, daemon=True)
        self.t_synth.start()
        self.t_play = None
        if play:
            self.t_play = threading.Thread(target=self._play_loop, daemon=True)
            self.t_play.start()

    def feed(self, text):
        self.buf += text
        self.fed += text
        while True:
            min_len = 20 if self.n == 0 else 80  # первая фраза — как можно раньше
            cut = next((m.end() for m in SENT_END_RE.finditer(self.buf) if m.end() >= min_len), None)
            if cut is None:
                if len(self.buf) <= 400:
                    return
                k = max(self.buf.rfind(", ", 0, 350), self.buf.rfind(" ", 0, 350))
                cut = k + 1 if k > 0 else 350
            chunk, self.buf = self.buf[:cut], self.buf[cut:]
            self._put(chunk)

    def _put(self, chunk):
        if chunk.strip():
            self.n += 1
            self.q_synth.put(chunk)

    def _synth_loop(self):
        pause = np.zeros(int(SAMPLE_RATE_TTS * 0.25), dtype=np.float32)
        while True:
            chunk = self.q_synth.get()
            if chunk is None or self.cancelled:
                self.q_play.put(None)
                return
            try:
                model = load_tts()
                for piece in split_long(clean_for_tts(chunk)):
                    if not re.search(r"[а-яё]", piece, re.I):
                        continue  # Silero падает на тексте без единой буквы
                    audio = model.apply_tts(text=piece, speaker=self.voice,
                                            sample_rate=SAMPLE_RATE_TTS)
                    arr = audio.numpy() if torch.is_tensor(audio) else np.asarray(audio)
                    arr = np.concatenate([np.asarray(arr, dtype=np.float32).flatten(), pause])
                    self.parts.append(arr)
                    if self.play:
                        self.q_play.put(arr)
            except Exception as e:
                print(f"[tts] {e}", flush=True)

    def _play_loop(self):
        started = False
        while True:
            arr = self.q_play.get()
            if arr is None:
                return
            if self.cancelled:
                continue
            if not started:
                started = True
                if self.on_start:
                    self.on_start()
            try:
                self.play(arr)
            except Exception as e:
                print(f"[play] {e}", flush=True)

    def finish(self):
        """Дождаться озвучки (и проигрывания) всего ответа -> путь к WAV."""
        self._put(self.buf)
        self.buf = ""
        self.q_synth.put(None)
        self.t_synth.join()
        if self.t_play:
            self.t_play.join()
        pause = np.zeros(int(SAMPLE_RATE_TTS * 0.25), dtype=np.float32)
        full = np.concatenate(self.parts) if self.parts else pause
        out = os.path.join(tempfile.gettempdir(), "ru_assistant_reply.wav")
        sf.write(out, full, SAMPLE_RATE_TTS)
        return out

    def cancel(self):
        self.cancelled = True
        self.q_synth.put(None)


# ----------------------------- Общий пайплайн --------------------------------

def llm_short(cfg, prompt, max_tokens=60):
    """Короткий служебный вопрос модели (оригинальное название фильма) — без истории."""
    msgs = [{"role": "system", "content": "Отвечай одной короткой строкой, без пояснений."},
            {"role": "user", "content": prompt}]
    return strip_think("".join(stream_text(cfg, msgs, max_tokens))).strip()


def run_pc_command(user_text, cmd, speech, cfg):
    """Команда компьютеру (pc_control.Reply): действие до озвучки, короткий ответ голосом,
    действие после (фильм стартует, когда ассистент договорил) -> (ответ, WAV)."""
    err = pc_control.run(cmd.before)
    text = err or cmd.text
    speech.feed(text)
    live_update(text=text, status="")
    wav = speech.finish()
    if not err:
        err = pc_control.run(cmd.after)
        if err:  # действие после озвучки не удалось — сказать и об этом
            text = err
            extra = SpeechStream(cfg["voice"], play=speech.play)
            extra.feed(err)
            wav = extra.finish()
    note = cmd.note + (f"\n   ⚠️ {err}" if err and cmd.note else (f"⚠️ {err}" if err else ""))
    history_append(user_text, text, note=note)
    return text, wav


def answer_and_speak(user_text, cfg, play=None, on_status=None, on_speaking=None):
    """Вопрос -> (поиск) -> LLM потоком -> Silero по предложениям -> (ответ, WAV).
    Потокобезопасно (общий для GUI и wake word). play(кусок) — сразу в динамики."""
    with PIPE_LOCK:
        live_update(user=user_text, text="", status="🧠 Думаю…")
        speech, shown = SpeechStream(cfg["voice"], play=play, on_start=on_speaking), []

        def emit(piece):
            shown.append(piece)
            speech.feed(piece)
            live_update(text=strip_think("".join(shown)), status="")

        def status(text):
            live_update(status=text)
            if on_status:
                on_status(text)

        try:
            if cfg.get("pc_control", True):
                def ask_llm(prompt):
                    status("🎬 Уточняю у модели название фильма…")
                    return llm_short(cfg, prompt)
                cmd = pc_control.handle(user_text, cfg, ask_llm=ask_llm)
                if cmd is not None:
                    return run_pc_command(user_text, cmd, speech, cfg)
            answer, sent, note = think_answer(user_text, cfg, emit=emit, status=status)
            if not answer:
                answer = "Не получилось ответить — попробуйте переформулировать вопрос."
                speech.feed(answer)
            if play is None:
                live_update(text=answer, status="🔊 Озвучиваю…")
            wav = speech.finish()
        except BaseException:
            speech.cancel()
            live_update(user="", text="", status="")
            raise
        history_append(user_text, answer, sent=sent, note=note)
        return answer, wav


# ----------------------------- Wake word -------------------------------------
#
# Микрофон компьютера -> «свободный» распознаватель Vosk, который расшифровывает
# всё подряд (в GUI это строка «Слышу: …»). Wake word ищется в расшифровке
# нечётко (падежи, «асистент», «систент»). Вопрос — текст ПОСЛЕ wake word в той же
# фразе («Ассистент, какая погода?») или следующая фраза, если после wake word
# была пауза (тогда звучит сигнал «слушаю»). Конец фразы определяет сам Vosk по
# паузе (endpointing), а не порог громкости: фиксированный порог RMS ломался от
# шума вентилятора — вопрос «не заканчивался» по 30 секунд.
# Vosk знает только русский словарь («MIDI» у него превращается в «видео»), поэтому
# звук самой фразы (по времени слов от Vosk) отдаётся Whisper — он и даёт текст вопроса.

QUESTION_TIMEOUT = 8.0   # сек ждать вопрос после одиночного wake word
SILENT_DB = -75.0        # тише — микрофон фактически отдаёт нули
RING_SEC = 45.0          # сколько секунд звука помнить для Whisper
WAKE_STATE = {"phase": "off", "level": -90.0, "heard": "", "info": "", "warn": "",
              "note": "", "stt_warn": "", "device": "", "wake": "", "silent": False}
_wake = {"thread": None, "stop": None, "gen": 0, "cfg": dict(DEFAULT_SETTINGS)}
MUTE_UNTIL = [0.0]  # до этого времени браузер озвучивает ответ — микрофон не слушаем


def _norm_word(w):
    return w.lower().replace("ё", "е")


def parse_wake_words(raw):
    """'ассистент, окей компьютер' -> [['ассистент'], ['окей', 'компьютер']]"""
    phrases = []
    for part in re.split(r"[,;|\n]+", raw or ""):
        words = [_norm_word(w) for w in re.findall(r"\w+", part)]
        if words and words not in phrases:
            phrases.append(words)
    return phrases or [["ассистент"]]


def _word_like(heard, wake):
    """Похоже ли распознанное слово на слово-триггер."""
    if heard == wake:
        return True
    if len(wake) < 5 or len(heard) < 4:
        return False  # короткие слова — только точное совпадение
    # падежи и окончания: «ассистента», «компьютеру»
    if len(os.path.commonprefix([heard, wake])) >= max(4, len(wake) - 2):
        return True
    # огрехи распознавания: «асистент», «систент»
    return difflib.SequenceMatcher(None, heard, wake).ratio() >= 0.8


def find_wake(words, phrases):
    """Первое вхождение wake word в списке слов -> (начало, конец) или None."""
    norm = [_norm_word(w) for w in words]
    for i in range(len(norm)):
        for ph in phrases:
            j = i + len(ph)
            if j <= len(norm) and all(_word_like(norm[i + k], ph[k]) for k in range(len(ph))):
                return i, j
    return None


def strip_wake_text(text, phrases, max_pos=2):
    """«Ассистент, как настроить MIDI?» -> «как настроить MIDI?» (wake word — в начале)."""
    toks = list(re.finditer(r"\w+", text or ""))
    longest = max(len(p) for p in phrases)
    pos = find_wake([m.group(0) for m in toks[:max_pos + longest]], phrases)
    if pos is None or pos[0] > max_pos:
        return (text or "").strip()
    return re.sub(r"^[\s,.!?:;—–\-]+", "", text[toks[pos[1] - 1].end():]).strip()


def missing_wake_words(model, phrases):
    """Слова, которых нет в словаре модели Vosk, — их она не услышит никогда."""
    finder = getattr(model, "vosk_model_find_word", None)
    out = []
    if finder is None:
        return out
    for ph in phrases:
        for w in ph:
            try:
                if finder(w) < 0:
                    out.append(w)
            except Exception:
                pass
    return out


def make_beep(sr=SAMPLE_RATE_TTS):
    """Короткий двухтоновый сигнал «слушаю»."""
    t = np.arange(int(sr * 0.08)) / sr
    tone = np.concatenate([np.sin(2 * np.pi * 660 * t), np.sin(2 * np.pi * 990 * t)])
    idx = np.arange(tone.size)
    ramp = np.minimum(1.0, np.minimum(idx, tone.size - idx) / (0.008 * sr))
    return (0.3 * tone * ramp).astype(np.float32)


def wake_loop(read_frame, sr, process, get_cfg, beep=lambda: None,
              flush=lambda: None, stop=lambda: False, report=None, muted=lambda: False,
              refine=None):
    """Ядро wake word; тестируется без микрофона.
    read_frame() -> bytes (int16 mono, частота sr) | b"" (пока пусто) | None (стоп).
    process(question) — LLM + TTS + проигрывание; блокирует (себя в это время не слушаем).
    get_cfg() -> текущие настройки: wake word можно менять на лету.
    report(**поля) — обновление состояния для GUI.
    muted() -> True, пока ответ звучит из браузера (иначе ассистент услышит сам себя).
    refine(pcm, sr) -> текст той же фразы от Whisper ("" — оставить текст Vosk)."""
    report = report or (lambda **kv: WAKE_STATE.update(kv))
    model = load_stt()

    def new_rec():
        r = vosk.KaldiRecognizer(model, sr)  # Vosk сам приведёт частоту к 16 кГц
        r.SetWords(True)  # время каждого слова — чтобы вырезать фразу для Whisper
        return r

    rec = new_rec()
    t = 0.0               # сколько секунд аудио обработано
    rec_t0 = 0.0          # с какого момента слушает текущий распознаватель (время слов Vosk — от него)
    ring = collections.deque()  # (начало, сэмплы) — последние RING_SEC секунд звука
    level = -90.0
    loud_at = 0.0         # когда последний раз с микрофона шли не нули
    raw_wake, phrases = None, []
    pending_until = None  # был одиночный wake word — ждём вопрос до этого момента
    was_muted = False

    def ring_slice(a, b):
        parts = []
        for t_start, smp in ring:
            t_end = t_start + smp.size / sr
            if t_end <= a or t_start >= b:
                continue
            parts.append(smp[max(0, int((a - t_start) * sr)):int(math.ceil((b - t_start) * sr))])
        return np.concatenate(parts) if parts else np.zeros(0, np.int16)

    def improve(question, winfo, first_word):
        """Текст вопроса от Whisper по звуку этой фразы; при сбое — текст Vosk."""
        if refine is None or not winfo:
            return question
        a = rec_t0 + winfo[first_word]["start"] - 0.3
        b = rec_t0 + winfo[-1]["end"] + 0.4
        try:
            better = refine(ring_slice(a, b), sr)
        except Exception as e:
            report(stt_warn=f"Whisper не сработал ({e}) — вопрос распознан Vosk.")
            print(f"[wake] Whisper не сработал: {e}", flush=True)
            return question
        report(stt_warn="")
        better = strip_wake_text(better, phrases) if better else ""
        if better:
            print(f"[wake] Vosk: «{question}» → Whisper: «{better}»", flush=True)
        return better or question

    report(phase="waiting", heard="", info="")
    while not stop():
        fr = read_frame()
        if fr is None:
            return
        if not fr:
            continue
        if muted():
            was_muted = True
            continue
        if was_muted:  # после озвучки — с чистого листа
            was_muted = False
            rec, rec_t0 = new_rec(), t
        cur = (get_cfg().get("wake_word") or "").strip() or "ассистент"
        if cur != raw_wake:
            raw_wake, phrases = cur, parse_wake_words(cur)
            miss = missing_wake_words(model, phrases)
            report(wake=" / ".join("«" + " ".join(p) + "»" for p in phrases),
                   warn=("Слов " + ", ".join(f"«{w}»" for w in miss)
                         + " нет в словаре модели — она их не распознает, выберите другое wake word."
                         ) if miss else "")
        samples = np.frombuffer(fr, dtype=np.int16)
        ring.append((t, samples))
        t += samples.size / sr
        while ring and ring[0][0] < t - RING_SEC:
            ring.popleft()
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) if samples.size else 0.0
        db = 20 * math.log10(max(rms, 1.0) / 32768.0)
        level = max(db, level - 2.0)  # индикатор: быстрая атака, плавный спад
        if db > SILENT_DB:
            loud_at = t
        report(level=level, silent=t - loud_at > 5.0)

        partial = ""
        if rec.AcceptWaveform(fr):
            res = json.loads(rec.Result())
            text = res.get("text", "").strip()
            winfo = res.get("result") or []
            question, first_word = None, 0
            if text:
                report(heard=text)
                print(f"[wake] слышу: {text}", flush=True)
                words = text.split()
                if len(winfo) != len(words):
                    winfo = []
                pos = find_wake(words, phrases)
                if pos is not None:
                    question, first_word = " ".join(words[pos[1]:]), pos[0]
                    if not question:  # только wake word: сигнал и ждём вопрос
                        pending_until = t + QUESTION_TIMEOUT
                        report(phase="listening", info="")
                        # распознаватель НЕ пересоздаём и очередь не чистим: если вопрос
                        # начали говорить сразу, его начало не потеряется; сам сигнал
                        # Vosk словами не считает (проверено тестом с эхом сигнала)
                        beep()
                        continue
                elif pending_until is not None and len(text) > 2:
                    question = text
            if question:
                pending_until = None
                question = improve(question, winfo, first_word)
                report(phase="thinking", info=f"Вопрос: «{question}»")
                try:
                    process(question)
                    report(info=f"Последний вопрос: «{question}»")
                except Exception as e:
                    report(info=f"Ошибка ответа: {e}")
                    print(f"[wake] ошибка ответа: {e}", flush=True)
                flush()               # выкинуть всё, что микрофон слышал во время ответа (эхо)
                rec, rec_t0 = new_rec(), t
                ring.clear()
                report(phase="waiting", heard="")
                continue
            if pending_until is None:
                report(phase="waiting")
        else:
            partial = json.loads(rec.PartialResult()).get("partial", "").strip()
            if partial:
                report(heard=partial + "…")
                if pending_until is None:
                    report(phase="listening" if find_wake(partial.split(), phrases) else "waiting")
        if pending_until is not None and t > pending_until and not partial:
            pending_until = None
            report(phase="waiting", info="Вопрос не прозвучал — снова жду wake word.")


def list_input_devices():
    """Микрофоны для списка — только из хост-API по умолчанию: на Windows иначе
    каждый микрофон виден 3–4 раза (MME / DirectSound / WASAPI / WDM-KS)."""
    names = [DEFAULT_DEVICE]
    try:
        import sounddevice as sd
        try:
            api = sd.query_devices(kind="input")["hostapi"]
        except Exception:
            api = None
        for d in sd.query_devices():
            if (d["max_input_channels"] > 0 and (api is None or d["hostapi"] == api)
                    and d["name"] not in names):
                names.append(d["name"])
    except Exception:
        pass
    return names


def resolve_input_device(sd, name):
    """Имя из настроек -> (индекс или None, имя для показа, примечание)."""
    note = ""
    if name and name != DEFAULT_DEVICE:
        for i, d in enumerate(sd.query_devices()):
            if d["name"] == name and d["max_input_channels"] > 0:
                return i, d["name"], ""
        note = f"Микрофон «{name}» не найден — использую системный по умолчанию."
    return None, sd.query_devices(kind="input")["name"], note


def pick_input_format(sd, dev):
    """Частота и число каналов, которые устройство точно примет. Сначала родная
    частота устройства (как в официальном примере Vosk): распознаватель Vosk
    принимает любую частоту и сам ресемплирует."""
    info = sd.query_devices(dev, "input")
    rates = []
    for r in (int(info["default_samplerate"]), 16000, 48000, 44100):
        if r > 0 and r not in rates:
            rates.append(r)
    chans = [1] + ([2] if info["max_input_channels"] >= 2 else [])
    err = None
    for ch in chans:
        for r in rates:
            try:
                sd.check_input_settings(device=dev, channels=ch, dtype="int16", samplerate=r)
                return r, ch
            except Exception as e:
                err = e
    raise RuntimeError(f"«{info['name']}» не принимает ни один формат записи ({err})")


def _wake_main(gen, stop_ev):
    def report(**kv):
        if _wake["gen"] != gen:
            return  # поток уже заменён новым
        if "phase" in kv and kv["phase"] != WAKE_STATE.get("phase"):
            extra = kv.get("info") or ""
            print(f"[wake] {kv['phase']} {extra}".rstrip(), flush=True)
        if kv.get("phase") == "waiting":
            pc_control.PLAYER.unduck()  # снова ждём wake word — фильму прежнюю громкость
        WAKE_STATE.update(kv)

    def get_cfg():
        return _wake["cfg"]

    report(phase="starting", info="", warn="", note="", stt_warn="", heard="", device="",
           silent=False)
    try:
        import sounddevice as sd
    except Exception as e:
        return report(phase="error", info=f"Не загрузился sounddevice: {e}")
    try:
        load_stt()
    except Exception as e:
        return report(phase="error", info=f"Модель распознавания не загрузилась: {e}")
    try:
        dev, dev_name, note = resolve_input_device(sd, get_cfg().get("wake_device"))
        sr, ch = pick_input_format(sd, dev)
    except Exception as e:
        return report(phase="error", info=f"Микрофон недоступен: {e}")

    audio_q = queue.Queue()

    def cb(indata, frames, time_info, status):
        data = bytes(indata)
        if ch > 1:  # стерео -> моно
            a = np.frombuffer(data, dtype=np.int16).reshape(-1, ch).astype(np.int32)
            data = (a.sum(axis=1) // ch).astype(np.int16).tobytes()
        audio_q.put(data)

    def read_frame():
        if stop_ev.is_set():
            return None
        try:
            return audio_q.get(timeout=0.5)
        except queue.Empty:
            return b""

    def flush():
        while True:
            try:
                audio_q.get_nowait()
            except queue.Empty:
                return

    beep_wav = make_beep()

    def beep():
        pc_control.PLAYER.duck()  # фильм потише, пока звучит вопрос
        try:
            sd.play(beep_wav, SAMPLE_RATE_TTS)
            sd.wait()
        except Exception:
            pass

    def refine(pcm, pcm_sr):
        cfg = get_cfg()
        if cfg.get("stt_engine") != STT_WHISPER:
            return ""
        report(phase="recognizing", info="")
        return whisper_pcm(resample_int16(pcm, pcm_sr), cfg)

    def play_now(arr):
        sd.play(arr, SAMPLE_RATE_TTS)
        sd.wait()

    def process(question):
        def on_status(text):
            if text.startswith("🌐"):
                report(phase="searching", info=text)
            elif text:
                report(phase="thinking", info=text)

        pc_control.PLAYER.duck()  # фильм потише, пока ассистент отвечает
        # первые предложения звучат, пока модель дописывает остальное
        answer, _ = answer_and_speak(question, get_cfg(), play=play_now, on_status=on_status,
                                     on_speaking=lambda: report(phase="speaking", info=""))
        report(phase="speaking", info=f"Ответ: «{answer[:150]}»")
        time.sleep(0.3)  # хвост эха из динамиков; затем wake_loop чистит очередь

    try:
        with sd.RawInputStream(samplerate=sr, blocksize=int(sr * 0.1), device=dev,
                               channels=ch, dtype="int16", callback=cb):
            report(device=f"{dev_name} · {sr} Гц" + (f" · {ch} кан." if ch > 1 else ""),
                   note=note)
            wake_loop(read_frame, sr, process, get_cfg, beep=beep, flush=flush,
                      stop=stop_ev.is_set, report=report,
                      muted=lambda: time.time() < MUTE_UNTIL[0], refine=refine)
    except Exception as e:
        report(phase="error", info=f"Ошибка микрофона: {e}")


def start_wake():
    stop_wake()
    ev = threading.Event()
    _wake["gen"] += 1
    _wake["stop"] = ev
    t = threading.Thread(target=_wake_main, args=(_wake["gen"], ev), daemon=True)
    _wake["thread"] = t
    t.start()


def stop_wake():
    if _wake["stop"] is not None:
        _wake["stop"].set()
    t = _wake["thread"]
    if t and t.is_alive():
        t.join(timeout=3)
    _wake["thread"] = _wake["stop"] = None
    _wake["gen"] += 1  # отчёты старого потока больше не принимаются
    WAKE_STATE.update(phase="off", info="", heard="", level=-90.0, silent=False)


PHASE_TEXT = {
    "off": "⚪ Wake word выключен",
    "starting": "⏳ Запускаю микрофон и распознавание…",
    "waiting": "🟢 Жду wake word {wake}",
    "listening": "🟡 Слушаю вопрос…",
    "recognizing": "✍️ Распознаю вопрос (Whisper)…",
    "thinking": "🧠 Думаю…",
    "searching": "🌐 Ищу в интернете…",
    "speaking": "🔊 Отвечаю…",
    "error": "🔴 Wake word не работает",
}


def render_wake_status():
    st = dict(WAKE_STATE)
    phase = st.get("phase", "off")
    lines = ["**" + PHASE_TEXT.get(phase, phase).format(wake=st.get("wake") or "") + "**"]
    if phase == "off":
        return lines[0]
    if st.get("device"):
        lines.append(f"🎙 {st['device']}")
    if phase not in ("starting", "error"):
        lvl = float(st.get("level", -90.0))
        n = int(round(max(0.0, min(1.0, (lvl + 70.0) / 60.0)) * 20))
        lines.append(f"Уровень: `{'█' * n}{'░' * (20 - n)}` {lvl:.0f} дБ")
        if st.get("heard"):
            lines.append(f"Слышу: «{st['heard'][-100:]}»")
    for key in ("note", "warn", "stt_warn"):
        if st.get(key):
            lines.append("⚠️ " + st[key])
    if st.get("silent") and phase in ("waiting", "listening"):
        lines.append("⚠️ С микрофона идёт полная тишина (уровень не растёт, даже если говорить?). "
                     "Выберите другой микрофон в списке или включите в Windows: Параметры → "
                     "Конфиденциальность → Микрофон → «Разрешить классическим приложениям "
                     "доступ к микрофону».")
    if st.get("info"):
        lines.append(st["info"])
    return "  \n".join(lines)


ENGINE_PHASE_TEXT = {
    "off": "⚪ не запущен — запустится сам при первом вопросе",
    "download": "⏬ скачиваю…",
    "starting": "⏳ запускается…",
    "ready": "🟢 работает",
    "error": "🔴 ошибка",
}


def render_engine_status(cfg=None):
    cfg = cfg or _wake["cfg"]
    st = dict(ENGINE_STATE)
    phase = st.get("phase", "off")
    if not any(engine_needs(cfg)) and phase in ("off", "error"):
        return ("**⚙️ Локальный движок не нужен:** ответы — через LiteLLM, "
                "вопрос распознаёт Vosk.")
    lines = ["**⚙️ Движок KoboldCpp: " + ENGINE_PHASE_TEXT.get(phase, phase) + "**"]
    if st.get("device"):
        lines.append("🖥 Считает: " + st["device"])
    if st.get("models"):
        lines.append("📦 " + st["models"])
    speed = []
    if st.get("stt_sec") is not None:
        speed.append(f"Whisper {st['stt_sec']:.1f} с")
    if st.get("llm_sec") is not None:
        speed.append(f"LLM {st['llm_sec']:.1f} с"
                     + (f" ({st['llm_tps']:.0f} ток/с)" if st.get("llm_tps") else ""))
    if st.get("search_sec") is not None:
        speed.append(f"из них поиск {st['search_sec']:.1f} с")
    if speed:
        lines.append("⏱ Последний вопрос: " + " · ".join(speed))
    try:
        changed = phase == "ready" and _eng["light"] is not None \
            and engine_light_key(cfg) != _eng["light"]
    except Exception:
        changed = False
    if changed:
        lines.append("⚠️ Настройки движка изменены — применятся при следующем вопросе "
                     "(или нажмите «Перезапустить движок»).")
    if st.get("warn"):
        lines.append("⚠️ " + st["warn"])
    if st.get("info"):
        lines.append(st["info"])
    return "  \n".join(lines)


# ----------------------------- UI-обработчики --------------------------------

SETTING_FIELDS = ["voice", "llm_source", "hf_repo", "hf_file", "local_gguf_path",
                  "litellm_base", "litellm_key", "litellm_model", "n_ctx",
                  "wake_word", "wake_enabled", "wake_device",
                  "stt_engine", "compute", "whisper_model", "whisper_prompt", "gpu_layers",
                  "system_prompt", "max_tokens", "web_search",
                  "pc_control", "movies_dir", "vlc_path", "pc_aliases"]


def make_cfg(*values):
    cfg = dict(DEFAULT_SETTINGS)
    # ключи без полей в GUI (kobold_path) — из текущих настроек
    cfg.update({k: v for k, v in _wake["cfg"].items() if k not in SETTING_FIELDS})
    cfg.update(zip(SETTING_FIELDS, values))
    cfg["gpu_layers"] = _int(cfg.get("gpu_layers"), -1)
    cfg["max_tokens"] = max(64, min(8192, _int(cfg.get("max_tokens"), DEFAULT_MAX_TOKENS)))
    return cfg


def apply_settings(*cfg_values):
    """Сохранить в settings.json и сразу отдать слушателю wake word."""
    cfg = make_cfg(*cfg_values)
    save_settings(cfg)
    _wake["cfg"] = cfg
    return cfg


def respond(audio_path, text_input, *cfg_values):
    """Вопрос из браузера. Текст ответа появляется в «Диалоге» по мере генерации
    (его обновляет таймер poll_ui), озвучка — когда ответ готов."""
    cfg = apply_settings(*cfg_values)

    def note(msg):
        return (render_log() + "\n" + msg).strip()

    try:
        user_text = (text_input or "").strip()
        if not user_text:
            if not audio_path:
                return None, note("⚠️ Запишите голос или напишите текст."), gr.skip()
            live_update(status="✍️ Распознаю речь…")
            try:
                user_text = transcribe(audio_path, cfg)
            finally:
                live_update(status="")
        if not user_text:
            return None, note("⚠️ Ничего не расслышал — попробуйте ещё раз."), gr.skip()
        answer, wav = answer_and_speak(user_text, cfg)
        # ответ сейчас зазвучит из браузера — wake word не должен слушать сам себя
        MUTE_UNTIL[0] = time.time() + sf.info(wav).duration + 1.5
        return wav, render_log(), ""
    except Exception as e:
        return None, note(f"⚠️ Ошибка: {e}"), gr.skip()


def on_wake_toggle(*cfg_values):
    cfg = apply_settings(*cfg_values)
    if cfg["wake_enabled"]:
        start_wake()
    else:
        stop_wake()
    return render_wake_status()


def on_device_change(*cfg_values):
    cfg = apply_settings(*cfg_values)
    if cfg["wake_enabled"]:
        start_wake()  # перезапуск с новым микрофоном
    return render_wake_status()


def on_engine_setting(*cfg_values):
    """Режим/распознавание/видеокарта/модель Whisper сменились — движок подстроится фоном."""
    apply_settings(*cfg_values)
    engine_bg()


def on_restart_engine(*cfg_values):
    apply_settings(*cfg_values)
    engine_bg(restart=True)


def refresh_devices():
    t = _wake["thread"]
    if not (t and t.is_alive()):
        try:  # перечитать список устройств (новый USB-микрофон и т.п.)
            import sounddevice as sd
            sd._terminate()
            sd._initialize()
        except Exception:
            pass
    return gr.Dropdown(choices=list_input_devices())


def poll_ui(seen):
    """Таймер: живой статус wake word и движка + диалог, если он изменился."""
    seen = seen or {}
    status, ver, eng = render_wake_status(), HISTORY_VER[0], render_engine_status()
    return (
        status if status != seen.get("status") else gr.skip(),
        render_log() if ver != seen.get("ver") else gr.skip(),
        eng if eng != seen.get("eng") else gr.skip(),
        {"status": status, "ver": ver, "eng": eng},
    )


def autosave(*cfg_values):
    apply_settings(*cfg_values)


def on_pc_refresh(*cfg_values):
    """Пересканировать папку фильмов и меню «Пуск», показать, что найдено."""
    cfg = apply_settings(*cfg_values)
    pc_control.reset_index()
    return pc_control.summary(cfg)


def clear_history():
    with HISTORY_LOCK:
        HISTORY.clear()
        HISTORY_VER[0] += 1
    return ""


# ----------------------------- UI --------------------------------------------

def build_ui():
    s = load_settings()
    devices = list_input_devices()
    if s["wake_device"] not in devices:
        devices.append(s["wake_device"])
    whisper_choices = list(WHISPER_MODELS)
    if s["whisper_model"] not in whisper_choices:
        whisper_choices.append(s["whisper_model"])
    with gr.Blocks(title="RU Voice Assistant") as demo:
        gr.Markdown(
            "# 🎙️ RU Voice Assistant\n"
            "Микрофон → Whisper / Vosk → LLM → Silero. Настройки сохраняются автоматически "
            "в `settings.json` рядом с программой."
        )
        mode = gr.Radio([SRC_HF, SRC_LOCAL, SRC_LITELLM],
                        value=s["llm_source"], label="Движок ответов")
        with gr.Row():
            stt = gr.Radio([STT_WHISPER, STT_VOSK], value=s["stt_engine"],
                           label="Распознавание вопроса")
            compute = gr.Radio([COMPUTE_GPU, COMPUTE_CPU], value=s["compute"],
                               label="Где считать LLM и Whisper")
        web_search = gr.Radio(WEB_MODES, value=s["web_search"], label="🌐 Поиск в интернете")
        engine_status = gr.Markdown(render_engine_status(s))
        with gr.Accordion("🧠 Характер и длина ответов (системный промпт)", open=False):
            system_prompt = gr.Textbox(
                label="Системный промпт — инструкция для модели", value=s["system_prompt"],
                lines=6, max_lines=20,
                info="Пустое поле — стандартный промпт. Текущие дата и время и правила "
                     "поиска добавляются к нему автоматически.")
            with gr.Row():
                preset_buttons = [gr.Button(name, size="sm") for name in PROMPT_PRESETS]
            max_tokens = gr.Slider(
                128, 4096, value=int(s["max_tokens"]), step=64,
                label="Максимальная длина ответа (токенов)",
                info="≈ 1000 токенов — 2500–3000 знаков, 2–3 минуты речи. Для локальной "
                     "модели не больше половины контекста.")
        with gr.Accordion("Настройки моделей", open=False):
            with gr.Row():
                hf_repo = gr.Textbox(label="HF репозиторий", value=s["hf_repo"],
                                     placeholder="например, bartowski/...-GGUF")
                hf_file = gr.Textbox(label="Файл .gguf", value=s["hf_file"])
            with gr.Row():
                local_gguf = gr.Textbox(label="Или путь к локальному .gguf",
                                        value=s["local_gguf_path"],
                                        placeholder=r"C:\models\model-q4_k_m.gguf")
                n_ctx = gr.Slider(2048, 16384, value=int(s["n_ctx"]), step=1024,
                                  label="Контекст (токенов)")
            with gr.Row():
                whisper_model = gr.Dropdown(
                    whisper_choices, value=s["whisper_model"], allow_custom_value=True,
                    label="Модель Whisper (из ggerganov/whisper.cpp или путь к .bin)",
                    info="large-v3-turbo — для видеокарты; на процессоре быстрее "
                         "ggml-small-q8_0.bin, но английские слова она узнаёт хуже")
                gpu_layers = gr.Number(value=s["gpu_layers"], precision=0,
                                       label="Слоёв LLM на видеокарте",
                                       info="−1 = авто (сколько влезет в видеопамять), "
                                            "0 = LLM на процессоре")
            whisper_prompt = gr.Textbox(
                label="Подсказка для Whisper — термины, которые вы часто говорите",
                value=s["whisper_prompt"], lines=2)
            with gr.Row():
                base_url = gr.Textbox(label="LiteLLM Base URL", value=s["litellm_base"])
                api_key = gr.Textbox(label="LiteLLM API key", type="password",
                                     value=s["litellm_key"])
                model_name = gr.Textbox(label="LiteLLM модель", value=s["litellm_model"],
                                        placeholder="например, gpt-4o-mini")
            restart = gr.Button("🔄 Перезапустить движок")
        with gr.Row():
            audio = gr.Audio(sources=["microphone"], type="filepath",
                             label="🎤 Голосовой вопрос")
            text_input = gr.Textbox(label="…или вопрос текстом",
                                    placeholder="Напишите и нажмите Enter",
                                    submit_btn=True)
        voice = gr.Dropdown(VOICES, value=s["voice"], label="Голос ассистента")
        with gr.Accordion("🔔 Голосовая активация (wake word)", open=True):
            with gr.Row(equal_height=True):
                wake_enabled = gr.Checkbox(value=bool(s["wake_enabled"]), scale=1,
                                           label="Слушать микрофон компьютера")
                wake_word = gr.Textbox(label="Wake word — можно несколько через запятую",
                                       value=s["wake_word"], scale=2)
                wake_device = gr.Dropdown(devices, value=s["wake_device"], scale=2,
                                          label="Микрофон (🔄 — обновить список)",
                                          allow_custom_value=True)
                refresh = gr.Button("🔄", scale=0, min_width=48)
            wake_status = gr.Markdown(render_wake_status())
            gr.Markdown("Скажите одной фразой «Ассистент, какая погода?» — или скажите "
                        "«Ассистент», дождитесь сигнала и задайте вопрос.")
        with gr.Accordion("🖥 Управление компьютером", open=False):
            pc_enabled = gr.Checkbox(value=bool(s["pc_control"]),
                                     label="Выполнять голосовые команды: фильмы, программы, громкость")
            with gr.Row():
                movies_dir = gr.Textbox(label="Папка с фильмами", value=s["movies_dir"],
                                        placeholder=r"D:\Фильмы")
                vlc_path = gr.Textbox(label="Путь к vlc.exe (пусто — найти автоматически)",
                                      value=s["vlc_path"],
                                      placeholder=r"C:\Program Files\VideoLAN\VLC\vlc.exe")
            pc_aliases = gr.Textbox(
                label="Свои названия: по одному на строку, «как говорю = что открыть»",
                value=s["pc_aliases"], lines=3,
                placeholder="бегущий по лезвию = Blade Runner\nбраузер = "
                            "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                info="Справа — название фильма, путь к видео, программе или ярлыку, адрес сайта.")
            with gr.Row(equal_height=True):
                pc_status = gr.Markdown(pc_control.summary(s))
                pc_refresh = gr.Button("🔄 Обновить списки", size="sm", scale=0)
            gr.Markdown(
                "Примеры: «Ассистент, включи Бегущий по лезвию на полный экран», «…пауза», "
                "«…продолжи», «…перемотай вперёд на 5 минут», «…громче», «…выключи фильм», "
                "«…запусти телеграм», «…заблокируй компьютер», «…выключи компьютер» "
                "(переспросит, ответьте «Ассистент, да»). Если похожих фильмов несколько, "
                "ассистент спросит какой — ответьте «Ассистент, второй» или назовите год.")
        with gr.Row():
            clear = gr.Button("🗑 Сбросить диалог")
        reply = gr.Audio(label="Ответ (озвучка)", type="filepath", autoplay=True)
        log = gr.Textbox(label="Диалог", lines=12, max_lines=40, autoscroll=True)
        seen = gr.State({})

        cfg_inputs = [voice, mode, hf_repo, hf_file, local_gguf, base_url, api_key,
                      model_name, n_ctx, wake_word, wake_enabled, wake_device,
                      stt, compute, whisper_model, whisper_prompt, gpu_layers,
                      system_prompt, max_tokens, web_search,
                      pc_enabled, movies_dir, vlc_path, pc_aliases]
        respond_inputs = [audio, text_input] + cfg_inputs
        respond_outputs = [reply, log, text_input]
        engine_controls = (mode, stt, compute, whisper_model)
        # индикатор загрузки — только на плеере: «Диалог» под ним показывает ответ вживую
        busy = ({"show_progress_on": [reply]}
                if "show_progress_on" in inspect.signature(text_input.submit).parameters
                else {"show_progress": "minimal"})

        text_input.submit(respond, inputs=respond_inputs, outputs=respond_outputs, **busy)
        audio.stop_recording(respond, inputs=respond_inputs, outputs=respond_outputs, **busy)
        for button, preset in zip(preset_buttons, PROMPT_PRESETS.values()):
            button.click(lambda p=preset: p, outputs=[system_prompt])
        wake_enabled.change(on_wake_toggle, inputs=cfg_inputs, outputs=[wake_status])
        wake_device.change(on_device_change, inputs=cfg_inputs, outputs=[wake_status])
        refresh.click(refresh_devices, outputs=[wake_device])
        for comp in engine_controls:
            comp.change(on_engine_setting, inputs=cfg_inputs, outputs=[])
        n_ctx.release(on_engine_setting, inputs=cfg_inputs, outputs=[])
        restart.click(on_restart_engine, inputs=cfg_inputs, outputs=[])
        for comp in cfg_inputs:
            if comp not in (wake_enabled, wake_device) + engine_controls:
                comp.change(autosave, inputs=cfg_inputs, outputs=[])
        clear.click(clear_history, outputs=[log])
        pc_refresh.click(on_pc_refresh, inputs=cfg_inputs, outputs=[pc_status])
        for comp in (movies_dir, vlc_path):
            comp.blur(on_pc_refresh, inputs=cfg_inputs, outputs=[pc_status])

        timer = gr.Timer(0.5)
        timer.tick(poll_ui, inputs=[seen], outputs=[wake_status, log, engine_status, seen],
                   show_progress="hidden")
    return demo


if __name__ == "__main__":
    settings = load_settings()
    _wake["cfg"] = settings
    atexit.register(_stop_engine_at_exit)
    if os.name != "nt":  # закрыли терминал или kill — atexit всё равно погасит KoboldCpp
        import signal as _signal  # (имя signal занято scipy.signal)

        def _exit_on_signal(*_):
            raise SystemExit(0)
        for _s in (_signal.SIGTERM, _signal.SIGHUP):
            _signal.signal(_s, _exit_on_signal)
    app = build_ui()
    if settings.get("wake_enabled"):
        start_wake()
    engine_bg()  # заранее поднять KoboldCpp — первый вопрос не будет ждать загрузки моделей
    app.launch(inbrowser=True, server_name="127.0.0.1", server_port=7861)
