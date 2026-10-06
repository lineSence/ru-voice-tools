#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RU Voice Assistant — локальный голосовой ассистент на русском.

STT : Vosk (русская модель, скачивается автоматически, ~45 МБ)
LLM : любая GGUF (Hugging Face repo+file или локальный .gguf) через llama-cpp-python,
      либо OpenAI-совместимый endpoint (LiteLLM-прокси)
TTS : Silero v5 (скачивается автоматически, ~140 МБ)
Wake word: Vosk расшифровывает микрофон компьютера (sounddevice), слово-триггер
           ищется в тексте нечётко; вопрос — та же фраза после него или следующая

Всё работает на CPU, GPU не требуется. Настройки автосохраняются в settings.json.
"""

import os

# Фикс для Windows с прокси/VPN: localhost не должен идти через прокси,
# иначе Gradio падает с "startup-events failed (code 502)".
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

import difflib
import gc
import inspect
import json
import math
import queue
import re
import tempfile
import threading
import time

import numpy as np
import soundfile as sf
from scipy import signal
import torch
import gradio as gr
import requests
import vosk

SAMPLE_RATE_TTS = 48000
SAMPLE_RATE_STT = 16000
VOICES = ["aidar", "baya", "kseniya", "xenia", "eugene", "random"]

SRC_HF = "Hugging Face GGUF"
SRC_LOCAL = "Локальный файл .gguf"
SRC_LITELLM = "LiteLLM-прокси (облако)"
DEFAULT_DEVICE = "Системный по умолчанию"

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
    "n_ctx": 4096,
    "wake_enabled": False,
    "wake_word": "ассистент",
    "wake_device": DEFAULT_DEVICE,
}

SYSTEM_PROMPT = (
    "Ты — дружелюбный русскоязычный голосовой помощник. "
    "Отвечай коротко: одно-три предложения, простым разговорным языком. "
    "Никаких списков, markdown, эмодзи, скобок и ссылок — ответ будет зачитан вслух. "
    "Если вопрос непонятен, вежливо переспроси."
)

MAX_TOKENS = 220
HISTORY_KEEP = 6  # пар реплик

_stt = {"model": None}
_llm = {"key": None, "obj": None}
_tts = {"model": None}

HISTORY = []
HISTORY_VER = [0]  # растёт при каждом изменении диалога — таймер GUI видит новое
HISTORY_LOCK = threading.Lock()
PIPE_LOCK = threading.Lock()


# ----------------------------- Настройки ------------------------------------

def load_settings():
    s = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            s.update(json.load(f))
    except Exception:
        pass
    return s


def save_settings(s):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def history_snapshot():
    with HISTORY_LOCK:
        return list(HISTORY)


def history_append(user_text, answer):
    with HISTORY_LOCK:
        HISTORY.append({"role": "user", "content": user_text})
        HISTORY.append({"role": "assistant", "content": answer})
        del HISTORY[:-HISTORY_KEEP * 2]
        HISTORY_VER[0] += 1


def render_log():
    return "\n".join(
        ("Вы: " if m["role"] == "user" else "Ассистент: ") + m["content"]
        for m in history_snapshot()
    )


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


def transcribe(audio_path):
    """wav-файл -> текст (Vosk)."""
    model = load_stt()
    pcm = to_16k_mono_int16(audio_path).tobytes()
    rec = vosk.KaldiRecognizer(model, SAMPLE_RATE_STT)
    parts = []
    for i in range(0, len(pcm), 8000):
        if rec.AcceptWaveform(pcm[i:i + 8000]):
            parts.append(json.loads(rec.Result()).get("text", ""))
    parts.append(json.loads(rec.FinalResult()).get("text", ""))
    return " ".join(p for p in parts if p).strip()


# ----------------------------- LLM ------------------------------------------

def load_llm(cfg):
    """GGUF из Hugging Face (repo+file) или локальный файл. Без хардкода моделей."""
    key = (cfg["llm_source"], cfg["hf_repo"], cfg["hf_file"],
           cfg["local_gguf_path"], int(cfg["n_ctx"]))
    if _llm["key"] == key and _llm["obj"] is not None:
        return _llm["obj"]
    from llama_cpp import Llama
    if cfg["llm_source"] == SRC_LOCAL:
        path = cfg["local_gguf_path"].strip().strip('"')
        if not path or not os.path.isfile(path):
            raise RuntimeError("Укажите существующий путь к .gguf файлу.")
    else:
        from huggingface_hub import hf_hub_download
        if not cfg["hf_repo"].strip() or not cfg["hf_file"].strip():
            raise RuntimeError("Укажите репозиторий и имя файла GGUF на Hugging Face.")
        path = hf_hub_download(repo_id=cfg["hf_repo"].strip(), filename=cfg["hf_file"].strip())
    if _llm["obj"] is not None:
        del _llm["obj"]
        gc.collect()
    _llm["obj"] = Llama(
        model_path=path,
        n_ctx=int(cfg["n_ctx"]),
        n_threads=max(1, (os.cpu_count() or 4) - 1),
        n_gpu_layers=0,
        verbose=False,
    )
    _llm["key"] = key
    return _llm["obj"]


def build_messages(history, user_text):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(history[-HISTORY_KEEP * 2:])
    msgs.append({"role": "user", "content": user_text})
    return msgs


def answer_local(cfg, messages):
    out = load_llm(cfg).create_chat_completion(
        messages=messages, max_tokens=MAX_TOKENS, temperature=0.6
    )
    return out["choices"][0]["message"]["content"].strip()


def answer_litellm(cfg, messages):
    base, model_name = cfg["litellm_base"].strip(), cfg["litellm_model"].strip()
    if not base or not model_name:
        raise RuntimeError("Заполните Base URL и имя модели в настройках LiteLLM.")
    r = requests.post(
        base.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {cfg['litellm_key']}"} if cfg["litellm_key"] else {},
        json={"model": model_name, "messages": messages,
              "max_tokens": MAX_TOKENS, "temperature": 0.6},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


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


def clean_for_tts(text):
    text = re.sub(r"https?://\S+", "ссылка", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"[*_#`>~|]", " ", text)
    # эмодзи и прочие символы вне алфавита/пунктуации — TTS их не читает корректно
    text = re.sub(r"[^\w\s.,!?…:;+%№«»()\-—–'\"]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


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
        audio = model.apply_tts(text=chunk, speaker=voice, sample_rate=SAMPLE_RATE_TTS)
        arr = audio.numpy() if torch.is_tensor(audio) else np.asarray(audio)
        pieces.append(np.asarray(arr, dtype=np.float32).flatten())
        pieces.append(pause.copy())
    full = np.concatenate(pieces)
    out = os.path.join(tempfile.gettempdir(), "ru_assistant_reply.wav")
    sf.write(out, full, SAMPLE_RATE_TTS)
    return out


# ----------------------------- Общий пайплайн --------------------------------

def answer_and_speak(user_text, cfg):
    """LLM + TTS + история. Потокобезопасно (общий для GUI и wake word)."""
    with PIPE_LOCK:
        messages = build_messages(history_snapshot(), user_text)
        if cfg["llm_source"] == SRC_LITELLM:
            answer = answer_litellm(cfg, messages)
        else:
            answer = answer_local(cfg, messages)
        wav = speak(answer, cfg["voice"])
        history_append(user_text, answer)
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

QUESTION_TIMEOUT = 8.0   # сек ждать вопрос после одиночного wake word
SILENT_DB = -75.0        # тише — микрофон фактически отдаёт нули
WAKE_STATE = {"phase": "off", "level": -90.0, "heard": "", "info": "", "warn": "",
              "note": "", "device": "", "wake": "", "silent": False}
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
              flush=lambda: None, stop=lambda: False, report=None, muted=lambda: False):
    """Ядро wake word; тестируется без микрофона.
    read_frame() -> bytes (int16 mono, частота sr) | b"" (пока пусто) | None (стоп).
    process(question) — LLM + TTS + проигрывание; блокирует (себя в это время не слушаем).
    get_cfg() -> текущие настройки: wake word можно менять на лету.
    report(**поля) — обновление состояния для GUI.
    muted() -> True, пока ответ звучит из браузера (иначе ассистент услышит сам себя)."""
    report = report or (lambda **kv: WAKE_STATE.update(kv))
    model = load_stt()

    def new_rec():
        return vosk.KaldiRecognizer(model, sr)  # Vosk сам приведёт частоту к 16 кГц

    rec = new_rec()
    t = 0.0               # сколько секунд аудио обработано
    level = -90.0
    loud_at = 0.0         # когда последний раз с микрофона шли не нули
    raw_wake, phrases = None, []
    pending_until = None  # был одиночный wake word — ждём вопрос до этого момента
    was_muted = False
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
            rec = new_rec()
        cur = (get_cfg().get("wake_word") or "").strip() or "ассистент"
        if cur != raw_wake:
            raw_wake, phrases = cur, parse_wake_words(cur)
            miss = missing_wake_words(model, phrases)
            report(wake=" / ".join("«" + " ".join(p) + "»" for p in phrases),
                   warn=("Слов " + ", ".join(f"«{w}»" for w in miss)
                         + " нет в словаре модели — она их не распознает, выберите другое wake word."
                         ) if miss else "")
        samples = np.frombuffer(fr, dtype=np.int16)
        t += samples.size / sr
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) if samples.size else 0.0
        db = 20 * math.log10(max(rms, 1.0) / 32768.0)
        level = max(db, level - 2.0)  # индикатор: быстрая атака, плавный спад
        if db > SILENT_DB:
            loud_at = t
        report(level=level, silent=t - loud_at > 5.0)

        partial = ""
        if rec.AcceptWaveform(fr):
            text = json.loads(rec.Result()).get("text", "").strip()
            question = None
            if text:
                report(heard=text)
                print(f"[wake] слышу: {text}", flush=True)
                words = text.split()
                pos = find_wake(words, phrases)
                if pos is not None:
                    question = " ".join(words[pos[1]:])
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
                report(phase="thinking", info=f"Вопрос: «{question}»")
                try:
                    process(question)
                    report(info=f"Последний вопрос: «{question}»")
                except Exception as e:
                    report(info=f"Ошибка ответа: {e}")
                    print(f"[wake] ошибка ответа: {e}", flush=True)
                flush()               # выкинуть всё, что микрофон слышал во время ответа (эхо)
                rec = new_rec()
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
        WAKE_STATE.update(kv)

    def get_cfg():
        return _wake["cfg"]

    report(phase="starting", info="", warn="", note="", heard="", device="", silent=False)
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
        try:
            sd.play(beep_wav, SAMPLE_RATE_TTS)
            sd.wait()
        except Exception:
            pass

    def process(question):
        answer, wav = answer_and_speak(question, get_cfg())
        report(phase="speaking", info=f"Ответ: «{answer[:150]}»")
        data, wav_sr = sf.read(wav, dtype="float32", always_2d=True)
        sd.play(data, wav_sr)
        sd.wait()
        time.sleep(0.3)  # хвост эха из динамиков; затем wake_loop чистит очередь

    try:
        with sd.RawInputStream(samplerate=sr, blocksize=int(sr * 0.1), device=dev,
                               channels=ch, dtype="int16", callback=cb):
            report(device=f"{dev_name} · {sr} Гц" + (f" · {ch} кан." if ch > 1 else ""),
                   note=note)
            wake_loop(read_frame, sr, process, get_cfg, beep=beep, flush=flush,
                      stop=stop_ev.is_set, report=report,
                      muted=lambda: time.time() < MUTE_UNTIL[0])
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
    "thinking": "🧠 Думаю…",
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
    for key in ("note", "warn"):
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


# ----------------------------- UI-обработчики --------------------------------

SETTING_FIELDS = ["voice", "llm_source", "hf_repo", "hf_file", "local_gguf_path",
                  "litellm_base", "litellm_key", "litellm_model", "n_ctx",
                  "wake_word", "wake_enabled", "wake_device"]


def make_cfg(*values):
    cfg = dict(DEFAULT_SETTINGS)
    cfg.update(zip(SETTING_FIELDS, values))
    return cfg


def apply_settings(*cfg_values):
    """Сохранить в settings.json и сразу отдать слушателю wake word."""
    cfg = make_cfg(*cfg_values)
    save_settings(cfg)
    _wake["cfg"] = cfg
    return cfg


def respond(audio_path, text_input, *cfg_values, progress=gr.Progress()):
    cfg = apply_settings(*cfg_values)
    try:
        user_text = (text_input or "").strip()
        if not user_text:
            if not audio_path:
                return None, "Запишите голос или напишите текст.", ""
            progress(0.1, desc="Распознаю речь (первый раз скачается ~45 МБ)...")
            user_text = transcribe(audio_path)
        if not user_text:
            return None, "Ничего не расслышал — попробуйте ещё раз.", ""
        progress(0.35, desc="Думаю (первый запуск скачает модель, это долго)...")
        answer, wav = answer_and_speak(user_text, cfg)
        # ответ сейчас зазвучит из браузера — wake word не должен слушать сам себя
        MUTE_UNTIL[0] = time.time() + sf.info(wav).duration + 1.5
        progress(0.9, desc="Готово")
        return wav, render_log(), ""
    except Exception as e:
        return None, f"Ошибка: {e}", ""


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
    """Таймер: живой статус wake word + диалог, если он изменился."""
    seen = seen or {}
    status, ver = render_wake_status(), HISTORY_VER[0]
    return (
        status if status != seen.get("status") else gr.skip(),
        render_log() if ver != seen.get("ver") else gr.skip(),
        {"status": status, "ver": ver},
    )


def autosave(*cfg_values):
    apply_settings(*cfg_values)


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
    with gr.Blocks(title="RU Voice Assistant") as demo:
        gr.Markdown(
            "# 🎙️ RU Voice Assistant\n"
            "Микрофон → Vosk → LLM → Silero. Настройки сохраняются автоматически "
            "в `settings.json` рядом с программой."
        )
        mode = gr.Radio([SRC_HF, SRC_LOCAL, SRC_LITELLM],
                        value=s["llm_source"], label="Движок ответов")
        with gr.Accordion("Настройки модели", open=False):
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
                base_url = gr.Textbox(label="LiteLLM Base URL", value=s["litellm_base"])
                api_key = gr.Textbox(label="LiteLLM API key", type="password",
                                     value=s["litellm_key"])
                model_name = gr.Textbox(label="LiteLLM модель", value=s["litellm_model"],
                                        placeholder="например, gpt-4o-mini")
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
        with gr.Row():
            clear = gr.Button("🗑 Сбросить диалог")
        reply = gr.Audio(label="Ответ (озвучка)", type="filepath", autoplay=True)
        log = gr.Textbox(label="Диалог", lines=10)
        seen = gr.State({})

        cfg_inputs = [voice, mode, hf_repo, hf_file, local_gguf, base_url, api_key,
                      model_name, n_ctx, wake_word, wake_enabled, wake_device]
        respond_inputs = [audio, text_input] + cfg_inputs
        respond_outputs = [reply, log, text_input]

        text_input.submit(respond, inputs=respond_inputs, outputs=respond_outputs)
        audio.stop_recording(respond, inputs=respond_inputs, outputs=respond_outputs)
        wake_enabled.change(on_wake_toggle, inputs=cfg_inputs, outputs=[wake_status])
        wake_device.change(on_device_change, inputs=cfg_inputs, outputs=[wake_status])
        refresh.click(refresh_devices, outputs=[wake_device])
        for comp in cfg_inputs:
            if comp not in (wake_enabled, wake_device):
                comp.change(autosave, inputs=cfg_inputs, outputs=[])
        clear.click(clear_history, outputs=[log])

        timer = gr.Timer(0.5)
        timer.tick(poll_ui, inputs=[seen], outputs=[wake_status, log, seen],
                   show_progress="hidden")
    return demo


if __name__ == "__main__":
    settings = load_settings()
    _wake["cfg"] = settings
    app = build_ui()
    if settings.get("wake_enabled"):
        start_wake()
    app.launch(inbrowser=True, server_name="127.0.0.1", server_port=7861)
