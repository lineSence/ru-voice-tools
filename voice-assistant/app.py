#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RU Voice Assistant — локальный голосовой ассистент на русском.

STT : Vosk (русская модель, скачивается автоматически, ~45 МБ)
LLM : любая GGUF (Hugging Face repo+file или локальный .gguf) через llama-cpp-python,
      либо OpenAI-совместимый endpoint (LiteLLM-прокси)
TTS : Silero v5 (скачивается автоматически, ~140 МБ)
Wake word: Vosk keyword spotting через системный микрофон (sounddevice)

Всё работает на CPU, GPU не требуется. Настройки автосохраняются в settings.json.
"""

import os

# Фикс для Windows с прокси/VPN: localhost не должен идти через прокси,
# иначе Gradio падает с "startup-events failed (code 502)".
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

import gc
import inspect
import json
import math
import queue
import re
import tempfile
import threading

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

WAKE_EVENTS = queue.Queue()
_wake = {"thread": None, "stop": threading.Event()}
WAKE_FRAME = 4000  # сэмплов на кадр (0.25 с при 16 кГц)


def wake_loop(read_frame, play_audio, process, cfg,
              flush=lambda: None, stop=lambda: False):
    """Ядро wake word, тестируется без микрофона.
    read_frame() -> bytes (int16 16kHz mono) или None для остановки.
    play_audio(float32_array, sample_rate) — проигрывание ответа.
    process(user_text) — вызов пайплайна LLM+TTS."""
    model = load_stt()
    wake_word = (cfg["wake_word"] or "ассистент").strip().lower()
    WAKE_EVENTS.put(("status", f"Слушаю wake word «{wake_word}»..."))
    while not stop():
        # ensure_ascii=False обязателен: парсер Vosk не декодирует \uXXXX,
        # и кириллическое слово "пропадает" из грамматики
        rec = vosk.KaldiRecognizer(
            model, SAMPLE_RATE_STT, json.dumps([wake_word, "[unk]"], ensure_ascii=False))
        spotted = False
        while not stop() and not spotted:
            fr = read_frame()
            if fr is None:
                return
            if not fr:
                continue
            if rec.AcceptWaveform(fr):
                spotted = wake_word in json.loads(rec.Result()).get("text", "")
            else:
                spotted = wake_word in json.loads(rec.PartialResult()).get("partial", "")
        if stop():
            return
        flush()  # выкинуть накопившийся фон
        # отбросить ~0.6 с: договаривается хвост wake word, он не должен
        # попасть в распознавание вопроса ("...сбер" от "компьютер")
        tail = int(0.6 * SAMPLE_RATE_STT / WAKE_FRAME)
        for _ in range(tail):
            fr = read_frame()
            if fr is None:
                return
        rec2 = vosk.KaldiRecognizer(model, SAMPLE_RATE_STT)
        heard, silent, total = False, 0, 0
        while not stop():
            fr = read_frame()
            if fr is None:
                return
            if not fr:
                continue
            total += 1
            arr = np.frombuffer(fr, dtype=np.int16)
            level = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2))) if arr.size else 0.0
            if level > 400:
                heard, silent = True, 0
            else:
                silent += 1
            rec2.AcceptWaveform(fr)
            if heard and silent * WAKE_FRAME / SAMPLE_RATE_STT > 1.5:
                break
            if total * WAKE_FRAME / SAMPLE_RATE_STT > 30:
                break
        if stop():
            return
        text = json.loads(rec2.FinalResult()).get("text", "").strip()
        # страховка: если wake word всё же попало в начало — срезать
        if text.startswith(wake_word):
            text = text[len(wake_word):].strip()
        if not text:
            WAKE_EVENTS.put(("status", "Не расслышал вопрос, слушаю дальше..."))
            continue
        WAKE_EVENTS.put(("status", f"Услышал: «{text}». Думаю..."))
        process(text)  # во время ответа и озвучки захват приостановлен — защита от эха
        flush()
        WAKE_EVENTS.put(("status", f"Слушаю wake word «{wake_word}»..."))


def _wake_main(cfg):
    try:
        import sounddevice as sd
    except Exception as e:
        WAKE_EVENTS.put(("status", f"sounddevice недоступен: {e}"))
        return
    audio_q = queue.Queue()

    def cb(indata, frames, time_info, status):
        audio_q.put(bytes(indata))

    def read_frame():
        if _wake["stop"].is_set():
            return None
        try:
            return audio_q.get(timeout=0.5)
        except queue.Empty:
            return b""

    def play_audio(arr, sr):
        sd.play(arr, sr)
        sd.wait()

    def process(text):
        answer, wav = answer_and_speak(text, cfg)
        WAKE_EVENTS.put(("exchange", (text, answer, wav)))
        data, sr = sf.read(wav, dtype="float32", always_2d=True)
        play_audio(data, sr)

    def flush():
        while not audio_q.empty():
            try:
                audio_q.get_nowait()
            except queue.Empty:
                break

    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE_STT, channels=1,
                               dtype="int16", blocksize=WAKE_FRAME, callback=cb):
            wake_loop(read_frame, play_audio, process, cfg,
                      flush=flush, stop=_wake["stop"].is_set)
    except Exception as e:
        WAKE_EVENTS.put(("status", f"Wake word остановлен: {e}"))


def start_wake(cfg):
    stop_wake()
    _wake["stop"].clear()
    t = threading.Thread(target=_wake_main, args=(dict(cfg),), daemon=True)
    _wake["thread"] = t
    t.start()


def stop_wake():
    _wake["stop"].set()
    t = _wake["thread"]
    if t and t.is_alive():
        t.join(timeout=5)
    _wake["thread"] = None


# ----------------------------- UI-обработчики --------------------------------

SETTING_FIELDS = ["voice", "llm_source", "hf_repo", "hf_file", "local_gguf_path",
                  "litellm_base", "litellm_key", "litellm_model", "n_ctx", "wake_word"]


def make_cfg(*values):
    return dict(zip(SETTING_FIELDS, values))


def respond(audio_path, text_input, *cfg_values, progress=gr.Progress()):
    cfg = make_cfg(*cfg_values)
    save_settings(cfg)
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
        progress(0.9, desc="Готово")
        return wav, render_log(), ""
    except Exception as e:
        return None, f"Ошибка: {e}", ""


def toggle_wake(enabled, *cfg_values):
    cfg = make_cfg(*cfg_values)
    save_settings(cfg)
    if enabled:
        start_wake(cfg)
        return f"Включаю… wake word «{cfg['wake_word']}» (микрофон этого компьютера)"
    stop_wake()
    return "Wake word выключен"


def poll_wake():
    out_audio, status, log = None, None, None
    while True:
        try:
            kind, payload = WAKE_EVENTS.get_nowait()
        except queue.Empty:
            break
        if kind == "status":
            status = payload
        elif kind == "exchange":
            _, _, wav = payload
            out_audio, log = wav, render_log()
    return (
        out_audio if out_audio else gr.skip(),
        log if log is not None else gr.skip(),
        status if status is not None else gr.skip(),
    )


def autosave(*cfg_values):
    save_settings(make_cfg(*cfg_values))


def clear_history():
    with HISTORY_LOCK:
        HISTORY.clear()
    return ""


# ----------------------------- UI --------------------------------------------

def build_ui():
    s = load_settings()
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
        with gr.Row():
            voice = gr.Dropdown(VOICES, value=s["voice"], label="Голос ассистента")
            wake_enabled = gr.Checkbox(value=bool(s["wake_enabled"]),
                                       label="🔔 Wake word (системный микрофон)")
            wake_word = gr.Textbox(label="Wake word", value=s["wake_word"])
        wake_status = gr.Markdown("Wake word выключен")
        with gr.Row():
            clear = gr.Button("🗑 Сбросить диалог")
        reply = gr.Audio(label="Ответ (озвучка)", type="filepath", autoplay=True)
        log = gr.Textbox(label="Диалог", lines=10)

        cfg_inputs = [voice, mode, hf_repo, hf_file, local_gguf,
                      base_url, api_key, model_name, n_ctx, wake_word]
        respond_inputs = [audio, text_input] + cfg_inputs
        respond_outputs = [reply, log, text_input]

        text_input.submit(respond, inputs=respond_inputs, outputs=respond_outputs)
        audio.stop_recording(respond, inputs=respond_inputs, outputs=respond_outputs)
        wake_enabled.change(toggle_wake, inputs=[wake_enabled] + cfg_inputs,
                            outputs=[wake_status])
        for comp in cfg_inputs:
            comp.change(autosave, inputs=cfg_inputs, outputs=[])
        clear.click(clear_history, outputs=[log])

        timer = gr.Timer(1.0)
        timer.tick(poll_wake, outputs=[reply, log, wake_status])
    return demo


if __name__ == "__main__":
    app = build_ui()
    s = load_settings()
    if s.get("wake_enabled"):
        start_wake(s)
    app.launch(inbrowser=True, server_name="127.0.0.1", server_port=7861)