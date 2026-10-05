#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RU Voice Assistant — локальный голосовой ассистент на русском.

STT : Vosk (русская модель, скачивается автоматически, ~45 МБ)
LLM : локальная GGUF через llama-cpp-python (Qwen3-4B или Gemma-3-12B)
      либо любой OpenAI-совместимый endpoint (LiteLLM-прокси)
TTS : Silero v5 (скачивается автоматически, ~140 МБ)

Всё работает на CPU, GPU не требуется.
"""

import os

# Фикс для Windows с прокси/VPN: localhost не должен идти через прокси,
# иначе Gradio падает с "startup-events failed (code 502)".
# LiteLLM на localhost тоже пойдёт напрямую.
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

import gc
import inspect
import json
import math
import re
import tempfile

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

MODE_QWEN = "Локальная Qwen3-4B (быстро, ~2.5 ГБ)"
MODE_GEMMA = "Локальная Gemma-3-12B (умнее, медленно, ~7.5 ГБ)"
MODE_LITELLM = "LiteLLM-прокси (облако)"

LOCAL_MODELS = {
    MODE_QWEN: (
        "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
        "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    ),
    MODE_GEMMA: (
        "bartowski/google_gemma-3-12b-it-GGUF",
        "google_gemma-3-12b-it-Q4_K_M.gguf",
    ),
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


# ----------------------------- STT (Vosk) ----------------------------------

VOSK_NAME = "vosk-model-small-ru-0.22"
VOSK_URL = f"https://alphacephei.com/vosk/models/{VOSK_NAME}.zip"


def vosk_model_path():
    """Возвращает путь к модели Vosk; при необходимости скачивает её сам.

    Своя загрузка вместо vosk.Model(lang="ru"): у штатного загрузчика
    таймаут 10 секунд, и на медленном соединении скачивание обрывается.
    Здесь — длинные таймауты, повторы и докачка.
    """
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
                        "Не удалось скачать модель распознавания с alphacephei.com. "
                        "Скачайте её вручную в браузере: " + VOSK_URL + " — "
                        "и распакуйте в папку .cache\\vosk в домашнем каталоге, "
                        "чтобы получилось .cache\\vosk\\" + VOSK_NAME
                    )

    for _ in range(2):
        download()
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(cache)
            zip_path.unlink(missing_ok=True)
            return str(target)
        except zipfile.BadZipFile:
            zip_path.unlink(missing_ok=True)  # битый недокачанный файл — с нуля
    raise RuntimeError("Архив модели повреждён даже после повторного скачивания.")


def load_stt():
    if _stt["model"] is None:
        vosk.SetLogLevel(-1)
        _stt["model"] = vosk.Model(vosk_model_path())
    return _stt["model"]


def transcribe(audio_path):
    """wav-файл -> текст (Vosk, 16 кГц моно)."""
    model = load_stt()
    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != SAMPLE_RATE_STT:
        g = math.gcd(int(sr), SAMPLE_RATE_STT)
        data = signal.resample_poly(data, SAMPLE_RATE_STT // g, sr // g)
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    rec = vosk.KaldiRecognizer(model, SAMPLE_RATE_STT)
    parts = []
    for i in range(0, len(pcm), 8000):
        if rec.AcceptWaveform(pcm[i:i + 8000]):
            parts.append(json.loads(rec.Result()).get("text", ""))
    parts.append(json.loads(rec.FinalResult()).get("text", ""))
    return " ".join(p for p in parts if p).strip()


# ----------------------------- LLM -----------------------------------------

def load_llm(mode):
    """Загружает локальную GGUF-модель (скачивает при первом запуске)."""
    if _llm["key"] == mode and _llm["obj"] is not None:
        return _llm["obj"]
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    repo, fname = LOCAL_MODELS[mode]
    path = hf_hub_download(repo_id=repo, filename=fname)
    if _llm["obj"] is not None:
        del _llm["obj"]
        gc.collect()
    _llm["obj"] = Llama(
        model_path=path,
        n_ctx=4096,
        n_threads=max(1, (os.cpu_count() or 4) - 1),
        n_gpu_layers=0,  # CPU
        verbose=False,
    )
    _llm["key"] = mode
    return _llm["obj"]


def build_messages(history, user_text):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(history[-HISTORY_KEEP * 2:])
    msgs.append({"role": "user", "content": user_text})
    return msgs


def answer_local(mode, messages):
    llm = load_llm(mode)
    out = llm.create_chat_completion(
        messages=messages, max_tokens=MAX_TOKENS, temperature=0.6
    )
    return out["choices"][0]["message"]["content"].strip()


def answer_litellm(base_url, api_key, model_name, messages):
    if not base_url or not model_name:
        raise RuntimeError("Заполните Base URL и имя модели в настройках LiteLLM.")
    r = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        json={
            "model": model_name,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.6,
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ----------------------------- TTS (Silero) --------------------------------

def load_tts():
    if _tts["model"] is None:
        hub_kwargs = {}
        if "trust_repo" in inspect.signature(torch.hub.load).parameters:
            hub_kwargs["trust_repo"] = True
        result = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="v5_ru",
            verbose=False,
            **hub_kwargs,
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
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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


# ----------------------------- Главный обработчик ---------------------------

def respond(audio_path, text_input, mode, base_url, api_key, model_name,
            voice, history, progress=gr.Progress()):
    try:
        # 1. Получаем текст вопроса
        user_text = (text_input or "").strip()
        if not user_text:
            if not audio_path:
                return None, "Запишите голос или напишите текст.", history, ""
            progress(0.1, desc="Распознаю речь (первый раз скачается ~45 МБ)...")
            user_text = transcribe(audio_path)
        if not user_text:
            return None, "Ничего не расслышал — попробуйте ещё раз, чётче и громче.", history, ""

        # 2. Ответ LLM
        progress(0.35, desc="Думаю (первый запуск скачает модель, это долго)...")
        messages = build_messages(history, user_text)
        if mode == MODE_LITELLM:
            answer = answer_litellm(base_url, api_key, model_name, messages)
        else:
            answer = answer_local(mode, messages)

        # 3. Озвучка
        progress(0.75, desc="Озвучиваю ответ...")
        wav = speak(answer, voice)

        history = (history or []) + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ]
        log = "\n".join(
            ("Вы: " if m["role"] == "user" else "Ассистент: ") + m["content"]
            for m in history[-HISTORY_KEEP * 2:]
        )
        return wav, log, history, ""
    except Exception as e:
        return None, f"Ошибка: {e}", history, ""


def clear_history():
    return [], ""


# ----------------------------- UI ------------------------------------------

def build_ui():
    with gr.Blocks(title="RU Voice Assistant") as demo:
        gr.Markdown(
            "# 🎙️ RU Voice Assistant\n"
            "Полностью локальный голосовой ассистент: запись → распознавание (Vosk) → "
            "ответ (LLM) → озвучка (Silero). Нажмите на микрофон, говорите, остановите запись."
        )
        with gr.Row():
            mode = gr.Radio(
                [MODE_QWEN, MODE_GEMMA, MODE_LITELLM],
                value=MODE_QWEN,
                label="Движок ответов",
            )
        with gr.Accordion("Настройки LiteLLM (только для режима прокси)", open=False):
            with gr.Row():
                base_url = gr.Textbox(label="Base URL", value="http://127.0.0.1:4000/v1")
                api_key = gr.Textbox(label="API key", type="password")
                model_name = gr.Textbox(label="Модель", placeholder="например, gpt-4o-mini")
        with gr.Row():
            audio = gr.Audio(sources=["microphone"], type="filepath",
                             label="🎤 Голосовой вопрос")
            text_input = gr.Textbox(label="…или вопрос текстом",
                                    placeholder="Напишите и нажмите Enter",
                                    submit_btn=True)
        with gr.Row():
            voice = gr.Dropdown(VOICES, value="kseniya", label="Голос ассистента")
            clear = gr.Button("🗑 Сбросить диалог")
        reply = gr.Audio(label="Ответ (озвучка)", type="filepath", autoplay=True)
        log = gr.Textbox(label="Диалог", lines=10)
        history = gr.State([])

        inputs = [audio, text_input, mode, base_url, api_key, model_name, voice, history]
        outputs = [reply, log, history, text_input]
        text_input.submit(respond, inputs=inputs, outputs=outputs)
        audio.stop_recording(respond, inputs=inputs, outputs=outputs)
        clear.click(clear_history, outputs=[history, log])
    return demo


if __name__ == "__main__":
    build_ui().launch(inbrowser=True, server_name="127.0.0.1", server_port=7861)