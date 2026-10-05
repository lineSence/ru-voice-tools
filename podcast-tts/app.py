#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RU Podcast TTS — многоголосые подкасты на русском языке.
Движок: Silero TTS v5 (работает на CPU, GPU не требуется).
Модель скачивается автоматически при первом запуске (~100 МБ).
"""

import os

# Фикс для Windows с прокси/VPN: localhost не должен идти через прокси,
# иначе Gradio падает с "startup-events failed (code 502)"
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

import re
import tempfile

import numpy as np
import soundfile as sf
import torch
import gradio as gr

SAMPLE_RATE = 48000
VOICES = ["aidar", "baya", "kseniya", "xenia", "eugene", "random"]
VOICE_HINTS = {
    "aidar": "мужской",
    "baya": "женский",
    "kseniya": "женский",
    "xenia": "женский",
    "eugene": "мужской",
    "random": "случайный",
}
DEFAULT_SPEAKERS = [("Аня", "kseniya"), ("Марк", "aidar"), ("Ольга", "baya"), ("Лев", "eugene")]

_state = {}


def load_model():
    """Скачивает и загружает Silero v5 (один раз за сессию)."""
    if "model" in _state:
        return
    import inspect
    hub_kwargs = {}
    # torch >= 2.6 интерактивно спрашивает доверие к репозиторию — отключаем вопрос
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
    # hubconf возвращает (model, example_text) для v5 или (model, symbols, sr, text, apply_tts) для старых
    model = result[0]
    model.to(torch.device("cpu"))
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    _state["model"] = model


def synthesize(text, voice):
    """Одна реплика -> numpy-массив float32."""
    audio = _state["model"].apply_tts(text=text, speaker=voice, sample_rate=SAMPLE_RATE)
    arr = audio.numpy() if torch.is_tensor(audio) else np.asarray(audio)
    return np.asarray(arr, dtype=np.float32).flatten()


def split_long(text, maxlen=350):
    """Режет длинные реплики на фразы, чтобы модель не 'спотыкалась'."""
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
    out = []
    for c in chunks:
        while len(c) > maxlen:
            cut = c.rfind(" ", 0, maxlen)
            cut = cut if cut > 0 else maxlen
            out.append(c[:cut].strip())
            c = c[cut:].strip()
        if c:
            out.append(c)
    return out


LINE_RE = re.compile(r"^\s*([^:：]{1,40}?)\s*[:：]\s*(.+)$")


def parse_script(script, mapping, default_voice):
    """Разбирает сценарий вида 'Имя: текст' в список (голос, текст)."""
    items = []
    for raw in script.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if m and m.group(1).strip().lower() in mapping:
            voice = mapping[m.group(1).strip().lower()]
            text = m.group(2).strip()
        else:
            voice, text = default_voice, line
        for chunk in split_long(text):
            items.append((voice, chunk))
    return items


def generate(script, s1n, s1v, s2n, s2v, s3n, s3v, s4n, s4v, pause_ms, progress=gr.Progress()):
    mapping = {}
    for n, v in [(s1n, s1v), (s2n, s2v), (s3n, s3v), (s4n, s4v)]:
        if n and n.strip():
            mapping[n.strip().lower()] = v
    default_voice = s1v
    try:
        progress(0, desc="Загрузка модели (первый запуск скачает ~100 МБ)...")
        load_model()
        items = parse_script(script, mapping, default_voice)
        if not items:
            return None, "Сценарий пуст. Напишите реплики в формате:  Имя: текст"
        sr = SAMPLE_RATE
        pause = np.zeros(int(sr * pause_ms / 1000), dtype=np.float32)
        pieces, log = [], []
        for i, (voice, text) in enumerate(items):
            progress((i + 1) / (len(items) + 1), desc=f"Реплика {i + 1}/{len(items)} ({voice})")
            pieces.append(synthesize(text, voice))
            pieces.append(pause.copy())
            log.append(f"{i + 1:>3}. [{voice}] {text[:70]}")
        full = np.concatenate(pieces)
        out = os.path.join(tempfile.gettempdir(), "ru_podcast_tts.wav")
        sf.write(out, full, sr)
        return out, f"Готово: {len(full) / sr:.1f} сек, {len(items)} реплик.\n" + "\n".join(log)
    except Exception as e:
        return None, f"Ошибка: {e}"


def build_ui():
    with gr.Blocks(title="RU Podcast TTS") as demo:
        gr.Markdown(
            "# 🎙️ RU Podcast TTS\n"
            "Каждая строка сценария: `Имя: текст реплики`. Строки без имени читает Спикер 1.\n\n"
            "Ударение можно задать вручную знаком `+` перед гласной: `молок+о`."
        )
        speaker_inputs = []
        with gr.Row():
            for i, (dn, dv) in enumerate(DEFAULT_SPEAKERS, start=1):
                with gr.Column():
                    n = gr.Textbox(label=f"Спикер {i} — имя в сценарии", value=dn)
                    v = gr.Dropdown(
                        label="Голос",
                        choices=[(f"{vv} ({VOICE_HINTS[vv]})", vv) for vv in VOICES],
                        value=dv,
                    )
                    speaker_inputs += [n, v]
        script = gr.Textbox(
            label="Сценарий",
            lines=12,
            value="Аня: Привет! Сегодня мы пробуем локальный синтез речи.\n"
                  "Марк: Да, всё работает прямо на процессоре, без видеокарты.\n"
                  "Аня: А ударения расставляются автоматически!",
        )
        pause = gr.Slider(0, 2000, value=400, step=50, label="Пауза между репликами, мс")
        btn = gr.Button("▶ Сгенерировать подкаст", variant="primary")
        audio = gr.Audio(label="Результат (можно скачать)", type="filepath")
        status = gr.Textbox(label="Статус", lines=8)
        btn.click(generate, inputs=[script] + speaker_inputs + [pause], outputs=[audio, status])
    return demo


if __name__ == "__main__":
    build_ui().launch(inbrowser=True, server_name="127.0.0.1", server_port=7860)