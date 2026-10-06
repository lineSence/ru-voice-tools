"""Wake word -> вопрос -> ответ звучит ПО ХОДУ генерации (первое предложение до конца
ответа модели). Поддельный sounddevice: «микрофон» отдаёт речь Silero в реальном
времени, play/wait — проигрывание в реальном времени. Настоящие Vosk, KoboldCpp + qwen05.
Запуск: python tests/test_wake_stream_v4.py [папка app.py]"""
import sys, threading, time, types
import numpy as np

from _paths import QWEN05, use_test_engine  # noqa: E402
import app  # noqa: E402

use_test_engine(app)
RES = []


def check(name, cond, info=""):
    RES.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f" — {info}" if info else ""), flush=True)


# речь для «микрофона»: Silero 48 кГц -> 16 кГц int16
tts = app.load_tts()
speech = tts.apply_tts(text="Ассистент, расскажи подробно, как правильно посадить дерево.", speaker="xenia",
                       sample_rate=48000).numpy()
from scipy import signal  # noqa: E402
pcm16 = (signal.resample_poly(speech, 1, 3) * 32767 * 0.8).astype(np.int16)
SR = 16000
mic = np.concatenate([np.zeros(SR, np.int16), pcm16, np.zeros(SR * 60, np.int16)])
EVENTS = []  # (время, событие)
T0 = time.time()


def ev(name, **kw):
    EVENTS.append((time.time() - T0, name, kw))


class RawInputStream:
    def __init__(self, samplerate, blocksize, device, channels, dtype, callback):
        assert samplerate == SR and channels == 1 and dtype == "int16"
        self.bs, self.cb, self.stop = blocksize, callback, threading.Event()

    def _run(self):
        pos = 0
        while not self.stop.is_set() and pos + self.bs <= len(mic):
            self.cb(mic[pos:pos + self.bs].tobytes(), self.bs, None, None)
            pos += self.bs
            time.sleep(self.bs / SR)

    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.t.join()


_playing = {"until": 0.0}


def play(arr, sr):
    dur = len(arr) / sr
    ev("play", dur=round(dur, 2))
    _playing["until"] = time.time() + dur


def wait():
    time.sleep(max(0.0, _playing["until"] - time.time()))


DEV = {"name": "Fake Mic", "max_input_channels": 1, "max_output_channels": 0,
       "default_samplerate": 16000.0, "hostapi": 0}
fake = types.ModuleType("sounddevice")
fake.RawInputStream = RawInputStream
fake.play, fake.wait, fake.stop = play, wait, lambda: None
fake.query_devices = lambda device=None, kind=None: DEV if (kind or device is not None) else [DEV]
fake.query_hostapis = lambda: [{"name": "MME"}]
fake.check_input_settings = lambda **kw: None
fake._terminate = fake._initialize = lambda: None
fake.default = types.SimpleNamespace(device=[0, 0])
sys.modules["sounddevice"] = fake

orig_think = app.think_answer


def think_spy(*a, **kw):
    ev("llm_start")
    try:
        return orig_think(*a, **kw)
    finally:
        ev("llm_end")


app.think_answer = think_spy
cfg = dict(app.DEFAULT_SETTINGS, llm_source=app.SRC_LOCAL, local_gguf_path=QWEN05,
           stt_engine=app.STT_VOSK, compute=app.COMPUTE_CPU, n_ctx=4096, max_tokens=220,
           web_search=app.WEB_OFF, wake_enabled=True, voice="baya")
app.ensure_engine(cfg)  # движок заранее, как фоновый старт в программе
app.clear_history()
phases = []
app._wake["cfg"] = cfg
T0 = time.time()
app.start_wake()
deadline = time.time() + 240
while time.time() < deadline:
    ph = app.WAKE_STATE.get("phase")
    if not phases or phases[-1] != ph:
        phases.append(ph)
        ev("phase", phase=ph)
    if len(app.HISTORY) >= 2 and ph in ("listening", "idle", "waiting") and \
            any(e[1] == "llm_end" for e in EVENTS):
        break
    if ph == "error":
        break
    time.sleep(0.05)
time.sleep(1.0)
app.stop_wake()
app.stop_engine()

plays = [e for e in EVENTS if e[1] == "play"]
llm_end = next((e[0] for e in EVENTS if e[1] == "llm_end"), None)
llm_start = next((e[0] for e in EVENTS if e[1] == "llm_start"), None)
print("фазы:", phases)
print("события:", [(round(t, 1), n, kw) for t, n, kw in EVENTS if n != "phase"][:20])
q = app.HISTORY[0].get("shown", "") if app.HISTORY else ""
a = app.HISTORY[1]["content"] if len(app.HISTORY) > 1 else ""
check("вопрос распознан после wake word", "дерев" in q.lower() and "ассистент" not in q.lower(), q)
check("ответ есть", len(a) > 20, a[:120])
check("несколько кусков озвучки", len(plays) >= 2, len(plays))
check("первый звук ДО конца генерации", plays and llm_end and plays[0][0] < llm_end,
      f"старт LLM {llm_start:.1f} с, первый звук {plays[0][0]:.1f} с, конец LLM {llm_end:.1f} с"
      if plays and llm_end and llm_start else "")
check("фаза «говорю» во время ответа", "speaking" in phases, phases)
check("после ответа снова слушает", phases[-1] in ("listening", "idle", "waiting", "off"), phases[-1:])
print(f"\nИТОГО: {sum(RES)}/{len(RES)} PASS")
sys.exit(0 if all(RES) else 1)
