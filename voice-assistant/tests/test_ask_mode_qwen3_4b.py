"""Режим «Только когда прошу» на Qwen3-4B: без просьбы — не выдумывает погоду, а предлагает
поискать; с просьбой — ПОИСК; обычные вопросы — сразу ответ."""
import subprocess, time
from _paths import QWEN4B, use_test_engine  # noqa: E402
import app  # noqa: E402

_Popen = subprocess.Popen


class Popen2(_Popen):
    def __init__(self, args, *a, **kw):
        if isinstance(args, list) and "--skiplauncher" in args:
            args = args + ["--threads", "2"]
        super().__init__(args, *a, **kw)


subprocess.Popen = Popen2
use_test_engine(app)
CFG = dict(app.DEFAULT_SETTINGS, llm_source=app.SRC_LOCAL,
           local_gguf_path=QWEN4B,
           stt_engine=app.STT_VOSK, compute=app.COMPUTE_CPU, n_ctx=3072, web_search=app.WEB_ASK)
app.ensure_engine(CFG)
for q, want in [("Какая погода будет завтра в Москве?", False), ("Какой сейчас курс евро?", False),
                ("Объясни, что такое фотосинтез.", False), ("Найди в интернете курс евро на сегодня.", True)]:
    system = app.build_system(CFG)
    msgs, mt = app.fit_messages(CFG, system, [], [{"role": "user", "content": f"[{app.now_line()}]\n{q}"}], 70)
    t = time.time()
    text, query = app._stream_answer(CFG, msgs, mt, lambda s: None, sniff=True)
    got = query is not None
    print(f"{'PASS' if got == want else 'FAIL'} {q} -> {'ПОИСК: ' + query if got else text.strip()[:220]}"
          f"  ({time.time() - t:.0f} с)", flush=True)
app.stop_engine()
