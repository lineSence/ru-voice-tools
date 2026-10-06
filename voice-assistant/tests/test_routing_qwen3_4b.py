"""Решения «искать / отвечать сразу» и длина ответа на настоящей Qwen3-4B-Instruct-2507
(та же модель, что по умолчанию у пользователя), CPU. Медленно (~10 мин на 2 vCPU).
Запуск: python tests/test_routing_qwen3_4b.py [папка app.py]"""
import subprocess, time
from _paths import QWEN4B, use_test_engine  # noqa: E402
import app  # noqa: E402

_Popen = subprocess.Popen


class Popen2(_Popen):  # все ядра песочницы (KoboldCpp сам берёт ядра − 1)
    def __init__(self, args, *a, **kw):
        if isinstance(args, list) and "--skiplauncher" in args:
            args = args + ["--threads", "2"]
        super().__init__(args, *a, **kw)


subprocess.Popen = Popen2
use_test_engine(app)
CFG = dict(app.DEFAULT_SETTINGS, llm_source=app.SRC_LOCAL, local_gguf_path=QWEN4B,
           stt_engine=app.STT_VOSK, compute=app.COMPUTE_CPU, n_ctx=3072, web_search=app.WEB_AUTO)
t0 = time.time()
app.ensure_engine(CFG)
print(f"движок: {time.time() - t0:.0f} с", flush=True)

CASES = [  # (вопрос, режим, нужен ли поиск)
    ("Какая погода будет завтра в Москве?", app.WEB_AUTO, True),
    ("Какой сейчас курс доллара к рублю?", app.WEB_AUTO, True),
    ("Что нового в мире технологий за эту неделю?", app.WEB_AUTO, True),
    ("Какая последняя версия Python?", app.WEB_AUTO, True),
    ("Сколько стоит айфон шестнадцать в России?", app.WEB_AUTO, True),
    ("Привет! Как у тебя дела?", app.WEB_AUTO, False),
    ("Объясни, что такое фотосинтез.", app.WEB_AUTO, False),
    ("Сколько будет семнадцать умножить на двадцать три?", app.WEB_AUTO, False),
    ("Придумай короткое стихотворение про кота.", app.WEB_AUTO, False),
    ("Как лучше организовать свой рабочий день?", app.WEB_AUTO, False),
    ("Который час?", app.WEB_AUTO, False),
    ("Какая погода будет завтра в Москве?", app.WEB_ASK, False),
    ("Поищи в интернете, какая погода завтра в Москве.", app.WEB_ASK, True),
]
ok = 0
for q, mode, want in CASES:
    cfg = dict(CFG, web_search=mode)
    system = app.build_system(cfg)
    msgs, mt = app.fit_messages(cfg, system, [], [{"role": "user", "content": f"[{app.now_line()}]\n{q}"}], 40)
    t = time.time()
    text, query = app._stream_answer(cfg, msgs, mt if q != "Который час?" else 60, lambda s: None, sniff=True)
    got = query is not None
    ok += got == want
    print(f"{'PASS' if got == want else 'FAIL'} [{'AUTO' if mode == app.WEB_AUTO else 'ASK'}] {q} -> "
          f"{'ПОИСК: ' + query if got else 'ответ: ' + text.strip()[:110]!s}  ({time.time() - t:.0f} с)",
          flush=True)
print(f"\nмаршрутизация: {ok}/{len(CASES)}", flush=True)

# длина ответа: простой вопрос — коротко, «расскажи подробно» — развёрнуто
for q, mt in [("Привет! Как у тебя дела?", 120),
              ("Расскажи подробно, как подготовить велосипед к зиме.", 420)]:
    cfg = dict(CFG, web_search=app.WEB_OFF, max_tokens=mt)
    t = time.time()
    ans, _, _ = app.think_answer(q, cfg)
    print(f"\n[{len(ans)} симв., {time.time() - t:.0f} с, {app.ENGINE_STATE.get('llm_tps')} т/с] {q}\n{ans}",
          flush=True)
app.stop_engine()
