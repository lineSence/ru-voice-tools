"""Интеграционные тесты v4: настоящий KoboldCpp + маленькая Qwen2.5-0.5B (CPU),
настоящий Silero, LiteLLM-заглушка (SSE / без потока / ошибка), перезапуск упавшего
движка. Запуск: python tests/test_integration_v4.py [папка app.py]"""
import json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from _paths import QWEN05, use_test_engine  # noqa: E402
import app  # noqa: E402
import requests  # noqa: E402
import soundfile as sf  # noqa: E402

use_test_engine(app)
RESULTS = []


def check(name, cond, info=""):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f" — {info}" if info else ""), flush=True)


LOCAL = dict(app.DEFAULT_SETTINGS, llm_source=app.SRC_LOCAL, local_gguf_path=QWEN05,
             stt_engine=app.STT_VOSK, compute=app.COMPUTE_CPU, n_ctx=4096, max_tokens=400,
             web_search=app.WEB_OFF)

# ------------------------------------------------------------------ LiteLLM-заглушка
REQS = []


def sse(pieces, finish="stop"):
    out = b""
    for p in pieces:
        out += b"data: " + json.dumps({"choices": [{"delta": {"content": p}, "finish_reason": None}]},
                                      ensure_ascii=False).encode() + b"\n\n"
    out += b"data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": finish}]}).encode() + b"\n\n"
    return out + b"data: [DONE]\n\n"


class Mock(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQS.append(body)
        model, msgs = body.get("model"), body["messages"]
        system, last = msgs[0]["content"], msgs[-1]["content"]
        if model == "err":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom from mock")
            return
        if model == "json":
            data = json.dumps({"choices": [{"message": {"content": "Ответ без потока."}}]},
                              ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        finish = "stop"
        if model == "long":
            pieces, finish = ["Очень ", "длинный ", "отв"], "length"
        elif model == "again":  # всегда просит поиск, пока в системном промпте есть правило
            pieces = (["ПОИСК: ещё раз"] if "У тебя есть поиск" in system
                      else ["Финальный ", "ответ ", "без поиска."])
        elif "Результаты поиска в интернете" in last:
            pieces = ["По данным ", "поиска, курс — ", "81,23 рубля. ", "Это всё."]
        elif "курс" in last:
            pieces = ["<think>надо искать</think>", "\n", "ПО", "ИСК", ": курс", " доллара", " сегодня\n"]
        else:
            pieces = ["<think>хм", "м</think>", "Привет! ", "Как дела?"]
        data = sse(pieces, finish)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(0, len(data), 7):  # мелкими TCP-кусками
            self.wfile.write(data[i:i + 7])
            self.wfile.flush()


srv = ThreadingHTTPServer(("127.0.0.1", 0), Mock)
threading.Thread(target=srv.serve_forever, daemon=True).start()
LITE = dict(app.DEFAULT_SETTINGS, llm_source=app.SRC_LITELLM, stt_engine=app.STT_VOSK,
            litellm_base=f"http://127.0.0.1:{srv.server_port}/v1", litellm_key="k",
            litellm_model="stream", web_search=app.WEB_AUTO)


def reset_history():
    app.clear_history()


def fake_search(results="1. ЦБ РФ: официальный курс доллара 81,23 рубля. cbr.ru",
                urls=("https://www.cbr.ru/currency_base/", "https://ru.wikipedia.org/wiki/x")):
    calls = []

    def ws(query, max_chars=6000):
        calls.append((query, max_chars))
        return results, list(urls)
    return ws, calls


real_ws = app.web_search

# ---- L1: LiteLLM поток, модель сама просит поиск (протокол «ПОИСК:»)
reset_history()
REQS.clear()
app.web_search, calls = fake_search()
emitted, statuses = [], []
ans, sent, note = app.think_answer("Какой сейчас курс доллара?", LITE, emit=emitted.append,
                                   status=statuses.append)
check("L1 поиск по команде модели", calls and calls[0][0] == "курс доллара сегодня", calls)
check("L1 ответ по результатам", "81,23" in ans and "ПОИСК" not in "".join(emitted), ans)
check("L1 заметка с доменами", "cbr.ru" in note and "wikipedia.org" in note, note)
check("L1 статусы", any(s.startswith("🌐") for s in statuses) and any("Читаю" in s for s in statuses),
      statuses)
check("L1 второй запрос: ПОИСК + результаты",
      len(REQS) == 2 and REQS[1]["messages"][-2] == {"role": "assistant", "content": "ПОИСК: курс доллара сегодня"}
      and "81,23" in REQS[1]["messages"][-1]["content"] and REQS[1].get("stream") is True)
check("L1 правило поиска в системном промпте", "ПОИСК: короткий" in REQS[0]["messages"][0]["content"])
check("L1 дата в вопросе", sent.startswith("[Сейчас ") and sent.endswith("Какой сейчас курс доллара?"), sent)
check("L1 бюджет LiteLLM", calls[0][1] == 8000, calls[0][1])

# ---- L2: think-блок не попадает в ответ, поток без поиска
REQS.clear()
emitted = []
ans, sent, note = app.think_answer("Привет", LITE, emit=emitted.append)
check("L2 think вырезан", ans == "Привет! Как дела?" and "".join(emitted).strip() == ans, repr(ans))
check("L2 один запрос, без заметки", len(REQS) == 1 and note == "", (len(REQS), note))

# ---- L3: модель снова просит поиск -> третий проход без правила поиска
REQS.clear()
app.web_search, calls = fake_search()
ans, sent, note = app.think_answer("Расскажи что-нибудь", dict(LITE, litellm_model="again"))
check("L3 третий проход", len(REQS) == 3 and ans == "Финальный ответ без поиска."
      and "У тебя есть поиск" not in REQS[2]["messages"][0]["content"], (len(REQS), ans))

# ---- L4: сервер без потока (JSON) и ошибка
ans, _, _ = app.think_answer("Привет", dict(LITE, litellm_model="json", web_search=app.WEB_OFF))
check("L4 JSON вместо SSE", ans == "Ответ без потока.", ans)
try:
    app.think_answer("Привет", dict(LITE, litellm_model="err"))
    check("L4 ошибка 500", False, "нет исключения")
except RuntimeError as e:
    check("L4 ошибка 500", "500" in str(e) and "boom" in str(e), str(e))

# ---- L6: ответ упёрся в лимит длины -> подсказка в «Диалоге»
ans, _, note = app.think_answer("Расскажи всё", dict(LITE, litellm_model="long", web_search=app.WEB_OFF))
check("L6 лимит длины -> заметка", ans == "Очень длинный отв" and note.startswith("✂️") and "1024" in note, note)

# ---- L5: поиск упал -> ответ по знаниям + предупреждение в заметке
REQS.clear()


def broken_ws(q, max_chars=6000):
    raise RuntimeError("Поисковики не ответили (нет интернета?)")


app.web_search = broken_ws
ans, sent, note = app.think_answer("Какой курс доллара?", LITE)
check("L5 поиск не удался", note.startswith("⚠️ Поиск") and "Поиск в интернете не удался"
      in REQS[1]["messages"][-1]["content"], note)
app.web_search = real_ws

# ------------------------------------------------------------------ настоящий KoboldCpp
t0 = time.time()
base = app.ensure_engine(LOCAL)
check("K0 движок запущен", app.ENGINE_STATE["phase"] == "ready", f"{time.time() - t0:.1f} с")

# ---- K1: поток реально кусочками
reset_history()
pieces, times = [], []
t0 = time.time()
ans, sent, note = app.think_answer("Расскажи подробно, как заварить чай.", dict(LOCAL, max_tokens=300),
                                   emit=lambda p: (pieces.append(p), times.append(time.time() - t0)))
total = time.time() - t0
check("K1 много кусочков", len(pieces) >= 20, len(pieces))
# на 2 vCPU чтение промпта ~10 с — главное, что текст идёт по мере генерации
check("K1 текст идёт по ходу генерации", times and total - times[0] > 2 and times[-1] - times[0] > 2,
      f"первый {times[0]:.2f} с, последний {times[-1]:.2f} с из {total:.2f} с" if times else "")
check("K1 ответ = сумма кусков", ans == app.strip_think("".join(pieces)).strip(), len(ans))
check("K1 скорость записана", app.ENGINE_STATE.get("llm_tps"), app.ENGINE_STATE.get("llm_tps"))

# ---- K2: прерывание генерации при закрытии потока
msgs = [app.build_system(LOCAL), {"role": "user", "content": "Напиши очень длинный рассказ о море."}]
gen = app.llm_stream(LOCAL, msgs, 2000)
got = [next(gen) for _ in range(5)]
t0 = time.time()
gen.close()
idle = None
for _ in range(30):
    perf = requests.get(base + "/api/extra/perf", timeout=2).json()
    if perf.get("idle") == 1:
        idle = time.time() - t0
        break
    time.sleep(0.1)
check("K2 KoboldCpp остановил генерацию", idle is not None and idle < 2, f"idle через {idle}")

# ---- K2b: лимит длины на настоящем KoboldCpp
for q in ["Напиши длинный рассказ о море.", "Перечисли подробно двадцать фактов о космосе.",
          "Расскажи очень подробно историю Москвы."]:  # 0.5B иногда отвечает коротко сама
    ans, _, note = app.think_answer(q, dict(LOCAL, max_tokens=64))
    if app.LAST_FINISH["reason"] == "length":
        break
    check("K2b без лимита — без заметки", "✂️" not in note, note)
check("K2b лимит длины -> заметка", app.LAST_FINISH["reason"] == "length" and "✂️" in note
      and "64 токенов" in note, (app.LAST_FINISH, note))

# ---- K3: принудительный поиск («найди…»), модель 0.5B отвечает сама -> прерываем и ищем
reset_history()
app.web_search, calls = fake_search()
statuses, pieces = [], []
cfg = dict(LOCAL, web_search=app.WEB_ASK)
ans, sent, note = app.think_answer("Найди в интернете, какой сейчас курс доллара", cfg,
                                   emit=pieces.append, status=statuses.append)
check("K3 поиск выполнен", calls, calls)
check("K3 запрос очищен", calls and "найди" not in calls[0][0].lower() and "доллар" in calls[0][0],
      calls[0][0] if calls else "")
check("K3 бюджет CPU <= 3000", calls and 1200 <= calls[0][1] <= 3000, calls[0][1] if calls else "")
check("K3 ответ использует результаты", "81" in ans, ans[:200])
check("K3 заметка", "cbr.ru" in note, note)
app.web_search = real_ws

# ---- K4: answer_and_speak — живой текст растёт, WAV настоящим Silero
reset_history()
seen, stop = [], threading.Event()


def watch():
    while not stop.is_set():
        seen.append((app.LIVE["text"], app.LIVE["status"]))
        time.sleep(0.02)


th = threading.Thread(target=watch)
th.start()
t0 = time.time()
ans, wav = app.answer_and_speak("Объясни, почему небо голубое.", dict(LOCAL, max_tokens=250))
stop.set()
th.join()
lens = sorted({len(t) for t, _ in seen})
check("K4 живой текст рос", len(lens) >= 5, f"{len(lens)} разных длин")
check("K4 статус «Думаю» был", any(s.startswith("🧠") for _, s in seen))
info = sf.info(wav)
check("K4 WAV 48 кГц, есть звук", info.samplerate == 48000 and info.duration > 3, f"{info.duration:.1f} с")
check("K4 история", app.HISTORY[-2]["content"].startswith("[Сейчас") and
      app.HISTORY[-2].get("shown") == "Объясни, почему небо голубое." and app.HISTORY[-1]["content"] == ans)
check("K4 LIVE очищен", app.LIVE["text"] == "" and app.LIVE["user"] == "", app.LIVE)
log = app.render_log()
check("K4 render_log показывает вопрос без даты", "Объясни, почему небо голубое." in log and "[Сейчас" not in log)
print(f"   K4 время {time.time() - t0:.1f} с, ответ {len(ans)} симв.", flush=True)

# ---- K5: второй вопрос — история отправляется в sent-виде (кэш префикса)
REQ_SPY = []
orig_req = app.engine_request


def spy(cfg, path, payload, timeout, stream=False):
    REQ_SPY.append(payload)
    return orig_req(cfg, path, payload, timeout, stream)


app.engine_request = spy
ans2, _ = app.answer_and_speak("А закат почему красный?", dict(LOCAL, max_tokens=120))
app.engine_request = orig_req
m = REQ_SPY[0]["messages"]
check("K5 история в запросе", m[1]["content"].startswith("[Сейчас") and m[2]["content"] == ans
      and set(m[1]) == {"role", "content"}, [x["role"] for x in m])

# ---- K6: движок упал до первого слова -> перезапуск и повтор
orig_stream = app.llm_stream
state = {"n": 0}


def crashing(cfg, messages, max_tokens):
    state["n"] += 1
    if state["n"] == 1:
        app._eng["proc"].kill()
        app._eng["proc"].wait()
        raise app.EngineError("Модель оборвала ответ — см. engine/koboldcpp.log")
    yield from orig_stream(cfg, messages, max_tokens)


app.llm_stream = crashing
t0 = time.time()
out = "".join(app.stream_text(LOCAL, msgs[:1] + [{"role": "user", "content": "Скажи привет."}], 30))
app.llm_stream = orig_stream
check("K6 перезапуск и ответ", state["n"] == 2 and out.strip() and app._eng["proc"].poll() is None,
      f"{time.time() - t0:.1f} с: {out[:60]!r}")

# ---- K7: движок убит между вопросами -> ensure_engine поднимает заново
app._eng["proc"].kill()
app._eng["proc"].wait()
ans, _, _ = app.think_answer("Сколько будет два плюс два?", dict(LOCAL, max_tokens=40))
check("K7 ответ после падения", ans.strip(), ans[:80])

app.stop_engine()
time.sleep(1)
check("K8 движок остановлен", app._eng["proc"] is None or app._eng["proc"].poll() is not None)
srv.shutdown()
bad = [n for n, ok in RESULTS if not ok]
print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} PASS" + (f"; FAIL: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
