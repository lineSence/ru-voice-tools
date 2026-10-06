"""Юнит-тесты: промпт, длина ответа, поиск (разбор команды, запрос, страницы), поток озвучки."""
import sys, os, json, time, tempfile
from _paths import APP_DIR  # noqa: E402  (путь к app.py)
sys.argv = ["x"]
import numpy as np
import app
print("app.py:", APP_DIR, flush=True)

RES = []
def check(name, cond, detail=""):
    RES.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  | {detail}" if detail else ""), flush=True)

def sniff(chunks):
    c, out = app.CommandSniffer(), []
    for ch in chunks:
        out.append(c.feed(ch))
        if c.done: break
    if not c.done: out.append(c.finish())
    return "".join(out), (c.query if c.done else None)

def pieces(text, n):
    return [text[i:i + n] for i in range(0, len(text), n)]

# --- команда ПОИСК в потоке, при любой нарезке на токены
cases = [
    ("ПОИСК: погода в Москве завтра\nНу а пока…", "", "погода в Москве завтра"),
    ("**ПОИСК:** курс доллара", "", "курс доллара"),
    ("[ПОИСК] новости IT", "", "новости IT"),
    ("Поиск: последняя версия Python", "", "последняя версия Python"),
    ("Сейчас поищу.\nПОИСК: счёт матча Спартак", "Сейчас поищу.\n", "счёт матча Спартак"),
    ("Поиски сокровищ — увлекательное дело.", "Поиски сокровищ — увлекательное дело.", None),
    ("Привет! Чем помочь?", "Привет! Чем помочь?", None),
    ("Да.", "Да.", None),
    ("MIDI — это протокол.\n\nПодключите кабель.", "MIDI — это протокол.\n\nПодключите кабель.", None),
]
for text, want_out, want_q in cases:
    ok = True
    for n in (1, 2, 3, 7, 100):
        out, q = sniff(pieces(text, n))
        q = app._norm_query(q) if q is not None else None
        if out != want_out or q != want_q:
            ok = False; detail = f"n={n}: out={out!r} q={q!r}"; break
    check(f"команда: {text[:40]!r}", ok, "" if ok else detail)
long = "Это длинный ответ. " * 30 + "\nПОИСК: не команда"
out, q = sniff(pieces(long, 5))
check("после 300 символов «ПОИСК:» — уже обычный текст", q is None and out == long)

# --- <think> вырезается из потока
def think(chunks):
    f = app.ThinkFilter(); return "".join(f.feed(c) for c in chunks) + f.flush()
check("think: рассуждения убраны", think(pieces("<think>\nНадо подумать</think>\n\nОтвет.", 3)) == "Ответ.")
check("think: обычный текст не тронут", think(pieces("Обычный ответ <b>", 2)) == "Обычный ответ <b>")
check("think: незакрытый тег — пусто", think(pieces("<think>думаю и думаю", 4)) == "")

# --- явная просьба поискать и запрос из неё
for q, want in [("найди в интернете, как настроить MIDI синтезатор", "как настроить MIDI синтезатор"),
                ("Поищи пожалуйста рецепт борща", "рецепт борща"),
                ("загугли курс биткоина", "курс биткоина"),
                ("что пишут в сети про Python 3.14", "что пишут про Python 3.14")]:
    check(f"просьба найти: {q!r}", bool(app.SEARCH_ASK_RE.search(q)) and app.clean_query(q) == want, app.clean_query(q))
for q in ["как настроить MIDI синтезатор", "расскажи сказку", "как подключиться к сети Wi-Fi"]:
    check(f"не просьба найти: {q!r}", not app.SEARCH_ASK_RE.search(q))

# --- системный промпт
cfg = dict(app.DEFAULT_SETTINGS)
sys_auto = app.build_system(cfg)["content"]
check("стандартный промпт — сбалансированный", sys_auto.startswith(app.PROMPT_BALANCED) and app.SEARCH_RULE_AUTO in sys_auto and app.TIME_RULE in sys_auto)
sys_off = app.build_system(dict(cfg, web_search=app.WEB_OFF, system_prompt="Ты пират."))["content"]
check("свой промпт + без поиска", sys_off.startswith("Ты пират.") and "ПОИСК" not in sys_off)
check("пустой промпт → стандартный", app.build_system(dict(cfg, system_prompt="  "))["content"].startswith(app.PROMPT_BALANCED))
check("режим «по просьбе»", app.SEARCH_RULE_ASK in app.build_system(dict(cfg, web_search=app.WEB_ASK))["content"])
check("дата по-русски", app.now_line(time.mktime((2026, 10, 6, 9, 5, 0, 0, 0, -1))) == "Сейчас вторник, 6 октября 2026 года, 9:05.", app.now_line(time.mktime((2026, 10, 6, 9, 5, 0, 0, 0, -1))))

# --- контекст: старые реплики выкидываются, длинные результаты подрезаются, ответ ≤ половины
system = app.build_system(cfg)
hist = []
for i in range(6):
    hist += [{"role": "user", "content": f"вопрос {i} " + "слово " * 50},
             {"role": "assistant", "content": f"ответ {i} " + "длинный ответ " * 300}]
tail = [{"role": "user", "content": "новый вопрос"}]
c4 = dict(cfg, llm_source=app.SRC_LOCAL, n_ctx=4096)
msgs, mt = app.fit_messages(c4, system, hist, tail, 1024)
tok = sum(app.est_tokens(m["content"]) for m in msgs)
check("4096: промпт + ответ влезают", tok + mt <= 4096 and msgs[0] is system and msgs[-1] == tail[0], f"{tok}+{mt}, сообщений {len(msgs)}")
check("4096: оставлены самые новые реплики", len(msgs) > 2 and msgs[-2]["content"].startswith("ответ 5"), msgs[1]["content"][:10] if len(msgs) > 2 else "")
msgs, mt = app.fit_messages(dict(c4, n_ctx=2048), system, hist, tail, 4096)
check("2048: длина ответа ≤ половины контекста", mt == 1024, mt)
big = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "ПОИСК: q"},
       {"role": "user", "content": "Результаты: " + "факт " * 6000 + " Ответь на мой вопрос «q»."}]
msgs, mt = app.fit_messages(c4, system, hist, big, 1024)
tok = sum(app.est_tokens(m["content"]) for m in msgs)
check("огромные результаты подрезаны, инструкция в конце цела", tok + mt <= 4096 and msgs[-1]["content"].endswith("Ответь на мой вопрос «q»."), f"{tok}+{mt}")
msgs, mt = app.fit_messages(dict(cfg, llm_source=app.SRC_LITELLM), system, hist, tail, 3000)
check("LiteLLM: без подрезки, 6 пар истории", len(msgs) == 14 and mt == 3000)

# --- бюджет результатов поиска
b_gpu = app.search_budget(dict(cfg, llm_source=app.SRC_LOCAL, n_ctx=8192), system, "вопрос", 1024)
b_cpu = app.search_budget(dict(cfg, llm_source=app.SRC_LOCAL, n_ctx=8192, compute=app.COMPUTE_CPU), system, "вопрос", 1024)
b_small = app.search_budget(dict(cfg, llm_source=app.SRC_LOCAL, n_ctx=2048), system, "вопрос", 1024)
check("бюджет поиска: видеокарта 6000, процессор 3000, малый контекст ≥1200", (b_gpu, b_cpu) == (6000, 3000) and 1200 <= b_small < 3000, (b_gpu, b_cpu, b_small))

# --- страница: текст из div-вёрстки, выбор кусков по словам запроса, кодировка
html_doc = """<html><head><meta charset="windows-1251"><title>x</title><script>var a=1;</script></head><body>
<nav>Главная | Новости | Погода | Контакты</nav>
<div class="w"><div>Погода в Москве на завтра</div><div>Днём +12°, ночью +5°, облачно, ветер 4 м/с</div></div>
<p>Реклама: купите слона прямо сейчас по выгодной цене в нашем магазине!</p>
<table><tr><td>Утро</td><td>+6°</td></tr><tr><td>День</td><td>+12°</td></tr></table>
<footer>© 2026 Все права защищены</footer></body></html>""".encode("cp1251")
blocks = app.page_blocks(html_doc)
joined = " ".join(blocks)
check("страница: текст из div, cp1251, без меню/скриптов/подвала", "Днём +12°" in joined and "var a" not in joined and "Контакты" not in joined and "права" not in joined, joined[:160])
ex = app.best_excerpt(blocks, "погода в Москве завтра", 400)
check("выдержка: про погоду, без рекламы", "+12°" in ex and "слона" not in ex, ex[:160])
utf = "<html><body><p>Python 3.14 вышел 7 октября 2025 года. Главное — свободные потоки.</p></body></html>".encode()
check("страница без charset (определение кодировки)", "свободные потоки" in " ".join(app.page_blocks(utf)))

# --- web_search: сборка результатов с моками (без сети)
orig_hits, orig_fetch = app.search_hits, app.fetch_page
app.search_hits = lambda q, max_results=6: [
    {"title": "Погода в Москве", "url": "https://www.gismeteo.ru/weather-moscow-4/tomorrow/", "body": "Завтра +12, облачно", "date": ""},
    {"title": "Прогноз", "url": "https://yandex.ru/pogoda/moscow", "body": "Облачно с прояснениями", "date": "2026-10-06"},
    {"title": "Видео", "url": "https://www.youtube.com/watch?v=1", "body": "ролик", "date": ""}]
fetched = []
def fake_fetch(url, limit_bytes=0):
    fetched.append(url)
    if "yandex" in url: raise RuntimeError("403")
    return app.page_blocks(html_doc)
app.fetch_page = fake_fetch
text, urls = app.web_search("погода в Москве завтра", 3000)
check("web_search: сниппеты + выдержка со страницы, ютуб не качаем",
      "[1] Погода в Москве (gismeteo.ru)" in text and "Со страницы:" in text and "+12°" in text
      and not any("youtube" in u for u in fetched) and len(urls) == 3 and len(text) <= 3000, text[:200].replace("\n", " / "))
check("web_search: дата новости и упавшая страница не мешают", "(yandex.ru, 2026-10-06)" in text)
text, _ = app.web_search("погода", 300)
check("web_search: укладывается в бюджет", len(text) <= 300, len(text))
app.search_hits, app.fetch_page = orig_hits, orig_fetch
check("домен без www", app._domain("https://www.gismeteo.ru/a") == "gismeteo.ru")

# --- настройки: миграция старого файла и проверка значений
tmp = tempfile.mkdtemp(); app.SETTINGS_PATH = os.path.join(tmp, "settings.json")
json.dump({"n_ctx": 4096, "voice": "aidar", "web_search": "чепуха", "max_tokens": "99999"}, open(app.SETTINGS_PATH, "w"))
s1 = app.load_settings()
check("старые настройки: 4096 → 8192, промпт и поиск по умолчанию", s1["n_ctx"] == 8192 and s1["voice"] == "aidar" and s1["system_prompt"] == app.PROMPT_BALANCED and s1["web_search"] == app.WEB_AUTO and s1["max_tokens"] == 8192, (s1["n_ctx"], s1["web_search"], s1["max_tokens"]))
json.dump(dict(s1, n_ctx=4096, settings_rev=2), open(app.SETTINGS_PATH, "w"))
check("новые настройки: свои 4096 не трогаем", app.load_settings()["n_ctx"] == 4096)
json.dump({"n_ctx": 2048}, open(app.SETTINGS_PATH, "w"))
check("старые настройки: свои 2048 не трогаем", app.load_settings()["n_ctx"] == 2048)
check("make_cfg: поля GUI на своих местах", (lambda c: c["system_prompt"] == "P" and c["max_tokens"] == 512 and c["web_search"] == app.WEB_OFF)(
    app.make_cfg(*[{"system_prompt": "P", "max_tokens": 512.0, "web_search": app.WEB_OFF}.get(k, app.DEFAULT_SETTINGS[k] if k in app.DEFAULT_SETTINGS else "") for k in app.SETTING_FIELDS])))

# --- диалог: живой ответ, заметка о поиске; модели уходят только role/content
app.HISTORY.clear()
app.history_append("какая погода", "Завтра +12.", sent="[Сейчас …]\nкакая погода", note="🔎 Искал в интернете «погода»: gismeteo.ru")
app.live_update(user="а послезавтра?", text="Послезавтра", status="")
log = app.render_log()
check("диалог: вопрос без даты, заметка, живой ответ", log == "Вы: какая погода\nАссистент: Завтра +12.\n   🔎 Искал в интернете «погода»: gismeteo.ru\nВы: а послезавтра?\nАссистент: Послезавтра", log)
check("модели: вопрос с датой, без лишних ключей", app.history_for_llm()[0] == {"role": "user", "content": "[Сейчас …]\nкакая погода"})
app.live_update(user="", text="", status="✍️ Распознаю речь…")
check("диалог: статус без вопроса", app.render_log().endswith("\n   ✍️ Распознаю речь…"))
app.live_update(status="")

# --- поисковики: каждый отдельно и параллельно, медленные не задерживают ответ
import types
class FakeDDGS:
    plan = {}  # backend -> (задержка, результат или исключение)
    def __init__(self, proxy=None, timeout=None): pass
    def _go(self, kind, backend):
        delay, res = FakeDDGS.plan[(kind, backend)]
        time.sleep(delay)
        if isinstance(res, Exception): raise res
        return res
    def text(self, q, region=None, safesearch=None, max_results=6, backend=None): return self._go("text", backend)
    def news(self, q, region=None, safesearch=None, timelimit=None, max_results=4, backend=None): return self._go("news", backend)
fake_mod = types.ModuleType("ddgs"); fake_mod.DDGS = FakeDDGS
fake_eng = types.ModuleType("ddgs.engines")
fake_eng.ENGINES = {"text": {"google": 1, "yahoo": 1, "brave": 1, "wikipedia": 1, "newengine": 1}, "news": {"bing": 1, "yahoo": 1}}
saved_mods = {k: sys.modules.get(k) for k in ("ddgs", "ddgs.engines")}
sys.modules["ddgs"], sys.modules["ddgs.engines"] = fake_mod, fake_eng
check("поисковики: только существующие, порядок, новые в конце, без энциклопедий",
      app._ddgs_backends("text", app.SEARCH_BACKENDS) == ["google", "yahoo", "brave", "newengine"],
      app._ddgs_backends("text", app.SEARCH_BACKENDS))
check("поисковики: чередование", app._interleave([[1, 2, 3], [4], [5, 6]]) == [1, 4, 5, 2, 6, 3])
def H(u, t="T"): return {"title": t, "href": u, "body": "b"}
FakeDDGS.plan = {("text", "google"): (0.05, RuntimeError("captcha")),
                 ("text", "yahoo"): (0.1, [H("https://a.ru/1"), H("https://b.ru/2")]),
                 ("text", "brave"): (0.4, [H("https://c.ru/3"), H("https://a.ru/1")]),
                 ("text", "newengine"): (5.0, [H("https://slow.ru/")])}
t0 = time.time(); hits = app.search_hits("рецепт борща"); dt = time.time() - t0
check("поиск: ответ не ждёт медленный поисковик", dt < 2.0, f"{dt:.2f} с")
check("поиск: результаты успевших чередуются, дубли убраны",
      [h["url"] for h in hits] == ["https://a.ru/1", "https://c.ru/3", "https://b.ru/2"], [h["url"] for h in hits])
FakeDDGS.plan.update({("news", "bing"): (0.1, [{"title": "N", "url": "https://news.ru/n", "body": "x", "date": "2026-10-06T10:00"}]),
                      ("news", "yahoo"): (0.1, RuntimeError("no"))})
hits = app.search_hits("последние новости спорта")
check("поиск: новости первыми, с датой", hits[0]["url"] == "https://news.ru/n" and hits[0]["date"] == "2026-10-06", hits[0])
FakeDDGS.plan = {k: (0.05, RuntimeError("blocked")) for k in [("text", "google"), ("text", "yahoo"), ("text", "brave"), ("text", "newengine")]}
real_wiki = app.wiki_hits
app.wiki_hits = lambda q: [{"title": "Борщ", "url": "https://ru.wikipedia.org/wiki/Борщ", "body": "суп", "date": ""}]
check("поиск: все молчат -> Википедия", app.search_hits("рецепт борща")[0]["title"] == "Борщ")
app.wiki_hits = lambda q: []
try:
    app.search_hits("рецепт борща"); check("поиск: ничего -> понятная ошибка", False)
except RuntimeError as e:
    check("поиск: ничего -> понятная ошибка", "ничего не нашлось" in str(e) and "blocked" in str(e), str(e))
app.wiki_hits = real_wiki
for k, v in saved_mods.items():
    if v is None: sys.modules.pop(k, None)
    else: sys.modules[k] = v
# меню сайта не вытесняет текст; не влезший абзац — началом
chunks = ["Новости · Бизнес · Экономика · Технологии · Медиа · Спорт · Политика",
          "Новости технологий: компания выпустила новый процессор 2026 года, " + "подробности " * 60]
ex = app.best_excerpt(chunks, "новости технологий", 400)
check("выдержка: меню ниже абзаца, абзац обрезан по слову", ex.startswith("Новости · Бизнес") and "процессор" in ex and ex.endswith("…") and len(ex) <= 420, ex[:120])

# --- поток озвучки: нарезка (без Silero — подменённая модель)
class FakeTTS:
    def __init__(self): self.calls = []
    def apply_tts(self, text, speaker, sample_rate):
        self.calls.append(text); return np.zeros(int(sample_rate * 0.01 * len(text)), np.float32)
fake = FakeTTS(); real_load = app.load_tts; app.load_tts = lambda: fake
played = []
st = app.SpeechStream("kseniya", play=lambda a: played.append(len(a)))
answer = ("Конечно! Сначала подключите синтезатор к компьютеру через USB-кабель. "
          "Затем откройте настройки звука в Windows и выберите MIDI-устройство. "
          "В конце запустите программу, например FL Studio, и проверьте, что клавиши звучат.")
for p in pieces(answer, 4): st.feed(p)
t_first = st.n
wav = st.finish()
check("озвучка: первая фраза ушла в синтез до конца ответа", t_first >= 1 and len(fake.calls) >= 2 and fake.calls[0].startswith("Конечно! Сначала"), f"до конца: {t_first}, всего: {len(fake.calls)}, первый: {fake.calls[0][:50] if fake.calls else ''}")
check("озвучка: всё проиграно и в WAV", len(played) == len(fake.calls) and os.path.getsize(wav) > 1000)
check("озвучка: латиница и числа по-русски в синтезе", any("ю-эс-би" in c or "юэсби" in c.replace(" ", "") for c in fake.calls) and not any("USB" in c for c in fake.calls), [c[:60] for c in fake.calls])
st2 = app.SpeechStream("kseniya")
for p in pieces("слово " * 200, 7): st2.feed(p)
st2.finish()
check("озвучка: длинный текст без точек режется ≤ 400", max(len(c) for c in fake.calls[-5:]) <= 400)
app.load_tts = real_load

print(f"\nИТОГО: {sum(RES)}/{len(RES)} PASS")
