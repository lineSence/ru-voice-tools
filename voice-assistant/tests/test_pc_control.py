"""Управление компьютером (pc_control.py): названия фильмов, нечёткий поиск, переспрос,
поддельный VLC (запуск, полный экран, пауза, перемотка, громкость, приглушение), программы
из «Пуска», система, встраивание в app.answer_and_speak. Без микрофона, GPU и Windows."""
import json, os, sys, tempfile, time
from _paths import HERE  # noqa: E402  (путь к app.py)
sys.argv = ["x"]
import pc_control as pc

RES = []
def check(name, cond, detail=""):
    RES.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  | {detail}" if detail else ""), flush=True)

# --- 1. имена файлов-релизов
for f, want in [("Blade.Runner.1982.Final.Cut.1080p.BluRay.x264.mkv", ("blade runner final cut", "1982")),
                ("Blade.Runner.2049.2017.2160p.HDR.mkv", ("blade runner 2049", "2017")),
                ("1917.2019.WEB-DL.mkv", ("1917", "2019")),
                ("Бегущий по лезвию (1982) BDRip [rutracker].avi", ("бегущий по лезвию", "1982")),
                ("Интерстеллар_2014_RUS.mkv", ("интерстеллар", "2014"))]:
    got = pc.clean_title(f)
    check(f"clean_title {f}", got == want, str(got))

# --- 2. похожесть: ошибки распознавания и латиница/кириллица
for a, b, lo in [("берущий по лезвия", "бегущий по лезвию", 0.85), ("телеграм", "Telegram", 0.9),
                 ("дискорд", "Discord", 0.9), ("стим", "Steam", 0.9), ("интерстелар", "интерстеллар", 0.85)]:
    s = pc.similarity(a, b)
    check(f"похоже «{a}» ~ «{b}»", s >= lo, f"{s:.2f}")
for a, b in [("телеграм", "Steam"), ("матрица", "интерстеллар"), ("2049", "2048")]:
    s = pc.similarity(a, b)
    check(f"непохоже «{a}» ≁ «{b}»", s < pc.MATCH_OK, f"{s:.2f}")

# --- папка фильмов
root = tempfile.mkdtemp(prefix="movies_")
for rel in ["Blade.Runner.1982.Final.Cut.1080p.BluRay.x264.mkv", "Blade.Runner.2049.2017.2160p.mkv",
            "Интерстеллар.2014.mkv", "Матрица (1999)/movie.mkv", "Матрица (1999)/sample.mkv",
            "Джентльмены.2019.BDRip.avi", "readme.txt"]:
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").close()
movies = pc.scan_movies(root)
check("scan_movies: 5 фильмов, sample и txt пропущены", len(movies) == 5, str([m["title"] for m in movies]))
check("имя из папки фильма", any("матрица" in m["names"] for m in movies))

log = os.path.join(tempfile.mkdtemp(), "vlc.log")
os.environ["FAKE_VLC_LOG"] = log
FAKE = os.path.join(HERE, "fake_vlc.py")
os.chmod(FAKE, 0o755)  # из git (push_files) файл приходит без флага исполнения
cfg = {"pc_control": True, "movies_dir": root, "vlc_path": FAKE, "pc_aliases": ""}
llm_calls = []
def ask_llm(prompt):
    llm_calls.append(prompt)
    return "Blade Runner | Blade Runner"

def records():
    if not os.path.exists(log):
        return []
    return [json.loads(l) for l in open(log, encoding="utf-8")]

def wait_for(pred, sec=8):
    t = time.time() + sec
    while time.time() < t:
        if pred():
            return True
        time.sleep(0.1)
    return False

# --- 3. главный пример: ошибка Whisper + русское название, английские файлы -> переспрос
r = pc.handle("включи Берущий по лезвия на полный экран", cfg, ask_llm)
check("LLM спрошена об оригинальном названии", len(llm_calls) == 1)
check("два «Бегущих» -> переспрос", r is not None and "Какой включить" in r.text and pc.PENDING["kind"] == "movie",
      repr(r))
r = pc.handle("второй", cfg, ask_llm)
check("«второй» -> 2049", r is not None and r.text.startswith("Включаю") and "2049" in r.text and r.after, repr(r))
check("действие после озвучки", pc.run(r.after) is None and pc.PLAYER.running())
check("VLC: полный экран, HTTP только 127.0.0.1, файл 2049",
      wait_for(lambda: any("args" in x for x in records())) and
      (lambda a: "--fullscreen" in a and "--http-host=127.0.0.1" in a and a[-1].endswith("Blade.Runner.2049.2017.2160p.mkv"))
      (next(x["args"] for x in records() if "args" in x)))
check("окно не развернулось само -> команда fullscreen",
      wait_for(lambda: any(x.get("cmd") == "fullscreen" for x in records())))

# --- 4. год в запросе, без LLM; кириллическое имя с опечаткой
llm_calls.clear()
r = pc.handle("включи фильм бегущий по лезвию 1982", dict(cfg), None)
check("без LLM русское название английского файла не находится", r is not None and "Не нашла" in r.text, repr(r))
r = pc.handle("поставь интерстелар", cfg, ask_llm)
check("«поставь интерстелар» -> Интерстеллар без LLM", r and "нтерстеллар" in r.text and not llm_calls, repr(r))
r = pc.handle("включи матрицу в окне", cfg, ask_llm)
check("«в окне» -> без полного экрана", r and "атриц" in r.text, repr(r))
pc.run(r.after)
wait_for(lambda: pc.PLAYER.status())
a = [x["args"] for x in records() if "args" in x][-1]
check("VLC без --fullscreen", "--fullscreen" not in a and a[-1].endswith("movie.mkv"), str(a[-2:]))
check("прежний VLC закрыт, работает один", pc.PLAYER.running())

# --- 5. управление идущим фильмом
def cmds():
    return [(x["cmd"], x["val"]) for x in records() if "cmd" in x]
n0 = len(cmds())
r = pc.handle("пауза", cfg); pc.run(r.before)
r2 = pc.handle("продолжи", cfg); pc.run(r2.after)
r3 = pc.handle("перемотай вперёд на 5 минут", cfg); pc.run(r3.after)
r4 = pc.handle("отмотай назад на 30 секунд", cfg); pc.run(r4.after)
r5 = pc.handle("сделай погромче", cfg); pc.run(r5.after)
got = cmds()[n0:]
check("пауза/продолжи/перемотка/громче", got == [("pl_forcepause", None), ("pl_forceresume", None),
      ("seek", "+300s"), ("seek", "-30s"), ("volume", "+26")], str(got))
check("тексты ответов", [x.text for x in (r, r2, r3, r4)] == ["Пауза.", "Продолжаю.", "Вперёд на 5 мин.", "Назад на 30 с."])

# приглушение: wake word во время фильма -> громкость 35 %, потом прежняя; «громче» в это время
pc.PLAYER.duck()
st = pc.PLAYER.status()
check("duck: громкость фильма понижена", st["volume"] == int(282 * pc.DUCK_TO), str(st["volume"]))
r = pc.handle("тише", cfg); pc.run(r.after)
pc.PLAYER.unduck()
check("unduck: прежняя громкость с учётом «тише»", pc.PLAYER.status()["volume"] == 282 - 26,
      str(pc.PLAYER.status()["volume"]))

r = pc.handle("выключи фильм", cfg); pc.run(r.before)
check("«выключи фильм» -> VLC закрыт", not pc.PLAYER.running() and r.text == "Выключаю фильм.")
check("без фильма «пауза» — не команда", pc.handle("пауза", cfg) is None)

# --- 6. программы из «Пуска»
sm = tempfile.mkdtemp(prefix="startmenu_")
for f in ["Telegram.lnk", "Steam/Steam.lnk", "Steam/Uninstall Steam.lnk", "OBS Studio.lnk", "Discord.url"]:
    p = os.path.join(sm, f)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").close()
pc.start_menu_dirs = lambda: [sm]
pc.reset_index()
opened = []
pc._open = opened.append
for phrase, want in [("запусти телеграм", "Telegram.lnk"), ("открой стим", "Steam.lnk"),
                     ("запусти программу обс студио", "OBS Studio.lnk"), ("открой дискорд", "Discord.url"),
                     ("открой блокнот", "notepad.exe")]:
    opened.clear()
    r = pc.handle(phrase, cfg, ask_llm)
    ok = r is not None and r.before and pc.run(r.before) is None and opened and opened[0].endswith(want)
    check(f"«{phrase}» -> {want}", ok, repr(r) + " " + str(opened))
check("деинсталлятор не в списке", not any("Uninstall" in a["title"] for a in pc.scan_apps([sm])))
cfg_al = dict(cfg, pc_aliases="браузер = C:\\Program Files\\Chrome\\chrome.exe\nкиношка = Interstellar")
opened.clear()
r = pc.handle("открой браузер", cfg_al); pc.run(r.before)
check("псевдоним программы", opened == ["C:\\Program Files\\Chrome\\chrome.exe"], str(opened))
r = pc.handle("включи киношку", dict(cfg_al, movies_dir=root), None)
check("псевдоним фильма (английское название к русскому файлу)", r is not None and "нтерстеллар" in r.text, repr(r))
cfg_al2 = dict(cfg, pc_aliases="мессенджер = Telegram")
opened.clear()
r = pc.handle("открой мессенджер", cfg_al2); pc.run(r.before)
check("псевдоним -> название программы из «Пуска»", opened and opened[0].endswith("Telegram.lnk"), str(opened))

# --- 7. система (на Linux — честный отказ), подтверждение выключения
r = pc.handle("выключи звук", cfg)
check("«выключи звук» -> mute (на Linux — объяснение)", r and "Windows" in (pc.run(r.after) or ""))
r = pc.handle("громче", cfg)
check("«громче» без фильма -> громкость системы", r and "Windows" in (pc.run(r.after) or ""))
check("«выключи компьютер» не на Windows", pc.handle("выключи компьютер", cfg).text.startswith("Это я умею"))
pc.IS_WIN, done = True, []
pc.power = done.append
r = pc.handle("выключи компьютер", cfg)
check("выключение — сначала вопрос", r and "Выключить компьютер?" in r.text and not done)
r = pc.handle("да", cfg); pc.run(r.after)
check("«да» -> shutdown", done == ["shutdown"], str(done))
pc.handle("перезагрузи компьютер", cfg)
r = pc.handle("нет", cfg)
check("«нет» -> отмена", r.text.startswith("Хорошо") and done == ["shutdown"])
pc.IS_WIN = os.name == "nt"

# --- 8. не команды -> ответит LLM
for q in ["какая погода завтра", "расскажи про фильм бегущий по лезвию", "что такое MIDI",
          "открой секрет как правильно варить борщ", "включи музыку", ""]:
    check(f"не команда: «{q}»", pc.handle(q, cfg, ask_llm) is None)
check("выключено в настройках", pc.handle("включи интерстелар", dict(cfg, pc_control=False)) is None)

# --- 9. встраивание в app.py: команда не идёт в LLM, ответ — в «Диалоге»
import app
spoken = []
class FakeSpeech:
    def __init__(self, voice, play=None, on_start=None): self.play = play
    def feed(self, t): spoken.append(t)
    def finish(self): return "/tmp/fake.wav"
    def cancel(self): pass
app.SpeechStream = FakeSpeech
app.think_answer = lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM не нужна"))
app.stream_text = lambda cfg, msgs, mt: iter(["Blade ", "Runner"])
acfg = dict(app.DEFAULT_SETTINGS, **cfg)
pc.reset_index()
ans, wav = app.answer_and_speak("Включи Интерстелар на полный экран", acfg)
check("app: команда -> «Включаю…», без LLM", ans.startswith("Включаю") and spoken == [ans], ans)
check("app: в истории с заметкой 🎬", app.HISTORY[-1]["note"].startswith("🎬"), app.HISTORY[-1]["note"])
check("app: фильм запущен после озвучки", pc.PLAYER.running())
spoken.clear()
ans, _ = app.answer_and_speak("включи берущий по лезвия", acfg)
check("app: llm_short -> переспрос", "Какой включить" in ans, ans)
pc.PLAYER.stop()
check("app: настройки по умолчанию", all(k in app.DEFAULT_SETTINGS for k in ("pc_control", "movies_dir", "vlc_path", "pc_aliases"))
      and app.SETTING_FIELDS[-4:] == ["pc_control", "movies_dir", "vlc_path", "pc_aliases"])

print(f"\n{sum(RES)}/{len(RES)} PASS", flush=True)
sys.exit(0 if all(RES) else 1)
