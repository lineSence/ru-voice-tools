# -*- coding: utf-8 -*-
"""
Управление компьютером голосом (без LLM для типовых команд — работает и на CPU).

  «включи Бегущий по лезвию»            → фильм из папки фильмов в VLC на полный экран
  «пауза», «продолжи», «выключи фильм»   → управление VLC (HTTP-интерфейс VLC, только 127.0.0.1)
  «перемотай вперёд на 5 минут», «громче» → перемотка / громкость фильма или системы
  «запусти телеграм», «открой блокнот»   → ярлыки меню «Пуск» + встроенные + псевдонимы
  «заблокируй компьютер», «выключи компьютер» (с подтверждением «да»)

Только белый список действий: никаких произвольных команд оболочки.
Модуль не импортирует app.py — тестируется отдельно (tests/test_pc_control.py).
handle(text, cfg, ask_llm) -> Reply | None (None — не команда, пусть отвечает LLM).
"""

import base64
import difflib
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

IS_WIN = os.name == "nt"

VIDEO_EXT = {".mkv", ".mp4", ".avi", ".mov", ".wmv", ".m4v", ".ts", ".m2ts", ".webm", ".flv",
             ".mpg", ".mpeg", ".vob", ".3gp", ".ogv"}
VLC_PORT = 7862  # 7860 подкасты, 7861 ассистент, 5011 KoboldCpp
INDEX_TTL = 60.0  # сек: папку фильмов и меню «Пуск» пересканируем не чаще
MATCH_OK = 0.78   # ниже — «не нашёл» (или спросить LLM об оригинальном названии)
AMBIG_GAP = 0.08  # два лучших ближе — переспросить, какой
PENDING_SEC = 60.0
DUCK_TO = 0.35    # громкость фильма, пока ассистент слушает / отвечает

# мусор в именах файлов-релизов: всё, начиная с первого такого слова, отрезается
RELEASE_TAGS = re.compile(
    r"\b(?:2160p|1080p|1080i|720p|576p|480p|4k|uhd|hdr10?|hdr|sdr|dv|bluray|blu ray|bdrip|bdremux|"
    r"brrip|remux|web dl|webdl|webrip|web|hdrip|dvdrip|dvd5|dvd9|dvd|hdtv|tvrip|satrip|camrip|ts|"
    r"x264|x265|h264|h 264|h265|h 265|hevc|avc|xvid|divx|aac|ac3|eac3|dts|dts hd|truehd|atmos|"
    r"flac|mp3|5 1|7 1|2 0|rus|eng|ukr|dub|mvo|avo|dvo|sub|subs|extended|unrated|directors cut|"
    r"theatrical|remastered|imax|proper|repack|lostfilm|newstudio|rutracker|by)\b")
YEAR_RE = re.compile(r"(?<!\d)(19[2-9]\d|20[0-4]\d)(?!\d)")

_RU2LAT = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh", "з": "z",
           "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
           "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch",
           "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya"}
_LAT_FIX = [("ks", "x"), ("kv", "qu"), ("dzh", "j"), ("yy", "y")]


def norm(s):
    """Нижний регистр, ё→е, без знаков препинания и подчёркиваний, одинарные пробелы."""
    s = (s or "").lower().replace("ё", "е")
    s = re.sub(r"[\W_]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def to_lat(s):
    """Грубая транслитерация для сравнения «телеграм» ~ «telegram», «дискорд» ~ «discord»."""
    out = "".join(_RU2LAT.get(ch, ch) for ch in s)
    for a, b in _LAT_FIX:
        out = out.replace(a, b)
    return out


def skel(s):
    """Латинский «скелет» звучания: «дискорд» и «discord», «стим» и «steam» совпадают."""
    s = to_lat(norm(s))
    for a, b in (("ph", "f"), ("ck", "k"), ("ch", "h"), ("kh", "h"), ("c", "k"), ("q", "k"),
                 ("w", "v"), ("x", "ks"), ("ea", "i"), ("ee", "i"), ("oo", "u"), ("ou", "au"),
                 ("y", "i"), ("tz", "ts")):
        s = s.replace(a, b)
    s = re.sub(r"(\w)\1+", r"\1", s)                    # двойные буквы
    s = re.sub(r"(?<=\w{3})e\b", "", s)                  # немая e на конце: chrome -> hrom
    return s


def clean_title(name):
    """Blade.Runner.1982.Final.Cut.1080p.BluRay.mkv -> ("blade runner final cut", "1982")."""
    base = os.path.splitext(name)[0] if os.path.splitext(name)[1].lower() in VIDEO_EXT else name
    base = re.sub(r"\[[^\]]*\]", " ", base)            # [rutracker], [Rus Eng]
    n = norm(base)
    year = ""
    # год выпуска — последний правдоподобный год не в начале («Blade Runner 2049 2017» -> 2017,
    # «2049» остаётся частью названия; «1917.2019.mkv» -> «1917», 2019)
    max_year = time.localtime().tm_year + 1
    ms = [m for m in YEAR_RE.finditer(n) if m.start() > 0 and int(m.group(1)) <= max_year]
    if ms:
        m = ms[-1]
        year, n = m.group(1), (n[:m.start()] + " " + n[m.end():])
    t = RELEASE_TAGS.search(n)
    if t and t.start() > 0:
        n = n[:t.start()]
    n = re.sub(r"\s+", " ", n).strip()
    return n, year


# ----------------------------- Нечёткое сравнение -----------------------------

def _word_sim(a, b):
    if a == b:
        return 1.0
    if a.isdigit() or b.isdigit():
        return 0.0                                      # «2049» ≠ «2048»
    return difflib.SequenceMatcher(None, a, b).ratio()


def _sim_words(q, c):
    qw, cw = q.split(), c.split()
    if not qw or not cw:
        return 0.0
    cover_q = sum(max(_word_sim(w, x) for x in cw) for w in qw) / len(qw)
    cover_c = sum(max(_word_sim(x, w) for w in qw) for x in cw) / len(cw)
    whole = difflib.SequenceMatcher(None, q.replace(" ", ""), c.replace(" ", "")).ratio()
    return max(0.85 * cover_q + 0.15 * cover_c, whole * 0.97)


def similarity(query, cand):
    """0..1. Сравниваем как есть и в латинице (русский запрос — английское имя файла)."""
    q, c = norm(query), norm(cand)
    if not q or not c:
        return 0.0
    best = _sim_words(q, c)
    if bool(re.search(r"[а-я]", q)) != bool(re.search(r"[а-я]", c)) or re.search(r"[а-я]", q + c) \
            and re.search(r"[a-z]", q + c):
        best = max(best, _sim_words(skel(q), skel(c)) * 0.97)
    return best


def rank(query, items, names=lambda it: it["names"]):
    """[(оценка, item)] по убыванию; у item несколько имён — берётся лучшее."""
    scored = []
    for it in items:
        s = max((similarity(query, n) for n in names(it)), default=0.0)
        if s > 0:
            scored.append((s, it))
    scored.sort(key=lambda p: -p[0])
    return scored


# ----------------------------- Индексы ----------------------------------------

_INDEX = {}


def _cached(key, ttl, build):
    now = time.time()
    hit = _INDEX.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = build()
    _INDEX[key] = (now, val)
    return val


def reset_index():
    _INDEX.clear()


def scan_movies(folder):
    """Фильмы в папке (рекурсивно). Имя — из файла и из папки фильма, если она своя."""
    out = []
    folder = (folder or "").strip().strip('"')
    if not folder or not os.path.isdir(folder):
        return out
    root = os.path.normcase(os.path.abspath(folder))
    for dirpath, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".") and norm(d) not in ("sample", "samples")]
        vids = [f for f in files if os.path.splitext(f)[1].lower() in VIDEO_EXT
                and "sample" not in f.lower()]
        for f in vids:
            path = os.path.join(dirpath, f)
            title, year = clean_title(f)
            names = [title] if title else []
            show = title
            parent = os.path.basename(dirpath)
            # папка «Бегущий по лезвию (1982)» с одним файлом внутри — тоже имя фильма
            if os.path.normcase(os.path.abspath(dirpath)) != root and len(vids) == 1:
                pt, py = clean_title(parent)
                if pt and pt not in names:
                    names.append(pt)
                    show = pt  # «Матрица (1999)/movie.mkv» — называем по папке
                year = year or py
            if not names:
                continue
            show = (os.path.splitext(f)[0] if len(show or "") < 2 else show).strip()
            show = " ".join(w[:1].upper() + w[1:] for w in show.split())
            out.append({"path": path, "names": names, "year": year,
                        "title": show + (f" ({year})" if year else "")})
    return out


START_MENU_SKIP = re.compile(r"uninstall|удал|readme|help|справк|documentation|документац|"
                             r"license|лиценз|release notes|website|веб сайт|support|faq|"
                             r"what s new|changelog", re.I)

# встроенные программы Windows (часть из них — UWP, ярлыков в «Пуске» у них нет)
BUILTIN_APPS = [
    (["блокнот", "notepad"], "notepad.exe"),
    (["калькулятор", "calculator"], "calc.exe"),
    (["проводник", "explorer", "мой компьютер", "этот компьютер"], "explorer.exe"),
    (["диспетчер задач", "task manager"], "taskmgr.exe"),
    (["панель управления", "control panel"], "control.exe"),
    (["параметры", "настройки windows", "settings"], "ms-settings:"),
    (["paint", "паинт", "пейнт"], "mspaint.exe"),
    (["командная строка", "cmd"], "cmd.exe"),
    (["ножницы", "snipping tool"], "snippingtool.exe"),
]


def start_menu_dirs():
    dirs = []
    for env, sub in (("ProgramData", r"Microsoft\Windows\Start Menu\Programs"),
                     ("APPDATA", r"Microsoft\Windows\Start Menu\Programs")):
        base = os.environ.get(env)
        if base:
            dirs.append(os.path.join(base, sub))
    return dirs


def scan_apps(dirs=None):
    out, seen = [], set()
    for d in (start_menu_dirs() if dirs is None else dirs):
        if not os.path.isdir(d):
            continue
        for dirpath, _, files in os.walk(d):
            for f in files:
                stem, ext = os.path.splitext(f)
                if ext.lower() not in (".lnk", ".url", ".appref-ms") or START_MENU_SKIP.search(stem):
                    continue
                n = norm(stem)
                if not n or n in seen:
                    continue
                seen.add(n)
                out.append({"names": [n], "target": os.path.join(dirpath, f), "title": stem})
    for names, target in BUILTIN_APPS:
        out.append({"names": names, "target": target, "title": names[0], "builtin": True})
    return out


def parse_aliases(raw):
    """Строки «телек = D:\\Фильмы\\...» / «браузер = chrome» -> {норм. имя: цель}.
    Цель — фильм (путь к видео или его название) или программа (путь, ярлык, адрес, имя)."""
    out = {}
    for line in (raw or "").splitlines():
        if "=" not in line or line.strip().startswith("#"):
            continue
        k, v = line.split("=", 1)
        k, v = norm(k), v.strip().strip('"')
        if k and v:
            out[k] = v
    return out


# ----------------------------- VLC --------------------------------------------

def find_vlc(custom=""):
    custom = (custom or "").strip().strip('"')
    if custom:
        return custom if os.path.isfile(custom) else None
    cands = []
    if IS_WIN:
        try:
            import winreg
            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                for key in (r"SOFTWARE\VideoLAN\VLC", r"SOFTWARE\WOW6432Node\VideoLAN\VLC"):
                    try:
                        with winreg.OpenKey(hive, key) as k:
                            v, _ = winreg.QueryValueEx(k, "")      # (по умолчанию) = путь к vlc.exe
                            cands.append(v)
                            d, _ = winreg.QueryValueEx(k, "InstallDir")
                            cands.append(os.path.join(d, "vlc.exe"))
                    except OSError:
                        pass
        except ImportError:
            pass
        for env in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
            if os.environ.get(env):
                cands.append(os.path.join(os.environ[env], "VideoLAN", "VLC", "vlc.exe"))
    w = shutil.which("vlc")
    if w:
        cands.append(w)
    return next((c for c in cands if c and os.path.isfile(c)), None)


class VLC:
    """Один процесс VLC, запущенный ассистентом, + его HTTP-интерфейс (127.0.0.1)."""

    def __init__(self):
        self.proc, self.port, self.pw = None, VLC_PORT, ""
        self.title, self.saved_vol, self.lock = "", None, threading.RLock()

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def _free_port(self):
        for p in [VLC_PORT] + list(range(VLC_PORT + 10, VLC_PORT + 30)):
            with socket.socket() as s:
                try:
                    s.bind(("127.0.0.1", p))
                    return p
                except OSError:
                    continue
        return VLC_PORT

    def play(self, exe, path, title, fullscreen=True):
        with self.lock:
            self.stop()
            self.port, self.pw = self._free_port(), secrets.token_hex(8)
            args = [exe, "--no-one-instance", "--no-playlist-enqueue", "--play-and-exit",
                    "--no-video-title-show", "--extraintf=http", "--http-host=127.0.0.1",
                    f"--http-port={self.port}", f"--http-password={self.pw}"]
            if fullscreen:
                args += ["--fullscreen", "--video-on-top"]
            args.append(path)
            self.proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL, close_fds=True)
            self.title, self.saved_vol = title, None
        if fullscreen:  # окно, открытое из фона, Windows может не развернуть — проверим
            threading.Thread(target=self._ensure_fullscreen, args=(self.proc,), daemon=True).start()

    def _ensure_fullscreen(self, proc, wait=15.0):
        t_end = time.time() + wait
        while time.time() < t_end and proc is self.proc and proc.poll() is None:
            st = self.status()
            if st and st.get("state") in ("playing", "paused"):
                if not st.get("fullscreen"):
                    self.cmd("fullscreen")
                return
            time.sleep(0.5)

    def _req(self, params, timeout=2.0):
        url = f"http://127.0.0.1:{self.port}/requests/status.json"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        auth = base64.b64encode(f":{self.pw}".encode()).decode()
        req = urllib.request.Request(url, headers={"Authorization": "Basic " + auth})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # мимо VPN-прокси
        with opener.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    def status(self):
        if not self.running():
            return None
        try:
            return self._req({})
        except (OSError, ValueError, urllib.error.URLError):
            return None

    def cmd(self, command, val=None):
        if not self.running():
            return None
        p = {"command": command}
        if val is not None:
            p["val"] = val
        try:
            return self._req(p)
        except (OSError, ValueError, urllib.error.URLError):
            return None

    def stop(self):
        with self.lock:
            if self.running():
                try:
                    self.proc.terminate()
                    self.proc.wait(3)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
            self.proc, self.title, self.saved_vol = None, "", None

    # громкость VLC: 0..512, 256 = 100 %
    def volume_delta(self, delta):
        with self.lock:
            if self.saved_vol is not None:  # сейчас приглушено — меняем то, что вернём
                self.saved_vol = max(0, min(512, self.saved_vol + delta))
                return True
            return self.cmd("volume", f"{delta:+d}") is not None

    def duck(self):
        with self.lock:
            if self.saved_vol is not None:
                return
            st = self.status()
            if not st or st.get("state") != "playing":
                return
            vol = int(st.get("volume") or 256)
            if self.cmd("volume", str(int(vol * DUCK_TO))) is not None:
                self.saved_vol = vol

    def unduck(self):
        with self.lock:
            if self.saved_vol is None:
                return
            vol, self.saved_vol = self.saved_vol, None
            self.cmd("volume", str(vol))


PLAYER = VLC()


# ----------------------------- Система ----------------------------------------

def _open(target):
    """Запуск ярлыка / exe / адреса. Аргументов не передаём — только то, что указал владелец."""
    if IS_WIN:
        os.startfile(target)  # noqa: (только Windows)
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    exe = target if (os.path.isfile(target) and os.access(target, os.X_OK)) else None
    subprocess.Popen([exe] if exe else [opener, target], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)


class NotSupported(RuntimeError):
    pass


VK_VOLUME_MUTE, VK_VOLUME_DOWN, VK_VOLUME_UP = 0xAD, 0xAE, 0xAF


def system_volume(key, times=1):
    """Мультимедийные клавиши громкости (шаг Windows — 2 %). Без сторонних пакетов."""
    if not IS_WIN:
        raise NotSupported("громкость системы меняю только на Windows")
    import ctypes
    u32 = ctypes.windll.user32
    for _ in range(times):
        u32.keybd_event(key, 0, 0, 0)
        u32.keybd_event(key, 0, 2, 0)  # KEYEVENTF_KEYUP


def lock_pc():
    if not IS_WIN:
        raise NotSupported("блокировку умею только на Windows")
    import ctypes
    ctypes.windll.user32.LockWorkStation()


def power(action):
    if not IS_WIN:
        raise NotSupported("выключение умею только на Windows")
    no_win = 0x08000000  # CREATE_NO_WINDOW
    if action == "shutdown":
        subprocess.Popen(["shutdown", "/s", "/t", "15"], creationflags=no_win)
    elif action == "restart":
        subprocess.Popen(["shutdown", "/r", "/t", "15"], creationflags=no_win)
    elif action == "abort":
        subprocess.Popen(["shutdown", "/a"], creationflags=no_win)
    elif action == "sleep":  # при включённой гибернации Windows уйдёт в неё
        subprocess.Popen(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
                         creationflags=no_win)


# ----------------------------- Разбор команд ----------------------------------

class Reply:
    """text — что сказать; before/after — действие до/после озвучки; note — строка в «Диалоге»."""

    def __init__(self, text, before=None, after=None, note=""):
        self.text, self.before, self.after, self.note = text, before, after, note

    def __repr__(self):
        return f"Reply({self.text!r}, note={self.note!r})"


PENDING = {"kind": None, "until": 0.0, "data": None}

VERB = (r"(?:включи(?:те)?|поставь(?:те)?|запусти(?:те)?|открой(?:те)?|покажи(?:те)?|"
        r"воспроизведи|врубай|вруби|давай(?:\s+посмотрим)?|хочу\s+посмотреть|"
        r"начни\s+(?:показ|смотреть))")
POLITE = r"(?:пожалуйста|ка|мне|нам|быстренько|скорее)"
MOVIE_WORD = r"(?:фильм|кино|кинофильм|мультфильм|мульт|мультик|сериал|видео|ролик)"
FULLSCREEN_RE = re.compile(r"\b(?:на|во)\s+(?:весь|полный|целый)\s+экран\b|\bв\s+полноэкранном"
                           r"(?:\s+режиме)?\b|\bполноэкранно\b|\bна\s+фулл?\s*скрин\b")
WINDOW_RE = re.compile(r"\bв\s+окне\b|\bне\s+на\s+(?:весь|полный)\s+экран\b")
PLAY_RE = re.compile(rf"^(?:{POLITE}\s+)*{VERB}(?:\s+{POLITE})*\s+(?:(?P<kind>{MOVIE_WORD}\w*)\s+)?"
                     r"(?P<what>.+)$")
APP_ONLY_RE = re.compile(r"^(?:запусти(?:те)?|открой(?:те)?)\b")
APP_WORD_RE = re.compile(r"^(?:программу|приложение|игру|прогу)\s+")

ORDINALS = {"первый": 0, "первую": 0, "первое": 0, "один": 0, "1": 0,
            "второй": 1, "вторую": 1, "второе": 1, "два": 1, "2": 1,
            "третий": 2, "третью": 2, "третье": 2, "три": 2, "3": 2,
            "четвертый": 3, "четвертую": 3, "четыре": 3, "4": 3,
            "последний": -1, "последнюю": -1}
YES_RE = re.compile(r"^(?:да|ага|конечно|подтверждаю|выключай|перезагружай|давай|точно)\b")
NO_RE = re.compile(r"^(?:нет|не надо|не нужно|отмена|отмени|никакой|ничего|стоп)\b")

NUM_WORDS = {"одну": 1, "один": 1, "одна": 1, "две": 2, "два": 2, "три": 3, "четыре": 4, "пять": 5,
             "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "пятнадцать": 15,
             "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "полминуты": 30,
             "пару": 2, "несколько": 3}


def _amount(text, default):
    m = re.search(r"\b(\d+)\b", text)
    if m:
        return int(m.group(1))
    for w, n in NUM_WORDS.items():
        if re.search(rf"\b{w}\b", text):
            return n
    return default


def _seek_seconds(t):
    if "полминуты" in t:
        return 30
    n = _amount(t, None)
    if re.search(r"\bчас", t):
        return (n or 1) * 3600
    if re.search(r"\bмин", t):
        return (n or 1) * 60
    if re.search(r"\bсек", t):
        return n or 10
    return (n * 60) if n else 30  # «перемотай вперёд» — на 30 с, «на 5» — минуты


def _say_title(item):
    return item.get("title") or item["names"][0]


def _ask_titles(ask_llm, what):
    """Русское (возможно, искажённое распознаванием) название -> варианты оригинального."""
    if not ask_llm:
        return []
    prompt = ("Пользователь голосом попросил включить фильм «" + what + "». Распознавание речи "
              "могло исказить слова. Назови, какой это фильм: его название на языке оригинала и "
              "английское название, через « | », одной строкой, без года и без пояснений. "
              "Если не знаешь такого фильма — напиши только: НЕТ")
    try:
        ans = (ask_llm(prompt) or "").strip()
    except Exception as e:
        print(f"[pc] LLM не ответила: {e}", flush=True)
        return []
    ans = ans.splitlines()[0] if ans else ""
    if not ans or re.match(r"^\W*нет\b", ans, re.I):
        return []
    return [p.strip(" «»\"'.*") for p in re.split(r"\||/|;", ans) if p.strip(" «»\"'.*")][:3]


def _movie_choice_reply(options, fullscreen, exe):
    PENDING.update(kind="movie", until=time.time() + PENDING_SEC,
                   data={"options": options, "fullscreen": fullscreen, "exe": exe})
    names = [_say_title(o) for o in options]
    listing = "; ".join(f"{i + 1} — {n}" for i, n in enumerate(names))
    return Reply(f"Нашлось несколько фильмов: {listing}. Какой включить? Скажите номер или год.",
                 note="🎬 Несколько похожих фильмов: " + " | ".join(names))


def _play_reply(item, fullscreen, exe):
    title = _say_title(item)

    def go():
        PLAYER.play(exe, item["path"], title, fullscreen=fullscreen)
    print(f"[pc] VLC: {item['path']}", flush=True)
    return Reply(f"Включаю {title}.", after=go, note=f"🎬 VLC{' (полный экран)' if fullscreen else ''}: "
                 f"{item['path']}")


def find_movie(what, cfg, ask_llm=None):
    """-> ("one", item) | ("many", [items]) | ("none", None) | ("nodir", None)."""
    folder = cfg.get("movies_dir", "")
    movies = _cached(("movies", folder), INDEX_TTL, lambda: scan_movies(folder))
    if not movies:
        return "nodir", None
    aliases = parse_aliases(cfg.get("pc_aliases", ""))
    q = norm(what)
    for k, v in aliases.items():  # псевдоним: «телек = Blade Runner» или путь к файлу
        if similarity(q, k) >= 0.85:
            if os.path.isfile(v):
                return "one", {"path": v, "names": [k], "title": clean_title(os.path.basename(v))[0]}
            q = norm(v)
            break
    year = (YEAR_RE.search(q) or [None])[0]
    # «бегущий по лезвию 2049» — год/номер в запросе не мешает и уточняет выбор
    scored = rank(q, movies)
    queries = [q]
    if not scored or scored[0][0] < MATCH_OK:
        for alt in _ask_titles(ask_llm, what):
            queries.append(norm(alt) + (f" {year}" if year and year not in norm(alt) else ""))
        if len(queries) > 1:
            best = {}
            for qq in queries:
                for s, it in rank(qq, movies):
                    if s > best.get(it["path"], (0, None))[0]:
                        best[it["path"]] = (s, it)
            scored = sorted(best.values(), key=lambda p: -p[0])
    if year:  # год назван — фильмы с этим годом вперёд
        scored = sorted(scored, key=lambda p: -(p[0] + (0.1 if p[1].get("year") == year else 0)))
    good = [p for p in scored if p[0] >= MATCH_OK]
    print(f"[pc] фильм «{what}» (запросы {queries}): "
          + ", ".join(f"{_say_title(it)}={s:.2f}" for s, it in scored[:3]), flush=True)
    if not good:
        return "none", None
    close = [it for s, it in good if good[0][0] - s < AMBIG_GAP]
    if len(close) > 1 and not (year and sum(1 for it in close if it.get("year") == year) == 1):
        close = sorted(close[:4], key=lambda it: it.get("year") or "9999")  # «первый» — старший
        return "many", close
    if year:
        same = [it for it in close if it.get("year") == year]
        if same:
            return "one", same[0]
    return "one", good[0][1]


def _looks_like_target(v):
    """Путь, ярлык, exe или адрес — а не название (фильма или программы)."""
    return bool(re.search(r"[\\/]|^\w+:|\.(?:exe|lnk|url|bat|cmd|appref-ms)$", v.strip(), re.I))


def find_app(what, cfg):
    aliases = parse_aliases(cfg.get("pc_aliases", ""))
    q = norm(APP_WORD_RE.sub("", norm(what)))
    apps = _cached(("apps",), INDEX_TTL, scan_apps)
    for k, v in sorted(aliases.items(), key=lambda kv: -similarity(q, kv[0])):
        s = similarity(q, k)
        if s < 0.85:
            break
        if os.path.splitext(v)[1].lower() in VIDEO_EXT:
            break  # псевдоним фильма — его найдёт find_movie
        if _looks_like_target(v):
            return s, {"names": [k], "target": v, "title": k}
        sc = rank(v, apps)  # «браузер = Google Chrome» — название программы из «Пуска»
        if sc and sc[0][0] >= 0.86:
            return s, sc[0][1]
        break  # «киношка = Interstellar» — название фильма
    scored = rank(q, apps)
    return scored[0] if scored else (0.0, None)


def _resolve_pending(t, cfg):
    kind, data = PENDING["kind"], PENDING["data"]
    if not kind or time.time() > PENDING["until"]:
        PENDING.update(kind=None, data=None)
        return None
    if NO_RE.match(t):
        PENDING.update(kind=None, data=None)
        return Reply("Хорошо, отменила.")
    if kind == "power":
        if YES_RE.match(t):
            PENDING.update(kind=None, data=None)
            act = data
            words = {"shutdown": "Выключаю компьютер через 15 секунд. Чтобы отменить, скажите: "
                                 "отмени выключение.",
                     "restart": "Перезагружаю компьютер через 15 секунд. Чтобы отменить, скажите: "
                                "отмени выключение.",
                     "sleep": "Перевожу компьютер в спящий режим."}
            return Reply(words[act], after=lambda: power(act), note=f"⏻ {act}")
        return None
    if kind == "movie":
        opts = data["options"]
        pick = None
        words = t.split()
        for w in words:
            if w in ORDINALS and (ORDINALS[w] < len(opts)):
                pick = opts[ORDINALS[w]]
                break
        if pick is None:
            y = YEAR_RE.search(t)
            if y:
                same = [o for o in opts if o.get("year") == y.group(1)
                        or y.group(1) in " ".join(o["names"])]
                pick = same[0] if len(same) == 1 else None
        if pick is None and len(t) > 3:
            sc = rank(t, opts)
            if sc and sc[0][0] >= MATCH_OK and (len(sc) == 1 or sc[0][0] - sc[1][0] >= AMBIG_GAP):
                pick = sc[0][1]
        if pick is None:
            return None
        PENDING.update(kind=None, data=None)
        return _play_reply(pick, data["fullscreen"], data["exe"])
    return None


def _player_reply(t):
    """Команды фильму, который сейчас идёт в VLC (запущенном ассистентом)."""
    if not PLAYER.running():
        return None
    if re.search(r"^(?:поставь\s+)?(?:на\s+)?пауз\w*|^(?:останови|приостанови|подожди|замри|стоп)\b"
                 r"(?!.*компьют)|^пауза\b", t) and not re.search(r"выключи|закрой", t):
        return Reply("Пауза.", before=lambda: PLAYER.cmd("pl_forcepause"))
    if re.search(r"^(?:продолж\w*|дальше|играй|воспроизвод\w*|сними\s+с\s+паузы|сними\s+паузу|"
                 r"сними\s+паузу|сними\s+с\s+пауз\w*|сними\s+с\s+пауз|убери\s+паузу|сними\s+паузу)\b", t):
        return Reply("Продолжаю.", after=lambda: PLAYER.cmd("pl_forceresume"))
    if re.search(r"^(?:выключи|закрой|останови|заверши|убери)\s+(?:этот\s+)?(?:фильм|кино|видео|"
                 r"плеер|vlc|влц|мультик|мультфильм|сериал)\b|^(?:хватит|выключи\s+все)$", t):
        title = PLAYER.title
        return Reply("Выключаю фильм.", before=PLAYER.stop, note=f"⏹ {title}")
    m = re.search(r"^(?:перемотай|промотай|мотни|отмотай|верни|прокрути|пропусти)\b(.*)$", t)
    if m:
        rest = m.group(1)
        back = bool(re.search(r"назад|отмотай|верни", t))
        sec = _seek_seconds(rest)
        val = f"{'-' if back else '+'}{sec}s"
        unit = (f"{sec // 3600} ч" if sec % 3600 == 0 and sec >= 3600 else
                f"{sec // 60} мин" if sec % 60 == 0 and sec >= 60 else f"{sec} с")
        return Reply(("Назад" if back else "Вперёд") + f" на {unit}.",
                     after=lambda: PLAYER.cmd("seek", val), note=f"⏩ seek {val}")
    if re.fullmatch(r"(?:сделай\s+|разверни\s+|открой\s+|включи\s+)?(?:фильм\s+|плеер\s+|кино\s+)?"
                    r"(?:на\s+|во\s+)?(?:весь|полный)\s+экран|(?:включи\s+)?полноэкранный(?:\s+режим)?", t):
        return Reply("Разворачиваю.", after=lambda: (PLAYER.status() or {}).get("fullscreen")
                     or PLAYER.cmd("fullscreen"))
    if re.fullmatch(r"сверни\s+(?:фильм|плеер|окно|кино)|выйди\s+из\s+полноэкранного(?:\s+режима)?|"
                    r"(?:сделай\s+|покажи\s+)?(?:фильм\s+)?(?:в\s+окно|в\s+окне)", t):
        return Reply("Хорошо.", after=lambda: (PLAYER.status() or {}).get("fullscreen")
                     and PLAYER.cmd("fullscreen"))
    v = _volume_dir(t)
    if v:
        step = 26 * (2 if re.search(r"намного|сильно|гораздо|ещё громче|ещe тише", t) else 1)
        delta = step if v > 0 else -step
        return Reply("Громче." if v > 0 else "Тише.",
                     after=lambda: PLAYER.volume_delta(delta), note=f"🔊 VLC {delta:+d}")
    return None


def _volume_dir(t):
    if re.search(r"^(?:сделай\s+|сделайте\s+|чуть\s+|немного\s+|намного\s+|еще\s+|звук\s+|погромче|"
                 r"прибавь|добавь|увеличь|подними|повысь)*.*\b(?:громче|погромче|прибавь\s+(?:звук|"
                 r"громкость)|увеличь\s+громкость|добавь\s+(?:звук|громкость)|подними\s+громкость)\b", t):
        return 1
    if re.search(r"\b(?:тише|потише|убавь\s+(?:звук|громкость)|уменьши\s+громкость|"
                 r"понизь\s+громкость|сделай\s+потише)\b", t):
        return -1
    return 0


def _system_reply(t):
    """Компьютер целиком: громкость Windows, блокировка, выключение (с подтверждением)."""
    v = _volume_dir(t)
    if v:
        n = 10 if re.search(r"намного|сильно|гораздо", t) else 5
        key = VK_VOLUME_UP if v > 0 else VK_VOLUME_DOWN
        return Reply("Громче." if v > 0 else "Тише.", after=lambda: system_volume(key, n))
    if re.search(r"^(?:выключи|отключи|убери)\s+звук\b|^без\s+звука$|^(?:включи|верни)\s+звук\b", t):
        return Reply("Хорошо.", after=lambda: system_volume(VK_VOLUME_MUTE))
    if re.search(r"(?:заблокируй|блокируй|заблокировать)\s+(?:компьютер|экран|комп|пк)", t):
        return Reply("Блокирую компьютер.", after=lock_pc, note="🔒 блокировка")
    if re.search(r"отмени(?:ть)?\s+(?:выключение|перезагрузку)", t):
        return Reply("Выключение отменено.", before=lambda: power("abort"))
    for pat, act, ask in (
            (r"^(?:выключи|выруби|отключи|заверши\s+работу)\s+(?:компьютер|комп|пк|систему)", "shutdown",
             "Выключить компьютер?"),
            (r"^(?:перезагрузи|перезапусти)\s+(?:компьютер|комп|пк|систему)", "restart",
             "Перезагрузить компьютер?"),
            (r"(?:спящий\s+режим|усыпи\s+(?:компьютер|комп|пк)|(?:отправь|переведи)\s+(?:компьютер\s+)?"
             r"в\s+сон)", "sleep", "Перевести компьютер в спящий режим?")):
        if re.search(pat, t):
            if not IS_WIN:
                return Reply("Это я умею только на Windows.")
            PENDING.update(kind="power", until=time.time() + 30, data=act)
            return Reply(ask + " Скажите: ассистент, да.")
    return None


def _launch_reply(t, cfg, ask_llm):
    m = PLAY_RE.match(t)
    if not m:
        return None
    what, kind = m.group("what"), m.group("kind")
    fullscreen = not WINDOW_RE.search(what)
    what = WINDOW_RE.sub(" ", FULLSCREEN_RE.sub(" ", what))
    what = re.sub(rf"\b{POLITE}\b", " ", what)
    what = re.sub(r"\s+", " ", what).strip()
    if not what or what in ("звук", "свет", "музыку", "музыка", "радио"):
        return None  # «включи звук» — системная команда, «включи музыку» — пока не умеем
    if re.search(r"^(?:звук|громкость)\b", what):
        return None
    app_only = bool(APP_ONLY_RE.match(t)) and not kind and not FULLSCREEN_RE.search(t)
    app_score, app = (0.0, None) if kind else find_app(what, cfg)
    exe = find_vlc(cfg.get("vlc_path", ""))
    if app and app_score >= 0.86 and (app_only or app_score >= 0.93):
        target = app["target"]
        print(f"[pc] программа «{what}» -> {target} ({app_score:.2f})", flush=True)
        return Reply(f"Запускаю {app['title']}.", before=lambda: _open(target),
                     note=f"🚀 {target}")
    # длинная фраза без слова «фильм» — скорее просьба к LLM («открой секрет, как…»): модель
    # про название не спрашиваем, годится только точное совпадение с фильмом из папки
    long_phrase = not kind and len(what.split()) > 4
    st, res = find_movie(what, cfg, ask_llm=None if (long_phrase or (app_only and app_score >= MATCH_OK))
                         else ask_llm)
    if st in ("one", "many") and not exe:
        return Reply("Не нашла плеер VLC. Установите VLC или укажите путь к vlc.exe в разделе "
                     "«Управление компьютером».", note="⚠️ VLC не найден")
    if st == "one":
        return _play_reply(res, fullscreen, exe)
    if st == "many":
        return _movie_choice_reply(res, fullscreen, exe)
    if app and app_score >= MATCH_OK:
        target = app["target"]
        print(f"[pc] программа «{what}» -> {target} ({app_score:.2f})", flush=True)
        return Reply(f"Запускаю {app['title']}.", before=lambda: _open(target),
                     note=f"🚀 {target}")
    if st == "nodir" and kind:
        return Reply("Папка с фильмами не указана или пуста. Укажите её в разделе «Управление "
                     "компьютером».", note="⚠️ папка фильмов не задана")
    if long_phrase:
        return None  # «открой секрет, как варить борщ» — не команда, пусть отвечает LLM
    if kind or not app_only:
        return Reply(f"Не нашла ни фильма, ни программы «{what}».",
                     note=f"🔍 не найдено: «{what}»")
    return Reply(f"Не нашла программу «{what}».", note=f"🔍 не найдено: «{what}»")


def handle(text, cfg, ask_llm=None):
    """Голосовая команда компьютеру -> Reply, или None (это не команда — ответит LLM).
    ask_llm(prompt) -> короткий ответ модели (оригинальное название фильма) или None."""
    if not cfg.get("pc_control", True):
        return None
    t = norm(text)
    t = re.sub(r"^(?:слушай|эй|так|ну|а|пожалуйста)\s+", "", t)
    if not t:
        return None
    for fn in (lambda: _resolve_pending(t, cfg), lambda: _player_reply(t), lambda: _system_reply(t),
               lambda: _launch_reply(t, cfg, ask_llm)):
        r = fn()
        if r is not None:
            return r
    return None


def run(fn):
    """Выполнить действие Reply; ошибка -> текст для пользователя (или None)."""
    if not fn:
        return None
    try:
        fn()
        return None
    except NotSupported as e:
        return str(e)
    except Exception as e:
        print(f"[pc] ошибка действия: {e}", flush=True)
        return f"Не получилось: {str(e)[:120]}"


def summary(cfg):
    """Строка для GUI: сколько фильмов и программ видно, найден ли VLC."""
    folder = cfg.get("movies_dir", "")
    movies = _cached(("movies", folder), INDEX_TTL, lambda: scan_movies(folder))
    apps = _cached(("apps",), INDEX_TTL, scan_apps)
    exe = find_vlc(cfg.get("vlc_path", ""))
    parts = [f"🎬 Фильмов: {len(movies)}" if folder else "🎬 Папка фильмов не указана",
             f"🚀 Программ: {len(apps)}",
             f"▶️ VLC: {exe}" if exe else "⚠️ VLC не найден — укажите путь к vlc.exe"]
    if PLAYER.running():
        parts.append(f"сейчас идёт: {PLAYER.title}")
    return " · ".join(parts)
