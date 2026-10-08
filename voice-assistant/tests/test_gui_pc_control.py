"""GUI раздела «🖥 Управление компьютером» (Playwright): поля сохраняются в settings.json,
«Обновить списки» показывает найденное, текстовая команда запускает фильм в поддельном VLC
без LLM (движок не нужен: режим LiteLLM без адреса)."""
import json, os, sys, tempfile, threading, time
from _paths import HERE, WORK  # noqa: E402
sys.argv = ["x"]
import app  # noqa: E402
import pc_control as pc  # noqa: E402
from playwright.sync_api import sync_playwright

RES = []
def check(name, cond, detail=""):
    RES.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  | {detail}" if detail else ""), flush=True)

root = tempfile.mkdtemp(prefix="movies_")
for f in ["Интерстеллар.2014.mkv", "Blade.Runner.1982.mkv"]:
    open(os.path.join(root, f), "wb").close()
os.environ["FAKE_VLC_LOG"] = os.path.join(WORK, "gui_vlc.log")
os.chmod(os.path.join(HERE, "fake_vlc.py"), 0o755)
app.SETTINGS_PATH = os.path.join(WORK, "gui_pc_settings.json")
json.dump(dict(app.DEFAULT_SETTINGS, llm_source=app.SRC_LITELLM, stt_engine=app.STT_VOSK),
          open(app.SETTINGS_PATH, "w", encoding="utf-8"), ensure_ascii=False)
app._wake["cfg"] = app.load_settings()
PORT = 7871
demo = app.build_ui()
threading.Thread(target=lambda: demo.launch(server_name="127.0.0.1", server_port=PORT,
                                            prevent_thread_lock=True), daemon=True).start()
time.sleep(6)
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=os.environ.get("CHROMIUM") or None)
    pg = b.new_page()
    pg.goto(f"http://127.0.0.1:{PORT}", wait_until="load")
    pg.wait_for_selector("text=Управление компьютером", timeout=30000)
    pg.get_by_text("🖥 Управление компьютером").click()
    pg.get_by_label("Папка с фильмами").fill(root)
    pg.get_by_label("Путь к vlc.exe (пусто — найти автоматически)").fill(os.path.join(HERE, "fake_vlc.py"))
    pg.get_by_role("button", name="🔄 Обновить списки").click()
    for _ in range(30):
        pg.wait_for_timeout(500)
        body = pg.inner_text("body")
        if "Фильмов: 2" in body:
            break
    check("статус: 2 фильма и VLC найден", "Фильмов: 2" in body and "VLC:" in body,
          next((l for l in body.splitlines() if "🎬" in l), ""))
    s = json.load(open(app.SETTINGS_PATH, encoding="utf-8"))
    check("настройки сохранены", s.get("movies_dir") == root and s.get("vlc_path", "").endswith("fake_vlc.py"))
    box = pg.get_by_label("…или вопрос текстом")
    box.fill("Включи интерстелар на полный экран")
    box.press("Enter")
    ok = False
    for _ in range(60):
        pg.wait_for_timeout(500)
        if "Включаю Интерстеллар" in pg.locator("textarea").all_inner_texts().__str__() or \
                "Включаю Интерстеллар" in str([t.input_value() for t in pg.locator("textarea").all()]):
            ok = True
            break
    check("«Диалог»: Включаю Интерстеллар", ok)
    pg.screenshot(path=os.path.join(WORK, "gui_pc.png"), full_page=True)
    for _ in range(200):  # VLC стартует после озвучки (первый запуск — ещё и скачивание Silero)
        if pc.PLAYER.status():
            break
        time.sleep(0.3)
    check("VLC запущен", pc.PLAYER.running() and "Интерстеллар" in pc.PLAYER.title, pc.PLAYER.title)
    b.close()
pc.PLAYER.stop()
demo.close()
print(f"\n{sum(RES)}/{len(RES)} PASS", flush=True)
sys.exit(0 if all(RES) else 1)
