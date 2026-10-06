"""E2E: браузер (Playwright + Chromium) против gui_server.py. Проверяет новые элементы,
кнопки-пресеты промпта, автосохранение, ответ «вживую» в «Диалоге», поиск, озвучку."""
import json, os, time
from playwright.sync_api import sync_playwright
from _paths import WORK  # noqa: E402
import app  # noqa: E402

SHOTS = os.path.join(WORK, "gui")
os.makedirs(SHOTS, exist_ok=True)
RES = []


def check(name, cond, info=""):
    RES.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f" — {info}" if info else ""), flush=True)


def settings():
    return json.load(open(os.path.join(WORK, "gui_settings.json"), encoding="utf-8"))


with sync_playwright() as p:
    exe = os.environ.get("CHROMIUM", "/usr/local/bin/chromium")
    br = p.chromium.launch(executable_path=exe if os.path.exists(exe) else None, args=["--no-sandbox"])
    page = br.new_page(viewport={"width": 1280, "height": 1700})
    page.goto("http://127.0.0.1:7861", wait_until="load")
    page.wait_for_selector("text=RU Voice Assistant")
    for m in app.WEB_MODES:
        check(f"радио поиска: «{m[:25]}…»", page.get_by_label(m).count() == 1)
    check("по умолчанию — автоматически", page.get_by_label(app.WEB_AUTO).is_checked())
    page.get_by_text("🧠 Характер и длина ответов").click()
    sp = page.get_by_label("Системный промпт — инструкция для модели")
    sp.wait_for()
    check("промпт по умолчанию — сбалансированный", sp.input_value() == app.PROMPT_BALANCED)
    check("ползунок длины ответа", page.get_by_text("Максимальная длина ответа (токенов)").count() >= 1)
    page.get_by_role("button", name="Коротко", exact=True).click()
    page.wait_for_timeout(1500)
    check("пресет «Коротко» -> текст", sp.input_value() == app.PROMPT_SHORT)
    check("пресет сохранён в settings.json", settings()["system_prompt"] == app.PROMPT_SHORT)
    page.get_by_role("button", name="Подробно", exact=True).click()
    page.wait_for_timeout(1500)
    check("пресет «Подробно»", sp.input_value() == app.PROMPT_DETAILED and
          settings()["system_prompt"] == app.PROMPT_DETAILED)
    sp.fill("Ты — пират. Отвечай по-русски, коротко, в пиратском стиле.")
    page.wait_for_timeout(1500)
    check("свой промпт сохраняется", settings()["system_prompt"].startswith("Ты — пират"))
    page.get_by_role("button", name="Сбалансированно (стандарт)").click()
    page.wait_for_timeout(1000)
    page.get_by_label(app.WEB_ASK).check()
    page.wait_for_timeout(1500)
    check("режим поиска сохраняется", settings()["web_search"] == app.WEB_ASK)
    page.screenshot(path=f"{SHOTS}/01_settings.png", full_page=True)

    # ждём движок
    t0 = time.time()
    while time.time() - t0 < 120 and app.ENGINE_PHASE_TEXT["ready"][:6] not in page.content():
        page.wait_for_timeout(1000)
    log = page.get_by_label("Диалог")
    q = page.get_by_placeholder("Напишите и нажмите Enter")
    q.fill("Расскажи подробно, как заварить вкусный чай.")
    q.press("Enter")
    lens, t0, shot = [], time.time(), False
    while time.time() - t0 < 180:
        v = log.input_value()
        lens.append(len(v))
        if not shot and len(v) > 120 and "Ассистент:" in v:
            page.screenshot(path=f"{SHOTS}/02_streaming.png", full_page=True)
            shot = True
        if page.locator("audio[src]").count() and q.input_value() == "":
            break
        page.wait_for_timeout(300)
    v = log.input_value()
    check("ответ рос в «Диалоге» по ходу генерации", len(set(lens)) >= 5, f"{len(set(lens))} разных длин")
    check("вопрос в диалоге без даты", "Вы: Расскажи подробно, как заварить вкусный чай." in v and "[Сейчас" not in v)
    check("озвучка появилась", page.locator("audio[src]").count() >= 1)
    check("поле вопроса очищено", q.input_value() == "")

    q.fill("Найди в интернете, какой сейчас курс доллара")
    q.press("Enter")
    seen_status, t0 = set(), time.time()
    while time.time() - t0 < 240:
        v = log.input_value()
        for mark in ("🌐 Ищу в интернете", "📖 Читаю найденное", "🔊 Озвучиваю"):
            if mark in v:
                seen_status.add(mark)
        if ("🔎 Искал в интернете" in v or "⚠️ Поиск" in v) and q.input_value() == "":
            break
        page.wait_for_timeout(250)
    v = log.input_value()
    check("статус поиска был виден", "🌐 Ищу в интернете" in seen_status, seen_status)
    check("заметка об источниках", "🔎 Искал в интернете" in v, v[-300:])
    page.screenshot(path=f"{SHOTS}/03_search.png", full_page=True)
    print("\n--- Диалог ---\n" + v)
    br.close()
print(f"\nИТОГО: {sum(RES)}/{len(RES)} PASS")
