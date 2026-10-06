"""Живой поиск из песочницы (нужен интернет): web_search() по типичным вопросам.
Запуск: python tests/test_search_live.py [папка app.py]"""
import time
from _paths import APP_DIR  # noqa: E402  (путь к app.py)
import app  # noqa: E402
print("app.py:", APP_DIR, flush=True)

QUERIES = ["погода в Москве завтра", "последняя версия Python", "курс доллара ЦБ сегодня",
           "новости технологий", "кто такой Линус Торвальдс", "KoboldCpp последний релиз"]
ok = 0
for q in QUERIES:
    t0 = time.time()
    try:
        text, urls = app.web_search(q, 3000)
        dt = time.time() - t0
        good = len(text) > 300 and urls and dt < 15
        ok += bool(good)
        pages = text.count("Со страницы")
        print(f"{'PASS' if good else 'FAIL'} «{q}»: {dt:.1f} с, {len(text)} симв., страниц {pages}, "
              f"{[app._domain(u) for u in urls[:6]]}")
        print("   " + text[:400].replace("\n", " | "))
    except Exception as e:
        print(f"FAIL «{q}»: {time.time() - t0:.1f} с: {e}")
print(f"\n{ok}/{len(QUERIES)} PASS")
