import json
import re

_NUM = re.compile(r"-?\d[\d\s ]*(?:[.,]\d+)?")


def collect_numbers(obj, out=None):
    """Рекурсивно збирає всі числа з довільної структури фактів."""
    if out is None:
        out = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_numbers(v, out)
    elif isinstance(obj, str):
        for m in _NUM.findall(obj):
            try:
                out.append(float(m.replace(" ", "").replace(" ", "").replace(",", ".")))
            except ValueError:
                pass
    return out


def unverified_numbers(text, facts, rel_tol=0.01):
    """Числа з тексту моделі, яких немає серед фактів (з допуском на округлення).
    Малі цілі (<=10) та роки пропускаються: це нумерація й службові слова."""
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)  # не str(): repr екранує нерозривні пробіли
    text = re.sub(r"[\u00a0\u2009\u202f]", " ", text)
    text = re.sub(r"[\u2010\u2011\u2012\u2212]", "-", text)  # різні дефіси та мінус -> ASCII
    text = re.sub(r"\d{4}-\d{2}-\d{2}", " ", text)  # дати з періодів не є метриками
    known = collect_numbers(facts)
    bad = []
    for m in _NUM.findall(text):
        s = m.replace(" ", "").replace(" ", "").replace(",", ".")
        try:
            x = float(s)
        except ValueError:
            continue
        if abs(x) <= 10 and x == int(x):
            continue
        if 1990 <= x <= 2100 and x == int(x):
            continue
        ok = False
        for v in known:
            for cand in (v, v * 100):
                if abs(abs(x) - abs(cand)) <= max(rel_tol * abs(cand), 0.006):  # знак не важливий: «знизився на 6.5%»
                    ok = True
                    break
            if ok:
                break
        if not ok:
            bad.append(m.strip())
    return bad
