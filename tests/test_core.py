import json
import os

import pandas as pd
import pytest

from ai_analyst.anomaly import decompose, explain_metric_change
from ai_analyst.eda import auto_eda
from ai_analyst.llm import ask_json, parse_json
from ai_analyst.nl2pandas import ask, check_code
from ai_analyst.profiling import profile
from ai_analyst.summary import executive_summary, kpi_snapshot
from ai_analyst.verify import unverified_numbers


class Fake:
    """Підставка замість LLM: повертає заготовлені відповіді по черзі."""
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return self.answers.pop(0)


@pytest.fixture
def pay():
    return pd.DataFrame({
        "pay_dt": pd.to_datetime(["2024-04-01", "2024-04-02", "2024-04-08", "2024-04-09", "2024-04-09"]),
        "amount_usd": [10.0, 10.0, 5.0, 5.0, 2.0],
        "country_name": ["UA", "US", "UA", "US", "US"],
    })


def test_parse_json_fenced():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Ось: {"a": 2} дякую') == {"a": 2}


def test_ask_json_retry():
    f = Fake("не json", '{"ok": true}')
    assert ask_json(f, "s", "u") == {"ok": True}
    assert f.calls == 2


def test_profile_empty_and_missing():
    assert profile(pd.DataFrame({"a": []}))["rows"] == 0
    p = profile(pd.DataFrame({"a": [1, None, 3, 100]}))
    assert p["column_profile"][0]["missing"] == 1


def test_decompose(pay):
    r = decompose(pay, "pay_dt", "amount_usd", ["country_name"], ("2024-04-01", "2024-04-07"), ("2024-04-08", "2024-04-14"))
    assert r["base_value"] == 20.0 and r["target_value"] == 12.0 and r["delta"] == -8.0
    us = next(s for s in r["segments"]["country_name"] if s["value"] == "US")
    assert us["delta"] == -3.0


def test_unverified_numbers():
    facts = {"revenue": 93750.0, "share": 0.4412}
    assert unverified_numbers("Виручка 93 750 $, частка 44,1%", facts) == []
    assert unverified_numbers("Виручка 120 000 $", facts) == ["120 000"]


def test_check_code_blocks_dangerous():
    for bad in ["import os", "open('x')", "x = ().__class__", "df.to_csv('a.csv')"]:
        with pytest.raises(ValueError):
            check_code(bad)


def test_ask_runs_visible_code(pay):
    f = Fake(json.dumps({"code": "result = payments['amount_usd'].sum()", "explanation": "сума"}))
    r = ask("скільки виручка", {"payments": pay}, f)
    assert r["result"] == 32.0 and "sum" in r["code"]


def test_ask_fixes_error(pay):
    f = Fake(json.dumps({"code": "result = payments['nope'].sum()", "explanation": "x"}),
             json.dumps({"code": "result = payments['amount_usd'].sum()", "explanation": "x"}))
    r = ask("q", {"payments": pay}, f)
    assert r["result"] == 32.0 and f.calls == 2


def test_ask_refuses_illogical(pay):
    f = Fake(json.dumps({"code": None, "explanation": "немає такої колонки"}))
    r = ask("яка погода", {"payments": pay}, f)
    assert r["refused"] and r["result"] is None


def test_ask_gives_error_after_failed_fix(pay):
    bad = json.dumps({"code": "result = payments['nope'].sum()", "explanation": "x"})
    r = ask("q", {"payments": pay}, Fake(bad, bad))
    assert r["error"] and r["result"] is None


def test_eda_flags_hallucination(pay):
    f = Fake(json.dumps({"quality_summary": "Рядків 777777", "issues": [], "key_findings": [], "next_steps": []}))
    r = auto_eda(pay, f)
    assert "777777" in r["unverified_numbers"]


def test_explain_and_summary(pay):
    ans = json.dumps({"headline": "Виручка впала на 8.0", "top_drivers": [], "hypotheses": [], "confidence": "low"})
    r = explain_metric_change(pay, "pay_dt", "amount_usd", ["country_name"], ("2024-04-01", "2024-04-07"), ("2024-04-08", "2024-04-14"), Fake(ans))
    assert r["unverified_numbers"] == []
    reg = pd.DataFrame({"user_id": [1, 2, 3], "reg_dt": pd.to_datetime(["2024-04-01", "2024-04-02", "2024-04-09"])})
    pay2 = pay.assign(user_id=[1, 2, 1, 3, 3])
    m = kpi_snapshot(reg, pay2, ("2024-04-01", "2024-04-07"), ("2024-04-08", "2024-04-14"))
    assert m["base"]["new_users"] == 2 and m["target"]["revenue_usd"] == 12.0
    s = executive_summary(m, Fake(json.dumps({"what_happened": "x", "why_it_matters": "y", "recommendations": [], "risks": []})))
    assert "summary" in s


def test_api_request(monkeypatch):
    import io
    from ai_analyst import llm as L
    seen = {}

    class R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def fake_urlopen(req, timeout=0):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data)
        return R(json.dumps({"choices": [{"message": {"content": "{\"a\": 1}"}}]}).encode())

    monkeypatch.setattr(L.urllib.request, "urlopen", fake_urlopen)
    out = L.LLM(api_key="k", model="m", base_url="https://x.test/v1/").complete("sys", "usr")
    assert out == '{"a": 1}' and seen["auth"] == "Bearer k" and seen["url"] == "https://x.test/v1/chat/completions"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        L.LLM(api_key=None).complete("s", "u")


def test_verify_handles_narrow_spaces_and_dates():
    facts = {"revenue": 25022.06, "users": 31879, "period": ["2024-04-22", "2024-04-28"]}
    out = {"text": "За 2024-04-22 \u2013 2024-04-28 виручка 25\u202f022.06, користувачів 31\u202f879"}
    assert unverified_numbers(out, facts) == []
