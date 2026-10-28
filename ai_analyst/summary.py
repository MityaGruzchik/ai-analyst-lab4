import pandas as pd

from .llm import ask_json
from .prompts import SUMMARY_SYSTEM
from .verify import unverified_numbers


def kpi_snapshot(registrations, payments, base, target):
    """KPI за два періоди: чистий pandas."""
    def one(p):
        s, e = pd.to_datetime(p[0]), pd.to_datetime(p[1]) + pd.Timedelta(days=1)
        r = registrations[(registrations["reg_dt"] >= s) & (registrations["reg_dt"] < e)]
        pay = payments[(payments["pay_dt"] >= s) & (payments["pay_dt"] < e)]
        return {
            "period": list(p),
            "new_users": int(r["user_id"].nunique()),
            "revenue_usd": round(float(pay["amount_usd"].sum()), 2),
            "payers": int(pay["user_id"].nunique()),
            "payments": int(len(pay)),
            "avg_check_usd": round(float(pay["amount_usd"].mean()), 2) if len(pay) else 0.0,
        }
    b, t = one(base), one(target)
    ch = {}
    for k in ["new_users", "revenue_usd", "payers", "avg_check_usd"]:
        ch[k + "_change_pct"] = round((t[k] - b[k]) / b[k] * 100, 1) if b[k] else None
    return {"base": b, "target": t, "change": ch}


def executive_summary(metrics, llm):
    out = ask_json(llm, SUMMARY_SYSTEM, f"Метрики:\n{metrics}")
    return {"metrics": metrics, "summary": out, "unverified_numbers": unverified_numbers(out, metrics)}
