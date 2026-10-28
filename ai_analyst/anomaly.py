import pandas as pd

from .llm import ask_json
from .prompts import ANOMALY_SYSTEM
from .verify import unverified_numbers


def decompose(df, date_col, value_col, dims, base, target, agg="sum", top=5):
    """Порівнює метрику у двох періодах і рахує внесок кожного сегмента у зміну (delta).
    base, target: (start, end) включно. Все це чистий pandas."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])

    def part(p):
        s, e = pd.to_datetime(p[0]), pd.to_datetime(p[1]) + pd.Timedelta(days=1)
        return d[(d[date_col] >= s) & (d[date_col] < e)]

    b, t = part(base), part(target)
    total_b = getattr(b[value_col], agg)()
    total_t = getattr(t[value_col], agg)()
    total_delta = total_t - total_b
    res = {
        "metric": f"{agg}({value_col})",
        "base_period": list(base), "target_period": list(target),
        "base_value": round(float(total_b), 2), "target_value": round(float(total_t), 2),
        "delta": round(float(total_delta), 2),
        "delta_pct": round(float(total_delta / total_b * 100), 2) if total_b else None,
        "days_base": int(b[date_col].dt.normalize().nunique()), "days_target": int(t[date_col].dt.normalize().nunique()),
        "segments": {},
    }
    for dim in dims:
        gb = getattr(b.groupby(dim)[value_col], agg)()
        gt = getattr(t.groupby(dim)[value_col], agg)()
        tab = pd.concat([gb.rename("base"), gt.rename("target")], axis=1).fillna(0)
        tab["delta"] = tab["target"] - tab["base"]
        tab["share_of_delta_pct"] = tab["delta"] / total_delta * 100 if total_delta else 0
        tab = tab.reindex(tab["delta"].abs().sort_values(ascending=False).index).head(top)
        res["segments"][dim] = [
            {"value": str(i), "base": round(float(r.base), 2), "target": round(float(r.target), 2),
             "delta": round(float(r.delta), 2), "share_of_delta_pct": round(float(r.share_of_delta_pct), 1)}
            for i, r in tab.iterrows()
        ]
    return res


def explain_metric_change(df, date_col, value_col, dims, base, target, llm, agg="sum"):
    facts = decompose(df, date_col, value_col, dims, base, target, agg)
    out = ask_json(llm, ANOMALY_SYSTEM, f"Факти:\n{facts}")
    return {"facts": facts, "explanation": out, "unverified_numbers": unverified_numbers(out, facts)}
