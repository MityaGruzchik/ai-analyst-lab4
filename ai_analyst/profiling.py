import numpy as np
import pandas as pd


def profile(df, top_k=5):
    """Повністю детермінований профіль таблиці: без LLM."""
    n = len(df)
    cols = []
    for c in df.columns:
        s = df[c]
        info = {
            "column": c,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            d = s.dropna()
            if len(d):
                q1, q3 = d.quantile(0.25), d.quantile(0.75)
                iqr = q3 - q1
                out = int(((d < q1 - 1.5 * iqr) | (d > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
                info.update(
                    min=float(d.min()), max=float(d.max()), mean=round(float(d.mean()), 3),
                    median=float(d.median()), std=round(float(d.std()), 3),
                    outliers_iqr=out, outliers_pct=round(out / len(d) * 100, 2),
                    negative=int((d < 0).sum()), zeros=int((d == 0).sum()),
                )
        elif pd.api.types.is_datetime64_any_dtype(s):
            info.update(min=str(s.min()), max=str(s.max()))
        else:
            vc = s.value_counts(dropna=True).head(top_k)
            info["top_values"] = {str(k): int(v) for k, v in vc.items()}
        cols.append(info)
    return {"rows": n, "columns": len(df.columns), "duplicate_rows": int(df.duplicated().sum()), "column_profile": cols}


def parse_dates(df):
    """Колонки, назва яких закінчується на _dt, перетворює у datetime."""
    df = df.copy()
    for c in df.columns:
        if c.endswith("_dt") and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def schema(tables):
    """Короткий опис таблиць для промпту NL->Pandas."""
    lines = []
    for name, df in tables.items():
        cols = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
        sample = df.head(2).to_dict("records")
        lines.append(f"Таблиця {name}, {len(df)} рядків. Колонки: {cols}. Приклад: {sample}")
    return "\n".join(lines)
