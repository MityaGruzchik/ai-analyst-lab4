from .llm import ask_json
from .profiling import profile
from .prompts import EDA_SYSTEM
from .verify import unverified_numbers


def auto_eda(df, llm, name="table"):
    """Профіль рахує pandas, LLM лише інтерпретує готові числа."""
    prof = profile(df)
    out = ask_json(llm, EDA_SYSTEM, f"Таблиця: {name}\nПрофіль:\n{prof}")
    return {"profile": prof, "interpretation": out, "unverified_numbers": unverified_numbers(out, prof)}
