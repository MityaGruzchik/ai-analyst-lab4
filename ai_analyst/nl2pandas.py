import ast
import builtins

import numpy as np
import pandas as pd

from .llm import ask_json
from .profiling import schema
from .prompts import NL2PANDAS_FIX, NL2PANDAS_SYSTEM

FORBIDDEN_NAMES = {"eval", "exec", "open", "compile", "__import__", "input", "globals", "locals", "getattr", "setattr", "delattr", "vars"}
SAFE_BUILTINS = {n: getattr(builtins, n)
                 for n in ["len", "range", "sum", "min", "max", "abs", "round", "sorted", "list", "dict", "set", "tuple",
                           "int", "float", "str", "bool", "enumerate", "zip", "any", "all", "map", "filter", "reversed", "print", "isinstance"]}


def check_code(code):
    """Статична перевірка: без імпортів, без dunder-атрибутів, без небезпечних викликів."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Імпорти заборонені")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Доступ до dunder-атрибутів заборонений")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"Заборонене ім'я: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in {"to_csv", "to_pickle", "to_excel", "to_parquet", "to_sql", "read_csv", "read_pickle"}:
            raise ValueError("Робота з файлами заборонена")


def run_code(code, tables):
    check_code(code)
    ns = {"pd": pd, "np": np, **{k: v.copy() for k, v in tables.items()}}
    exec(compile(code, "<generated>", "exec"), {"__builtins__": SAFE_BUILTINS}, ns)
    if "result" not in ns:
        raise ValueError("Код не записав відповідь у змінну result")
    return ns["result"]


def ask(question, tables, llm, max_fix=1):
    """NL -> pandas. Повертає код (щоб його можна було перевірити), результат і пояснення."""
    sch = schema(tables)
    user = f"Схема даних:\n{sch}\n\nПитання: {question}"
    resp = ask_json(llm, NL2PANDAS_SYSTEM, user)
    code, expl = resp.get("code"), resp.get("explanation", "")
    if not code:
        return {"question": question, "code": None, "result": None, "explanation": expl, "error": None, "refused": True}
    error = None
    for attempt in range(max_fix + 1):
        try:
            result = run_code(code, tables)
            return {"question": question, "code": code, "result": result, "explanation": expl, "error": None, "refused": False}
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            if attempt < max_fix:
                resp = ask_json(llm, NL2PANDAS_SYSTEM, user + "\n\n" + NL2PANDAS_FIX.format(error=error, code=code))
                code = resp.get("code") or code
                expl = resp.get("explanation", expl)
    return {"question": question, "code": code, "result": None, "explanation": expl, "error": error, "refused": False}
