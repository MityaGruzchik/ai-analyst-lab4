import argparse
import json

import pandas as pd

from .anomaly import explain_metric_change
from .data import load_csv, load_demo
from .eda import auto_eda
from .llm import LLM
from .nl2pandas import ask
from .summary import executive_summary, kpi_snapshot


def _tables(args):
    return load_csv(args.csv) if args.csv else load_demo()


def _show(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def main(argv=None):
    p = argparse.ArgumentParser(prog="ai_analyst", description="AI-асистент аналітика")
    p.add_argument("--csv", help="шлях до власного CSV (за замовчуванням використовується демо-датасет)")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eda", help="автоматичний EDA таблиці")
    e.add_argument("--table", default=None, help="назва таблиці (для демо: registrations або payments)")

    a = sub.add_parser("ask", help="питання природною мовою -> pandas")
    a.add_argument("question")

    x = sub.add_parser("explain", help="пояснити зміну виручки між двома періодами (демо-дані)")
    x.add_argument("--base", nargs=2, required=True, metavar=("START", "END"))
    x.add_argument("--target", nargs=2, required=True, metavar=("START", "END"))

    s = sub.add_parser("summary", help="executive summary за KPI двох періодів (демо-дані)")
    s.add_argument("--base", nargs=2, required=True, metavar=("START", "END"))
    s.add_argument("--target", nargs=2, required=True, metavar=("START", "END"))

    args = p.parse_args(argv)
    llm = LLM()
    tables = _tables(args)

    if args.cmd == "eda":
        name = args.table or next(iter(tables))
        r = auto_eda(tables[name], llm, name)
        _show(r["interpretation"])
        if r["unverified_numbers"]:
            print("\n[!] Числа, яких немає в профілі (можлива галюцинація):", r["unverified_numbers"])
    elif args.cmd == "ask":
        r = ask(args.question, tables, llm)
        print("Код:\n" + (r["code"] or "(не згенеровано)"))
        print("\nПояснення:", r["explanation"])
        if r["error"]:
            print("\nПомилка виконання:", r["error"])
        elif not r["refused"]:
            print("\nРезультат:\n", r["result"])
    elif args.cmd == "explain":
        pay = tables["payments"]
        r = explain_metric_change(pay, "pay_dt", "amount_usd", ["country_name", "app", "feature_type_id"],
                                  args.base, args.target, llm)
        _show(r["explanation"])
        if r["unverified_numbers"]:
            print("\n[!] Числа, яких немає у фактах:", r["unverified_numbers"])
    elif args.cmd == "summary":
        m = kpi_snapshot(tables["registrations"], tables["payments"], args.base, args.target)
        r = executive_summary(m, llm)
        _show(r["summary"])
        if r["unverified_numbers"]:
            print("\n[!] Числа, яких немає у метриках:", r["unverified_numbers"])


if __name__ == "__main__":
    main()
