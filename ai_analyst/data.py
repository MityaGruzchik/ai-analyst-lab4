import os

import pandas as pd

from .profiling import parse_dates

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_demo(data_dir=DATA_DIR):
    """Демо-датасет: реєстрації, платежі та довідник країн мобільного застосунку (квітень-травень 2024)."""
    reg = parse_dates(pd.read_csv(os.path.join(data_dir, "registrations.csv")))
    pay = parse_dates(pd.read_csv(os.path.join(data_dir, "payments.csv")))
    cty = pd.read_csv(os.path.join(data_dir, "country_info_dict.csv"))
    reg = reg.merge(cty, on="country_code", how="left")
    # у платежі додаємо країну та платформу користувача, щоб можна було розрізати виручку
    pay = pay.merge(reg[["user_id", "country_name", "app"]].drop_duplicates("user_id"), on="user_id", how="left")
    return {"registrations": reg, "payments": pay}


def load_csv(path):
    return {os.path.splitext(os.path.basename(path))[0]: parse_dates(pd.read_csv(path))}
