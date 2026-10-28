import json
import os
import re
import time
import urllib.error
import urllib.request


class LLM:
    """Клієнт для будь-якого OpenAI-сумісного API (за замовчуванням Groq), без зовнішніх бібліотек.
    Налаштування: LLM_API_KEY (або GROQ_API_KEY), LLM_BASE_URL, AI_ANALYST_MODEL.
    Для тестів підставляється будь-який об'єкт з методом complete()."""

    def __init__(self, model=None, api_key=None, base_url=None, max_tokens=4096, timeout=120):
        self.model = model or os.environ.get("AI_ANALYST_MODEL", "openai/gpt-oss-120b")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")).rstrip("/")
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, system, user, retries=3):
        if not self.api_key:
            raise RuntimeError("Не задано LLM_API_KEY (див. README, розділ «Запуск»).")
        body = {"model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key,
                   "User-Agent": "ai-analyst/1.0"}
        for attempt in range(retries + 1):
            req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                         headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "ignore")[:300]
                # 429 (ліміт безкоштовного тарифу), 5xx: чекаємо та повторюємо
                if e.code in (429, 500, 502, 503) and attempt < retries:
                    wait = 5 * 2 ** attempt
                    print(f"[{self.model}] {e.code}, чекаю {wait} с і повторюю...", flush=True)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Помилка API {e.code} (модель {self.model}): {msg}")
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise RuntimeError("Порожня відповідь моделі: " + json.dumps(data, ensure_ascii=False)[:300])


def parse_json(text):
    """Дістає JSON із відповіді моделі (прибирає ```json ... ``` і зайвий текст навколо)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"[\{\[].*[\}\]]", t, re.S)
        if not m:
            raise ValueError("У відповіді моделі немає JSON: " + text[:200])
        return json.loads(m.group(0))


def ask_json(llm, system, user, retries=1):
    """Запит із очікуваним JSON-виходом; при невдалому парсингу один раз перепитує."""
    last = None
    for _ in range(retries + 1):
        raw = llm.complete(system, user)
        try:
            return parse_json(raw)
        except (ValueError, json.JSONDecodeError) as e:
            last = e
            user = user + "\n\nПопередня відповідь не була валідним JSON. Поверни ЛИШЕ JSON без пояснень."
    raise ValueError(f"Модель не повернула валідний JSON: {last}")
