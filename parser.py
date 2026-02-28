"""
Pizza order parser — two-layer strategy:
  1. HuggingFace free Inference API  (Mistral/Zephyr via huggingface_hub)
  2. Rule-based keyword fallback      (always works, even offline)

Set HF_TOKEN as a Space secret (free HF account) for best results.
If HF_TOKEN is absent the rule-based engine runs automatically.
"""

import os, re, json, logging
from catalogue import (SIZES, CRUSTS, SAUCES, CHEESES,
                       TOPPINGS, EXTRAS, TAX_RATE)

logger = logging.getLogger(__name__)

# ── HuggingFace client (optional) ───────────────────────────────────────────
_client = None

def _get_client():
    global _client
    if _client is not None:
        return _client
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None
    try:
        from huggingface_hub import InferenceClient
        # Free serverless models — try in order until one works
        for model in [
            "mistralai/Mistral-7B-Instruct-v0.3",
            "HuggingFaceH4/zephyr-7b-beta",
            "microsoft/Phi-3-mini-4k-instruct",
        ]:
            try:
                _client = InferenceClient(model=model, token=token)
                logger.info(f"HF client ready: {model}")
                return _client
            except Exception:
                continue
    except ImportError:
        pass
    return None


# ── LLM parser ───────────────────────────────────────────────────────────────
_SYSTEM = """You are a pizza order extractor. 
Given a customer's message, extract ONLY a JSON object with these keys:
  size        - one of: personal, small, medium, large, extra large, xl
  crust       - one of: thin, crispy, thick, hand tossed, stuffed, cauliflower, gluten free
  sauce       - one of: marinara, tomato, bbq, alfredo, white, pesto, ranch, buffalo, garlic butter
  cheese      - one of: mozzarella, cheddar, parmesan, feta, gouda, ricotta, no cheese, dairy-free, vegan
  toppings    - list of topping names (only from: pepperoni, mushrooms, spinach, jalapeños, olives,
                bell peppers, red onion, grilled chicken, ground beef, sausage, bacon, ham, pineapple,
                tomatoes, basil, garlic, arugula, broccoli, corn, artichoke, anchovies, avocado,
                prosciutto, truffle, zucchini, sun dried tomato)
  extras      - list from: extra cheese, extra sauce, well done, light sauce
  quantity    - integer (default 1)

Output ONLY the raw JSON object. No explanation. No markdown. No backticks."""


def _llm_parse(text: str) -> dict | None:
    client = _get_client()
    if not client:
        return None
    try:
        prompt = f"[INST] {_SYSTEM}\n\nCustomer said: \"{text}\" [/INST]"
        resp = client.text_generation(
            prompt,
            max_new_tokens=300,
            temperature=0.1,
            do_sample=False,
        )
        # Strip markdown fences if model adds them
        clean = re.sub(r"```json|```", "", resp).strip()
        # Extract first {...} block
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if not m:
            return None
        data = json.loads(m.group())
        return data
    except Exception as e:
        logger.warning(f"LLM parse failed: {e}")
        return None


# ── Rule-based parser (fallback) ─────────────────────────────────────────────
_QTY = {"one":1,"two":2,"three":3,"four":4,"five":5,
        "six":6,"seven":7,"eight":8,"nine":9,"ten":10}

def _rule_parse(text: str) -> dict:
    t = text.lower()
    result = {}

    # quantity
    qty = 1
    for w, n in _QTY.items():
        if re.search(rf"\b{w}\b", t):
            qty = n; break
    m = re.search(r"\b([2-9])\s+pizza", t)
    if m: qty = int(m.group(1))
    result["quantity"] = qty

    # helpers
    def pick(catalogue):
        for k in sorted(catalogue, key=len, reverse=True):
            if k in t:
                return k
        return None

    result["size"]   = pick(SIZES)   or "medium"
    result["crust"]  = pick(CRUSTS)  or "hand tossed"
    result["sauce"]  = pick(SAUCES)  or "tomato"
    result["cheese"] = pick(CHEESES) or "mozzarella"

    seen, toppings = set(), []
    for k in sorted(TOPPINGS, key=len, reverse=True):
        if k in t:
            lbl = TOPPINGS[k][0]
            if lbl not in seen:
                seen.add(lbl)
                toppings.append(k)
    result["toppings"] = toppings

    seen_ex, extras = set(), []
    for k in sorted(EXTRAS, key=len, reverse=True):
        if k in t:
            lbl = EXTRAS[k][0]
            if lbl not in seen_ex:
                seen_ex.add(lbl)
                extras.append(k)
    result["extras"] = extras

    return result


# ── Normalise LLM output into same shape as rule parser ──────────────────────
def _normalise_llm(data: dict) -> dict:
    def match(val, catalogue):
        if not val: return None
        v = str(val).lower().strip()
        # exact key match
        if v in catalogue: return v
        # partial match
        for k in sorted(catalogue, key=len, reverse=True):
            if k in v or v in k: return k
        return None

    qty = 1
    try: qty = max(1, int(data.get("quantity", 1)))
    except: pass

    size   = match(data.get("size"),   SIZES)   or "medium"
    crust  = match(data.get("crust"),  CRUSTS)  or "hand tossed"
    sauce  = match(data.get("sauce"),  SAUCES)  or "tomato"
    cheese = match(data.get("cheese"), CHEESES) or "mozzarella"

    raw_tops = data.get("toppings", [])
    if isinstance(raw_tops, str):
        raw_tops = [x.strip() for x in raw_tops.split(",")]
    seen, toppings = set(), []
    for item in raw_tops:
        k = match(item, TOPPINGS)
        if k and TOPPINGS[k][0] not in seen:
            seen.add(TOPPINGS[k][0])
            toppings.append(k)

    raw_ex = data.get("extras", [])
    if isinstance(raw_ex, str):
        raw_ex = [x.strip() for x in raw_ex.split(",")]
    seen_ex, extras = set(), []
    for item in raw_ex:
        k = match(item, EXTRAS)
        if k and EXTRAS[k][0] not in seen_ex:
            seen_ex.add(EXTRAS[k][0])
            extras.append(k)

    return dict(size=size, crust=crust, sauce=sauce, cheese=cheese,
                toppings=toppings, extras=extras, quantity=qty)


# ── Build structured order object ─────────────────────────────────────────────
def _build_order(parsed: dict, raw_text: str) -> dict:
    def spec(key, catalogue):
        lbl, price = catalogue[key]
        return {"label": lbl, "price": price}

    tops = []
    seen = set()
    for k in parsed["toppings"]:
        if k in TOPPINGS:
            lbl, emoji, price = TOPPINGS[k]
            if lbl not in seen:
                seen.add(lbl)
                tops.append({"label": lbl, "emoji": emoji, "price": price})

    exs = []
    seen_ex = set()
    for k in parsed["extras"]:
        if k in EXTRAS:
            lbl, price = EXTRAS[k]
            if lbl not in seen_ex:
                seen_ex.add(lbl)
                exs.append({"label": lbl, "price": price})

    return {
        "size":     spec(parsed["size"],   SIZES),
        "crust":    spec(parsed["crust"],  CRUSTS),
        "sauce":    spec(parsed["sauce"],  SAUCES),
        "cheese":   spec(parsed["cheese"], CHEESES),
        "toppings": tops,
        "extras":   exs,
        "quantity": parsed["quantity"],
        "raw":      raw_text,
    }


def parse_order(text: str) -> tuple[dict, str]:
    """Returns (order_dict, engine_used)"""
    # Try LLM first
    llm_data = _llm_parse(text)
    if llm_data:
        parsed = _normalise_llm(llm_data)
        return _build_order(parsed, text), "🤖 AI (HuggingFace Free)"

    # Fallback to rules
    parsed = _rule_parse(text)
    return _build_order(parsed, text), "⚙️ Rule Engine"


def calc_price(order: dict) -> dict:
    q   = order["quantity"]
    b   = order["size"]["price"]
    cr  = order["crust"]["price"]
    sa  = order["sauce"]["price"]
    ch  = order["cheese"]["price"]
    tp  = sum(x["price"] for x in order["toppings"])
    ex  = sum(x["price"] for x in order["extras"])
    unit    = b + cr + sa + ch + tp + ex
    sub     = unit * q
    tax     = sub * TAX_RATE
    return {
        "unit":     round(unit, 2),
        "quantity": q,
        "subtotal": round(sub, 2),
        "tax":      round(tax, 2),
        "total":    round(sub + tax, 2),
        "breakdown": {
            "base": round(b, 2), "crust": round(cr, 2),
            "sauce": round(sa, 2), "cheese": round(ch, 2),
            "toppings": round(tp, 2), "extras": round(ex, 2),
        },
    }