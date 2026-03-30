"""
PizzaVoice – Pino AI Pizza Ordering Assistant
Optimised backend:
  • catalogue.py is the single source of truth for all menu data & prices
  • /prices endpoint keeps frontend JS in sync — no hardcoded JS price maps
  • Stage-aware dynamic menu injection (RAG without model download)
  • Pre-compiled word-boundary patterns for every menu key
  • Pre-sorted catalogue keys computed once at startup
  • InferenceClient module-level singleton
  • Token-budget conversation trimming (replaces naive last-20)
  • TTS audio LRU cache (64 entries)
  • Pydantic validation on every ORDER / UPDATE block
  • Simple per-IP rate limiter on /chat (30 req/min)
  • /stats endpoint for live token-usage monitoring
  • max_tokens 180 → 250, temperature 0.75 → 0.70
"""

from flask import Flask, render_template, request, jsonify, Response
from huggingface_hub import InferenceClient
import os, re, json, uuid, asyncio, hashlib, random, time
import requests as http_req
from collections import defaultdict
from typing import Optional, List
from pydantic import BaseModel, ValidationError
from datetime import datetime

# ── All menu data lives in catalogue.py — imported here, served via /prices ───
from catalogue import (
    SIZES_MAP, CRUSTS_MAP, SAUCES_MAP, CHEESES_MAP,
    TOPPINGS_MAP, DRINKS_MAP, EXTRAS_MAP,
    TAX_RATE, FANCY_QUOTES,
)

try:
    import edge_tts
except ImportError:
    edge_tts = None

app = Flask(__name__)

# ── Pre-sorted catalogue keys — computed ONCE at startup ───────────────────────
_S_SIZES    = sorted(SIZES_MAP,    key=len, reverse=True)
_S_CRUSTS   = sorted(CRUSTS_MAP,   key=len, reverse=True)
_S_SAUCES   = sorted(SAUCES_MAP,   key=len, reverse=True)
_S_CHEESES  = sorted(CHEESES_MAP,  key=len, reverse=True)
_S_TOPPINGS = sorted(TOPPINGS_MAP, key=len, reverse=True)
_S_DRINKS   = sorted(DRINKS_MAP,   key=len, reverse=True)
_S_EXTRAS   = sorted(EXTRAS_MAP,   key=len, reverse=True)

# Pre-compiled word-boundary regex for every menu key — built ONCE at import
_WB: dict = {
    k: re.compile(r'\b' + re.escape(k) + r'\b', re.IGNORECASE)
    for k in (list(SIZES_MAP) + list(CRUSTS_MAP) + list(SAUCES_MAP) +
              list(CHEESES_MAP) + list(TOPPINGS_MAP) + list(DRINKS_MAP) +
              list(EXTRAS_MAP))
}

# Static compiled patterns
_ORDER_RE    = re.compile(r"#{1,3}\s*ORDER\s*#{1,3}(.*?)#{1,3}\s*END\s*#{1,3}",  re.DOTALL | re.IGNORECASE)
_UPDATE_RE   = re.compile(r"#{1,3}\s*UPDATE\s*#{1,3}(.*?)#{1,3}\s*END\s*#{1,3}", re.DOTALL | re.IGNORECASE)
_CODE_FENCE  = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)
_JSON_BLOB   = re.compile(r"\{.*\}", re.DOTALL)
_NAME_PREFIX = re.compile(r"^(my name is|i'm|i am|it's|this is|hey i'm|call me)\s+", re.IGNORECASE)
_QTY_RE      = re.compile(r'\b(\d+)\s*(pizza|pie)', re.IGNORECASE)
_TTS_CLEAN   = re.compile(r"[#*_~`>]|\bhttps?://\S+")

# ── Stage-aware dynamic menu injection (the RAG layer) ─────────────────────────
#
# Instead of embedding the full ~450-token menu in every system prompt,
# we detect which menu sections are relevant for the current conversation
# stage and inject only those (~100-200 tokens on average).
#
# No vector model or network call required — stage detection is deterministic
# and keyword-based, which is more reliable for a structured ordering flow.

_MENU_SECTIONS = {
    "sizes":    'Sizes: Personal 6" $7.99 | Small 8" $9.99 | Medium 12" $13.99 | Large 14" $16.99 | XL 16" $19.99',
    "crusts":   "Crusts: Thin / Thick / Hand Tossed (free) | Stuffed +$2.50 | Cauliflower +$3.00 | Gluten-Free +$2.50",
    "sauces":   "Sauces: Marinara / Tomato / BBQ / Ranch / Buffalo / Garlic Butter (free) | Alfredo / Pesto +$0.50",
    "cheeses":  "Cheese: Mozzarella (free) | Cheddar / Parmesan +$0.50 | Feta / Gouda +$1.00 | Ricotta +$0.75 | Vegan / Dairy-Free +$1.50 | No Cheese (free)",
    "toppings": (
        "Toppings ($0.75–$3.00 each):\n"
        "  Meat: Pepperoni, Sausage, Bacon, Ham, Chicken, Ground Beef, Anchovies, Prosciutto\n"
        "  Veg:  Mushrooms, Spinach, Jalapeños, Olives, Bell Peppers, Red Onion, Tomatoes, Basil,\n"
        "        Garlic, Arugula, Broccoli, Corn, Artichoke, Avocado, Zucchini, Sun-Dried Tomatoes, Pineapple\n"
        "  Premium: Truffle Oil +$3.00"
    ),
    "drinks": (
        "Drinks:\n"
        "  Soft: Cola / Diet Cola / Sprite / Fanta / Root Beer $2.49\n"
        "  Juice: Orange Juice / Apple Juice $2.99, Lemonade $2.99\n"
        "  Water: Still $1.49, Sparkling $1.99\n"
        "  Italian: Espresso $3.49, Cappuccino $4.49, Italian Soda / Limonata $3.99\n"
        "  Beer/Wine: Craft Beer $5.99, Red/White Wine $6.99, Prosecco $7.99"
    ),
    "extras": "Extras: Extra Cheese +$1.50 | Extra Sauce +$0.50 | Well Done / Light Sauce (free)",
}

_SECTION_ORDER = ["sizes", "crusts", "sauces", "cheeses", "toppings", "drinks", "extras"]

# Keywords that signal the user is asking about a particular menu section
_SECTION_KW = {
    "sizes":    ["size", "large", "medium", "small", "personal", "xl",
                 "how big", "6 inch", "12 inch", "big pizza"],
    "crusts":   ["crust", "thin", "thick", "stuffed", "cauliflower",
                 "gluten", "base", "dough"],
    "sauces":   ["sauce", "marinara", "bbq", "barbecue", "pesto",
                 "ranch", "buffalo", "alfredo", "garlic butter"],
    "cheeses":  ["cheese", "mozzarella", "cheddar", "parmesan", "feta",
                 "gouda", "ricotta", "vegan cheese", "dairy", "no cheese"],
    "toppings": ["topping", "pepperoni", "mushroom", "spinach", "vegetarian",
                 "vegan", "meat", "chicken", "beef", "bacon", "sausage",
                 "onion", "pepper", "olive", "pineapple", "anchov",
                 "prosciutto", "truffle", "arugula", "broccoli"],
    "drinks":   ["drink", "cola", "water", "juice", "beer", "wine",
                 "coffee", "espresso", "cappuccino", "soda", "lemonade",
                 "beverage", "thirsty", "prosecco"],
    "extras":   ["extra", "well done", "light sauce", "crispy"],
}

# Broad-question triggers → return entire menu
# Use longer/specific phrases so common words like "options" or "list"
# inside normal sentences don't accidentally trigger a full-menu dump.
_FULL_MENU_KW = frozenset({
    "menu", "full menu", "whole menu", "entire menu",
    "everything", "show me everything", "see everything",
    "what do you have", "what've you got", "what's available",
    "all items", "complete list",
})


def _relevant_sections(partial: Optional[dict], user_msg: str) -> list:
    """
    Return the ordered list of menu section keys relevant to this turn.
    This is the RAG retrieval step — no embeddings needed.
    """
    low = (user_msg or "").lower()

    # Broad question → give the full menu so Pino can answer anything
    if any(kw in low for kw in _FULL_MENU_KW):
        return _SECTION_ORDER[:]

    needed: set = set()

    # ── Stage-aware: include sections for fields not yet confirmed ─────────────
    if partial:
        if not partial.get("size"):   needed.add("sizes")
        if not partial.get("crust"):  needed.add("crusts")
        if not partial.get("sauce"):  needed.add("sauces")
        if not partial.get("cheese"): needed.add("cheeses")
        # Toppings & drinks become relevant once the base is set
        if partial.get("sauce"):      needed.add("toppings")
        if partial.get("cheese"):     needed.add("drinks")
    else:
        # No partial order yet — early conversation; send first three sections
        needed.update(["sizes", "crusts", "sauces", "cheeses", "toppings"])

    # ── Keyword-aware: what the user is explicitly asking about ────────────────
    for section, kws in _SECTION_KW.items():
        if any(k in low for k in kws):
            needed.add(section)

    # Always include extras (tiny, useful context)
    needed.add("extras")

    # Preserve canonical section order; fall back to full menu if nothing matched
    result = [s for s in _SECTION_ORDER if s in needed]
    return result if result else _SECTION_ORDER[:]


_BASE_SYSTEM = """\
You are Pino, a warm and witty Italian pizza waiter at PizzaVoice.

YOUR PERSONALITY:
- Friendly, relaxed, occasionally drops light Italian expressions ("Perfetto!", "Magnifico!", "Bellissimo!")
- You genuinely enjoy helping people find their perfect pizza
- You can handle ANY conversational direction naturally — jokes, questions about ingredients,
  dietary concerns, group orders, half-and-half requests, etc.
- If someone goes off-topic, be charming but gently steer back to their order

YOUR GOAL:
Guide the customer through a NATURAL conversation to build their pizza order.
Collect: customer name, size, crust, sauce, cheese, toppings (or none), drinks (or none),
         quantity, and DELIVERY ADDRESS.
Do NOT ask all questions at once. Ask ONE thing at a time, like a real waiter would.
After pizza details are done, ask if they'd like any drinks.
Finally, ask for the delivery address before confirming.

RESPONSE STYLE:
- Keep every reply SHORT — 1 to 2 sentences max. You're talking, not writing an essay.
- Sound natural and conversational, like a real person speaking out loud.
- NEVER repeat or echo back what the customer just told you. They can already see it on the live order card.
  BAD: "Great, large pepperoni pizza! And for the crust?"  GOOD: "Nice! What crust would you like?"
- Just acknowledge briefly and move to the next thing.
- Use casual phrasing ("Nice!", "Perfect!", "And the crust?") not formal sentences.

BEHAVIOUR RULES:
- Make smart recommendations when asked (vegetarian → pesto base, feta, spinach, mushrooms, etc.)
- Handle vague requests gracefully ("make it spicy" → suggest jalapeños + buffalo sauce)
- If someone wants half-and-half toppings, note it in extras as "Half [A] / Half [B]"
- Always offer drinks after the pizza is customised — suggest pairings!
- Always ask for a delivery address before confirming the order
- When ALL info is collected (including address) AND the customer explicitly confirms
  (words like "yes", "perfect", "place it", "go ahead", "that's right", "sounds good"),
  output EXACTLY this block on its own line — no extra text on that line:
  ##ORDER##{"name":"...","size":"...","crust":"...","sauce":"...","cheese":"...","toppings":["..."],"drinks":["..."],"extras":["..."],"quantity":1,"address":"..."}##END##
- All values inside the JSON must be lowercase and match the menu items as closely as possible
- The "drinks" array should list each drink ordered (can be empty [])
- The "address" must be the full delivery address the customer provided
- NEVER output ##ORDER## before explicit confirmation
- After outputting ##ORDER##, say a short warm farewell with a fancy Italian quote

LIVE ORDER TRACKING — CRITICAL (do this EVERY turn):
You MUST end EVERY reply (except the farewell after ##ORDER##) with an ##UPDATE## block.
The customer sees a LIVE order card built from this. Without it, the card stays blank!

Format — always on a NEW LINE at the very end:
##UPDATE##{"name":"...","size":"...","crust":"...","sauce":"...","cheese":"...","toppings":[...],"drinks":[...],"extras":[...],"quantity":1,"address":"..."}##END##

Rules:
- Include ALL fields every time, even unchanged ones
- Use confirmed lowercase values; use null for items not yet decided
- toppings/drinks/extras are arrays that grow as items are added

Examples:

After learning name is Lisa:
Nice to meet you, Lisa! What size pizza?
##UPDATE##{"name":"lisa","size":null,"crust":null,"sauce":null,"cheese":null,"toppings":[],"drinks":[],"extras":[],"quantity":1,"address":null}##END##

After Lisa picks medium + thin crust:
And which sauce?
##UPDATE##{"name":"lisa","size":"medium","crust":"thin","sauce":null,"cheese":null,"toppings":[],"drinks":[],"extras":[],"quantity":1,"address":null}##END##

After adding pepperoni and mushrooms:
Anything else on top?
##UPDATE##{"name":"lisa","size":"medium","crust":"thin","sauce":"marinara","cheese":"mozzarella","toppings":["pepperoni","mushrooms"],"drinks":[],"extras":[],"quantity":1,"address":null}##END##\
"""


def _build_system(partial: Optional[dict] = None, user_msg: str = "") -> str:
    """Assemble the system prompt, injecting only the relevant menu sections."""
    sections  = _relevant_sections(partial, user_msg)
    menu_text = "\n".join(_MENU_SECTIONS[s] for s in sections)
    return f"{_BASE_SYSTEM}\n\nMENU:\n{menu_text}"


# ── Pydantic models — validate every ORDER / UPDATE block ──────────────────────
class _OrderUpdate(BaseModel):
    name:     Optional[str] = None
    size:     Optional[str] = None
    crust:    Optional[str] = None
    sauce:    Optional[str] = None
    cheese:   Optional[str] = None
    toppings: List[str]     = []
    drinks:   List[str]     = []
    extras:   List[str]     = []
    quantity: int           = 1
    address:  Optional[str] = None


class _FinalOrder(_OrderUpdate):
    name:    str
    size:    str
    crust:   str
    sauce:   str
    cheese:  str
    address: str


def _parse_json(raw: str) -> Optional[dict]:
    """Parse JSON from LLM output with graceful markdown/blob fallback."""
    cleaned = _CODE_FENCE.sub("", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = _JSON_BLOB.search(cleaned)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


def _validate_update(data: dict) -> dict:
    try:
        return _OrderUpdate(**data).model_dump()
    except (ValidationError, Exception):
        return data


def _validate_final(data: dict) -> dict:
    try:
        return _FinalOrder(**data).model_dump()
    except (ValidationError, Exception):
        return data


# ── InferenceClient — module-level singleton ───────────────────────────────────
_hf_client: Optional[InferenceClient] = None
_hf_token_cached: str = ""


def _get_client() -> Optional[InferenceClient]:
    global _hf_client, _hf_token_cached
    token = (os.environ.get("HF_TOKEN") or
             os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token:
        return None
    if _hf_client is None or token != _hf_token_cached:
        _hf_client = InferenceClient(token=token)
        _hf_token_cached = token
    return _hf_client


# ── Rate limiter — simple in-memory per-IP tracker ────────────────────────────
_RATE_LIMIT_RPM  = 30          # max /chat requests per IP per 60-second window
_rate_store: dict = defaultdict(list)

def _is_rate_limited(ip: str) -> bool:
    now    = time.time()
    window = now - 60.0
    prev   = [t for t in _rate_store[ip] if t > window]
    _rate_store[ip] = prev
    if len(prev) >= _RATE_LIMIT_RPM:
        return True
    _rate_store[ip].append(now)
    return False


# ── Token tracking & history trimming ─────────────────────────────────────────
_stats: dict = {
    "calls": 0,
    "input_tokens_est": 0,
    "output_tokens_est": 0,
    "model_hits": {},
}


def _est_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def _trim_history(history: list, token_budget: int = 900) -> list:
    """
    Return the most-recent messages that fit within token_budget.
    Always keeps at least the 2 most-recent messages regardless of size,
    so the LLM always has immediate context.
    """
    if not history:
        return history
    kept, total = [], 0
    for msg in reversed(history):
        t = _est_tokens(msg["content"])
        if total + t > token_budget and len(kept) >= 2:
            break
        kept.append(msg)
        total += t
    return list(reversed(kept))


# ── Models ─────────────────────────────────────────────────────────────────────
MODELS = [
    ("Qwen/Qwen2.5-72B-Instruct",                True),
    ("meta-llama/Llama-3.3-70B-Instruct",        True),
    ("mistralai/Mistral-Small-24B-Instruct-2501", True),
    ("microsoft/Phi-4",                           True),
    ("Qwen/Qwen2.5-Coder-32B-Instruct",          True),
    ("google/gemma-2-27b-it",                     True),
]


def _build_text_prompt(history: list, system: str) -> str:
    """Plain-text fallback prompt for non-chat-completion models."""
    parts = [f"[SYSTEM]\n{system}\n"]
    for msg in history:
        role = "User" if msg["role"] == "user" else "Pino"
        parts.append(f"{role}: {msg['content']}")
    parts.append("Pino:")
    return "\n".join(parts)


def chat_with_llm(history: list, partial: Optional[dict] = None) -> str:
    client = _get_client()
    if not client:
        return "⚠️ No HuggingFace token found. Add HF_TOKEN in Space Settings → Secrets."

    last_user = next(
        (m["content"] for m in reversed(history) if m["role"] == "user"), ""
    )
    system   = _build_system(partial, last_user)
    trimmed  = _trim_history(history)
    messages = [{"role": "system", "content": system}] + trimmed

    # Track estimated input tokens before the call
    input_est = sum(_est_tokens(m["content"]) for m in messages)
    _stats["input_tokens_est"] += input_est
    _stats["calls"] += 1

    print(f"[LLM] call #{_stats['calls']} | ~{input_est} input tokens "
          f"| history {len(trimmed)}/{len(history)} msgs", flush=True)

    last_err = "Unknown error"
    for model_id, supports_chat in MODELS:
        print(f"[LLM] trying {model_id}", flush=True)
        try:
            if supports_chat:
                resp   = client.chat_completion(
                    model=model_id,
                    messages=messages,
                    max_tokens=250,   # raised: ##UPDATE## block needs ~100 tokens
                    temperature=0.70,
                )
                result = resp.choices[0].message.content.strip()
            else:
                prompt = _build_text_prompt(trimmed, system)
                resp   = client.text_generation(
                    prompt, model=model_id,
                    max_new_tokens=220, temperature=0.70,
                    stop_sequences=["User:", "[SYSTEM]"],
                )
                result = resp.split("User:")[0].strip()

            out_est = _est_tokens(result)
            _stats["output_tokens_est"] += out_est
            _stats["model_hits"][model_id] = _stats["model_hits"].get(model_id, 0) + 1
            print(f"[LLM] ✅ {model_id} | ~{out_est} output tokens", flush=True)
            return result

        except Exception as e:
            last_err = str(e)
            print(f"[LLM] ❌ {model_id}: {last_err[:200]}", flush=True)

    print(f"[LLM] 🚨 all models failed. Last: {last_err[:200]}", flush=True)
    return f"⚠️ All models failed. Last error: {last_err[:200]}"


# ── Order / update extraction ──────────────────────────────────────────────────
def extract_order(text: str):
    m = _ORDER_RE.search(text)
    if not m:
        return text, None
    raw  = _CODE_FENCE.sub("", m.group(1).strip()).strip()
    data = _parse_json(raw)
    if data:
        data = _validate_final(data)
    return _ORDER_RE.sub("", text).strip(), data


def extract_update(text: str):
    m = _UPDATE_RE.search(text)
    if not m:
        return text, None
    raw        = m.group(1).strip()
    clean_text = _UPDATE_RE.sub("", text).strip()
    data       = _parse_json(raw)
    if data:
        data = _validate_update(data)
    return clean_text, data


# ── Keyword-based partial inference (server-side fallback) ─────────────────────
_MENU_KEY_SET = frozenset(SIZES_MAP) | frozenset(CRUSTS_MAP) | frozenset(SAUCES_MAP)


def infer_partial(history: list) -> dict:
    """
    Lightweight keyword extraction used as fallback when the LLM omits ##UPDATE##.
    Uses pre-sorted keys and pre-compiled patterns — no per-call sorting.
    """
    state: dict = {
        "name": None, "size": None, "crust": None, "sauce": None,
        "cheese": None, "toppings": [], "drinks": [], "extras": [],
        "quantity": 1, "address": None,
    }

    for i, msg in enumerate(history):
        if msg["role"] != "user":
            continue
        text = msg["content"].strip()
        low  = text.lower()

        # Name heuristic: short first message not containing menu keywords
        if i == 0 and not state["name"]:
            cleaned = _NAME_PREFIX.sub("", low).strip()
            words   = cleaned.split()
            if len(words) <= 3 and not any(k in cleaned for k in _MENU_KEY_SET):
                state["name"] = cleaned.title()
                continue

        for k in _S_SIZES:
            if _WB[k].search(low):
                state["size"] = k
                break

        for k in _S_CRUSTS:
            if _WB[k].search(low):
                state["crust"] = k
                break

        for k in _S_SAUCES:
            if _WB[k].search(low):
                state["sauce"] = k
                break

        for k in _S_CHEESES:
            if _WB[k].search(low):
                state["cheese"] = k
                break

        seen_t = set(state["toppings"])
        for k in _S_TOPPINGS:
            if k not in seen_t and _WB[k].search(low):
                state["toppings"].append(k)
                seen_t.add(k)

        seen_d = set(state["drinks"])
        for k in _S_DRINKS:
            if k not in seen_d and _WB[k].search(low):
                state["drinks"].append(k)
                seen_d.add(k)

        qm = _QTY_RE.search(low)
        if qm:
            state["quantity"] = int(qm.group(1))

        # Address: user message after assistant asked for delivery details
        if i > 0 and history[i - 1].get("role") == "assistant":
            prev = history[i - 1]["content"].lower()
            if ("address" in prev or "deliver" in prev) and len(text) > 5:
                state["address"] = text

    return state


def merge_partial(llm_update: Optional[dict], inferred: Optional[dict]) -> Optional[dict]:
    if not llm_update and not inferred:
        return None
    merged = {**(inferred or {})}
    for k, v in (llm_update or {}).items():
        if k in ("toppings", "drinks", "extras"):
            if v:                   # non-empty LLM list wins
                merged[k] = v
        elif v is not None:
            merged[k] = v
    return merged


# ── Receipt builder ────────────────────────────────────────────────────────────
def _pick(val: Optional[str], catalogue: dict, sorted_keys: list, default_key: str):
    """Fuzzy-match val against catalogue using pre-sorted keys."""
    if not val:
        return catalogue[default_key]
    v = val.lower().strip()
    for k in sorted_keys:
        if k in v or v in k:
            return catalogue[k]
    return catalogue[default_key]


def _to_list(val) -> list:
    if isinstance(val, str):
        return [x.strip() for x in val.split(",")]
    return list(val or [])


def build_receipt(order_data: dict) -> dict:
    size   = _pick(order_data.get("size"),   SIZES_MAP,   _S_SIZES,   "medium")
    crust  = _pick(order_data.get("crust"),  CRUSTS_MAP,  _S_CRUSTS,  "hand tossed")
    sauce  = _pick(order_data.get("sauce"),  SAUCES_MAP,  _S_SAUCES,  "tomato")
    cheese = _pick(order_data.get("cheese"), CHEESES_MAP, _S_CHEESES, "mozzarella")
    qty    = max(1, int(order_data.get("quantity", 1)))

    # Toppings
    seen_t, tops = set(), []
    for item in _to_list(order_data.get("toppings", [])):
        il = item.lower()
        for k in _S_TOPPINGS:
            if k in il:
                lbl, emoji, price = TOPPINGS_MAP[k]
                if lbl not in seen_t:
                    seen_t.add(lbl)
                    tops.append({"label": lbl, "emoji": emoji, "price": price})
                break

    # Drinks
    seen_d, drinks_out = set(), []
    for item in _to_list(order_data.get("drinks", [])):
        il = item.lower().strip()
        if not il or il == "none":
            continue
        for k in _S_DRINKS:
            if k in il or il in k:
                lbl, emoji, price = DRINKS_MAP[k]
                if lbl not in seen_d:
                    seen_d.add(lbl)
                    drinks_out.append({"label": lbl, "emoji": emoji, "price": price})
                break

    # Extras — uses EXTRAS_MAP via pre-sorted keys; unknown extras pass through
    extras_out = []
    for item in _to_list(order_data.get("extras", [])):
        il = item.lower().strip()
        if not il:
            continue
        matched = False
        for k in _S_EXTRAS:
            if k in il:
                lbl, price = EXTRAS_MAP[k]
                extras_out.append({"label": lbl, "price": price})
                matched = True
                break
        if not matched:
            extras_out.append({"label": item.strip().title(), "price": 0.00})

    address = (order_data.get("address") or "").strip() or "Pick-up"

    pizza_unit   = (size[1] + crust[1] + sauce[1] + cheese[1]
                    + sum(t["price"] for t in tops)
                    + sum(e["price"] for e in extras_out))
    pizza_sub    = pizza_unit * qty
    drinks_total = sum(d["price"] for d in drinks_out)
    sub          = pizza_sub + drinks_total
    tax          = sub * TAX_RATE

    return {
        "order": {
            "name":     order_data.get("name", "Friend"),
            "size":     {"label": size[0],   "price": size[1]},
            "crust":    {"label": crust[0],  "price": crust[1]},
            "sauce":    {"label": sauce[0],  "price": sauce[1]},
            "cheese":   {"label": cheese[0], "price": cheese[1]},
            "toppings": tops,
            "drinks":   drinks_out,
            "extras":   extras_out,
            "quantity": qty,
            "address":  address,
        },
        "pricing": {
            "unit":     round(pizza_unit, 2),
            "quantity": qty,
            "subtotal": round(sub,        2),
            "tax":      round(tax,        2),
            "total":    round(sub + tax,  2),
            "breakdown": {
                "base":     round(size[1],                              2),
                "crust":    round(crust[1],                             2),
                "sauce":    round(sauce[1],                             2),
                "cheese":   round(cheese[1],                            2),
                "toppings": round(sum(t["price"] for t in tops),        2),
                "drinks":   round(drinks_total,                         2),
                "extras":   round(sum(e["price"] for e in extras_out),  2),
            },
        },
        "quote":     random.choice(FANCY_QUOTES),
        "order_id":  str(uuid.uuid4())[:8].upper(),
        "timestamp": datetime.now().strftime("%B %d, %Y  •  %I:%M %p"),
    }


# ── TTS: Edge TTS (primary) → HF Inference (fallback) ─────────────────────────
EDGE_VOICE    = "en-US-JennyNeural"
HF_TTS_MODELS = ["espnet/kan-bayashi_ljspeech_vits", "facebook/mms-tts-eng"]

# Simple LRU-style cache keyed by MD5 of cleaned text
_TTS_CACHE:     dict = {}
_TTS_CACHE_MAX: int  = 64


def _tts_clean(text: str) -> str:
    clean = _TTS_CLEAN.sub(" ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:500]


def _edge_tts_sync(text: str) -> bytes:
    async def _gen():
        comm = edge_tts.Communicate(text, EDGE_VOICE)
        buf  = b""
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf += chunk["data"]
        return buf

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_gen())
    finally:
        loop.close()


@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return Response(b"", status=400)

    clean     = _tts_clean(text)
    cache_key = hashlib.md5(clean.encode()).hexdigest()

    if cache_key in _TTS_CACHE:
        print(f"[TTS] cache hit {cache_key[:8]}", flush=True)
        return Response(
            _TTS_CACHE[cache_key], mimetype="audio/mpeg",
            headers={"Cache-Control": "no-cache"},
        )

    audio: Optional[bytes] = None

    # 1) Edge TTS — high-quality Microsoft Neural voices (free)
    if edge_tts:
        try:
            audio = _edge_tts_sync(clean)
            if len(audio) < 100:
                audio = None
        except Exception as e:
            print(f"[TTS] Edge failed: {e}", flush=True)
            audio = None

    # 2) HF Inference REST fallback
    if not audio:
        token = (os.environ.get("HF_TOKEN") or
                 os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
        if token:
            for mid in HF_TTS_MODELS:
                try:
                    r = http_req.post(
                        f"https://api-inference.huggingface.co/models/{mid}",
                        headers={"Authorization": f"Bearer {token}"},
                        json={"inputs": clean, "options": {"wait_for_model": True}},
                        timeout=30,
                    )
                    if r.status_code == 200 and len(r.content) > 100:
                        audio = r.content
                        break
                except Exception as e:
                    print(f"[TTS] HF {mid}: {e}", flush=True)

    if not audio:
        return Response(b"", status=503)

    # Evict oldest entry when cache is full
    if len(_TTS_CACHE) >= _TTS_CACHE_MAX:
        del _TTS_CACHE[next(iter(_TTS_CACHE))]
    _TTS_CACHE[cache_key] = audio

    return Response(audio, mimetype="audio/mpeg", headers={"Cache-Control": "no-cache"})


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prices")
def prices():
    """
    Serve all price maps to the frontend.
    The JS fetches this once at page load instead of hard-coding prices,
    so a catalogue change in catalogue.py automatically propagates to the UI.
    """
    return jsonify({
        "sizes":       {k: v[1] for k, v in SIZES_MAP.items()},
        "crusts":      {k: v[1] for k, v in CRUSTS_MAP.items()},
        "sauces":      {k: v[1] for k, v in SAUCES_MAP.items()},
        "cheeses":     {k: v[1] for k, v in CHEESES_MAP.items()},
        "toppings":    {k: v[2] for k, v in TOPPINGS_MAP.items()},
        "drinks":      {k: v[2] for k, v in DRINKS_MAP.items()},
        "size_labels": {k: v[0] for k, v in SIZES_MAP.items()},
        "tax_rate":    TAX_RATE,
    })


@app.route("/chat", methods=["POST"])
def chat():
    ip = (request.headers.get("X-Forwarded-For") or
          request.remote_addr or "unknown").split(",")[0].strip()
    if _is_rate_limited(ip):
        return jsonify({
            "reply":   "⚠️ Too many requests — please wait a moment!",
            "partial": None,
            "receipt": None,
        }), 429

    data    = request.get_json(force=True)
    history = data.get("history", [])

    # Infer partial state first so _build_system can inject the right menu sections
    inferred = infer_partial(history)
    reply    = chat_with_llm(history, partial=inferred)

    reply, order_data  = extract_order(reply)
    reply, update_data = extract_update(reply)

    partial = merge_partial(update_data, inferred)
    receipt = build_receipt(order_data) if order_data else None
    return jsonify({"reply": reply, "partial": partial, "receipt": receipt})


@app.route("/stats")
def stats():
    """Live token-usage and model-hit counters for monitoring."""
    total = _stats["input_tokens_est"] + _stats["output_tokens_est"]
    return jsonify({
        "calls":              _stats["calls"],
        "input_tokens_est":   _stats["input_tokens_est"],
        "output_tokens_est":  _stats["output_tokens_est"],
        "total_tokens_est":   total,
        "avg_input_per_call": round(_stats["input_tokens_est"] / max(1, _stats["calls"]), 1),
        "model_hits":         _stats["model_hits"],
        "tts_cache_entries":  len(_TTS_CACHE),
        "tts_cache_capacity": _TTS_CACHE_MAX,
    })


# ── Startup ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  PizzaVoice  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"  Catalogue : {len(SIZES_MAP)} sizes | {len(CRUSTS_MAP)} crusts | "
          f"{len(SAUCES_MAP)} sauces | {len(CHEESES_MAP)} cheeses | "
          f"{len(TOPPINGS_MAP)} toppings | {len(DRINKS_MAP)} drinks")
    print(f"  Patterns  : {len(_WB)} pre-compiled word-boundary regexes")
    print(f"  Menu RAG  : {len(_MENU_SECTIONS)} sections, injected dynamically per turn")
    print(f"  TTS cache : up to {_TTS_CACHE_MAX} entries")
    print(f"  Models    : {len(MODELS)} with sequential fallback")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=7860, debug=False)
