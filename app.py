from flask import Flask, render_template, request, jsonify, Response
from huggingface_hub import InferenceClient
import os, re, json, uuid, io, asyncio
import requests as http_req
try:
    import edge_tts
except ImportError:
    edge_tts = None
from datetime import datetime

app = Flask(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM = """You are Pino, a warm and witty Italian pizza waiter at PizzaVoice.

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

MENU:
  Sizes:   Personal 6" $7.99 | Small 8" $9.99 | Medium 12" $13.99 | Large 14" $16.99 | XL 16" $19.99
  Crusts:  Thin / Thick / Hand Tossed (free) | Stuffed +$2.50 | Cauliflower +$3.00 | Gluten-Free +$2.50
  Sauces:  Marinara / Tomato / BBQ / Ranch / Buffalo / Garlic Butter (free) | Alfredo / Pesto +$0.50
  Cheese:  Mozzarella (free) | Cheddar/Parmesan +$0.50 | Feta/Gouda +$1.00 | Ricotta +$0.75
           Vegan/Dairy-Free +$1.50 | No Cheese (free)
  Toppings ($0.75–$3.00 each):
    Meat:    Pepperoni, Italian Sausage, Bacon, Ham, Grilled Chicken, Ground Beef, Anchovies, Prosciutto
    Veg:     Mushrooms, Spinach, Jalapeños, Black Olives, Bell Peppers, Red Onion, Fresh Tomatoes,
             Fresh Basil, Roasted Garlic, Arugula, Broccoli, Sweet Corn, Artichoke Hearts,
             Avocado, Zucchini, Sun-Dried Tomatoes, Pineapple
    Premium: Truffle Oil +$3.00
  Drinks:
    Soft Drinks: Cola $2.49 | Diet Cola $2.49 | Sprite $2.49 | Fanta $2.49 | Root Beer $2.49
    Juice:       Orange Juice $2.99 | Apple Juice $2.99 | Lemonade $2.99
    Water:       Still Water $1.49 | Sparkling Water $1.99
    Italian:     Espresso $3.49 | Cappuccino $4.49 | Italian Soda $3.99 | Limonata $3.99
    Beer/Wine:   Craft Beer $5.99 | House Red Wine $6.99 | House White Wine $6.99 | Prosecco $7.99
  Extras:  Extra Cheese +$1.50 | Extra Sauce +$0.50 | Well Done / Light Sauce (free)

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
##UPDATE##{"name":"lisa","size":"medium","crust":"thin","sauce":"marinara","cheese":"mozzarella","toppings":["pepperoni","mushrooms"],"drinks":[],"extras":[],"quantity":1,"address":null}##END##
"""

# ── Catalogue maps ─────────────────────────────────────────────────────────────
SIZES_MAP = {
    "personal": ("Personal 6\"", 7.99), "small": ("Small 8\"", 9.99),
    "medium": ("Medium 12\"", 13.99), "large": ("Large 14\"", 16.99),
    "extra large": ("XL 16\"", 19.99), "xl": ("XL 16\"", 19.99),
}
CRUSTS_MAP = {
    "thin": ("Thin Crust", 0.00), "crispy": ("Crispy Thin", 0.00),
    "thick": ("Thick Crust", 0.00), "hand tossed": ("Hand Tossed", 0.00),
    "stuffed": ("Stuffed Crust", 2.50), "cauliflower": ("Cauliflower Crust", 3.00),
    "gluten free": ("Gluten-Free Crust", 2.50), "gluten-free": ("Gluten-Free Crust", 2.50),
}
SAUCES_MAP = {
    "marinara": ("Marinara", 0.00), "tomato": ("Classic Tomato", 0.00),
    "bbq": ("BBQ", 0.00), "barbecue": ("BBQ", 0.00),
    "alfredo": ("Alfredo", 0.50), "white": ("White Alfredo", 0.50),
    "pesto": ("Basil Pesto", 0.50), "ranch": ("Ranch", 0.00),
    "buffalo": ("Buffalo", 0.00), "garlic butter": ("Garlic Butter", 0.00),
}
CHEESES_MAP = {
    "no cheese": ("No Cheese", 0.00), "dairy-free": ("Dairy-Free", 1.50),
    "vegan": ("Vegan Cheese", 1.50), "mozzarella": ("Mozzarella", 0.00),
    "cheddar": ("Cheddar", 0.50), "parmesan": ("Parmesan", 0.50),
    "feta": ("Feta", 1.00), "gouda": ("Gouda", 1.00), "ricotta": ("Ricotta", 0.75),
}
TOPPINGS_MAP = {
    "pepperoni": ("Pepperoni", "🍕", 1.50), "mushrooms": ("Mushrooms", "🍄", 1.00),
    "mushroom": ("Mushrooms", "🍄", 1.00), "spinach": ("Spinach", "🥬", 1.00),
    "jalapeños": ("Jalapeños", "🌶️", 0.75), "jalapenos": ("Jalapeños", "🌶️", 0.75),
    "jalapeno": ("Jalapeños", "🌶️", 0.75), "olives": ("Black Olives", "🫒", 1.00),
    "black olives": ("Black Olives", "🫒", 1.00),
    "bell peppers": ("Bell Peppers", "🫑", 1.00), "bell pepper": ("Bell Peppers", "🫑", 1.00),
    "red onion": ("Red Onion", "🧅", 0.75), "onion": ("Red Onion", "🧅", 0.75),
    "grilled chicken": ("Grilled Chicken", "🍗", 2.00), "chicken": ("Grilled Chicken", "🍗", 2.00),
    "ground beef": ("Ground Beef", "🥩", 2.00), "beef": ("Ground Beef", "🥩", 2.00),
    "sausage": ("Italian Sausage", "🌭", 1.75), "bacon": ("Bacon", "🥓", 1.75),
    "ham": ("Ham", "🍖", 1.50), "pineapple": ("Pineapple", "🍍", 1.00),
    "fresh tomatoes": ("Fresh Tomatoes", "🍅", 1.00), "tomatoes": ("Fresh Tomatoes", "🍅", 1.00),
    "basil": ("Fresh Basil", "🌿", 0.75), "garlic": ("Roasted Garlic", "🧄", 0.75),
    "arugula": ("Arugula", "🥗", 1.00), "broccoli": ("Broccoli", "🥦", 1.00),
    "corn": ("Sweet Corn", "🌽", 0.75), "artichoke": ("Artichoke Hearts", "🌱", 1.50),
    "anchovies": ("Anchovies", "🐟", 1.50), "avocado": ("Avocado", "🥑", 1.50),
    "prosciutto": ("Prosciutto", "🍖", 2.50), "truffle": ("Truffle Oil", "✨", 3.00),
    "zucchini": ("Zucchini", "🥒", 1.00),
    "sun-dried tomatoes": ("Sun-Dried Tomatoes", "☀️", 1.25),
    "sun dried tomato": ("Sun-Dried Tomatoes", "☀️", 1.25),
}
DRINKS_MAP = {
    "cola":            ("Cola",            "🥤", 2.49),
    "diet cola":       ("Diet Cola",       "🥤", 2.49),
    "sprite":          ("Sprite",          "🥤", 2.49),
    "fanta":           ("Fanta",           "🥤", 2.49),
    "root beer":       ("Root Beer",       "🍺", 2.49),
    "orange juice":    ("Orange Juice",    "🍊", 2.99),
    "apple juice":     ("Apple Juice",     "🍏", 2.99),
    "lemonade":        ("Lemonade",        "🍋", 2.99),
    "still water":     ("Still Water",     "💧", 1.49),
    "water":           ("Still Water",     "💧", 1.49),
    "sparkling water": ("Sparkling Water", "✨", 1.99),
    "sparkling":       ("Sparkling Water", "✨", 1.99),
    "espresso":        ("Espresso",        "☕", 3.49),
    "cappuccino":      ("Cappuccino",      "☕", 4.49),
    "italian soda":    ("Italian Soda",    "🧊", 3.99),
    "limonata":        ("Limonata",        "🍋", 3.99),
    "craft beer":      ("Craft Beer",      "🍺", 5.99),
    "beer":            ("Craft Beer",      "🍺", 5.99),
    "red wine":        ("House Red Wine",  "🍷", 6.99),
    "white wine":      ("House White Wine","🥂", 6.99),
    "prosecco":        ("Prosecco",        "🥂", 7.99),
}
TAX_RATE = 0.13

# ── Fancy quotes for the receipt ───────────────────────────────────────────────
import random
FANCY_QUOTES = [
    "\"In pizza we trust.\" — Every Italian ever 🍕",
    "\"Life is too short for bad pizza.\" — Pino 🇮🇹",
    "\"You can't make everyone happy. You're not pizza.\" — Unknown 😄",
    "\"A slice a day keeps the sadness away.\" — Ancient Proverb 🧡",
    "\"La vita è bella… especially with pizza!\" — PizzaVoice 💫",
    "\"Pizza is a lot like love — when it's good, it's really good.\" — Chef Pino 🍷",
    "\"Eat pizza. Be happy. Repeat.\" — The Italian Way 🌟",
    "\"There's no 'we' in pizza… wait, yes there is. Share the love!\" — Pino 🤌",
]

# ── Models: (model_id, supports_chat_completion) ───────────────────────────────
# Ordered by reliability on HF Serverless Inference API (free tier)
# Updated Feb 2026 — old models returned 410 Gone on the legacy endpoint.
# huggingface_hub >= 0.31 auto-routes via Inference Providers.
MODELS = [
    ("Qwen/Qwen2.5-72B-Instruct",                True),
    ("meta-llama/Llama-3.3-70B-Instruct",        True),
    ("mistralai/Mistral-Small-24B-Instruct-2501", True),
    ("microsoft/Phi-4",                           True),
    ("Qwen/Qwen2.5-Coder-32B-Instruct",          True),
    ("google/gemma-2-27b-it",                     True),
]


def _build_prompt(history):
    """Plain-text prompt for models that only support text_generation."""
    parts = [f"[SYSTEM]\n{SYSTEM}\n"]
    for msg in history:
        role = "User" if msg["role"] == "user" else "Pino"
        parts.append(f"{role}: {msg['content']}")
    parts.append("Pino:")
    return "\n".join(parts)


def chat_with_llm(history):
    import sys

    # HF Spaces auto-injects HUGGING_FACE_HUB_TOKEN; also accept manual HF_TOKEN secret
    hf_token_raw   = os.environ.get("HF_TOKEN") or ""
    hfhub_token_raw = os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    token = (hf_token_raw or hfhub_token_raw).strip()

    print(f"[DEBUG] HF_TOKEN present: {bool(hf_token_raw.strip())}", flush=True)
    print(f"[DEBUG] HUGGING_FACE_HUB_TOKEN present: {bool(hfhub_token_raw.strip())}", flush=True)
    print(f"[DEBUG] Token resolved: {bool(token)} (len={len(token)})", flush=True)
    sys.stdout.flush()

    if not token:
        return "⚠️ No HuggingFace token found. Add HF_TOKEN in Space Settings → Secrets."

    trimmed  = history[-20:]
    messages = [{"role": "system", "content": SYSTEM}] + trimmed
    client   = InferenceClient(token=token)
    last_err = "Unknown error"

    for model_id, supports_chat in MODELS:
        print(f"[DEBUG] Trying model: {model_id} (chat={supports_chat})", flush=True)
        try:
            if supports_chat:
                resp = client.chat_completion(
                    model=model_id,
                    messages=messages,
                    max_tokens=180,
                    temperature=0.75,
                )
                result = resp.choices[0].message.content.strip()
                print(f"[DEBUG] ✅ Success with {model_id} — response length: {len(result)}", flush=True)
                return result
            else:
                prompt = _build_prompt(trimmed)
                resp   = client.text_generation(
                    prompt,
                    model=model_id,
                    max_new_tokens=150,
                    temperature=0.75,
                    stop_sequences=["User:", "[SYSTEM]"],
                )
                result = resp.split("User:")[0].strip()
                print(f"[DEBUG] ✅ Success with {model_id} — response length: {len(result)}", flush=True)
                return result

        except Exception as e:
            last_err = str(e)
            print(f"[DEBUG] ❌ Failed {model_id}: {last_err[:300]}", flush=True)
            continue

    print(f"[DEBUG] 🚨 All models exhausted. Last error: {last_err[:300]}", flush=True)
    return f"⚠️ All models failed. Last error: {last_err[:200]}"


# ── Order extraction ───────────────────────────────────────────────────────────
ORDER_RE = re.compile(
    r"#{1,3}\s*ORDER\s*#{1,3}(.*?)#{1,3}\s*END\s*#{1,3}",
    re.DOTALL | re.IGNORECASE,
)

def extract_order(text):
    m = ORDER_RE.search(text)
    if not m:
        return text, None
    raw = re.sub(r"^```[a-z]*\n?|```$", "", m.group(1).strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        inner = re.search(r"\{.*\}", raw, re.DOTALL)
        if not inner:
            return ORDER_RE.sub("", text).strip(), None
        try:
            data = json.loads(inner.group())
        except Exception:
            return ORDER_RE.sub("", text).strip(), None
    return ORDER_RE.sub("", text).strip(), data


# ── Live-order partial extraction ──────────────────────────────────────────────
UPDATE_RE = re.compile(
    r"#{1,3}\s*UPDATE\s*#{1,3}(.*?)#{1,3}\s*END\s*#{1,3}",
    re.DOTALL | re.IGNORECASE,
)

def extract_update(text):
    """Pull the optional ##UPDATE##…##END## block, return (clean_text, dict|None)."""
    m = UPDATE_RE.search(text)
    if not m:
        return text, None
    raw = m.group(1).strip()
    clean_text = UPDATE_RE.sub("", text).strip()
    try:
        return clean_text, json.loads(raw)
    except json.JSONDecodeError:
        inner = re.search(r"\{.*\}", raw, re.DOTALL)
        if inner:
            try:
                return clean_text, json.loads(inner.group())
            except Exception:
                pass
    return clean_text, None


# ── Server-side fallback: infer partial order from conversation ────────────────
def infer_partial(history):
    """Lightweight keyword extraction as fallback when LLM skips ##UPDATE##."""
    state = {"name": None, "size": None, "crust": None, "sauce": None,
             "cheese": None, "toppings": [], "drinks": [], "extras": [],
             "quantity": 1, "address": None}

    for i, msg in enumerate(history):
        if msg["role"] != "user":
            continue
        text = msg["content"].strip()
        low = text.lower()

        # Name: first user message if short and not a menu keyword
        if i == 0 and not state["name"]:
            cleaned = re.sub(r"^(my name is|i'm|i am|it's|this is|hey i'm|call me)\s+", "", low).strip()
            words = cleaned.split()
            menu_keys = set(SIZES_MAP) | set(CRUSTS_MAP) | set(SAUCES_MAP)
            if len(words) <= 3 and not any(k in cleaned for k in menu_keys):
                state["name"] = cleaned.title()
                continue

        # Size
        for k in sorted(SIZES_MAP, key=len, reverse=True):
            if re.search(r'\b' + re.escape(k) + r'\b', low):
                state["size"] = k
                break

        # Crust
        for k in sorted(CRUSTS_MAP, key=len, reverse=True):
            if re.search(r'\b' + re.escape(k) + r'\b', low):
                state["crust"] = k
                break

        # Sauce
        for k in sorted(SAUCES_MAP, key=len, reverse=True):
            if re.search(r'\b' + re.escape(k) + r'\b', low):
                state["sauce"] = k
                break

        # Cheese
        for k in sorted(CHEESES_MAP, key=len, reverse=True):
            if re.search(r'\b' + re.escape(k) + r'\b', low):
                state["cheese"] = k
                break

        # Toppings
        seen_t = set(state["toppings"])
        for k in sorted(TOPPINGS_MAP, key=len, reverse=True):
            if re.search(r'\b' + re.escape(k) + r'\b', low) and k not in seen_t:
                state["toppings"].append(k)
                seen_t.add(k)

        # Drinks
        seen_d = set(state["drinks"])
        for k in sorted(DRINKS_MAP, key=len, reverse=True):
            if re.search(r'\b' + re.escape(k) + r'\b', low) and k not in seen_d:
                state["drinks"].append(k)
                seen_d.add(k)

        # Quantity
        qm = re.search(r'\b(\d+)\s*(pizza|pie)', low)
        if qm:
            state["quantity"] = int(qm.group(1))

        # Address: if previous assistant message asked about address/delivery
        if i > 0 and history[i-1].get("role") == "assistant":
            prev_a = history[i-1]["content"].lower()
            if ("address" in prev_a or "deliver" in prev_a) and len(text) > 5:
                state["address"] = text

    return state


def merge_partial(llm_update, inferred):
    """Merge LLM UPDATE data with inferred data, preferring LLM non-null values."""
    if not llm_update and not inferred:
        return None
    base = inferred or {}
    overlay = llm_update or {}
    merged = {**base}
    for k, v in overlay.items():
        if k in ("toppings", "drinks", "extras"):
            if v:  # non-empty list from LLM wins
                merged[k] = v
        elif v is not None:
            merged[k] = v
    return merged


# ── Receipt builder ────────────────────────────────────────────────────────────
def _pick(val, catalogue, default_key):
    if not val:
        return catalogue[default_key]
    v = val.lower().strip()
    for k in sorted(catalogue, key=len, reverse=True):
        if k in v or v in k:
            return catalogue[k]
    return catalogue[default_key]


def build_receipt(order_data):
    size   = _pick(order_data.get("size"),   SIZES_MAP,   "medium")
    crust  = _pick(order_data.get("crust"),  CRUSTS_MAP,  "hand tossed")
    sauce  = _pick(order_data.get("sauce"),  SAUCES_MAP,  "tomato")
    cheese = _pick(order_data.get("cheese"), CHEESES_MAP, "mozzarella")
    qty    = max(1, int(order_data.get("quantity", 1)))

    raw_tops = order_data.get("toppings", [])
    if isinstance(raw_tops, str):
        raw_tops = [x.strip() for x in raw_tops.split(",")]
    seen, tops = set(), []
    for item in raw_tops:
        il = item.lower()
        for k in sorted(TOPPINGS_MAP, key=len, reverse=True):
            if k in il:
                lbl, emoji, price = TOPPINGS_MAP[k]
                if lbl not in seen:
                    seen.add(lbl)
                    tops.append({"label": lbl, "emoji": emoji, "price": price})
                break

    # ── Drinks ──
    raw_drinks = order_data.get("drinks", [])
    if isinstance(raw_drinks, str):
        raw_drinks = [x.strip() for x in raw_drinks.split(",")]
    seen_drinks, drinks_out = set(), []
    for item in raw_drinks:
        il = item.lower().strip()
        if not il or il == "none":
            continue
        for k in sorted(DRINKS_MAP, key=len, reverse=True):
            if k in il or il in k:
                lbl, emoji, price = DRINKS_MAP[k]
                if lbl not in seen_drinks:
                    seen_drinks.add(lbl)
                    drinks_out.append({"label": lbl, "emoji": emoji, "price": price})
                break

    raw_ex = order_data.get("extras", [])
    if isinstance(raw_ex, str):
        raw_ex = [x.strip() for x in raw_ex.split(",")]
    extras_out = []
    for item in raw_ex:
        il = item.lower()
        if "extra cheese" in il:
            extras_out.append({"label": "Extra Cheese", "price": 1.50})
        elif "extra sauce" in il:
            extras_out.append({"label": "Extra Sauce",  "price": 0.50})
        elif "well done"   in il:
            extras_out.append({"label": "Well Done",    "price": 0.00})
        elif "light sauce" in il:
            extras_out.append({"label": "Light Sauce",  "price": 0.00})
        elif item.strip():
            extras_out.append({"label": item.strip().title(), "price": 0.00})

    # ── Address ──
    address = order_data.get("address", "").strip() or "Pick-up"

    pizza_unit = (size[1] + crust[1] + sauce[1] + cheese[1]
                  + sum(t["price"] for t in tops)
                  + sum(e["price"] for e in extras_out))
    pizza_sub   = pizza_unit * qty
    drinks_total = sum(d["price"] for d in drinks_out)
    sub  = pizza_sub + drinks_total
    tax  = sub * TAX_RATE

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
            "subtotal": round(sub,  2),
            "tax":      round(tax,  2),
            "total":    round(sub + tax, 2),
            "breakdown": {
                "base":     round(size[1],  2),
                "crust":    round(crust[1], 2),
                "sauce":    round(sauce[1], 2),
                "cheese":   round(cheese[1], 2),
                "toppings": round(sum(t["price"] for t in tops),      2),
                "drinks":   round(drinks_total, 2),
                "extras":   round(sum(e["price"] for e in extras_out), 2),
            },
        },
        "quote":     random.choice(FANCY_QUOTES),
        "order_id":  str(uuid.uuid4())[:8].upper(),
        "timestamp": datetime.now().strftime("%B %d, %Y  •  %I:%M %p"),
    }


# ── Natural TTS (Edge TTS primary → HF Inference fallback) ────────────────────
EDGE_VOICE = "en-US-JennyNeural"          # warm, natural Microsoft Neural voice
HF_TTS_MODELS = [
    "espnet/kan-bayashi_ljspeech_vits",
    "facebook/mms-tts-eng",
]


def _edge_tts_sync(text, voice=EDGE_VOICE):
    """Run Edge TTS and return MP3 bytes synchronously."""
    async def _generate():
        comm = edge_tts.Communicate(text, voice)
        buf = b""
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf += chunk["data"]
        return buf

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_generate())
    finally:
        loop.close()


@app.route("/tts", methods=["POST"])
def tts():
    """Natural speech: Edge TTS (Microsoft Neural) → HF Inference → 503."""
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return Response(b"", status=400)

    # Clean text for speech
    clean = re.sub(r"[#*_~`>]", "", text)
    clean = re.sub(r"\bhttps?://\S+", "link", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) > 500:
        clean = clean[:500]

    # 1) Edge TTS — very natural Microsoft Neural voices (free)
    if edge_tts:
        try:
            audio = _edge_tts_sync(clean)
            if len(audio) > 100:
                return Response(audio, mimetype="audio/mpeg",
                                headers={"Cache-Control": "no-cache"})
        except Exception as e:
            print(f"[TTS] Edge failed: {str(e)[:200]}", flush=True)

    # 2) HF Inference API direct REST as fallback
    token = (os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if token:
        for mid in HF_TTS_MODELS:
            try:
                r = http_req.post(
                    f"https://api-inference.huggingface.co/models/{mid}",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"inputs": clean,
                          "options": {"wait_for_model": True}},
                    timeout=30,
                )
                if r.status_code == 200 and len(r.content) > 100:
                    ct = r.headers.get("content-type", "audio/flac")
                    return Response(r.content, mimetype=ct,
                                    headers={"Cache-Control": "no-cache"})
            except Exception as e:
                print(f"[TTS] HF {mid}: {str(e)[:200]}", flush=True)

    return Response(b"", status=503)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data        = request.get_json(force=True)
    history     = data.get("history", [])
    reply       = chat_with_llm(history)
    reply, order_data = extract_order(reply)
    reply, update_data = extract_update(reply)
    # Fallback: infer partial from conversation if LLM didn't include UPDATE
    inferred = infer_partial(history)
    partial  = merge_partial(update_data, inferred)
    receipt  = build_receipt(order_data) if order_data else None
    return jsonify({"reply": reply, "partial": partial, "receipt": receipt})


if __name__ == "__main__":
    print(f"\n===== Application Startup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    app.run(host="0.0.0.0", port=7860, debug=False)