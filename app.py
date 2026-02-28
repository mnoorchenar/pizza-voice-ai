from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient
import os, re, json, uuid
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
Collect: customer name, size, crust, sauce, cheese, toppings (or none), quantity.
Do NOT ask all questions at once. Ask ONE thing at a time, like a real waiter would.

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
  Extras:  Extra Cheese +$1.50 | Extra Sauce +$0.50 | Well Done / Light Sauce (free)

BEHAVIOUR RULES:
- Make smart recommendations when asked (vegetarian → pesto base, feta, spinach, mushrooms, etc.)
- Handle vague requests gracefully ("make it spicy" → suggest jalapeños + buffalo sauce)
- If someone wants half-and-half toppings, note it in extras as "Half [A] / Half [B]"
- When ALL info is collected AND the customer explicitly confirms
  (words like "yes", "perfect", "place it", "go ahead", "that's right", "sounds good"),
  output EXACTLY this block on its own line — no extra text on that line:
  ##ORDER##{"name":"...","size":"...","crust":"...","sauce":"...","cheese":"...","toppings":["..."],"extras":["..."],"quantity":1}##END##
- All values inside the JSON must be lowercase and match the menu items as closely as possible
- NEVER output ##ORDER## before explicit confirmation
- After outputting ##ORDER##, say a short warm farewell
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
TAX_RATE = 0.13

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
                    max_tokens=350,
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
                    max_new_tokens=300,
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

    unit = (size[1] + crust[1] + sauce[1] + cheese[1]
            + sum(t["price"] for t in tops)
            + sum(e["price"] for e in extras_out))
    sub  = unit * qty
    tax  = sub * TAX_RATE

    return {
        "order": {
            "name":     order_data.get("name", "Friend"),
            "size":     {"label": size[0],   "price": size[1]},
            "crust":    {"label": crust[0],  "price": crust[1]},
            "sauce":    {"label": sauce[0],  "price": sauce[1]},
            "cheese":   {"label": cheese[0], "price": cheese[1]},
            "toppings": tops,
            "extras":   extras_out,
            "quantity": qty,
        },
        "pricing": {
            "unit":     round(unit, 2),
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
                "extras":   round(sum(e["price"] for e in extras_out), 2),
            },
        },
        "order_id":  str(uuid.uuid4())[:8].upper(),
        "timestamp": datetime.now().strftime("%B %d, %Y  •  %I:%M %p"),
    }


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
    receipt     = build_receipt(order_data) if order_data else None
    return jsonify({"reply": reply, "receipt": receipt})


if __name__ == "__main__":
    print(f"\n===== Application Startup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    app.run(host="0.0.0.0", port=7860, debug=False)