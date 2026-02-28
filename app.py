from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient
import os, re, json, uuid
from datetime import datetime

app   = Flask(__name__)
TOKEN = os.environ.get("HF_TOKEN", "")

# ── LLM client ────────────────────────────────────────────────────────────────
def get_client():
    if not TOKEN:
        return None
    for model in [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "HuggingFaceH4/zephyr-7b-beta",
        "microsoft/Phi-3-mini-4k-instruct",
    ]:
        try:
            return InferenceClient(model=model, token=TOKEN)
        except Exception:
            continue
    return None

CLIENT = get_client()

# ── System prompt — Pino the waiter ───────────────────────────────────────────
SYSTEM = """You are Pino, a warm and friendly Italian pizza waiter at PizzaVoice restaurant.
Your job is to have a NATURAL conversation with the customer to take their pizza order.

MENU:
Sizes:   Personal $7.99 | Small $9.99 | Medium $13.99 | Large $16.99 | XL $19.99
Crusts:  Thin, Thick, Hand Tossed (free) | Stuffed +$2.50 | Cauliflower +$3.00 | Gluten-Free +$2.50
Sauces:  Tomato, Marinara, BBQ, Ranch, Buffalo, Garlic Butter (free) | Alfredo/Pesto +$0.50
Cheese:  Mozzarella (free) | Cheddar/Parmesan +$0.50 | Feta/Gouda +$1.00 | Ricotta +$0.75 | No Cheese
Toppings (+$0.75-$3.00): Pepperoni, Mushrooms, Spinach, Jalapeños, Black Olives, Bell Peppers,
  Red Onion, Grilled Chicken, Ground Beef, Italian Sausage, Bacon, Ham, Pineapple,
  Fresh Tomatoes, Fresh Basil, Roasted Garlic, Arugula, Broccoli, Sweet Corn,
  Artichoke Hearts, Anchovies, Avocado, Prosciutto, Truffle Oil, Zucchini, Sun-Dried Tomatoes
Extras:  Extra Cheese +$1.50 | Extra Sauce +$0.50 | Well Done / Light Sauce (free)

RULES:
- Be warm, short, and conversational — like a real waiter talking out loud.
- Make recommendations when asked (e.g. vegetarian → suggest pesto, spinach, feta, mushrooms).
- Ask only ONE question at a time — never bombard the customer.
- You MUST collect: customer name, size, crust, sauce, cheese, toppings (or none), quantity.
- When you have all details AND the customer has confirmed, output their order like this on its own line:
  ##ORDER##{"name":"...","size":"...","crust":"...","sauce":"...","cheese":"...","toppings":["..."],"extras":["..."],"quantity":1}##END##
- Keep all topping/size/etc values lowercase and matching the menu exactly.
- Do NOT output ##ORDER## until the customer explicitly confirms (yes/correct/place it/etc).
- After outputting ##ORDER##, say a warm goodbye line."""

# ── Fallback rule-based extractor (if no HF_TOKEN) ────────────────────────────
SIZES_MAP    = {"personal":("Personal 6\"",7.99),"small":("Small 8\"",9.99),"medium":("Medium 12\"",13.99),"large":("Large 14\"",16.99),"extra large":("XL 16\"",19.99),"xl":("XL 16\"",19.99)}
CRUSTS_MAP   = {"thin":("Thin Crust",0),"thick":("Thick Crust",0),"hand tossed":("Hand Tossed",0),"stuffed":("Stuffed Crust",2.50),"cauliflower":("Cauliflower Crust",3.00),"gluten free":("Gluten-Free",2.50),"gluten-free":("Gluten-Free",2.50)}
SAUCES_MAP   = {"tomato":("Classic Tomato",0),"marinara":("Marinara",0),"bbq":("BBQ",0),"ranch":("Ranch",0),"buffalo":("Buffalo",0),"garlic butter":("Garlic Butter",0),"alfredo":("Alfredo",0.50),"pesto":("Basil Pesto",0.50)}
CHEESES_MAP  = {"mozzarella":("Mozzarella",0),"cheddar":("Cheddar",0.50),"parmesan":("Parmesan",0.50),"feta":("Feta",1.00),"gouda":("Gouda",1.00),"ricotta":("Ricotta",0.75),"no cheese":("No Cheese",0),"vegan":("Vegan Cheese",1.50)}
TOPPINGS_MAP = {
    "pepperoni":("Pepperoni","🍕",1.50),"mushrooms":("Mushrooms","🍄",1.00),"mushroom":("Mushrooms","🍄",1.00),
    "spinach":("Spinach","🥬",1.00),"jalapeños":("Jalapeños","🌶️",0.75),"jalapenos":("Jalapeños","🌶️",0.75),
    "olives":("Black Olives","🫒",1.00),"bell peppers":("Bell Peppers","🫑",1.00),
    "red onion":("Red Onion","🧅",0.75),"onion":("Red Onion","🧅",0.75),
    "chicken":("Grilled Chicken","🍗",2.00),"grilled chicken":("Grilled Chicken","🍗",2.00),
    "beef":("Ground Beef","🥩",2.00),"sausage":("Italian Sausage","🌭",1.75),
    "bacon":("Bacon","🥓",1.75),"ham":("Ham","🍖",1.50),"pineapple":("Pineapple","🍍",1.00),
    "tomatoes":("Fresh Tomatoes","🍅",1.00),"basil":("Fresh Basil","🌿",0.75),
    "garlic":("Roasted Garlic","🧄",0.75),"arugula":("Arugula","🥗",1.00),
    "broccoli":("Broccoli","🥦",1.00),"corn":("Sweet Corn","🌽",0.75),
    "artichoke":("Artichoke Hearts","🌱",1.50),"anchovies":("Anchovies","🐟",1.50),
    "avocado":("Avocado","🥑",1.50),"prosciutto":("Prosciutto","🍖",2.50),
    "truffle":("Truffle Oil","✨",3.00),"zucchini":("Zucchini","🥒",1.00),
    "feta":("Feta Crumbles","🧀",1.25),"sun dried tomato":("Sun-Dried Tomatoes","☀️",1.25),
}
TAX = 0.13


def build_receipt(order_data: dict) -> dict:
    """Turn raw LLM JSON into a full priced receipt."""
    def pick(val, catalogue, default_key):
        if not val: return catalogue[default_key]
        v = val.lower().strip()
        for k in sorted(catalogue, key=len, reverse=True):
            if k in v or v in k:
                return catalogue[k]
        return catalogue[default_key]

    size   = pick(order_data.get("size",""),   SIZES_MAP,  "medium")
    crust  = pick(order_data.get("crust",""),  CRUSTS_MAP, "hand tossed")
    sauce  = pick(order_data.get("sauce",""),  SAUCES_MAP, "tomato")
    cheese = pick(order_data.get("cheese",""), CHEESES_MAP,"mozzarella")
    qty    = max(1, int(order_data.get("quantity", 1)))

    raw_tops = order_data.get("toppings", [])
    if isinstance(raw_tops, str):
        raw_tops = [x.strip() for x in raw_tops.split(",")]
    seen, tops = set(), []
    for item in raw_tops:
        item = item.lower().strip()
        for k in sorted(TOPPINGS_MAP, key=len, reverse=True):
            if k in item:
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
        if "extra cheese" in il: extras_out.append({"label":"Extra Cheese","price":1.50})
        elif "extra sauce" in il: extras_out.append({"label":"Extra Sauce","price":0.50})
        elif "well done" in il:   extras_out.append({"label":"Well Done","price":0.00})

    unit = size[1] + crust[1] + sauce[1] + cheese[1] + sum(t["price"] for t in tops) + sum(e["price"] for e in extras_out)
    sub  = unit * qty
    tax  = sub * TAX
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
            "unit": round(unit,2), "quantity": qty,
            "subtotal": round(sub,2), "tax": round(tax,2),
            "total": round(sub+tax, 2),
            "breakdown": {"base":round(size[1],2),"crust":round(crust[1],2),
                          "sauce":round(sauce[1],2),"cheese":round(cheese[1],2),
                          "toppings":round(sum(t["price"] for t in tops),2),
                          "extras":round(sum(e["price"] for e in extras_out),2)},
        },
        "order_id":  str(uuid.uuid4())[:8].upper(),
        "timestamp": datetime.now().strftime("%B %d, %Y  •  %I:%M %p"),
    }


def chat_with_llm(history: list) -> str:
    """Send full conversation to LLM and get Pino's reply."""
    if not CLIENT:
        return "I'm sorry, I need an HF_TOKEN to work properly! Please add it in Space Secrets. 🍕"
    try:
        messages = [{"role": "system", "content": SYSTEM}] + history
        resp = CLIENT.chat_completion(messages=messages, max_tokens=300, temperature=0.7)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Oops, I had a little trouble there! Could you repeat that? 😅 ({str(e)[:60]})"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data    = request.get_json(force=True)
    history = data.get("history", [])   # full [{role, content}] list from frontend
    receipt = None

    # Get LLM response
    reply = chat_with_llm(history)

    # Check if order is ready
    m = re.search(r"##ORDER##(.*?)##END##", reply, re.DOTALL)
    if m:
        try:
            order_json = json.loads(m.group(1).strip())
            receipt    = build_receipt(order_json)
        except Exception:
            pass
        # Clean the tag from the spoken reply
        reply = re.sub(r"##ORDER##.*?##END##", "", reply, flags=re.DOTALL).strip()

    return jsonify({"reply": reply, "receipt": receipt})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)