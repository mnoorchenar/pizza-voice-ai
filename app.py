from flask import Flask, render_template, request, jsonify, session
import uuid, re, os
from datetime import datetime
from catalogue import SIZES, CRUSTS, SAUCES, CHEESES, TOPPINGS, EXTRAS, TAX_RATE
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# ── Conversation state store (in-memory, per session) ────────────────────────
_sessions = {}

# ── Conversational bot replies ────────────────────────────────────────────────
BOT = {
    "welcome":    "👋 Hey there! Welcome to PizzaVoice — I'm Pino, your pizza waiter! What's your name?",
    "size":       "Great to meet you, {name}! 🍕 Let's build your perfect pizza. What size would you like?\n👉 Personal, Small, Medium, Large, or XL?",
    "crust":      "Excellent choice! 🥨 What crust are you feeling today?\n👉 Thin, Thick, Hand Tossed, Stuffed (+$2.50), Cauliflower (+$3.00), or Gluten-Free (+$2.50)?",
    "sauce":      "Love it! 🥫 Now, which sauce would you like?\n👉 Tomato, Marinara, BBQ, Alfredo, Pesto, Ranch, Buffalo, or Garlic Butter?",
    "cheese":     "Delicious! 🧀 What cheese are you going with?\n👉 Mozzarella, Cheddar, Parmesan, Feta, Gouda, Ricotta, or No Cheese?",
    "toppings":   "Amazing! 🍄 What toppings would you like? You can name as many as you want!\n👉 e.g. pepperoni, mushrooms, jalapeños, chicken, bacon…",
    "no_toppings":"No worries — keeping it classic! 😄",
    "extras":     "Nice picks! ⭐ Any extras?\n👉 Extra Cheese (+$1.50), Extra Sauce (+$0.50), Well Done, Light Sauce — or just say 'none'!",
    "quantity":   "Almost there! 🔢 How many pizzas would you like?",
    "confirm":    "Perfect! Let me read that back to you, {name}:\n\n🍕 {qty}× {size} pizza\n🥨 {crust} crust\n🥫 {sauce} sauce\n🧀 {cheese}\n🍄 Toppings: {toppings}\n⭐ Extras: {extras}\n\n💰 Estimated total: {total}\n\nShall I place this order? Say **yes** to confirm or **no** to start over!",
    "placed":     "🎉 Order placed, {name}! Your pizza is being made with love. Here's your receipt below! 🧾",
    "restart":    "No problem! Let's start fresh. 😊 What's your name?",
    "not_understood": "Sorry, I didn't catch that! Could you say that again? 😊",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
_QTY_WORDS = {"one":1,"two":2,"three":3,"four":4,"five":5,
              "six":6,"seven":7,"eight":8,"nine":9,"ten":10}

def pick(text, catalogue):
    t = text.lower()
    for k in sorted(catalogue, key=len, reverse=True):
        if k in t:
            return k
    return None

def extract_name(text):
    t = text.strip()
    # Remove common phrases
    for p in ["my name is","i'm","i am","call me","it's","its","name's","name is"]:
        t = re.sub(rf"\b{p}\b", "", t, flags=re.I).strip()
    # Take first 1-2 words, capitalised
    words = [w.capitalize() for w in t.split() if len(w) > 1]
    return " ".join(words[:2]) if words else None

def extract_qty(text):
    t = text.lower()
    for w, n in _QTY_WORDS.items():
        if re.search(rf"\b{w}\b", t):
            return n
    m = re.search(r"\b([1-9])\b", t)
    return int(m.group(1)) if m else 1

def extract_toppings(text):
    t = text.lower()
    seen, result = set(), []
    for k in sorted(TOPPINGS, key=len, reverse=True):
        if k in t:
            lbl, emoji, price = TOPPINGS[k]
            if lbl not in seen:
                seen.add(lbl)
                result.append({"label": lbl, "emoji": emoji, "price": price})
    return result

def extract_extras(text):
    t = text.lower()
    seen, result = set(), []
    for k in sorted(EXTRAS, key=len, reverse=True):
        if k in t:
            lbl, price = EXTRAS[k]
            if lbl not in seen:
                seen.add(lbl)
                result.append({"label": lbl, "price": price})
    return result

def is_yes(text):
    return bool(re.search(r"\b(yes|yeah|yep|sure|correct|confirm|place|ok|okay|yup|absolutely)\b", text.lower()))

def is_no(text):
    return bool(re.search(r"\b(no|nope|restart|start over|cancel|nah)\b", text.lower()))

def is_none(text):
    return bool(re.search(r"\b(none|nothing|no|skip|plain|that'?s? ?(it|all))\b", text.lower()))

def calc_price(order):
    q  = order["quantity"]
    b  = SIZES[order["size"]][1]
    cr = CRUSTS[order["crust"]][1]
    sa = SAUCES[order["sauce"]][1]
    ch = CHEESES[order["cheese"]][1]
    tp = sum(x["price"] for x in order["toppings"])
    ex = sum(x["price"] for x in order["extras"])
    unit = b + cr + sa + ch + tp + ex
    sub  = unit * q
    tax  = sub * TAX_RATE
    return {
        "unit": round(unit, 2), "quantity": q,
        "subtotal": round(sub, 2), "tax": round(tax, 2),
        "total": round(sub + tax, 2),
        "breakdown": {"base": round(b,2), "crust": round(cr,2),
                      "sauce": round(sa,2), "cheese": round(ch,2),
                      "toppings": round(tp,2), "extras": round(ex,2)},
    }

def new_order():
    return {"name": None, "size": None, "crust": None, "sauce": None,
            "cheese": None, "toppings": [], "extras": [], "quantity": 1}

# ── Main chat endpoint ────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data    = request.get_json(force=True)
    sid     = data.get("session_id")
    user_msg= data.get("message", "").strip()

    # Get or create session
    if sid not in _sessions:
        _sessions[sid] = {"state": "welcome", "order": new_order()}
    s = _sessions[sid]
    state = s["state"]
    order = s["order"]

    reply    = ""
    receipt  = None
    advance  = True

    # ── State machine ─────────────────────────────────────────────────────────
    if state == "welcome":
        reply = BOT["welcome"]
        s["state"] = "name"
        advance = False

    elif state == "name":
        name = extract_name(user_msg)
        if not name:
            reply = "I didn't catch your name — could you tell me again? 😊"
            advance = False
        else:
            order["name"] = name
            reply = BOT["size"].format(name=name)
            s["state"] = "size"
            advance = False

    elif state == "size":
        key = pick(user_msg, SIZES)
        if not key:
            reply = "Hmm, I didn't get that size! Try: Personal, Small, Medium, Large, or XL 😊"
            advance = False
        else:
            order["size"] = key
            reply = BOT["crust"]
            s["state"] = "crust"
            advance = False

    elif state == "crust":
        key = pick(user_msg, CRUSTS)
        if not key:
            reply = "Sorry, didn't catch the crust! Try: Thin, Thick, Hand Tossed, Stuffed, Cauliflower, or Gluten-Free 😊"
            advance = False
        else:
            order["crust"] = key
            reply = BOT["sauce"]
            s["state"] = "sauce"
            advance = False

    elif state == "sauce":
        key = pick(user_msg, SAUCES)
        if not key:
            reply = "Didn't get that sauce! Try: Tomato, Marinara, BBQ, Alfredo, Pesto, Ranch, Buffalo, or Garlic Butter 😊"
            advance = False
        else:
            order["sauce"] = key
            reply = BOT["cheese"]
            s["state"] = "cheese"
            advance = False

    elif state == "cheese":
        key = pick(user_msg, CHEESES)
        if not key:
            reply = "Didn't catch that! Try: Mozzarella, Cheddar, Parmesan, Feta, Gouda, Ricotta, or No Cheese 😊"
            advance = False
        else:
            order["cheese"] = key
            reply = BOT["toppings"]
            s["state"] = "toppings"
            advance = False

    elif state == "toppings":
        if is_none(user_msg):
            order["toppings"] = []
            reply = BOT["no_toppings"] + "\n\n" + BOT["extras"]
        else:
            tops = extract_toppings(user_msg)
            if not tops:
                reply = "I couldn't find any toppings I know! Try: pepperoni, mushrooms, chicken, bacon, jalapeños… or say 'none'!"
                advance = False
            else:
                order["toppings"] = tops
                names = ", ".join(t["label"] for t in tops)
                reply = f"Great choices — {names}! 😋\n\n" + BOT["extras"]
        if advance:
            s["state"] = "extras"
            advance = False

    elif state == "extras":
        if is_none(user_msg):
            order["extras"] = []
        else:
            order["extras"] = extract_extras(user_msg)
        reply = BOT["quantity"]
        s["state"] = "quantity"
        advance = False

    elif state == "quantity":
        qty = extract_qty(user_msg)
        order["quantity"] = qty
        pricing = calc_price(order)
        top_str   = ", ".join(t["label"] for t in order["toppings"]) or "None"
        extra_str = ", ".join(e["label"] for e in order["extras"]) or "None"
        reply = BOT["confirm"].format(
            name    = order["name"],
            qty     = qty,
            size    = SIZES[order["size"]][0],
            crust   = CRUSTS[order["crust"]][0],
            sauce   = SAUCES[order["sauce"]][0],
            cheese  = CHEESES[order["cheese"]][0],
            toppings= top_str,
            extras  = extra_str,
            total   = f"${pricing['total']:.2f}",
        )
        s["state"] = "confirm"
        advance = False

    elif state == "confirm":
        if is_yes(user_msg):
            pricing = calc_price(order)
            receipt = {
                "order":     {
                    "name":     order["name"],
                    "size":     {"label": SIZES[order["size"]][0],    "price": SIZES[order["size"]][1]},
                    "crust":    {"label": CRUSTS[order["crust"]][0],  "price": CRUSTS[order["crust"]][1]},
                    "sauce":    {"label": SAUCES[order["sauce"]][0],  "price": SAUCES[order["sauce"]][1]},
                    "cheese":   {"label": CHEESES[order["cheese"]][0],"price": CHEESES[order["cheese"]][1]},
                    "toppings": order["toppings"],
                    "extras":   order["extras"],
                    "quantity": order["quantity"],
                },
                "pricing":   pricing,
                "order_id":  str(uuid.uuid4())[:8].upper(),
                "timestamp": datetime.now().strftime("%B %d, %Y  •  %I:%M %p"),
            }
            reply = BOT["placed"].format(name=order["name"])
            s["state"] = "done"
            advance = False
        elif is_no(user_msg):
            _sessions[sid] = {"state": "name", "order": new_order()}
            reply = BOT["restart"]
            advance = False
        else:
            reply = "Just say **yes** to place the order or **no** to start over! 😊"
            advance = False

    elif state == "done":
        # Start a new order
        _sessions[sid] = {"state": "name", "order": new_order()}
        reply = "Starting a new order! 🍕 What's your name?"
        advance = False

    return jsonify({"reply": reply, "receipt": receipt, "state": s["state"]})


@app.route("/start", methods=["POST"])
def start():
    sid = str(uuid.uuid4())
    _sessions[sid] = {"state": "welcome", "order": new_order()}
    return jsonify({"session_id": sid, "reply": BOT["welcome"], "state": "name"})


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)