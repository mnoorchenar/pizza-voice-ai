from flask import Flask, render_template, request, jsonify
import uuid
from datetime import datetime
from parser import parse_order, calc_price

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/parse", methods=["POST"])
def parse():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    order, engine = parse_order(text)
    pricing       = calc_price(order)

    return jsonify({
        "order":     order,
        "pricing":   pricing,
        "engine":    engine,
        "order_id":  str(uuid.uuid4())[:8].upper(),
        "timestamp": datetime.now().strftime("%B %d, %Y  •  %I:%M %p"),
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)