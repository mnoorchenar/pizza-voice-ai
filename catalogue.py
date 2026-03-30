"""
PizzaVoice — Menu catalogue
Single source of truth for all prices and menu data.
Imported by app.py; the /prices endpoint serialises this for the frontend
so JS never has hardcoded prices that can drift out of sync.
"""

SIZES_MAP = {
    "personal":    ("Personal 6\"",  7.99),
    "small":       ("Small 8\"",     9.99),
    "medium":      ("Medium 12\"",  13.99),
    "large":       ("Large 14\"",   16.99),
    "extra large": ("XL 16\"",      19.99),
    "xl":          ("XL 16\"",      19.99),
}
CRUSTS_MAP = {
    "thin":        ("Thin Crust",        0.00),
    "crispy":      ("Crispy Thin",       0.00),
    "thick":       ("Thick Crust",       0.00),
    "hand tossed": ("Hand Tossed",       0.00),
    "stuffed":     ("Stuffed Crust",     2.50),
    "cauliflower": ("Cauliflower Crust", 3.00),
    "gluten free": ("Gluten-Free Crust", 2.50),
    "gluten-free": ("Gluten-Free Crust", 2.50),
}
SAUCES_MAP = {
    "marinara":     ("Marinara",       0.00),
    "tomato":       ("Classic Tomato", 0.00),
    "bbq":          ("BBQ",            0.00),
    "barbecue":     ("BBQ",            0.00),
    "alfredo":      ("Alfredo",        0.50),
    "white":        ("White Alfredo",  0.50),
    "pesto":        ("Basil Pesto",    0.50),
    "ranch":        ("Ranch",          0.00),
    "buffalo":      ("Buffalo",        0.00),
    "garlic butter":("Garlic Butter",  0.00),
}
CHEESES_MAP = {
    "no cheese":  ("No Cheese",    0.00),
    "dairy-free": ("Dairy-Free",   1.50),
    "vegan":      ("Vegan Cheese", 1.50),
    "mozzarella": ("Mozzarella",   0.00),
    "cheddar":    ("Cheddar",      0.50),
    "parmesan":   ("Parmesan",     0.50),
    "feta":       ("Feta",         1.00),
    "gouda":      ("Gouda",        1.00),
    "ricotta":    ("Ricotta",      0.75),
}
# Toppings: key → (label, emoji, price)
TOPPINGS_MAP = {
    "pepperoni":          ("Pepperoni",         "🍕", 1.50),
    "mushrooms":          ("Mushrooms",          "🍄", 1.00),
    "mushroom":           ("Mushrooms",          "🍄", 1.00),
    "spinach":            ("Spinach",            "🥬", 1.00),
    "jalapeños":          ("Jalapeños",          "🌶️", 0.75),
    "jalapenos":          ("Jalapeños",          "🌶️", 0.75),
    "jalapeno":           ("Jalapeños",          "🌶️", 0.75),
    "olives":             ("Black Olives",       "🫒", 1.00),
    "black olives":       ("Black Olives",       "🫒", 1.00),
    "bell peppers":       ("Bell Peppers",       "🫑", 1.00),
    "bell pepper":        ("Bell Peppers",       "🫑", 1.00),
    "red onion":          ("Red Onion",          "🧅", 0.75),
    "onion":              ("Red Onion",          "🧅", 0.75),
    "grilled chicken":    ("Grilled Chicken",    "🍗", 2.00),
    "chicken":            ("Grilled Chicken",    "🍗", 2.00),
    "ground beef":        ("Ground Beef",        "🥩", 2.00),
    "beef":               ("Ground Beef",        "🥩", 2.00),
    "sausage":            ("Italian Sausage",    "🌭", 1.75),
    "bacon":              ("Bacon",              "🥓", 1.75),
    "ham":                ("Ham",                "🍖", 1.50),
    "pineapple":          ("Pineapple",          "🍍", 1.00),
    "fresh tomatoes":     ("Fresh Tomatoes",     "🍅", 1.00),
    "tomatoes":           ("Fresh Tomatoes",     "🍅", 1.00),
    "basil":              ("Fresh Basil",        "🌿", 0.75),
    "garlic":             ("Roasted Garlic",     "🧄", 0.75),
    "arugula":            ("Arugula",            "🥗", 1.00),
    "broccoli":           ("Broccoli",           "🥦", 1.00),
    "corn":               ("Sweet Corn",         "🌽", 0.75),
    "artichoke":          ("Artichoke Hearts",   "🌱", 1.50),
    "anchovies":          ("Anchovies",          "🐟", 1.50),
    "avocado":            ("Avocado",            "🥑", 1.50),
    "prosciutto":         ("Prosciutto",         "🍖", 2.50),
    "truffle":            ("Truffle Oil",        "✨", 3.00),
    "zucchini":           ("Zucchini",           "🥒", 1.00),
    "sun-dried tomatoes": ("Sun-Dried Tomatoes", "☀️", 1.25),
    "sun dried tomato":   ("Sun-Dried Tomatoes", "☀️", 1.25),
}
# Drinks: key → (label, emoji, price)
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
EXTRAS_MAP = {
    "extra cheese": ("Extra Cheese", 1.50),
    "extra sauce":  ("Extra Sauce",  0.50),
    "well done":    ("Well Done",    0.00),
    "light sauce":  ("Light Sauce",  0.00),
}

TAX_RATE: float = 0.13

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
