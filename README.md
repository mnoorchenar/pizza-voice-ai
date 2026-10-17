---
title: PizzaVoice
emoji: 🍕
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# 🍕 PizzaVoice

A conversational AI pizza ordering assistant powered by open-source LLMs via the HuggingFace Inference API. Talk or type your order — Pino, your Italian AI waiter, guides you through every step and builds a live order card in real time.

## Features

- **Voice & text ordering** — uses the Web Speech API for speech-to-text (STT) and Microsoft Edge TTS (Neural) for natural text-to-speech
- **Live order card** — updates in real time as the conversation progresses, showing each item and a running price total
- **Smart AI character** — Pino is a warm, witty Italian waiter who asks one question at a time, handles dietary requests, makes recommendations, and suggests drink pairings
- **Resilient LLM backend** — tries 6 open-source models in sequence (Qwen 72B → Llama 70B → Mistral 24B → Phi-4 → Qwen Coder 32B → Gemma 27B) with automatic fallback
- **Itemised receipt** — final order confirmation with full pricing breakdown, tax, and a randomly selected Italian quote

## Architecture

```
Browser (HTML + Vanilla JS)
  ├── Web Speech API  ──► STT (voice input)
  ├── /tts            ◄── Edge TTS → HF TTS fallback (voice output)
  ├── /chat           ◄── Flask backend → HF Inference API (LLM)
  └── /prices         ◄── Flask backend → catalogue.py (price maps)

Flask Backend (app.py)
  ├── Stage-aware menu injection  — injects only relevant menu sections
  │   per conversation turn instead of the full menu every time
  ├── InferenceClient singleton   — created once, reused across requests
  ├── Token-budget history trim   — keeps history within ~900 tokens
  ├── TTS audio cache (64 slots)  — MD5-keyed, avoids re-generating
  │   audio for repeated phrases
  ├── Pydantic validation         — validates every ##ORDER## / ##UPDATE##
  │   JSON block before use
  └── Rate limiter                — 30 requests/min per IP (no deps)

catalogue.py  ← single source of truth for all menu data & prices
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Main app UI |
| `POST` | `/chat` | Send conversation history, get Pino's reply + order state |
| `POST` | `/tts` | Convert text to MP3 audio (Edge TTS → HF fallback) |
| `GET` | `/prices` | Serve all price maps as JSON (consumed by the frontend) |
| `GET` | `/stats` | Live token-usage counters and model-hit stats |

## Menu

| Category | Options |
|---|---|
| Sizes | Personal 6" · Small 8" · Medium 12" · Large 14" · XL 16" |
| Crusts | Thin · Thick · Hand Tossed (free) · Stuffed · Cauliflower · Gluten-Free |
| Sauces | Marinara · Tomato · BBQ · Ranch · Buffalo · Garlic Butter (free) · Alfredo · Pesto |
| Cheeses | Mozzarella (free) · Cheddar · Parmesan · Feta · Gouda · Ricotta · Vegan · Dairy-Free · No Cheese |
| Toppings | 26 options — Meat, Veg, and Premium (Truffle Oil) |
| Drinks | Soft drinks · Juices · Water · Italian coffees · Beer · Wine · Prosecco |
| Extras | Extra Cheese · Extra Sauce · Well Done · Light Sauce |

Tax rate: 13%

## Stack

- **Backend:** Python 3.11 · Flask 3.1 · HuggingFace Hub · Pydantic v2
- **TTS:** Edge TTS (Microsoft Neural) · HF Inference REST fallback · Web Speech API (browser)
- **STT:** Web Speech API (browser-native, no server cost)
- **Deployment:** Docker on HuggingFace Spaces (port 7860)

## Setup

Each visitor enters their own Hugging Face token on the start screen (stored only in their browser), so the Space never uses the owner's quota. The token needs access to the serverless Inference API.

For local testing with your own server-side token:

```bash
pip install -r requirements.txt
USE_SERVER_TOKEN=1 HF_TOKEN=hf_... python app.py
```

To update menu items or prices, edit **`catalogue.py`** only — the backend and frontend both derive their data from it automatically.
