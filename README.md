---
title: PizzaVoice
emoji: 🍕
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

<h1>🍕 PizzaVoice</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=dc2626&center=true&vCenter=true&width=700&lines=Talk+to+Pino,+Your+AI+Italian+Waiter;Voice+++Text+Ordering+with+Open+LLMs;Live+Order+Card+·+Itemised+Receipt" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-4f46e5?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/docs/inference-providers)
[![Edge TTS](https://img.shields.io/badge/Edge_TTS-Voice-0078D4?style=for-the-badge)](https://github.com/rany2/edge-tts)
[![Docker](https://img.shields.io/badge/Docker-Ready-3b82f6?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

[![Live Demo](https://img.shields.io/badge/%E2%96%B6_Live_Demo-Hugging_Face_Space-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/mnoorchenar/pizza-voice-ai)
[![Source Code](https://img.shields.io/badge/Source_Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar/pizza-voice-ai)

<br/>

**🍕 PizzaVoice** — A conversational AI pizza ordering assistant powered by open-source LLMs via the HuggingFace Inference API. Talk or type your order — Pino, your Italian AI waiter, guides you through every step and builds a live order card in real time.

<br/>

---

</div>

## Table of Contents

- [Features](#-features)
- [Architecture](#️-architecture)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [Endpoints](#-endpoints)
- [Menu](#-menu)
- [Stack](#-stack)
- [Setup](#-setup)
- [Project Structure](#-project-structure)
- [Author](#-author)
- [Contributing](#-contributing)
- [Disclaimer](#️-disclaimer)
- [License](#-license)

---

## ✨ Features

- **Voice & text ordering** — uses the Web Speech API for speech-to-text (STT) and Microsoft Edge TTS (Neural) for natural text-to-speech
- **Live order card** — updates in real time as the conversation progresses, showing each item and a running price total
- **Smart AI character** — Pino is a warm, witty Italian waiter who asks one question at a time, handles dietary requests, makes recommendations, and suggests drink pairings
- **Resilient LLM backend** — tries 6 open-source models in sequence (Qwen 72B → Llama 70B → Mistral 24B → Phi-4 → Qwen Coder 32B → Gemma 27B) with automatic fallback
- **Itemised receipt** — final order confirmation with full pricing breakdown, tax, and a randomly selected Italian quote

---

## 🏗️ Architecture

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

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git
- Docker (optional)

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/mnoorchenar/pizza-voice-ai.git
cd pizza-voice-ai

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```

Open your browser at `http://localhost:7860` 🎉

---

## 🐳 Docker Deployment

```bash
docker build -t pizza-voice-ai .
docker run -p 7860:7860 pizza-voice-ai
```

---

## 📡 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Main app UI |
| `POST` | `/chat` | Send conversation history, get Pino's reply + order state |
| `POST` | `/tts` | Convert text to MP3 audio (Edge TTS → HF fallback) |
| `GET` | `/prices` | Serve all price maps as JSON (consumed by the frontend) |
| `GET` | `/stats` | Live token-usage counters and model-hit stats |

---

## 🍕 Menu

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

---

## 🧰 Stack

- **Backend:** Python 3.11 · Flask 3.1 · HuggingFace Hub · Pydantic v2
- **TTS:** Edge TTS (Microsoft Neural) · HF Inference REST fallback · Web Speech API (browser)
- **STT:** Web Speech API (browser-native, no server cost)
- **Deployment:** Docker on HuggingFace Spaces (port 7860)

---

## 🚀 Setup

Each visitor enters their own Hugging Face token on the start screen (stored only in their browser), so the Space never uses the owner's quota. The token needs access to the serverless Inference API.

For local testing with your own server-side token:

```bash
pip install -r requirements.txt
USE_SERVER_TOKEN=1 HF_TOKEN=hf_... python app.py
```

To update menu items or prices, edit **`catalogue.py`** only — the backend and frontend both derive their data from it automatically.

---

## 📁 Project Structure

```
pizza-voice-ai/
│
├── 📂 templates/                 # HTML templates
│   └── index.html
├── 📄 Dockerfile                 # Container definition (port 7860)
├── 📄 app.py                     # Application entry point
├── 📄 catalogue.py
└── 📄 requirements.txt           # Python dependencies
```

---

## 👨‍💻 Author

<div align="center">

<table>
<tr>
<td align="center" width="100%">

<img src="https://avatars.githubusercontent.com/mnoorchenar" width="120" style="border-radius:50%; border: 3px solid #4f46e5;" alt="Mohammad Noorchenarboo"/>

<h3>Mohammad Noorchenarboo</h3>

<code>Data Scientist</code> &nbsp;|&nbsp; <code>AI Researcher</code> &nbsp;|&nbsp; <code>Biostatistician</code>

📍 &nbsp;Ontario, Canada &nbsp;&nbsp; 📧 &nbsp;[mohammadnoorchenarboo@gmail.com](mailto:mohammadnoorchenarboo@gmail.com)

──────────────────────────────────────

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mnoorchenar)&nbsp;
[![Personal Site](https://img.shields.io/badge/Website-mnoorchenar.github.io-4f46e5?style=for-the-badge&logo=githubpages&logoColor=white)](https://mnoorchenar.github.io/)&nbsp;
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar)&nbsp;
[![Google Scholar](https://img.shields.io/badge/Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white)](https://scholar.google.ca/citations?user=nn_Toq0AAAAJ&hl=en)&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar)

</td>
</tr>
</table>

</div>

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## ⚠️ Disclaimer

> [!WARNING]
> This project is developed strictly for educational and research purposes and does not constitute professional advice of any kind. All datasets used are either synthetically generated or publicly available — no real user data is stored. This software is provided "as is" without warranty of any kind; use at your own risk.

---

## 📜 License

Distributed under the **MIT License**.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:dc2626,100:06b6d4&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20by%20Mohammad%20Noorchenarboo&fontColor=ffffff&fontSize=18&fontAlignY=80" width="100%"/>

[![GitHub Stars](https://img.shields.io/github/stars/mnoorchenar/pizza-voice-ai?style=social)](https://github.com/mnoorchenar/pizza-voice-ai/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/mnoorchenar/pizza-voice-ai?style=social)](https://github.com/mnoorchenar/pizza-voice-ai/fork)

<sub>The name "PizzaVoice" is used purely for academic and research purposes. Any similarity to existing product names or trademarks is entirely coincidental. This project has no affiliation with any commercial entity.</sub>

</div>
