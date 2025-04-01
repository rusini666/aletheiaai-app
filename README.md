# 📘 AletheiaAI – Detecting AI-Generated Text with User-Centric Explainability

AletheiaAI is a lightweight AI vs. Human text detection tool that classifies text and explains its decisions to the user using a combination of visual and sentence-level explanations. It uses a LoRA-finetuned OPT-1.3b model for classification, LIME for visual explanation, and Mistral-7B-Instruct for smart sentence-level explanation.

---
## 📁 Dataset

The system was trained on a monolingual English dataset derived from the **M4 corpus**, including:

- **AI-generated text**: Sourced from models such as ChatGPT, GPT-3.5, OPT, and GPT-2.
- **Human-written text**: Sourced from Wikipedia, Reddit, WikiHow, and arXiv.

### Benchmark Datasets:
- [AI Text Detection Pile](https://huggingface.co/datasets/artem9k/ai-text-detection-pile)
- [HC3 Dataset](https://huggingface.co/datasets/Hello-SimpleAI/HC3)

All data is preprocessed using tokenization, cleaning, stopword filtering, and lowercasing.

---

## 🧠 Model Details

- **Base model**: OPT-1.3b
- **Fine-tuning technique**: LoRA (Low-Rank Adaptation)
- **Loss Function**: CrossEntropyLoss
- **Split**: 70% train / 20% validation / 10% test
- **Optimization**: AdamW

---

## 💡 Explainability

- 🟩 **LIME** – Highlights top influential words per classification
- 🧠 **Mistral-7B-Instruct** – Produces natural language sentence-level explanations

Both explanation types are integrated in the user interface and accessible through API.

---

## 🛠️ Project Setup

## Backend (Python)
1. `cd backend/`
2. Create a virtual environment (`python -m venv .venv`)
3. Activate it (`source .venv/bin/activate` on Mac/Linux, or `.\.venv\Scripts\activate` on Windows)
4. `pip install -r requirements.txt`
5. `python main.py` (or however you start your server)

## Frontend (Vite, React, etc.)
1. `cd frontend/`
2. `npm install` (or `yarn install`)
3. `npm run dev`
4. Open http://localhost:5173 (or whichever port).
