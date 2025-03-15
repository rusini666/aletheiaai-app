#!/usr/bin/env python3

import os
import re
import sys
import time
import json
import string
import threading


import torch
from torch import nn
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from sklearn.metrics import (f1_score, accuracy_score, classification_report, 
                             confusion_matrix, ConfusionMatrixDisplay, 
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

import shap
from lime.lime_text import LimeTextExplainer

import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from PyPDF2 import PdfReader
try:
    import pdfplumber
except ImportError:
    pass

from peft import get_peft_model, LoraConfig, TaskType
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM,
                          get_linear_schedule_with_warmup)

from huggingface_hub import login

##############################################################################
# Global Config
##############################################################################
app_configs = {
    "base_model": "facebook/opt-1.3b",
    "models_path": "./models",
    "model_name": "202409230028_subtaskA_monolingual_facebook_opt-1.3b",
    "hf_token_file": "hf_token.txt",
    "device": None  # We will set this in target_device()
}

##############################################################################
# Device Selection
##############################################################################
def target_device():
    """Use GPU if available, else CPU or MPS on Apple Silicon."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    app_configs["device"] = device
    print(f"[DEBUG] Using device: {device}")
    return device

##############################################################################
# Optional: Hugging Face Login
##############################################################################
def login_to_huggingface():
    """Log in to Hugging Face using a token from hf_token.txt (if needed)."""
    print("[DEBUG] Attempting to log in to Hugging Face...")
    try:
        with open(app_configs["hf_token_file"], "r") as token_file:
            hf_token = token_file.read().strip()
        login(token=hf_token)
        print("[DEBUG] Successfully logged in to Hugging Face.")
    except FileNotFoundError:
        print(f"[ERROR] {app_configs['hf_token_file']} not found. Skipping login.")
    except Exception as e:
        print(f"[ERROR] Error during Hugging Face login: {e}")

##############################################################################
# Classification Preprocessor
##############################################################################
class PreprocessDataset:
    """
    Basic text preprocessing for classification. 
    Removes URLs, mentions, non-alphanumeric chars, stopwords, etc.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.custom_stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        start_prep = time.time()
        original_text = text

        # Replace fancy dashes/quotes
        text = text.replace("—", "-").replace("“", '"').replace("”", '"')

        # Remove usernames, URLs
        text = re.sub(r'(@\w+)|https?://\S+|www\.\S+', ' ', text)
        # Remove non-alphanumeric chars
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        # Lowercase
        text = text.lower()

        # Remove stopwords
        tokens = text.split()
        tokens = [token for token in tokens if token not in self.custom_stop_words]
        processed_text = ' '.join(tokens)

        # Tokenize
        tokenized = self.tokenizer(
            processed_text,
            padding='max_length',
            max_length=150,
            truncation=True,
            return_tensors="pt"
        )
        end_prep = time.time()
        print(f"[DEBUG] Preprocessing done. Original length: {len(original_text)}, "
              f"Processed length: {len(processed_text)}. Time: {end_prep - start_prep:.2f}s")
        return tokenized["input_ids"][0], tokenized["attention_mask"][0]

##############################################################################
# OPT-based Classifier
##############################################################################
class CustomOPTClassifier(nn.Module):
    """
    Feed-forward classifier on top of an OPT-like model's final logits.
    """
    def __init__(self, pretrained_model):
        super().__init__()
        print("[DEBUG] Initializing CustomOPTClassifier...")
        self.opt = pretrained_model
        self.fc1 = nn.Linear(pretrained_model.config.vocab_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        print("[DEBUG] Running classifier forward pass...")
        start_forward = time.time()
        
        # The base model => (batch_size, seq_len, vocab_size)
        opt_logits = self.opt(input_ids=input_ids, attention_mask=attention_mask).logits

        # OPTION 1: Using last token's logits (original approach)
        # last_token_logits = opt_logits[:, -1, :]

        # OPTION 2: Pooling over all tokens (often better)
        pooled_logits = opt_logits.mean(dim=1)  # Average over all tokens

        # Pass through classifier head
        x = self.fc1(pooled_logits)
        x = self.relu(x)
        x = self.fc2(x)

        end_forward = time.time()
        print(f"[DEBUG] Forward pass completed in {end_forward - start_forward:.2f}s")
        return torch.sigmoid(x)

##############################################################################
# Load Pretrained Model for Classification
##############################################################################
def classification_get_pretrained_model():
    """
    Load tokenizer + base model + LoRA checkpoint for classification.
    """
    start_time = time.time()
    print("[DEBUG] Loading classifier tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(app_configs["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        print("[DEBUG] Added pad_token to tokenizer")

    print("[DEBUG] Loading classifier base model...")
    base_model = AutoModelForCausalLM.from_pretrained(app_configs["base_model"], cache_dir="./cache")
    
    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model_with_lora = get_peft_model(base_model, lora_config)

    print("[DEBUG] Initializing custom classifier head...")
    classifier = CustomOPTClassifier(model_with_lora)

    # Load the checkpoint
    model_path = os.path.join(app_configs["models_path"], app_configs["model_name"] + ".pt")
    if os.path.exists(model_path):
        print(f"[DEBUG] Loading classifier checkpoint from {model_path} ...")
        state_dict = torch.load(model_path, map_location="cpu")
        classifier.load_state_dict(state_dict)
    else:
        print(f"[WARNING] No checkpoint found at {model_path}. Classification might be random.")

    classifier.to(app_configs["device"])
    classifier.eval()
    
    end_time = time.time()
    print(f"[DEBUG] Classifier model & tokenizer loaded in {end_time - start_time:.2f}s")
    return tokenizer, classifier

##############################################################################
# Explanation Model (Mistral) - For generating free-text explanations
##############################################################################
stop_spinner = False

def spinner():
    """A simple console spinner to show progress."""
    chars = ['|', '/', '-', '\\']
    idx = 0
    while not stop_spinner:
        sys.stdout.write('\r[DEBUG] Generating explanation, please wait... ' + chars[idx])
        sys.stdout.flush()
        idx = (idx + 1) % len(chars)
        time.sleep(0.2)
    sys.stdout.write('\r[DEBUG] Explanation generation completed.          \n')
    sys.stdout.flush()

def generate_explanation(text, prediction, explanation_tokenizer, explanation_model):
    """
    Generates 2 distinct explanations (free text) for why the text is AI/Human.
    """
    if explanation_model is None:
        return ("No explanation model available.", "No explanation model available.")

    print("[DEBUG] Entering generate_explanation...")

    prompt = f"""
Below is a piece of text and its classification:

Text: {text}
Classification: {prediction}

Please provide exactly TWO DISTINCT explanations, each prefixed by:
"Explanation #1:" 
and 
"Explanation #2:".

Each explanation should describe unique reasons for why the text is considered {prediction}.
Do not restate the text verbatim.
"""
    global stop_spinner
    stop_spinner = False
    spin_thread = threading.Thread(target=spinner)
    spin_thread.start()

    start_exp = time.time()
    try:
        inputs = explanation_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=False,
            padding=True
        ).to(app_configs["device"])

        # For faster generation, reduce max_new_tokens if desired
        outputs = explanation_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1024,
            temperature=0.7,    
            top_k=50,          
            do_sample=True,    
            pad_token_id=explanation_tokenizer.pad_token_id
        )
        raw_text = explanation_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[ERROR] Explanation generation error: {e}")
        raw_text = "Explanation generation failed."
    finally:
        stop_spinner = True
        spin_thread.join()
        end_exp = time.time()
        print(f"[DEBUG] Explanation generated in {end_exp - start_exp:.2f}s")

    return raw_text

def explanation_get_pretrained_model():
    """
    Load Mistral (or another LLM) for free-text explanations.
    """
    explanation_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"[DEBUG] Loading explanation tokenizer: {explanation_model_name}")
    explanation_tokenizer = AutoTokenizer.from_pretrained(explanation_model_name)
    if explanation_tokenizer.pad_token is None:
        explanation_tokenizer.add_special_tokens({"pad_token": explanation_tokenizer.eos_token})
        print("[DEBUG] Added pad_token to explanation tokenizer")

    print("[DEBUG] Loading explanation model (this can be large)...")
    try:
        explanation_model = AutoModelForCausalLM.from_pretrained(
            explanation_model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir="./cache"
        ).to(app_configs["device"])
        explanation_model.resize_token_embeddings(len(explanation_tokenizer))
        print("[DEBUG] Explanation model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Error loading explanation model: {e}")
        explanation_model = None

    return explanation_tokenizer, explanation_model

def predict_ai_probability(text, preprocessor, classifier_model, device):
    """
    Preprocess the text, run the classifier, and return the probability
    that the text is AI-generated (or human).
    """
    input_ids, attention_mask = preprocessor.preprocess(text)
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier_model(input_ids, attention_mask)
        p_ai = logits.squeeze().item()  # Probability of AI

    return p_ai

##############################################################################
# Classification + Explanation
##############################################################################
def classify_with_explanation(text, tokenizer, classifier_model, preprocessor,
                              explanation_tokenizer, explanation_model):
    print("[DEBUG] classify_with_explanation called...")

    p_ai = predict_ai_probability(text, preprocessor, classifier_model, app_configs["device"])
    p_human = 1.0 - p_ai
    label_str = "AI-generated" if p_ai > 0.5 else "Human-written"

    print(f"[DEBUG] p_ai = {p_ai:.4f}, p_human = {p_human:.4f}")
    print(f"[DEBUG] Final classification: {label_str}")

    # Generate free-text explanation via Mistral or fallback
    if explanation_model is not None:
        combined_explanation = generate_explanation(
            text, 
            label_str, 
            explanation_tokenizer, 
            explanation_model
        )
    else:
        combined_explanation = "No explanation model available."

    return label_str, combined_explanation

##############################################################################
# SHAP / LIME Utility
##############################################################################
force_shap_cpu = False

def predict_proba_for_explanations(texts, model, tokenizer, device):
    """
    Return [p(Human), p(AI)] for each text in texts.
    """
    model.eval()
    all_probs = []
    for txt in texts:
        inputs = tokenizer(
            txt,
            padding="max_length",
            truncation=True,
            max_length=100,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(inputs["input_ids"], inputs["attention_mask"])
            p_ai = torch.sigmoid(logits).item()
            p_human = 1.0 - p_ai
        all_probs.append([p_human, p_ai])
    return np.array(all_probs)

def lime_explanation(model, tokenizer, device, text_sample):
    """
    LIME on the full text => color-coded bar chart + text.
    """
    try:
        original_device = device
        if force_shap_cpu:
            model_cpu = model.to("cpu")
            use_device = torch.device("cpu")
        else:
            model_cpu = model
            use_device = original_device

        def lime_predict(txts):
            return predict_proba_for_explanations(txts, model_cpu, tokenizer, use_device)

        explainer = LimeTextExplainer(class_names=["Human", "AI"])
        exp = explainer.explain_instance(
            text_sample,
            classifier_fn=lime_predict,
            labels=[1],    # label=1 => "AI"
            num_features=15,
            num_samples=200
        )

        if force_shap_cpu:
            model.to(original_device)

        return exp.as_html(labels=[1])

    except Exception as e:
        print("[ERROR] LIME explanation error:", e)
        raise e

##############################################################################
# Combine Classification + LIME => final_report.html
##############################################################################
def get_strong_lime_words(lime_exp, top_n=5):
    """
    Extract the top N strongest words from the LIME explanation.
    Avoids stopwords, punctuation, and ensures meaningful words.
    """
    stop_words = set(stopwords.words("english")) | set(string.punctuation)

    lime_words = [(word, weight) for word, weight in lime_exp.as_list(label=1)]
    lime_words.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"[DEBUG] LIME Extracted Words (Before Filtering): {lime_words}")

    strong_lime_words = [
        (word, weight) for word, weight in lime_words
        if (word.lower() not in stop_words) 
           and (len(word) > 2)
           and (abs(weight) > 0.005)
    ]
    print(f"[DEBUG] LIME Strong Words (After Filtering): {strong_lime_words}")

    if len(strong_lime_words) < top_n:
        print(f"[WARNING] Less than {top_n} strong words found. Showing top available words.")
        strong_lime_words = lime_words[:top_n]

    return strong_lime_words[:top_n]

def classify_and_explain(user_text, model, tokenizer, device):
    """
    Classifies the input text and generates a LIME explanation,
    then saves an HTML report.
    """
    p_ai = predict_ai_probability(user_text, preprocessor, model, device)
    p_human = 1.0 - p_ai
    label_str = "AI-generated" if p_ai > 0.5 else "Human-written"
    print(f"[DEBUG] Predicted label: {label_str}, p_ai={p_ai:.3f}, p_human={p_human:.3f}")

    # LIME explanation
    def lime_predict(txts):
        probs = predict_proba_for_explanations(txts, model, tokenizer, device)
        return probs

    explainer = LimeTextExplainer(class_names=["Human", "AI"])
    lime_exp = explainer.explain_instance(
        user_text,
        classifier_fn=lime_predict,
        labels=[1],
        num_features=15,
        num_samples=500
    )

    strong_lime_words = get_strong_lime_words(lime_exp, top_n=5)
    strong_lime_html = "<h3>Strongest Words in LIME Explanation</h3><ul>"
    for word, weight in strong_lime_words:
        strong_lime_html += f"<li><strong>{word}</strong> (Influence: {weight:.3f})</li>"
    strong_lime_html += "</ul>"

    classification_banner = (
        f"<b>Predicted Label:</b> {label_str}<br>"
        f"<b>Prob(AI) = {p_ai:.3f}, Prob(Human) = {p_human:.3f}</b>"
    )

    final_html = f"""
<html>
<head>
  <meta charset="utf-8"/>
  <title>Classification & Explanation Report</title>
  <style>
    /* Basic resets */
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}
    
    body {{
      font-family: 'Helvetica Neue', Arial, sans-serif;
      background: #f9f9f9;
      padding: 20px;
      color: #333;
    }}

    .container {{
      max-width: 900px;
      margin: 0 auto;
      background: #fff;
      padding: 20px 30px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
    }}

    h1, h2, h3 {{
        margin-bottom: 12px;
      font-weight: bold;
      color: #444;
    }}

    .banner {{
      background: #f0f8ff;
      padding: 15px;
      border-radius: 6px;
      margin-bottom: 20px;
      border: 1px solid #e0e0e0;
    }}

    .banner p {{
      font-size: 1.1rem;
      line-height: 1.5;
    }}

    .lime-container {{
      margin-top: 20px;
      padding: 20px;
      background: #fafafa;
      border: 1px solid #e0e0e0;
      border-radius: 4px;
    }}

    .lime-container * {{
      line-height: 1.4;
    }}

    .lime-container ul {{
      margin-left: 25px; 
      margin-top: 10px;
      list-style-type: disc;
    }}

    .lime-container li {{
      margin-bottom: 5px;
    }}

    .lime-container .lime-label {{
      font-weight: bold;
      margin-right: 8px;
    }}

    .lime-container td {{
      vertical-align: top;
      padding: 8px;
    }}

    .footer {{
      margin-top: 30px;
      font-size: 0.9rem;
      color: #888;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Classification &amp; Explanation Report</h1>

    <div class="banner">
      <h2>Classification Result</h2>
      <p>{classification_banner}</p>
    </div>

    <div class="lime-container">
      <h2>LIME Explanation</h2>
      {strong_lime_html}

      <h3>Original LIME Visualization</h3>
      {lime_exp.as_html(labels=[1])}
    </div>

    <div class="footer">
      <p>If you do not see the bar chart or color highlights, 
      your environment may be blocking inline scripts.</p>
    </div>
  </div>
</body>
</html>
    """

    with open("final_report.html", "w") as f:
        f.write(final_html)
    print("[INFO] Report generated with meaningful LIME explanation.")

# ---------------------------------------------------------------------------
# -- New SHAP helper functions
# ---------------------------------------------------------------------------
def shap_explanation(text, model, tokenizer, device, background_samples=None):
    """
    Generate SHAP KernelExplainer for a single text and return (shap_values, tokens).
    """
    if background_samples is None:
        # Some small baseline texts
        background_samples = [
            "This is a sample sentence.",
            "It contains some typical words.",
            "We use this for baseline reference."
        ]

    def shap_predict(texts):
        # Return [p(Human), p(AI)] for each text
        return predict_proba_for_explanations(texts, model, tokenizer, device)

    explainer = shap.KernelExplainer(shap_predict, background_samples)
    # Evaluate shap_values for a single input
    shap_values = explainer.shap_values([text], nsamples=100)  # can adjust nsamples
    # Tokenize text
    tokens = tokenizer.tokenize(text)
    return shap_values, tokens

def shap_summary_human_readable(shap_values, tokens, top_n=5):
    """
    Create a plain-English summary from the SHAP values for class "AI".
    """
    # shap_values is a list [class0_values, class1_values]
    ai_shap = shap_values[1].flatten()
    token_shap_pairs = list(zip(tokens, ai_shap))

    # Sort by absolute contribution
    token_shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_contributors = token_shap_pairs[:top_n]

    pushes_ai = [(tok, val) for (tok, val) in top_contributors if val > 0]
    pushes_human = [(tok, val) for (tok, val) in top_contributors if val < 0]

    lines = []
    if pushes_ai:
        lines.append("Tokens pushing classification towards AI:")
        for (tok, val) in pushes_ai:
            lines.append(f"  - {tok} (influence={val:.3f})")
    if pushes_human:
        lines.append("Tokens pushing classification towards Human:")
        for (tok, val) in pushes_human:
            lines.append(f"  - {tok} (influence={val:.3f})")

    if not pushes_ai and not pushes_human:
        lines.append("No strong tokens found (all near zero).")

    return "\n".join(lines)

def lime_get_top_tokens(lime_exp, top_n=10):
    """
    Parse the LIME explanation object to get top tokens for AI label=1
    as (token, weight).
    """
    all_tokens = lime_exp.as_list(label=1)  # e.g. [("word", weight), ...]
    # Sort by absolute weight
    all_tokens.sort(key=lambda x: abs(x[1]), reverse=True)
    return all_tokens[:top_n]

def shap_get_top_tokens(shap_values, tokens, top_n=10):
    """
    From the shap_values for AI (index=1), return top (token, shap_value).
    """
    ai_shap = shap_values[1].flatten()
    pairs = list(zip(tokens, ai_shap))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:top_n]

def combine_explanations(lime_tokens, shap_tokens, top_n=5):
    """
    Merge LIME and SHAP tokens into a combined importance list.
    Return list of (token, combined_score, lime_score, shap_score).
    """
    lime_dict = {t[0]: t[1] for t in lime_tokens}  # token -> weight
    shap_dict = {t[0]: t[1] for t in shap_tokens}  # token -> shap_val
    combined_set = set(lime_dict.keys()) | set(shap_dict.keys())

    combined_list = []
    for token in combined_set:
        l_score = lime_dict.get(token, 0)
        s_score = shap_dict.get(token, 0)
        combined_score = abs(l_score) + abs(s_score)
        combined_list.append((token, combined_score, l_score, s_score))

    # Sort by combined_score desc
    combined_list.sort(key=lambda x: x[1], reverse=True)
    return combined_list[:top_n]

# ---------------------------------------------------------------------------
# -- Combined LIME + SHAP Route
# ---------------------------------------------------------------------------
@app.route("/api/combined_explanation", methods=["POST"])
def combined_explanation_route():
    """
    Runs both LIME and SHAP on the input text, merges top tokens,
    and returns a simple non-technical explanation.
    """
    data = request.json or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400

    # 1) Classify + get probabilities
    p_ai = predict_ai_probability(text, preprocessor, classifier_model, app_configs["device"])
    p_human = 1.0 - p_ai
    label_str = "AI-generated" if p_ai > 0.5 else "Human-written"

    # 2) LIME Explanation
    def lime_predict(txts):
        return predict_proba_for_explanations(txts, classifier_model, tokenizer, app_configs["device"])

    lime_explainer = LimeTextExplainer(class_names=["Human", "AI"])
    lime_exp = lime_explainer.explain_instance(
        text,
        classifier_fn=lime_predict,
        labels=[1],
        num_features=15,
        num_samples=300
    )
    lime_top = lime_get_top_tokens(lime_exp, top_n=10)

    # 3) SHAP Explanation
    shap_vals, shap_tokens = shap_explanation(
        text=text,
        model=classifier_model,
        tokenizer=tokenizer,
        device=app_configs["device"]
    )
    shap_top = shap_get_top_tokens(shap_vals, shap_tokens, top_n=10)

    # 4) Combine them
    combined_top = combine_explanations(lime_top, shap_top, top_n=5)

    # 5) Create a user-friendly bullet list
    bullets = []
    for (token, cscore, lscore, sscore) in combined_top:
        direction = "AI" if (lscore + sscore) > 0 else "Human"
        bullets.append(f"- *{token}* pushes classification towards **{direction}** (combined importance={cscore:.3f})")

    # 6) Optional short summary at the top
    summary_text = (
        f"This text is predicted as **{label_str}** "
        f"(Prob(AI)={p_ai:.2f}, Prob(Human)={p_human:.2f}).\n\n"
        "Below are tokens that both LIME and SHAP found most influential:\n"
    )
    bullet_points = "\n".join(bullets)

    # Return as JSON or as an HTML snippet
    response = {
        "classification": label_str,
        "prob_ai": p_ai,
        "prob_human": p_human,
        "combined_explanation": summary_text + bullet_points
    }
    return jsonify(response), 200

##############################################################################
# Flask App Initialization
##############################################################################
app = Flask(__name__)
CORS(app)

# 1) Device
device = target_device()

# 2) (Optional) HF login
# login_to_huggingface()  # Uncomment if needed

# 3) Load classification model
print("[DEBUG] Loading classification model...")
tokenizer, classifier_model = classification_get_pretrained_model()

# 4) Preprocessor
preprocessor = PreprocessDataset(tokenizer)

# 5) Explanation model (Mistral)
print("[DEBUG] Loading explanation model...")
explanation_tokenizer, explanation_model = explanation_get_pretrained_model()

##############################################################################
# Flask Routes (Existing)
##############################################################################
@app.route("/api/classify", methods=["POST"])
def classify_text():
    """
    Classify text & generate 2 short explanations from Mistral.
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Invalid JSON or missing 'text' field"}), 400
    
    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text provided"}), 400

    try:
        pred, explanation = classify_with_explanation(
            text=text,
            tokenizer=tokenizer,
            classifier_model=classifier_model,
            preprocessor=preprocessor,
            explanation_tokenizer=explanation_tokenizer,
            explanation_model=explanation_model
        )
        return jsonify({
            "prediction": pred,
            "explanation": explanation
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route("/api/explain", methods=["POST"])
def explain_text():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text required"}), 400
    
    classify_and_explain(text, classifier_model, tokenizer, device)
    return send_file("final_report.html", mimetype="text/html")

@app.route("/api/lime_explanation", methods=["POST"])
def lime_explanation_route():
    """
    Returns a LIME-only HTML page with the classic bar chart on the left 
    and color-highlighted text on the right.
    """
    data = request.json or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided."}), 400

    # 1) Classify
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = classifier_model(inputs["input_ids"], inputs["attention_mask"])
        p_ai = torch.sigmoid(logits).item()
    p_human = 1 - p_ai
    label_str = "AI-generated" if p_ai > 0.5 else "Human-written"

    # 2) LIME Explanation
    def lime_predict(txts):
        return predict_proba_for_explanations(txts, classifier_model, tokenizer, device)

    explainer = LimeTextExplainer(class_names=["Human-written","AI-generated"])
    exp = explainer.explain_instance(
        text,
        classifier_fn=lime_predict,
        labels=[1],           # label=1 => "AI-generated"
        num_features=20,
        num_samples=500
    )
    lime_html = exp.as_html(labels=[1])

    # 3) Build final HTML
    final_html = f"""
<html>
<head>
  <meta charset="utf-8"/>
  <title>LIME Explanation Demo</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      background-color: #f9f9f9;
      padding: 20px;
      color: #333;
    }}
    .container {{
      max-width: 900px;
      margin: 0 auto;
      background: #fff;
      padding: 20px 30px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }}
    h1 {{
      margin-bottom: 12px;
      color: #444;
    }}
    .classification-result {{
      background: #f0f8ff;
      border: 1px solid #e0e0e0;
      padding: 15px;
      border-radius: 6px;
      margin-bottom: 20px;
    }}
    .lime-report {{
      background: #fafafa;
      border: 1px solid #e0e0e0;
      padding: 15px;
      border-radius: 6px;
    }}
    .footer {{
      margin-top: 20px;
      font-size: 0.9rem;
      color: #999;
    }}
    .lime-report .lime-label {{
      font-weight: bold;
      margin-right: 8px;
    }}
    .lime-report td {{
      vertical-align: top;
      padding: 8px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>LIME Explanation Demo</h1>

    <div class="classification-result">
      <h2>Classification Result</h2>
      <p><b>Text Classification:</b> {label_str}</p>
      <p>Prob(Human)={p_human:.2f}, Prob(AI)={p_ai:.2f}</p>
    </div>

    <div class="lime-report">
      <h2>LIME Explanation Report</h2>
      {lime_html}
    </div>

    <div class="footer">
      <hr/>
      <p>If you do not see the bar chart or highlights, 
      your environment may be blocking inline JavaScript. 
      Try saving this HTML to a file and opening it in a normal browser.</p>
    </div>
  </div>
</body>
</html>
"""
    return final_html, 200

#############################################################################
# MAIN
##############################################################################
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
