#!/usr/bin/env python3

import os
import re
import sys
import time
import ssl
import json
import base64
import string
import threading
import pickle
import types

import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Must be done before importing pyplot
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
    """Use GPU if available, else MPS on Apple Silicon, else CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    app_configs["device"] = device
    print(f"[DEBUG] Using device: {app_configs['device']}")
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
        """Preprocess the input text for classification."""
        start_prep = time.time()
        original_text = text

        # Replace fancy dashes/quotes that can break some libs
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
    Simple feed-forward classifier on top of an OPT-like model's final logits.
    We take the last token's logits => dimension: vocab_size => feed into small MLP => 1 output.
    """
    def __init__(self, pretrained_model):
        super(CustomOPTClassifier, self).__init__()
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
        # Take last token's logits => (batch_size, vocab_size)
        opt_out = opt_logits[:, -1, :]
        x = self.fc1(opt_out)
        x = self.relu(x)
        x = self.fc2(x)
        end_forward = time.time()
        print(f"[DEBUG] Forward pass completed in {end_forward - start_forward:.2f}s")
        return x

##############################################################################
# Load Pretrained Model for Classification
##############################################################################
def classification_get_pretrained_model():
    """Load tokenizer + base model + LoRA checkpoint for classification."""
    start_time = time.time()
    print("[DEBUG] Loading classifier tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(app_configs["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        print("[DEBUG] Added pad_token to tokenizer")

    print("[DEBUG] Loading classifier base model...")
    pretrained_model = AutoModelForCausalLM.from_pretrained(app_configs["base_model"])
    
    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model_with_lora = get_peft_model(pretrained_model, lora_config)

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

def generate_explanation(
    text: str, 
    prediction: str, 
    explanation_tokenizer, 
    explanation_model
):
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

        outputs = explanation_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1024,
            temperature=0.0, # 0.7
            top_k=1, # 40
            do_sample=False, # True
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

    # Now actually return the decoded text
    return raw_text

def explanation_get_pretrained_model():
    """
    Load the explanation model (e.g., Mistral-7B) and tokenizer, 
    used for generating free-text explanations.
    """
    explanation_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"[DEBUG] Loading explanation tokenizer: {explanation_model_name}")
    explanation_tokenizer = AutoTokenizer.from_pretrained(explanation_model_name)
    if explanation_tokenizer.pad_token is None:
        explanation_tokenizer.add_special_tokens({"pad_token": explanation_tokenizer.eos_token})
        print("[DEBUG] Added pad_token to explanation tokenizer")

    print("[DEBUG] Loading explanation model (may be large)...")
    try:
        explanation_model = AutoModelForCausalLM.from_pretrained(
            explanation_model_name,
            torch_dtype=torch.float16,        # Adjust if needed
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir="./cache"
        ).to(app_configs["device"])
        explanation_model.resize_token_embeddings(len(explanation_tokenizer))
        print("[DEBUG] Explanation model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Error loading explanation model: {e}")
        explanation_model = None

    return explanation_tokenizer, explanation_model

##############################################################################
# Classification + Explanation (Short)
##############################################################################
def classify_with_explanation(
    text: str,
    tokenizer,
    classifier_model,
    preprocessor,
    explanation_tokenizer,
    explanation_model
):
    """
    1) Preprocess & classify => 'AI generated' or 'Human generated'
    2) Call generate_explanation(...) which returns a SINGLE string
       containing both explanations.
    3) Return (prediction, combined_explanation)
    """
    print("[DEBUG] classify_with_explanation called...")

    # Preprocessing
    input_ids, attention_mask = preprocessor.preprocess(text)
    input_ids = input_ids.unsqueeze(0).to(app_configs["device"])
    attention_mask = attention_mask.unsqueeze(0).to(app_configs["device"])

    # Classification
    with torch.no_grad():
        logits = classifier_model(input_ids, attention_mask)
    pred_score = torch.sigmoid(logits).item()
    prediction = "AI generated" if pred_score > 0.5 else "Human generated"
    print(f"[DEBUG] Classification => {prediction}, score={pred_score:.4f}")

    # Explanation: a single string containing two labeled explanations
    if explanation_model is not None:
        combined_explanation = generate_explanation(
            text, 
            prediction, 
            explanation_tokenizer, 
            explanation_model
        )
    else:
        combined_explanation = "No explanation model available."

    # (Optional) log to file
    with open("classification_results.txt", "a", encoding="utf-8") as f:
        f.write("=== Classification + Explanations ===\n")
        f.write(f"Text:\n{text}\n\n")
        f.write(f"Prediction: {prediction}\n\n")
        f.write("Combined Explanation:\n")
        f.write(f"{combined_explanation}\n\n---\n")

    # Return both the classification label + the entire explanation text
    return prediction, combined_explanation

##############################################################################
# Fast(er) SHAP/LIME Explanation
##############################################################################

# Set this to True if you suspect MPS issues with SHAP/LIME.
# This will force CPU for the shap + lime steps.
force_shap_cpu = False

def predict_proba_for_explanations(texts, model, tokenizer, device):
    """
    Return [p(Human), p(AI)] for each text in 'texts'.
    We reuse the classification model but just get probabilities for each snippet.
    """
    model.eval()
    probs = []
    for txt in texts:
        inputs = tokenizer(
            txt,
            padding="max_length",
            truncation=True,
            max_length=100,  # Reduced for speed
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(inputs["input_ids"], inputs["attention_mask"])
            p_ai = torch.sigmoid(logits).item()
            p_human = 1.0 - p_ai
        probs.append([p_human, p_ai])
    return np.array(probs)


def pos_filter_tokenize(text):
    """
    Tokenize text and keep only certain parts of speech (nouns, verbs, adjectives).
    Remove punctuation & stopwords.
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    keep_pos_prefixes = ("NN", "VB", "JJ")

    sw = set(stopwords.words("english"))
    punctuation_set = set(string.punctuation)

    filtered = []
    for (word, pos) in tagged:
        if any(pos.startswith(pref) for pref in keep_pos_prefixes):
            # Also remove stopwords and punctuation
            if (word.lower() not in sw) and (word not in punctuation_set):
                filtered.append(word)
    return filtered


def pos_filter_text(text, max_tokens=50):
    """Return a 'cleaned' string by re-joining the POS-filtered tokens."""
    tokens = pos_filter_tokenize(text)
    tokens = tokens[:max_tokens]
    return " ".join(tokens)


def shap_explanation(model, tokenizer, device, text_sample):
    """
    1) Convert raw text -> POS-filtered text
    2) Create shap_values with reduced computations (algorithm='permutation', max_evals=200)
    3) Save text-based explanation => shap_text_explanation.html
    4) Save bar chart => shap_bar.png
    """
    try:
        cleaned_text = pos_filter_text(text_sample)

        # If we suspect MPS issues, or want to ensure no GPU overhead:
        # Move model to CPU for the SHAP call only.
        original_device = device
        if force_shap_cpu:
            model_cpu = model.to("cpu")
            use_device = torch.device("cpu")
        else:
            model_cpu = model
            use_device = original_device

        text_masker = shap.maskers.Text()
        explainer = shap.Explainer(
            lambda T: predict_proba_for_explanations(T, model_cpu, tokenizer, use_device),
            masker=text_masker,
            algorithm="partition"
        )

        # Limit the number of evaluations
        shap_values = explainer([cleaned_text], max_evals=600)

        # If we forced CPU, move model back
        if force_shap_cpu:
            model.to(original_device)

        # A) SHAP text highlight => HTML string
        shap_text_html = shap.plots.text(shap_values[0], display=False)
        with open("shap_text_explanation.html", "w", encoding="utf-8") as f:
            f.write(shap_text_html)
        print("[INFO] SHAP text explanation saved to shap_text_explanation.html")

        # B) SHAP bar chart => shap_bar.png
        fig = plt.figure()
        shap.plots.bar(shap_values[0][:, 1], show=False)
        plt.title("SHAP Bar Chart (AI class)")
        plt.savefig("shap_bar.png", dpi=200, bbox_inches='tight')
        plt.close(fig)
        print("[INFO] SHAP bar chart saved to shap_bar.png")

        return shap_values[0], shap_text_html, "shap_bar.png"

    except Exception as e:
        print("[ERROR] SHAP explanation error:", e)
        raise e


def lime_explanation(model, tokenizer, device, text_sample):
    """
    LIME Explanation on POS-filtered text.
    Returns => (lime_exp_object, 'lime_explanation.png')
    """
    try:
        cleaned_text = pos_filter_text(text_sample)

        # If we suspect device issues for LIME:
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
            cleaned_text,
            classifier_fn=lime_predict,
            labels=[1],
            num_features=5,
            num_samples=200  # drastically reduced from default
        )

        # Move back if forced CPU
        if force_shap_cpu:
            model.to(original_device)

        fig = exp.as_pyplot_figure(label=1)
        plt.title("LIME Explanation (AI class) - POS Filtered")
        fig.savefig("lime_explanation.png", dpi=200, bbox_inches='tight')
        plt.close(fig)
        print("[INFO] LIME explanation saved to lime_explanation.png")

        return exp, "lime_explanation.png"

    except Exception as e:
        print("[ERROR] LIME explanation error:", e)
        raise e


def generate_natural_language_summary(
    text_sample,
    shap_values_for_text,
    lime_exp_for_text,
    top_n=5,
    classification_banner=""
):
    """
    Summarize top tokens from SHAP & LIME in plain English, plus classification results at the top.
    """
    # Extract token-level SHAP contributions for AI class => shap_values_for_text.values[:, 1]
    token_contribs = []
    for i, token in enumerate(shap_values_for_text.data):
        shap_ai = shap_values_for_text.values[i, 1]
        token_contribs.append((token, shap_ai))

    # Sort by absolute contribution
    token_contribs.sort(key=lambda x: abs(x[1]), reverse=True)

    # Top positive => push toward AI
    top_positive = [t for t in token_contribs if t[1] > 0]
    top_positive.sort(key=lambda x: x[1], reverse=True)
    top_positive = top_positive[:top_n]

    # Top negative => push toward Human
    top_negative = [t for t in token_contribs if t[1] < 0]
    top_negative.sort(key=lambda x: x[1])
    top_negative = top_negative[:top_n]

    # LIME results
    lime_list = lime_exp_for_text.as_list(label=1)
    lime_list.sort(key=lambda x: abs(x[1]), reverse=True)
    lime_top = lime_list[:top_n]

    lines = []
    if classification_banner:
        lines.append(classification_banner)

    lines.append("====================================================")
    lines.append("                    EXPLANATION SUMMARY")
    lines.append("====================================================\n")

    # SHAP tokens - positive
    if top_positive:
        tokens_only = [tok for (tok, _) in top_positive]
        lines.append(
            "Based on SHAP, the words that pushed the classification toward 'AI-generated' are: "
            f"{', '.join(tokens_only)}."
        )
    else:
        lines.append("No words strongly pushing toward AI according to SHAP.")

    lines.append("")

    # SHAP tokens - negative
    if top_negative:
        tokens_only = [tok for (tok, _) in top_negative]
        lines.append(
            "The words pushing the classification toward 'Human-written' are: "
            f"{', '.join(tokens_only)}."
        )
    else:
        lines.append("No words strongly pushing toward Human according to SHAP.")

    lines.append("")

    # LIME perspective
    if lime_top:
        lime_pos = [tok for (tok, w) in lime_top if w > 0]
        lime_neg = [tok for (tok, w) in lime_top if w < 0]
        lines.append("From the LIME perspective:")
        if lime_pos:
            lines.append(f"- {', '.join(lime_pos)} also push the classifier toward 'AI'.")
        if lime_neg:
            lines.append(f"- {', '.join(lime_neg)} push the classifier toward 'Human'.")
    else:
        lines.append("LIME didn't find any strong features to push the text to AI or Human.")

    lines.append("\nIn summary, these keywords gave the model important cues.\n")
    return "\n".join(lines)


def embed_image_base64(img_path):
    """
    Read an image (PNG) from img_path, convert to base64,
    and return an <img> tag embedding it inline.
    """
    with open(img_path, "rb") as f:
        img_data = f.read()
    encoded = base64.b64encode(img_data).decode("utf-8")
    return f"<img src='data:image/png;base64,{encoded}' style='max-width:600px;'/>"


def classify_and_explain(user_text, model, tokenizer, device):
    """
    1) Classify user_text
    2) Run SHAP & LIME with reduced complexity
    3) Generate a combined HTML report (final_report.html) with black text
    """
    # === 1) Classification ===
    inputs = tokenizer(
        user_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=100  # reduced for speed
    ).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        p_ai = torch.sigmoid(logits).item()
    p_human = 1 - p_ai
    label_str = "AI" if p_ai > 0.5 else "Human"
    classification_banner = (
        f"=== CLASSIFICATION RESULT ===<br>"
        f"Predicted label: {label_str}<br>"
        f"Prob(AI)={p_ai:.3f}, Prob(Human)={p_human:.3f}<br><br>"
    )
    print(classification_banner.replace("<br>", "\n"))  # console display

    # === 2) SHAP Explanation ===
    shap_vals, shap_text_html, shap_bar_png = shap_explanation(
        model, tokenizer, device, user_text
    )

    # === 3) LIME Explanation ===
    lime_exp, lime_png = lime_explanation(
        model, tokenizer, device, user_text
    )

    # === 4) Explanation Summary (Plain text)
    explanation_report = generate_natural_language_summary(
        text_sample=user_text,
        shap_values_for_text=shap_vals,
        lime_exp_for_text=lime_exp,
        top_n=5,
        classification_banner=""
    )
    print(explanation_report)

    # === 5) Build Final HTML Report ===
    shap_bar_img_tag = embed_image_base64(shap_bar_png)
    lime_img_tag     = embed_image_base64(lime_png)

    # Wrap the SHAP text HTML in .force-black for black text
    final_html = f"""
<html>
<head>
    <meta charset="utf-8"/>
    <title>Classification & Explanation Report</title>
    <style>
      body {{
        font-family: Arial, sans-serif;
        margin: 20px;
        color: #000 !important; /* Force black text for body */
      }}
      .banner {{
        background-color: #f0f0f0;
        padding: 10px;
        margin-bottom: 20px;
      }}
      .section {{
        margin-bottom: 30px;
      }}
      h2 {{
        color: #444;
      }}
      .explanation-summary {{
        white-space: pre-wrap;
        background: #f9f9f9;
        padding: 10px;
      }}
      img {{
        display: block;
        margin: 10px 0;
        max-width: 600px;
      }}

      /* Force all text (and fills) inside .force-black to be black */
      .force-black, .force-black * {{
        color: #000 !important;
        fill: #000 !important;
      }}

      /* Basic modal styling */
      .modal-overlay {{
        display: none; /* hidden by default */
        position: fixed; 
        top: 0; 
        left: 0; 
        width: 100%; 
        height: 100%;
        background: rgba(0,0,0,0.5);
        z-index: 9999;
      }}
      .modal-content {{
        background: #fff;
        padding: 20px;
        margin: 100px auto;
        width: 80%;
        max-width: 800px;
        position: relative;
      }}
      .close-btn {{
        position: absolute;
        top: 10px; 
        right: 10px;
        cursor: pointer;
        font-weight: bold;
      }}
    </style>
</head>
<body>

<div class="banner force-black">
  <h2>Classification Result</h2>
  <p>{classification_banner}</p>
</div>

<div class="section force-black">
  <h2>SHAP Explanation (Text Highlight)</h2>
  {shap_text_html}
</div>

<div class="section force-black">
  <h2>SHAP Bar Chart</h2>
  {shap_bar_img_tag}
</div>

<div class="section force-black">
  <h2>LIME Explanation</h2>
  {lime_img_tag}
</div>

<div class="section explanation-summary force-black">
  <h2>Explanation Summary</h2>
  <div>{explanation_report}</div>
</div>

</body>
</html>
"""

    with open("final_report.html", "w", encoding="utf-8") as f:
        f.write(final_html)
    print("[INFO] Wrote combined explanation to final_report.html")

##############################################################################
# Flask App
##############################################################################
app = Flask(__name__)
CORS(app)

# 1) Select device
device = target_device()

# 2) (Optional) Hugging Face login
# login_to_huggingface()  # Uncomment if you need HF private model access

# 3) Load classification model & tokenizer
print("[DEBUG] Loading classification model ...")
tokenizer, classifier_model = classification_get_pretrained_model()

# 4) Create Preprocessor for classification
preprocessor = PreprocessDataset(tokenizer)

# 5) Load the explanation model & tokenizer
print("[DEBUG] Loading explanation model ...")
explanation_tokenizer, explanation_model = explanation_get_pretrained_model()

##############################################################################
# Flask Routes
##############################################################################
@app.route("/api/classify", methods=["POST"])
def classify_text():
    data = request.json or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        # Returns (prediction, explanation)
        prediction, explanation = classify_with_explanation(
            text=text,
            tokenizer=tokenizer,
            classifier_model=classifier_model,
            preprocessor=preprocessor,
            explanation_tokenizer=explanation_tokenizer,
            explanation_model=explanation_model
        )
        return jsonify({
            "prediction": prediction,
            "explanation": explanation
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route("/api/shap_html", methods=["GET"])
def serve_shap_html():
    shap_html_path = os.path.join(os.getcwd(), "shap_text_explanation.html")
    return send_file(shap_html_path, mimetype="text/html")

@app.route("/api/explain", methods=["POST"])
def explain_text():
    """
    Full classification + SHAP/LIME visualization, 
    returns an HTML report (final_report.html).
    """
    try:
        data = request.json or {}
        user_text = data.get("text", "")
        if not user_text:
            return jsonify({"error": "Text input is required."}), 400

        # Classification + SHAP+LIME explanation
        classify_and_explain(user_text, classifier_model, tokenizer, device)

        # Return final report
        report_path = os.path.join(os.getcwd(), "final_report.html")
        if os.path.exists(report_path):
            return send_file(report_path, mimetype="text/html")
        else:
            return jsonify({"error": "Failed to generate explanation report."}), 500

    except Exception as e:
        # Now you see the REAL reason it failed:
        return jsonify({"error": f"SHAP + LIME explanation error: {str(e)}"}), 500

##############################################################################
# Main Entrypoint
##############################################################################
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)