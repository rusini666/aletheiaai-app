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

        # OPTION 1: Using last token's logits (your current approach)
        last_token_logits = opt_logits[:, -1, :]

        # OPTION 2: Pooling over all tokens (might be better)
        pooled_logits = opt_logits.mean(dim=1)  # Average over all tokens

        # Pass through classifier head
        x = self.fc1(pooled_logits)  # <- Change here if using pooled_logits
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

        # For faster generation, reduce max_new_tokens
        outputs = explanation_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=1024,  # or even 50
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
    # 1) Preprocess
    input_ids, attention_mask = preprocessor.preprocess(text)
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    # 2) Classify
    with torch.no_grad():
        logits = classifier_model(input_ids, attention_mask)
        # logits is shape (1, 1) because your classifier returns a single sigmoid
        p_ai = logits.squeeze().item()  # Probability of AI
        # p_human = 1.0 - p_ai  # If you need it

    return p_ai

##############################################################################
# Classification + Explanation
##############################################################################
def classify_with_explanation(text, classifier_model, preprocessor, explanation_tokenizer, explanation_model):
    print("[DEBUG] classify_with_explanation called...")

    # (1) Probability of AI
    p_ai = predict_ai_probability(text, preprocessor, classifier_model, app_configs["device"])
    p_human = 1.0 - p_ai
    label_str = "AI-generated" if p_ai > 0.5 else "Human-written"

    print(f"[DEBUG] p_ai = {p_ai:.4f}, p_human = {p_human:.4f}")
    print(f"[DEBUG] Final classification: {label_str}")

    # (2) Generate explanation if explanation_model is available
    if explanation_model is not None:
        # For example, your custom 'generate_explanation' with Mistral
        combined_explanation = generate_explanation(
            text, 
            label_str, 
            explanation_tokenizer, 
            explanation_model
        )
    else:
        combined_explanation = "No explanation model available."

    # (3) Return
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
    Avoids stopwords, punctuation, and ensures meaningful words are kept.
    """
    stop_words = set(stopwords.words("english")) | set(string.punctuation)

    # Extract LIME words and their weights
    lime_words = [(word, weight) for word, weight in lime_exp.as_list(label=1)]

    # Sort words by absolute weight (importance)
    lime_words.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"[DEBUG] LIME Extracted Words (Before Filtering): {lime_words}")

    # Filter words: remove stopwords, punctuation, short words, and very weak contributions
    strong_lime_words = [
        (word, weight) for word, weight in lime_words
        if word.lower() not in stop_words
           and len(word) > 2
           and abs(weight) > 0.005
    ]
    print(f"[DEBUG] LIME Strong Words (After Filtering): {strong_lime_words}")

    # If filtering removes too many words, take the top N regardless
    if len(strong_lime_words) < top_n:
        print(f"[WARNING] Less than {top_n} strong words found. Showing top available words.")
        strong_lime_words = lime_words[:top_n]

    # Return only the top N (after filtering, or fallback above)
    return strong_lime_words[:top_n]

def classify_and_explain(user_text, model, tokenizer, device):
    """
    Classifies the input text and generates a LIME explanation.
    Extracts the strongest words from LIME and displays them properly.
    """
    # ========== (1) Classification (Shared Logic) ==========
    p_ai = predict_ai_probability(user_text, preprocessor, model, device)
    p_human = 1.0 - p_ai
    label_str = "AI-generated" if p_ai > 0.5 else "Human-written"
    print(f"[DEBUG] Predicted label: {label_str}, p_ai={p_ai:.3f}, p_human={p_human:.3f}")

    # (2) Define LIME Prediction Function
    def lime_predict(txts):
        probs = predict_proba_for_explanations(txts, model, tokenizer, device)
        print("[DEBUG] LIME predict probs:", probs)
        
        # Ensure it is always 2D with shape (N, 2)
        if isinstance(probs, list):
            probs = np.array(probs)

        if probs.ndim == 1:  # If it's a 1D array, reshape it
            probs = probs.reshape(-1, 2)

        return probs

    # (3) Run LIME Explanation
    explainer = LimeTextExplainer(class_names=["Human", "AI"])
    lime_exp = explainer.explain_instance(
        user_text,
        classifier_fn=lime_predict,
        labels=[1],
        num_features=15,  # Generate more words initially
        num_samples=500  # More samples = better accuracy
    )

    # (4) Extract the strongest words
    strong_lime_words = get_strong_lime_words(lime_exp, top_n=5)

    # (5) Build the Strongest Words HTML
    strong_lime_html = "<h3>Strongest Words in LIME Explanation</h3><ul>"
    for word, weight in strong_lime_words:
        strong_lime_html += f"<li><strong>{word}</strong> (Influence: {weight:.3f})</li>"
    strong_lime_html += "</ul>"

    # (6) Classification Summary
    classification_banner = (
        f"<b>Predicted Label:</b> {label_str}<br>"
        f"<b>Prob(AI) = {p_ai:.3f}, Prob(Human) = {p_human:.3f}</b>"
    )

    # (7) Build the Final Report
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
      font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
      background: linear-gradient(to bottom right, #eef2ff, #ffffff, #ebf4ff);
      padding: 20px;
      color: #333;
      min-height: 100vh;
    }}

    .container {{
      max-width: 900px;
      margin: 0 auto;
      background: #fff;
      padding: 30px;
      border-radius: 12px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
      border: 1px solid rgba(99, 102, 241, 0.1);
    }}

    h1, h2, h3 {{
      margin-bottom: 16px;
      font-weight: bold;
      color: #4338ca; /* indigo-700 */
    }}

    h1 {{
      font-size: 28px;
      text-align: center;
      margin-bottom: 24px;
      background: linear-gradient(90deg, #4338ca, #3b82f6);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}

    h2 {{
      font-size: 22px;
      border-bottom: 1px solid #e5e7eb;
      padding-bottom: 8px;
      margin-top: 28px;
    }}

    h3 {{
      font-size: 18px;
      margin-top: 20px;
      color: #4f46e5; /* indigo-600 */
    }}

    .banner {{
      background: #f5f7ff;
      padding: 20px;
      border-radius: 8px;
      margin-bottom: 24px;
      border: 1px solid #e0e7ff; /* indigo-100 */
    }}

    .banner p {{
      font-size: 1.1rem;
      line-height: 1.6;
      margin-bottom: 8px;
    }}

    .prediction-label {{
      display: inline-block;
      padding: 6px 12px;
      border-radius: 6px;
      font-weight: 600;
      margin-bottom: 12px;
      color: white;
    }}
    
    .prediction-ai {{
      background: linear-gradient(90deg, #4f46e5, #4338ca);
    }}
    
    .prediction-human {{
      background: linear-gradient(90deg, #047857, #065f46);
    }}

    .probability-bar {{
      display: flex;
      height: 30px;
      border-radius: 6px;
      overflow: hidden;
      margin: 15px 0;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}

    .probability-ai {{
      background: linear-gradient(90deg, #818cf8, #6366f1);
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: 600;
      text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }}

    .probability-human {{
      background: linear-gradient(90deg, #34d399, #10b981);
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: 600;
      text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }}

    .lime-container {{
      margin-top: 24px;
      padding: 24px;
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
    }}

    .lime-container * {{
      line-height: 1.5;
    }}

    .lime-container ul {{
      margin-left: 25px; 
      margin-top: 12px;
      list-style-type: disc;
    }}

    .lime-container li {{
      margin-bottom: 8px;
      font-size: 15px;
    }}

    .strong-word {{
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 500;
      background: #e0e7ff; /* indigo-100 */
      margin-right: 4px;
    }}

    .influence-positive {{
      color: #4338ca; /* indigo-700 */
    }}

    .influence-negative {{
      color: #b91c1c; /* red-700 */
    }}

    .lime-container .lime-label {{
      font-weight: 600;
      margin-right: 8px;
    }}

    .lime-container td {{
      vertical-align: top;
      padding: 10px;
    }}

    .footer {{
      margin-top: 36px;
      text-align: center;
      font-size: 0.9rem;
      color: #6b7280;
      padding-top: 16px;
      border-top: 1px solid #e5e7eb;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Classification &amp; Explanation Report</h1>

    <div class="banner">
      <h2>Classification Result</h2>
      <div class="prediction-label prediction-{label_str.lower().split('-')[0]}">
        {label_str}
      </div>
      
      <div class="probability-bar">
        <div class="probability-human" style="width: {p_human*100:.1f}%;">
          Human: {p_human*100:.1f}%
        </div>
        <div class="probability-ai" style="width: {p_ai*100:.1f}%;">
          AI: {p_ai*100:.1f}%
        </div>
      </div>
    </div>

    <div class="lime-container">
      <h2>Key Insights</h2>
      <h3>Strongest Words in Analysis</h3>
      <ul>
        {' '.join([f'<li><span class="strong-word">{word}</span> <span class="influence-{"positive" if weight > 0 else "negative"}">Influence: {weight:.3f}</span></li>' for word, weight in strong_lime_words])}
      </ul>

      <h3>Original LIME Visualization</h3>
      {lime_exp.as_html(labels=[1])}
    </div>

    <div class="footer">
      <p>AletheiaAI helps detect AI-generated content with user-centric explainability</p>
    </div>
  </div>
</body>
</html>
"""

    # (8) Save Report to File
    with open("final_report.html", "w") as f:
        f.write(final_html)
    print("[INFO] Report generated with meaningful LIME explanation.")

##############################################################################
# Flask App
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
# Flask Routes
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
        pred, raw_explanation = classify_with_explanation(
            text=text,
            classifier_model=classifier_model,
            preprocessor=preprocessor,
            explanation_tokenizer=explanation_tokenizer,
            explanation_model=explanation_model
        )
        
        # Parse the explanations to get clean versions
        explanation1, explanation2 = parse_explanations(raw_explanation)
        
        # Format them nicely for the frontend
        formatted_explanation = f"Explanation #1:\n{explanation1}\n\nExplanation #2:\n{explanation2}"
        
        return jsonify({
            "prediction": pred,
            "explanation": formatted_explanation
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

def parse_explanations(raw_explanation_text):
    """
    Parse the raw explanation text to extract just the explanations,
    removing the prompt and other artifacts.
    """
    # Remove any instruction blocks that might be in the generated text
    cleaned_text = re.sub(r'Each explanation should describe.*?verbatim\.', '', raw_explanation_text, flags=re.DOTALL)
    
    exp1 = ""
    exp2 = ""
    
    # Extract Explanation #1
    exp1_match = re.search(r"Explanation #1:(.*?)(?:Explanation #2:|$)", 
                          cleaned_text, re.DOTALL)
    if exp1_match:
        exp1 = exp1_match.group(1).strip()
    
    # Extract Explanation #2
    exp2_match = re.search(r"Explanation #2:(.*?)$", 
                          cleaned_text, re.DOTALL)
    if exp2_match:
        exp2 = exp2_match.group(1).strip()
    
    # If parsing failed, use some fallbacks
    if not exp1:
        exp1 = "No explanation could be extracted."
    if not exp2:
        exp2 = "No second explanation could be extracted."
        
    return exp1, exp2

@app.route("/api/smart_explanation_html", methods=["POST"])
def smart_explanation_html():
    """Return a clean HTML report with only valid explanations"""
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text required"}), 400
        
    # (1) Probability of AI
    p_ai = predict_ai_probability(text, preprocessor, classifier_model, app_configs["device"])
    p_human = 1.0 - p_ai
    label_str = "AI-generated" if p_ai > 0.5 else "Human-written"
    
    # (2) Generate explanation
    if explanation_model is not None:
        raw_explanation = generate_explanation(
            text, 
            label_str, 
            explanation_tokenizer, 
            explanation_model
        )
        # Extract only the valid explanations, completely ignoring any junk
        valid_explanations = []
        
        # Try to find real explanations with proper content
        exp1_match = re.search(r"Explanation #1:(.*?)(?:Explanation #2:|$)", raw_explanation, re.DOTALL)
        exp2_match = re.search(r"Explanation #2:(.*?)$", raw_explanation, re.DOTALL)
        
        if exp1_match:
            exp1 = exp1_match.group(1).strip()
            if len(exp1) > 10 and exp1 not in ['"', '"and"', '"."']:  # Only include if substantial
                valid_explanations.append(("Explanation #1", exp1))
                
        if exp2_match:
            exp2 = exp2_match.group(1).strip()
            if len(exp2) > 10 and exp2 not in ['"', '"."', '".']:  # Only include if substantial
                valid_explanations.append(("Explanation #2", exp2))
    else:
        valid_explanations = []
    
    # Generate explanation HTML only for valid explanations
    explanations_html = ""
    for title, content in valid_explanations:
        explanations_html += f"""
        <div class="explanation">
          <h3>{title}</h3>
          <p>{content}</p>
        </div>
        """
    
    label_class = "ai" if label_str.lower().startswith("ai") else "human"
    
    html_report = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Smart AI Explanation</title>
  <style>
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}
    
    body {{
      font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
      background: linear-gradient(to bottom right, #eef2ff, #ffffff, #ebf4ff);
      padding: 20px;
      color: #333;
      min-height: 100vh;
    }}

    .container {{
      max-width: 900px;
      margin: 0 auto;
      background: #fff;
      padding: 30px;
      border-radius: 12px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
      border: 1px solid rgba(99, 102, 241, 0.1);
    }}

    h1, h2, h3 {{
      margin-bottom: 16px;
      font-weight: bold;
      color: #4338ca;
    }}

    h1 {{
      font-size: 28px;
      text-align: center;
      margin-bottom: 24px;
      background: linear-gradient(90deg, #4338ca, #3b82f6);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}

    h2 {{
      font-size: 22px;
      border-bottom: 1px solid #e5e7eb;
      padding-bottom: 8px;
      margin-top: 28px;
    }}

    h3 {{
      font-size: 18px;
      margin-top: 20px;
      color: #4f46e5;
    }}

    .content-box {{
      background: #f5f7ff;
      border: 1px solid #e0e7ff;
      padding: 20px;
      border-radius: 8px;
      margin-bottom: 24px;
    }}
    
    .prediction-label {{
      display: inline-block;
      padding: 6px 12px;
      border-radius: 6px;
      font-weight: 600;
      margin-bottom: 12px;
      color: white;
    }}
    
    .prediction-ai {{
      background: linear-gradient(90deg, #4f46e5, #4338ca);
    }}
    
    .prediction-human {{
      background: linear-gradient(90deg, #047857, #065f46);
    }}

    .probability-bar {{
      display: flex;
      height: 30px;
      border-radius: 6px;
      overflow: hidden;
      margin: 15px 0;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}

    .probability-ai {{
      background: linear-gradient(90deg, #818cf8, #6366f1);
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: 600;
      text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }}

    .probability-human {{
      background: linear-gradient(90deg, #34d399, #10b981);
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: 600;
      text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }}

    .input-text {{
      padding: 15px;
      line-height: 1.6;
      max-height: 300px;
      overflow-y: auto;
    }}

    .explanation-section {{
      margin-top: 24px;
      padding: 20px;
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
    }}

    .explanation {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 15px;
      margin-bottom: 15px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}

    .explanation h3 {{
      margin-top: 0;
      margin-bottom: 10px;
      font-size: 16px;
      color: #4f46e5;
    }}

    .explanation p {{
      margin: 0;
      line-height: 1.6;
    }}

    .footer {{
      margin-top: 36px;
      text-align: center;
      font-size: 0.9rem;
      color: #6b7280;
      padding-top: 16px;
      border-top: 1px solid #e5e7eb;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Smart AI Explanation</h1>

    <div class="content-box">
      <h2>Input Text</h2>
      <div class="input-text">
        {text}
      </div>
    </div>

    <div class="content-box">
      <h2>Classification Result</h2>
      <div class="prediction-label prediction-{label_class}">
        {label_str}
      </div>
      
      <div class="probability-bar">
        <div class="probability-human" style="width: {p_human*100:.1f}%;">
          Human: {p_human*100:.1f}%
        </div>
        <div class="probability-ai" style="width: {p_ai*100:.1f}%;">
          AI: {p_ai*100:.1f}%
        </div>
      </div>
    </div>

    {f'''
    <div class="explanation-section">
      <h2>AI-Generated Explanations</h2>
      {explanations_html}
    </div>
    ''' if explanations_html else ''}

    <div class="footer">
      <p>AletheiaAI helps detect AI-generated content with user-centric explainability</p>
    </div>
  </div>
</body>
</html>
"""
    return html_report, 200, {'Content-Type': 'text/html'}

@app.route("/api/explain", methods=["POST"])
def explain_text():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text required"}), 400
    
    classify_and_explain(text, classifier_model, tokenizer, device)
    return send_file("final_report.html", mimetype="text/html")

##############################################################################
# (NEW) LIME Explanation Route (Bar Chart + color-coded text)
##############################################################################
@app.route("/api/lime_explanation", methods=["POST"])
def lime_explanation_route():
    """
    Returns a LIME-only HTML page with the classic bar chart on the left 
    and color-highlighted text on the right, exactly like your screenshot.
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
        num_features=20,      # show more "meaningful" words
        num_samples=500       # can tweak for speed/accuracy
    )
    lime_html = exp.as_html(labels=[1])

    # 3) Build final HTML
    final_html = f"""
<html>
<head>
  <meta charset="utf-8"/>
  <title>Detailed Insights Report</title>
  <style>
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}
    
    body {{
      font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
      background: linear-gradient(to bottom right, #eef2ff, #ffffff, #ebf4ff);
      padding: 20px;
      color: #333;
      min-height: 100vh;
    }}

    .container {{
      max-width: 900px;
      margin: 0 auto;
      background: #fff;
      padding: 30px;
      border-radius: 12px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
      border: 1px solid rgba(99, 102, 241, 0.1);
    }}

    h1, h2, h3 {{
      margin-bottom: 16px;
      font-weight: bold;
      color: #4338ca; /* indigo-700 */
    }}

    h1 {{
      font-size: 28px;
      text-align: center;
      margin-bottom: 24px;
      background: linear-gradient(90deg, #4338ca, #3b82f6);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}

    h2 {{
      font-size: 22px;
      border-bottom: 1px solid #e5e7eb;
      padding-bottom: 8px;
      margin-top: 24px;
    }}

    .classification-result {{
      background: #f5f7ff;
      border: 1px solid #e0e7ff; /* indigo-100 */
      padding: 20px;
      border-radius: 8px;
      margin-bottom: 24px;
    }}
    
    .prediction-label {{
      display: inline-block;
      padding: 6px 12px;
      border-radius: 6px;
      font-weight: 600;
      margin-bottom: 12px;
      color: white;
    }}
    
    .prediction-ai {{
      background: linear-gradient(90deg, #4f46e5, #4338ca);
    }}
    
    .prediction-human {{
      background: linear-gradient(90deg, #047857, #065f46);
    }}

    .probability-bar {{
      display: flex;
      height: 30px;
      border-radius: 6px;
      overflow: hidden;
      margin: 15px 0;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}

    .probability-ai {{
      background: linear-gradient(90deg, #818cf8, #6366f1);
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: 600;
      text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }}

    .probability-human {{
      background: linear-gradient(90deg, #34d399, #10b981);
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: 600;
      text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }}

    .lime-report {{
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      padding: 24px;
      border-radius: 8px;
    }}

    .lime-report .lime-label {{
      font-weight: 600;
      margin-right: 8px;
    }}

    .lime-report td {{
      vertical-align: top;
      padding: 10px;
    }}

    .footer {{
      margin-top: 36px;
      text-align: center;
      font-size: 0.9rem;
      color: #6b7280;
      padding-top: 16px;
      border-top: 1px solid #e5e7eb;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Detailed Insights Report</h1>

    <div class="classification-result">
      <h2>Classification Result</h2>
      <div class="prediction-label prediction-{label_str.lower().split('-')[0]}">
        {label_str}
      </div>
      
      <div class="probability-bar">
        <div class="probability-human" style="width: {p_human*100:.1f}%;">
          Human: {p_human*100:.1f}%
        </div>
        <div class="probability-ai" style="width: {p_ai*100:.1f}%;">
          AI: {p_ai*100:.1f}%
        </div>
      </div>
    </div>

    <div class="lime-report">
      <h2>LIME Analysis</h2>
      <p>The following visualization highlights which parts of the text influenced the classification decision:</p>
      {lime_html}
    </div>

    <div class="footer">
      <p>AletheiaAI helps detect AI-generated content with user-centric explainability</p>
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