import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
import re
import os
import time
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
import sys
import threading

# Custom stop words
custom_stop_words = set(stopwords.words('english'))

# Initialize configurations
app_configs = {
    'base_model': 'facebook/opt-1.3b',
    'models_path': './models',
    'model_name': '202409230028_subtaskA_monolingual_facebook_opt-1.3b',
    'device': torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"),
    'hf_token_file': 'hf_token.txt'  # File containing the Hugging Face token
}

print(f"[DEBUG] Using device: {app_configs['device']}")

def login_to_huggingface():
    """Log in to Hugging Face using a token from hf_token.txt."""
    print("[DEBUG] Attempting to log in to Hugging Face...")
    try:
        with open(app_configs['hf_token_file'], 'r') as token_file:
            hf_token = token_file.read().strip()
        login(token=hf_token)
        print("[DEBUG] Successfully logged in to Hugging Face.")
    except FileNotFoundError:
        print(f"[ERROR] {app_configs['hf_token_file']} not found. Please create the file and add your Hugging Face token.")
    except Exception as e:
        print(f"[ERROR] Error during Hugging Face login: {e}")

# Preprocess Dataset
class PreprocessDataset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def preprocess(self, text):
        print("[DEBUG] Preprocessing text...")
        start_prep = time.time()
        original_text = text
        # Remove usernames and URLs
        text = re.sub(r'(@\w+)|https?://\S+|www\.\S+', ' ', text)
        # Remove non-alphanumeric chars
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        # Lowercase
        text = text.lower()
        # Remove stopwords
        tokens = text.split()
        tokens = [token for token in tokens if token not in custom_stop_words]
        processed_text = ' '.join(tokens)

        tokenized = self.tokenizer(
            processed_text,
            padding='max_length',
            max_length=150,
            truncation=True,
            return_tensors="pt"
        )
        end_prep = time.time()
        print(f"[DEBUG] Preprocessing completed. Original text length: {len(original_text)} chars, Processed text length: {len(processed_text)} chars. Time taken: {end_prep - start_prep:.2f}s")
        return tokenized['input_ids'][0], tokenized['attention_mask'][0]

# OPT-based Classifier
class CustomOPTClassifier(nn.Module):
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
        opt_out = self.opt(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        x = self.fc1(opt_out)
        x = self.relu(x)
        x = self.fc2(x)
        end_forward = time.time()
        print(f"[DEBUG] Forward pass completed in {end_forward - start_forward:.2f}s")
        return x

def classify_with_explanation(text, tokenizer, classifier, preprocessor, explanation_tokenizer, explanation_model):
    """Classify the given text and optionally generate two complete explanations."""
    print("[DEBUG] classify_with_explanation called...")
    start_classify = time.time()
    input_ids, attention_mask = preprocessor.preprocess(text)

    print("[DEBUG] Moving data to device...")
    input_ids = input_ids.unsqueeze(0).to(app_configs['device'])
    attention_mask = attention_mask.unsqueeze(0).to(app_configs['device'])

    with torch.no_grad():
        logits = classifier(input_ids, attention_mask)

    pred_score = torch.sigmoid(logits).item()

    # >0.5 = AI generated, else Human generated
    prediction = "AI generated" if pred_score > 0.5 else "Human generated"
    print(f"[DEBUG] Classification prediction: {prediction}, score: {pred_score:.4f}")

    explanation = "No explanation model available."
    if explanation_model is not None:
        print("[DEBUG] Generating explanation...")
        explanation = generate_explanation(text, prediction, explanation_tokenizer, explanation_model)
    end_classify = time.time()
    print(f"[DEBUG] classify_with_explanation completed in {end_classify - start_classify:.2f}s")

    # Save results to a text file
    with open("classification_results.txt", "a", encoding="utf-8") as f:
        f.write(f"Text: {text}\nPrediction: {prediction}\nExplanation:\n{explanation}\n---\n")

    return prediction, explanation

def get_pretrained_model():
    print("[DEBUG] Starting to load models and tokenizers...")
    start_time = time.time()

    # Load classifier tokenizer and model
    print("[DEBUG] Loading classifier tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(app_configs['base_model'])
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        print("[DEBUG] Added pad_token to tokenizer")

    print("[DEBUG] Loading classifier model...")
    pretrained_model = AutoModelForCausalLM.from_pretrained(app_configs['base_model'])
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model_with_lora = get_peft_model(pretrained_model, lora_config)
    classifier = CustomOPTClassifier(model_with_lora)

    print("[DEBUG] Loading classifier checkpoint...")
    model_path = os.path.join(app_configs['models_path'], app_configs['model_name'] + ".pt")
    state_dict = torch.load(model_path, map_location='cpu')
    classifier.load_state_dict(state_dict)
    classifier = classifier.to(app_configs['device'])
    classifier.eval()

    # Load explanation model and tokenizer
    print("[DEBUG] Loading explanation tokenizer...")
    explanation_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    explanation_tokenizer = AutoTokenizer.from_pretrained(explanation_model_name)
    if explanation_tokenizer.pad_token is None:
        explanation_tokenizer.add_special_tokens({'pad_token': explanation_tokenizer.eos_token})
        print("[DEBUG] Added pad_token to explanation tokenizer")

    print("[DEBUG] Loading explanation model (This might be slow for large models)...")
    try:
        explanation_model = AutoModelForCausalLM.from_pretrained(
            explanation_model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir="./cache"
        ).to(app_configs['device'])

        explanation_model.resize_token_embeddings(len(explanation_tokenizer))
        print("[DEBUG] Explanation model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Error loading explanation model: {e}")
        explanation_model = None

    end_time = time.time()
    print(f"[DEBUG] Models and tokenizers loaded in {end_time - start_time:.2f} seconds")
    return tokenizer, classifier, explanation_tokenizer, explanation_model

# Spinner/loader for explanation generation
stop_spinner = False
def spinner():
    """A simple spinner to indicate processing."""
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
    """Generate a single, complete explanation focusing on why the text was classified as AI or Human generated."""
    print("[DEBUG] Entering generate_explanation...")
    start_exp = time.time()

    # A larger context size to ensure the full prompt is included
    prompt = (
        f"Below is a piece of text and its classification:\n\n"
        f"Text: {text}\n"
        f"Classification: {prediction}\n\n"
        f"Provide two distinct explanations detailing the reasoning behind this classification decision. "
        f"For each explanation, focus on the linguistic patterns, style, coherence, complexity, and any indicative markers "
        f"that suggest the text was {'produced by an AI model' if prediction=='AI generated' else 'written by a human'}. "
        f"Do not simply restate the text. Each explanation should provide unique details and rationale for this classification."
    )

    # Increase max_length for the prompt to avoid truncation of the input
    inputs = explanation_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
        max_length=1024,  # Increase if needed
        padding=True
    ).to(app_configs['device'])

    print("[DEBUG] Explanation generation inputs prepared.")
    print(f"[DEBUG] Input shape: {inputs['input_ids'].shape}")
    print("[DEBUG] Starting explanation generation...")

    global stop_spinner
    stop_spinner = False
    spin_thread = threading.Thread(target=spinner)
    spin_thread.start()

    try:
        # Generate one complete explanation with plenty of room
        outputs = explanation_model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=1024,  # Large enough for a full explanation
            temperature=0.0,       # Deterministic output
            top_k=1,               # Always choose the most likely token
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=explanation_tokenizer.pad_token_id
        )

        explanation = explanation_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as gen_e:
        print(f"[ERROR] Error during explanation generation: {gen_e}")
        explanation = "Explanation generation failed."
    finally:
        # Stop the spinner
        stop_spinner = True
        spin_thread.join()

    end_exp = time.time()
    print(f"[DEBUG] Explanation generated in {end_exp - start_exp:.2f}s")
    return explanation


def process_file(file_path):
    print(f"[DEBUG] Processing file: {file_path}")
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"[DEBUG] Extracted {len(lines)} lines from file")
            return lines
    elif file_path.endswith('.pdf'):
        reader = PdfReader(file_path)
        pages = [page.extract_text() for page in reader.pages]
        print(f"[DEBUG] Extracted {len(pages)} pages from PDF")
        return pages
    else:
        raise ValueError("[ERROR] Unsupported file format. Please provide a TXT or PDF file.")