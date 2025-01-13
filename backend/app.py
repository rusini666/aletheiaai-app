from flask import Flask, request, jsonify
from classification_code import (
    get_pretrained_model,
    PreprocessDataset,
    classify_with_explanation
)
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load models and preprocessors on startup
print("[DEBUG] Loading models and preprocessors...")
tokenizer, classifier, explanation_tokenizer, explanation_model = get_pretrained_model()
preprocessor = PreprocessDataset(tokenizer)

@app.route('/api/classify', methods=['POST'])
def classify_text():
    """API endpoint for text classification with free-text explanation."""
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        # Generate prediction and explanation
        prediction, explanation = classify_with_explanation(
            text, tokenizer, classifier, preprocessor, explanation_tokenizer, explanation_model
        )
        return jsonify({
            "prediction": prediction,
            "explanation": explanation
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error during classification: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)