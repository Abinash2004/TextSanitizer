from flask import Flask, render_template, request, jsonify
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Load the trained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('sanitizer_model_650')
tokenizer = T5Tokenizer.from_pretrained('sanitizer_model_650')

def replace_explicit_words(text):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=512, num_beams=5, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sanitize', methods=['POST'])
def sanitize_text():
    data = request.json
    input_text = data.get("text", "")
    sanitized_text = replace_explicit_words(input_text)
    return jsonify({"sanitized_text": sanitized_text})

if __name__ == '__main__':
    app.run(debug=True)
