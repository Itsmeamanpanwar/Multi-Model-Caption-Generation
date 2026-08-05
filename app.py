# app.py
# Flask backend for the image captioning model.
#
# Fixes applied vs. the original file:
#   1. Missing `pad_sequences` import (would crash on the first request)
#   2. Model filename mismatch: train.py saves "image_caption_model.keras",
#      this used to try to load "image_caption_model.h5"
#   3. Was doing plain greedy decoding here (different from predict.py),
#      which meant the live demo didn't match the beam-search results
#      reported in evaluate.py / the paper. Now reuses the exact same
#      beam_search_predictions() function from predict.py.
#   4. Added CORS so a frontend served from a different port/origin
#      (e.g. the index.html in /frontend) is allowed to call this API.

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

from predict import load_tokenizer, extract_features, beam_search_predictions, MAX_LENGTH, BEAM_WIDTH

app = Flask(__name__)
CORS(app)  # allow browser requests from a different origin (the frontend)

print("Loading tokenizer and model...")
tokenizer = load_tokenizer('tokenizer.json')
model = load_model('image_caption_model.keras')
print("Model loaded. Ready to serve requests.")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    temp_path = 'temp_upload.jpg'
    image_file.save(temp_path)

    try:
        features = extract_features(temp_path)
        caption = beam_search_predictions(
            model, features, tokenizer,
            max_length=MAX_LENGTH, beam_index=BEAM_WIDTH
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify({'caption': caption})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
