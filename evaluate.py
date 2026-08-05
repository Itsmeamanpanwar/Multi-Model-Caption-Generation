import numpy as np
import pickle
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from predict import beam_search_predictions

# --- CONFIGURATION ---
MAX_LENGTH = 40 
BEAM_WIDTHS_TO_TEST = [1, 3, 5] 

def load_resources():
    print("Loading resources...")
    with open('tokenizer.json', 'r') as f:
        tokenizer = tokenizer_from_json(json.load(f))
        
    model = load_model('image_caption_model.keras')
    
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
        
    return tokenizer, model, features

def get_test_data(features):
    print("Loading caption mappings and splitting data...")
    filename = os.path.join('flickr8k', 'captions.txt')
    
    with open(filename, 'r', encoding='utf-8') as f:
        captions_doc = f.read()
        
    mapping = {}
    lines = captions_doc.split('\n')
    if lines[0].startswith('image,caption'): lines = lines[1:]
    
    for line in lines:
        if len(line) < 2: continue
        tokens = line.split(',', 1)
        image_id = tokens[0].split('.')[0]
        
        # Raw ground truths must include words like "a" to match training
        caption_raw = tokens[1].lower().split()
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption_raw)

    # BUG FIX 1: Sorted to guarantee deterministic splits matching train.py
    available_ids = sorted(list(set(features.keys()) & set(mapping.keys())))
    
    # STANDARD KARPATHY SPLIT (Test set is the chunk from 7000 to 8000)
    test_ids = available_ids[7000:8000]
    
    return test_ids, mapping

def evaluate_model(model, tokenizer, test_ids, features, mapping, beam_width):
    actual, predicted = list(), list()
    
    print(f"\n--- Evaluating with Beam Width k={beam_width} ---")
    
    for key in tqdm(test_ids): 
        references = mapping[key] 
        feature = features[key]
        yhat = beam_search_predictions(model, feature, tokenizer, max_length=MAX_LENGTH, beam_index=beam_width)
        
        actual.append(references)
        predicted.append(yhat.split())

    b1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    b2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    b3 = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0))
    b4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    
    return b1, b2, b3, b4

if __name__ == "__main__":
    tokenizer, model, features = load_resources()
    test_ids, mapping = get_test_data(features)
    
    print(f"Total Test Images (Unseen): {len(test_ids)}")
    
    results = {}
    
    for k in BEAM_WIDTHS_TO_TEST:
        b1, b2, b3, b4 = evaluate_model(model, tokenizer, test_ids, features, mapping, k)
        results[k] = {'BLEU-1': b1, 'BLEU-2': b2, 'BLEU-3': b3, 'BLEU-4': b4}
        print(f"k={k} -> B-1: {b1:.4f}, B-4: {b4:.4f}")

    print("\n\n================ FINAL RESULTS ================")
    print(f"{'Beam Width':<12} | {'BLEU-1':<10} | {'BLEU-2':<10} | {'BLEU-3':<10} | {'BLEU-4':<10}")
    print("-" * 65)
    for k, metrics in results.items():
        print(f"{k:<12} | {metrics['BLEU-1']:.4f}     | {metrics['BLEU-2']:.4f}     | {metrics['BLEU-3']:.4f}     | {metrics['BLEU-4']:.4f}")
    print("===============================================")



























