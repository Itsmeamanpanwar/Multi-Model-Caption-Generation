import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import os

# --- CONFIGURATION ---
BEAM_WIDTH = 5        # Higher = more accurate but slower (3, 5, 7 are good)
ALPHA = 0.7           # Length normalization factor (0.6 to 0.7 is standard)
MAX_LENGTH = 40       # Must match your training max_length
IMAGE_SIZE = (224, 224)

def load_tokenizer(path='tokenizer.json'):
    with open(path, 'r') as f:
        data = json.load(f)
        return tokenizer_from_json(data)

def extract_features(filename):
    model = MobileNetV2(weights='imagenet')
    # Remove the classification layer
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    
    # Load and process image
    image = load_img(filename, target_size=IMAGE_SIZE)
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    
    # Get features (1, 1280)
    feature = model.predict(image, verbose=0)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def beam_search_predictions(model, image, tokenizer, max_length=MAX_LENGTH, beam_index=BEAM_WIDTH):
    """
    Comparison: 
    - Greedy Search picks the highest probability word at step t.
    - Beam Search maintains 'k' most likely sequences at step t.
    """
    start = [tokenizer.word_index['startseq']]
    
    # beam_list structure: [[sequence_list, score]]
    # score is log probability (sum of logs)
    start_word = [start, 0.0]
    
    beam_list = [start_word]
    
    # Loop for max_length
    for i in range(max_length):
        temp_list = []
        
        for cur_seq, cur_score in beam_list:
            # If the sequence already ended, keep it as is (but don't expand it)
            if cur_seq[-1] == tokenizer.word_index.get('endseq'):
                temp_list.append([cur_seq, cur_score])
                continue

            # Prepare input for model
            cur_seq_padded = pad_sequences([cur_seq], maxlen=max_length)
            
            # Predict next word probabilities
            preds = model.predict([image, cur_seq_padded], verbose=0)[0]
            
            # Take top k probabilities (Greedy would just take argmax)
            # We use argsort to get indices of top k, then flip to get descending order
            top_words_indices = np.argsort(preds)[-beam_index:] 
            
            for word_index in top_words_indices:
                # Calculate new score: old_score + log(probability of new word)
                # We use log because multiplying small probabilities leads to underflow
                next_score = cur_score + np.log(preds[word_index] + 1e-20) # 1e-20 avoids log(0)
                
                next_seq = cur_seq + [word_index]
                temp_list.append([next_seq, next_score])
                
        # Determine the top k sequences from the expanded candidates
        beam_list = sorted(temp_list, key=lambda x: x[1], reverse=True)
        beam_list = beam_list[:beam_index]
        
        # Optimization: If all top k beams have ended, stop early
        if all(seq[0][-1] == tokenizer.word_index.get('endseq') for seq in beam_list):
            break
            
    # Final Selection with Length Normalization
    # Longer sentences naturally have lower sums of logs (more negative numbers added).
    # We divide by length^alpha to normalize this bias.
    best_score = -float('inf')
    best_caption = None
    
    for seq, score in beam_list:
        # Don't count startseq in length
        actual_length = len(seq) - 1
        if actual_length == 0: actual_length = 1 
        
        # Normalized Score formula
        normalized_score = score / (actual_length ** ALPHA)
        
        if normalized_score > best_score:
            best_score = normalized_score
            best_caption = seq

    # Convert indices back to words
    final_caption = []
    for i in best_caption:
        word = word_for_id(i, tokenizer)
        if word is None: continue
        if word != 'startseq' and word != 'endseq':
            final_caption.append(word)
            
    return ' '.join(final_caption)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Load resources
    print("Loading tokenizer and model...")
    tokenizer = load_tokenizer()
    model = load_model('image_caption_model.keras')
    
    # Image to predict
    img_path = 'test_image.jpg' # REPLACE THIS WITH YOUR IMAGE PATH
    
    if os.path.exists(img_path):
        print(f"Extracting features from {img_path}...")
        photo = extract_features(img_path)
        
        print(f"Generating caption with Beam Search (k={BEAM_WIDTH})...")
        caption = beam_search_predictions(model, photo, tokenizer)
        
        print("\n------------------------------------------------")
        print("Generated Caption:", caption)
        print("------------------------------------------------")
    else:
        print(f"Error: Image '{img_path}' not found. Please place an image in the folder.")






