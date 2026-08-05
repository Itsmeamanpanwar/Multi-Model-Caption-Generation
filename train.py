import os
import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
import json

# Import architecture
from model import create_model

# --- CONFIGURATION ---
# Was hardcoded with Windows backslashes (r'flickr8k\Images'), which breaks
# on Colab/Mac/Linux. os.path.join builds the right separator for whatever
# OS this actually runs on.
IMAGE_DIR = os.path.join('flickr8k', 'Images')
CAPTION_FILE = os.path.join('flickr8k', 'captions.txt')

WORKING_DIR = '.'      
EPOCHS = 60            
BATCH_SIZE = 32        

# --- 1. FEATURE EXTRACTION ---
def extract_features(directory):
    model = MobileNetV2(weights='imagenet')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = {}
    
    if os.path.exists(os.path.join(WORKING_DIR, 'features.pkl')):
        print("Loading features from pickle file...")
        with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
            return pickle.load(f)

    print("Extracting features from images...")
    for img_name in tqdm(os.listdir(directory)):
        img_path = os.path.join(directory, img_name)
        if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        try:
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            feature = model.predict(image, verbose=0)
            image_id = img_name.split('.')[0]
            features[image_id] = feature
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))
    return features

# --- 2. TEXT PROCESSING ---
def load_captions(filename):
    print(f"Loading captions from {filename}...")
    with open(filename, 'r', encoding='utf-8') as f:
        captions_doc = f.read()
    
    mapping = {}
    lines = captions_doc.split('\n')
    if lines[0].startswith('image,caption'):
        lines = lines[1:]
        
    for line in tqdm(lines):
        if len(line) < 2: continue
        tokens = line.split(',', 1)
        if len(tokens) < 2: continue
        image_id_full, caption = tokens[0], tokens[1]
        image_id = image_id_full.split('.')[0]
        
        # BUG FIX 2: Removed len(word)>1 so we keep words like "a" and "I"
        caption = caption.lower()
        caption = 'startseq ' + " ".join([word for word in caption.split()]) + ' endseq'
        
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
        
    return mapping

# --- 3. DATA GENERATOR ---
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            if key not in mapping or key not in features: continue
            
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            
            if n == batch_size:
                yield (np.array(X1), np.array(X2)), np.array(y)
                X1, X2, y = list(), list(), list()
                n = 0

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    features = extract_features(IMAGE_DIR)
    mapping = load_captions(CAPTION_FILE)
    
    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)
            
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(c.split()) for c in all_captions)
    print(f"Vocab Size: {vocab_size}, Max Length: {max_length}")
    
    with open('tokenizer.json', 'w') as f:
        f.write(json.dumps(tokenizer.to_json()))
    
    # BUG FIX 1: Sorted to guarantee deterministic splits
    available_ids = sorted(list(set(features.keys()) & set(mapping.keys())))
    print(f"Total valid images: {len(available_ids)}")
    
    # STANDARD KARPATHY SPLIT (6000 Train, 1000 Val, 1000 Test)
    # Assumes dataset has exactly 8000 matched images. 
    # If slight variations exist (e.g., 8091), we strictly bound it.
    train_ids = available_ids[:6000]
    val_ids = available_ids[6000:7000]
    
    print(f"Training on {len(train_ids)} images, Validating on {len(val_ids)} images.")
    
    model = create_model(max_length, vocab_size)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    def create_dataset(keys):
        return tf.data.Dataset.from_generator(
            lambda: data_generator(keys, mapping, features, tokenizer, max_length, vocab_size, BATCH_SIZE),
            output_signature=(
                (tf.TensorSpec(shape=(None, 1280), dtype=tf.float32), 
                 tf.TensorSpec(shape=(None, max_length), dtype=tf.float32)), 
                tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32) 
            )
        )

    train_dataset = create_dataset(train_ids)
    val_dataset = create_dataset(val_ids)

    checkpoint = ModelCheckpoint(
        'image_caption_model.keras', 
        monitor='val_loss', save_best_only=True, mode='min', verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1
    )
    
    steps = len(train_ids) // BATCH_SIZE
    val_steps = len(val_ids) // BATCH_SIZE
    
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )












    































