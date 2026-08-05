from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

def create_model(max_length, vocab_size):
    """
    Creates the Image Captioning Model architecture.
    
    Args:
        max_length (int): The maximum length of the caption sequences (e.g., 35).
        vocab_size (int): The size of the vocabulary (total unique words).
        
    Returns:
        model: A Keras Model instance ready for compiling.
    """
    
    # --- Feature Extractor (Image) Model ---
    # Expects a 1280-dimensional vector from MobileNetV2
    inputs1 = Input(shape=(1280,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # --- Sequence Processor (Text) Model ---
    # Expects a sequence of integers
    inputs2 = Input(shape=(max_length,))
    
    # Embedding: Maps integer indices to dense vectors
    # mask_zero=True tells LSTM to ignore the 0-padding
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # --- Decoder (Merge) ---
    # Add the Image context and Text context together
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    
    # Output Layer: Probabilities for the next word in the sequence
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # Tie it all together
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    
    # (Optional) Print summary to verify architecture when called
    # model.summary()
    
    return model











