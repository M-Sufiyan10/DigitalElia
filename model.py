import streamlit as st
import numpy as np
import pickle
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model as keras_load_model

# Load the trained model
@st.cache_resource
def load_trained_model(path):
    return keras_load_model(path)

# Load tokenizer (Ensure it's the same tokenizer used during training)
@st.cache_data
def load_tokenizer(path):
    with open(path, "rb") as f:
        return pickle.load(f)


path="/home/sufi/Downloads/roman_urdu_poetry_model(1).h5"
tokenizer_path="/home/sufi/Downloads/tokenizer(1).pkl"
model = load_trained_model(path)
tokenizer = load_tokenizer(tokenizer_path)


def generate_poetry(seed_text, next_words=20):
    # ... existing code ...
    generated_text = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=100, padding='pre')
        # Check what token_list looks like
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        # Check predicted value
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
            
        if output_word:  # Only append if a valid word is found
            generated_text += " " + output_word
        else:
            break  # Exit if no valid word is found
    return generated_text

print(generate_poetry("muhabbat"))