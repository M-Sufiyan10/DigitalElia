import streamlit as st
import numpy as np
from model import generate_poetry
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.preprocessing.text import Tokenizer
import pickle
#from loadmodel import load_models


st.title("Digital Elia")

seed_text = st.text_input("Enter a seed phrase")
poetry_length = st.slider("Select poetry length (number of words)", 
                          min_value=10, 
                          max_value=100, 
                          value=30, 
                          step=5)


if st.button("Generate Poetry"):
    poetry = generate_poetry(seed_text, poetry_length)
    st.write(poetry)