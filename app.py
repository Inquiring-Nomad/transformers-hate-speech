import streamlit as st
import numpy as np
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import time
import os
# import dvc.api


def load_tokenizer_and_model():


    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    model = TFAutoModelForSequenceClassification.from_pretrained('web-app/hate-speech-tranformers')
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_auth_token=True)
    #
    # model = TFAutoModelForSequenceClassification.from_pretrained("akislou/hate-speech", use_auth_token=True)
    return (model, tokenizer)


def classify_text(text):

    data_load_state = st.text('Calculating...')
    (model, tokenizer) = load_tokenizer_and_model()

    model.config.id2label = {0: 'Hate Speech',
                             1: 'Offensive Language',
                             2: 'Neither hate speech or offensive language'}
    preds = model(tokenizer(text, return_tensors="tf"))['logits']
    class_preds = np.argmax(preds, axis=1)
    data_load_state.text('Calculating...done!')

    return (model.config.id2label[class_preds[0]])


st.title("Detect hate speech or offensive language")
st.header("using :hugging_face: Transformers")

st.text("This is a simple example app of text classification.")
st.text("The app would detect if the text is classified as hate speech, offensive language or  neither.")

with st.form("tweet"):
    # st.write("Inside the form")
    tweet_val = st.text_area("Enter text here", max_chars=200)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        if not tweet_val.strip():
            st.error("Please enter some text")
        else:

            st.write(f"This text is classified as {classify_text(tweet_val)}")
