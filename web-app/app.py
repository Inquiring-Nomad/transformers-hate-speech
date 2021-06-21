import streamlit as st
import numpy as np
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import time
import dvc.api


def load_tokenizer_and_model():
    with st.spinner('Please wait...'):
        time.sleep(5)
    hatespeech = dvc.api.read(
        'hate-speech-tranformers',
        remote='myremote',
        mode='rb'
    )
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    model = TFAutoModelForSequenceClassification.from_pretrained(hatespeech)
    return (model, tokenizer)


def classify_text(text):
    # my_bar = st.progress(0)
    # for percent_complete in range(100):
    #     time.sleep(0.1)
    #     my_bar.progress(percent_complete + 1)
    (model, tokenizer) = load_tokenizer_and_model()

    model.config.id2label = {0: 'Hate Speech',
                             1: 'Offensive Language',
                             2: 'Neither'}
    preds = model(tokenizer(text, return_tensors="tf"))['logits']
    class_preds = np.argmax(preds, axis=1)

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

            st.write(classify_text(tweet_val))
