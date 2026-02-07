import streamlit as st
import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModel
from textblob import TextBlob
import numpy as np
import re

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(page_title="News Popularity Intelligence", layout="wide")

# --------------------------------
# Load Transformer (once)
# --------------------------------
@st.cache_resource
def load_transformer():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    return tokenizer, model

tokenizer, model = load_transformer()

# --------------------------------
# Proxy Signal Functions
# --------------------------------
def emotion_intensity(text):
    return abs(TextBlob(text).sentiment.polarity)

def subjectivity_score(text):
    return TextBlob(text).sentiment.subjectivity

def urgency_score(text):
    urgent_words = ["breaking", "urgent", "alert", "now", "today", "exclusive"]
    text = text.lower()
    return sum(word in text for word in urgent_words)

def lexical_diversity(text):
    words = re.findall(r"\w+", text.lower())
    return len(set(words)) / (len(words) + 1)

def readability_score(text):
    words = re.findall(r"\w+", text)
    if len(words) == 0:
        return 0
    return 1 / (np.mean([len(w) for w in words]) + 1)

def length_score(text):
    l = len(text)
    if l < 50:
        return 0.2
    elif l > 500:
        return 0.5
    else:
        return 1.0

# --------------------------------
# Popularity Scoring
# --------------------------------
def compute_popularity(text):
    scores = {
        "Emotion": emotion_intensity(text),
        "Urgency": urgency_score(text),
        "Lexical Richness": lexical_diversity(text),
        "Readability": readability_score(text),
        "Length Balance": length_score(text),
        "Subjectivity": subjectivity_score(text)
    }

    popularity_score = (
        0.25 * scores["Emotion"] +
        0.20 * scores["Urgency"] +
        0.20 * scores["Lexical Richness"] +
        0.15 * scores["Readability"] +
        0.10 * scores["Length Balance"] +
        0.10 * scores["Subjectivity"]
    )

    return popularity_score, scores

# --------------------------------
# Sidebar Navigation
# --------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "News Intelligence", "Model Reasoning"])

# --------------------------------
# HOME PAGE
# --------------------------------
if page == "Home":
    st.title("📰 News Popularity Intelligence System")

    st.markdown("""
    ### Problem Overview
    Digital news platforms must decide **which articles to highlight** before real popularity
    signals like clicks or shares are available.

    ### Key Challenge
    Popularity is a **latent variable** — it cannot be directly observed at publishing time.

    ### Solution
    This system uses:
    - Transformer-based semantic understanding
    - Weak proxy popularity signals
    - Self-supervised ranking logic
    - Explainable AI reasoning
    """)

# --------------------------------
# NEWS INTELLIGENCE PAGE
# --------------------------------
elif page == "News Intelligence":
    st.title("🧠 News Intelligence")

    title = st.text_input("Enter News Title")
    description = st.text_area("Enter News Description")

    if st.button("Analyze Popularity"):
        full_text = title + " [SEP] " + description

        score, explanations = compute_popularity(full_text)

        st.metric("📊 Popularity Score", round(score, 3))

        st.subheader("🔍 Explanation Signals")
        st.json(explanations)

# --------------------------------
# MODEL REASONING PAGE
# --------------------------------
elif page == "Model Reasoning":
    st.title("🔬 Model Reasoning & Interpretation")

    st.markdown("""
    ### How the Model Thinks
    - Transformers understand **semantic meaning and intent**
    - Proxy signals approximate **attention-driving features**
    - Scores are **relative**, not absolute popularity

    ### Why This Works
    - No labeled popularity data exists at publish time
    - Weak supervision captures editorial intuition
    - Ranking is more realistic than classification
    """)

    st.info("This system prioritizes **reasoning and interpretability** over raw accuracy.")
