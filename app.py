import streamlit as st
import joblib
import re
import numpy as np
from textblob import TextBlob
from scipy.sparse import hstack, csr_matrix

# ---- Page Config ----
st.set_page_config(
    page_title="Mental Health Text Classifier",
    page_icon="🧠",
    layout="centered"
)

# ---- Load Models ----
@st.cache_resource
def load_models():
    lr      = joblib.load("models_v3/logistic_regression_model.pkl")
    tfidf   = joblib.load("models_v3/tfidf_vectorizer.pkl")
    encoder = joblib.load("models_v3/label_encoder.pkl")
    return lr, tfidf, encoder

lr, tfidf, encoder = load_models()

# ---- Lexicons ----
SUICIDAL_WORDS = [
    "die", "death", "dead", "kill", "suicide", "suicidal",
    "end my life", "end it all", "no reason to live", "not worth living",
    "better off without me", "want to disappear", "cant go on",
    "giving up", "farewell", "no point living", "point in living",
    "dont want to exist", "want to die", "tired of living",
    "falling apart", "cant take it anymore", "done with life",
    "affairs in order", "giving away", "no way out", "disappear"
]
DEPRESSION_WORDS = [
    "hopeless", "worthless", "empty", "numb", "pointless", "meaningless",
    "exhausted", "drained", "hollow", "defeated", "burden", "broken",
    "helpless", "miserable", "despair", "trapped", "stuck", "sad",
    "depressed", "depression", "crying", "unmotivated", "no motivation",
    "no energy", "no appetite", "no interest", "no joy", "used to love",
    "dont enjoy", "stopped caring", "gave up", "no future", "no hope"
]
ANXIETY_WORDS = [
    "panic", "panicking", "anxious", "anxiety", "worry", "worried",
    "scared", "fear", "nervous", "overwhelmed", "racing", "overthinking",
    "restless", "tense", "dreading", "terrified", "uneasy",
    "heart racing", "cant breathe", "shaking", "replaying",
    "what if", "stomach drops", "cant relax", "on edge", "spiral"
]
LONELINESS_WORDS = [
    "alone", "lonely", "loneliness", "isolated", "invisible",
    "disconnected", "abandoned", "unwanted", "forgotten", "nobody",
    "no one", "unloved", "rejected", "no friends", "no one cares",
    "nobody checks", "misunderstood", "all alone", "by myself",
    "no one to talk", "no one notices", "unheard", "unseen"
]
POSITIVE_WORDS = [
    "happy", "happiness", "excited", "grateful", "blessed", "wonderful",
    "fantastic", "amazing", "great", "joy", "joyful", "love", "loved",
    "cheerful", "delighted", "peaceful", "content", "proud", "thrilled",
    "relaxed", "energized", "refreshed", "enjoy", "fun", "laugh",
    "smile", "celebrate", "positive", "optimistic", "motivated"
]

# ---- Preprocessing ----
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---- Feature Extraction ----
def extract_features(text):
    text_lower = text.lower()
    words      = text_lower.split()
    word_count = len(words) if len(words) > 0 else 1
    blob       = TextBlob(text)

    return np.array([
        sum(1 for w in SUICIDAL_WORDS   if w in text_lower) / word_count,
        sum(1 for w in DEPRESSION_WORDS if w in text_lower) / word_count,
        sum(1 for w in ANXIETY_WORDS    if w in text_lower) / word_count,
        sum(1 for w in LONELINESS_WORDS if w in text_lower) / word_count,
        sum(1 for w in POSITIVE_WORDS   if w in text_lower) / word_count,
        sum(1 for w in words if w in ["i","me","my","myself","mine"]) / word_count,
        sum(1 for w in words if w in ["not","no","never","nobody",
                                       "nothing","nowhere","cannot","nor"]) / word_count,
        text.count("!") / word_count,
        text.count("?") / word_count,
        sum(1 for c in text if c.isupper()) / (len(text) + 1),
        np.mean([len(w) for w in words]) if words else 0,
        blob.sentiment.polarity,
        blob.sentiment.subjectivity,
        word_count / 500
    ])

# ---- Predict ----
def predict(text, threshold=0.40):
    cleaned    = preprocess(text)
    tfidf_feat = tfidf.transform([cleaned])

    proba      = lr.predict_proba(tfidf_feat)
    pred_enc   = np.argmax(proba)
    confidence = float(proba.max())
    pred_label = encoder.inverse_transform([pred_enc])[0]

    scores = {
        cls: round(float(proba[0][i]) * 100, 2)
        for i, cls in enumerate(encoder.classes_)
    }
    scores = dict(sorted(scores.items(),
                        key=lambda x: x[1], reverse=True))

    if confidence < threshold:
        pred_label = "uncertain"

    risk_flag = scores.get("suicidal", 0) > 30 and pred_label != "suicidal"
    return pred_label, round(confidence * 100, 2), scores, risk_flag

#---------UI---------
st.title("🧠 Mental Health Text Classifier")
st.markdown("Enter any text and the model will classify it into one of five categories.")
st.markdown("---")

with st.expander("ℹ️ What do the categories mean?"):
    st.markdown("""
    | Category | Description |
    |---|---|
    | 😊 Normal | Everyday language, no mental health concern |
    | 😰 Anxiety | Excessive worry, nervousness, racing thoughts |
    | 😔 Depression | Persistent sadness, hopelessness, low energy |
    | 😞 Loneliness | Feeling isolated, invisible, disconnected |
    | 🆘 Suicidal | Thoughts of self harm or ending life |
    """)

with st.expander("⚙️ Model Information"):
    st.markdown("""
    **Architecture:** Ensemble of 3 classical ML models
    - Logistic Regression (weight: 0.4)
    - Linear SVM (weight: 0.4)  
    - Naive Bayes (weight: 0.2)
    
    **Features:** TF-IDF (50,000 terms) + 14 handcrafted features
    
    **Performance:**
    - Accuracy: 76%
    - Macro F1: 0.76
    - Mean AUC: 0.935
    
    **Limitation:** Works best with direct expressions of emotion.
    Sarcastic or highly indirect language may return UNCERTAIN.
    """)

text_input = st.text_area(
    "Enter your text here:",
    height=150,
    placeholder="e.g. I have been feeling really down and hopeless for weeks now..."
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_btn = st.button("🔍 Analyze", use_container_width=True)

st.markdown("---")

if analyze_btn:
    if not text_input.strip():
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing..."):
            pred_label, confidence, scores, risk_flag = predict(text_input)

        color_map = {
            "normal"    : "🟢",
            "anxiety"   : "🟡",
            "depression": "🟠",
            "loneliness": "🔵",
            "suicidal"  : "🔴",
            "uncertain" : "⚪"
        }

        emoji = color_map.get(pred_label, "⚪")
        st.markdown(f"### {emoji} Prediction: **{pred_label.upper()}**")
        st.markdown(f"**Confidence: {confidence}%**")

        if pred_label == "uncertain":
            st.info("ℹ️ The model is not confident enough to make a clear prediction. "
                   "Try providing more detail or context.")

        if risk_flag:
            st.error("⚠️ HIGH RISK FLAG: Elevated suicidal indicators detected. "
                    "Please consider reaching out to a mental health professional.")

        st.markdown("#### Confidence Scores")
        for cls, score in scores.items():
            e = color_map.get(cls, "⚪")
            st.markdown(f"{e} **{cls.capitalize()}**")
            st.progress(score / 100)
            st.caption(f"{score}%")

        if pred_label == "suicidal" or risk_flag:
            st.markdown("---")
            st.markdown("### 🆘 Crisis Resources")
            st.markdown("""
            If you or someone you know is struggling, please reach out:
            - **iCall (India):** 9152987821
            - **Vandrevala Foundation:** 1860-2662-345 (24/7)
            - **AASRA:** 91-22-27546669
            - **International:** [findahelpline.com](https://findahelpline.com)
            """)

st.markdown("---")
st.caption("⚠️ This tool is for educational purposes only and is not a "
           "substitute for professional mental health diagnosis or treatment.")