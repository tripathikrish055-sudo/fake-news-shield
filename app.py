import os
import re
import string

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.pipeline import Pipeline


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s",
    "same", "she", "should", "so", "some", "such", "t", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "you", "your",
    "yours", "yourself", "yourselves",
}


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/HTML/emails/digits/punctuation, remove stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


def strip_source_tags(text: str) -> str:
    """Remove wire-service tags like 'WASHINGTON (Reuters) -' that leak the label."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"^[A-Z\s]+\(Reuters\)\s*-\s*", "", text)
    text = re.sub(r"\(Reuters\)", "", text)
    text = re.sub(r"21st Century Wire", "", text, flags=re.IGNORECASE)
    return text


def basic_stats(text: str) -> dict:
    words = text.split()
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return {
        "characters": len(text),
        "words": len(words),
        "sentences": max(len(sentences), 1),
        "avg_word_length": round(sum(len(w) for w in words) / len(words), 2) if words else 0,
    }


# ---------------------------- DATA LOADING (Kaggle CSVs) ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAKE_CSV_PATH = os.path.join(BASE_DIR, "fake.csv")
TRUE_CSV_PATH = os.path.join(BASE_DIR, "true.csv")


def load_demo_dataset() -> pd.DataFrame:
    fake_df = pd.read_csv(FAKE_CSV_PATH)
    real_df = pd.read_csv(TRUE_CSV_PATH)

    fake_df["text"] = fake_df["title"].fillna("") + " " + fake_df["text"].fillna("")
    real_df["text"] = real_df["title"].fillna("") + " " + real_df["text"].fillna("")

    # Strip the Reuters wire-tag so the model can't "cheat" off source formatting
    fake_df["text"] = fake_df["text"].apply(strip_source_tags)
    real_df["text"] = real_df["text"].apply(strip_source_tags)

    fake_df["label"] = 0   # 0 = FAKE
    real_df["label"] = 1   # 1 = REAL

    df = pd.concat([fake_df[["text", "label"]], real_df[["text", "label"]]], ignore_index=True)
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


LABELS = {0: "FAKE", 1: "REAL"}


@st.cache_resource(show_spinner="Training model...")
def get_model():
    df = load_demo_dataset()
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    x_train, x_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.25, random_state=42, stratify=df["label"],
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=6000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=5.0, class_weight="balanced")),
    ])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred, target_names=["FAKE", "REAL"], zero_division=0),
        "train_size": len(x_train),
        "test_size": len(x_test),
    }
    return pipeline, metrics


def predict_news(pipeline: Pipeline, raw_text: str) -> dict:
    cleaned = clean_text(raw_text)
    if not cleaned:
        return {"label": "UNKNOWN", "confidence": 0.0, "top_fake_words": [], "top_real_words": []}

    proba = pipeline.predict_proba([cleaned])[0]
    pred_class = int(np.argmax(proba))
    confidence = round(float(np.max(proba)) * 100, 2)
    top_fake_words, top_real_words = explain(pipeline, cleaned)

    return {
        "label": LABELS[pred_class],
        "confidence": confidence,
        "prob_fake": round(float(proba[0]) * 100, 2),
        "prob_real": round(float(proba[1]) * 100, 2),
        "top_fake_words": top_fake_words,
        "top_real_words": top_real_words,
    }


def explain(pipeline: Pipeline, cleaned_text: str, top_n: int = 6):
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    clf: LogisticRegression = pipeline.named_steps["clf"]

    doc_vector = vectorizer.transform([cleaned_text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]

    nonzero_idx = doc_vector.nonzero()[1]
    if len(nonzero_idx) == 0:
        return [], []

    word_scores = sorted([(feature_names[i], coefs[i]) for i in nonzero_idx], key=lambda x: x[1])
    top_fake_words = [w for w, s in word_scores if s < 0][:top_n]
    top_real_words = [w for w, s in reversed(word_scores) if s > 0][:top_n]
    return top_fake_words, top_real_words


def top_keywords_overall(pipeline: Pipeline, label: int, top_n: int = 15):
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    clf: LogisticRegression = pipeline.named_steps["clf"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]
    order = np.argsort(coefs)
    idx = order[:top_n] if label == 0 else order[::-1][:top_n]
    return list(feature_names[idx])


st.set_page_config(page_title="SmartNews Shield | Fake News Detector",
                    page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

if "page" not in st.session_state:
    st.session_state.page = "Home"


def go_to(page_name: str):
    st.session_state.page = page_name


st.sidebar.title("🛡️ SmartNews Shield")
st.sidebar.caption("AI-powered Fake News Detection")

page = st.sidebar.radio(
    "Navigate", ["Home", "Analyze News", "Model Performance", "About"],
    index=["Home", "Analyze News", "Model Performance", "About"].index(st.session_state.page),
)
st.session_state.page = page

pipeline, metrics = get_model()


# ---------------------------- HOME ----------------------------
if st.session_state.page == "Home":
    st.title("🛡️ SmartNews Shield")
    st.subheader("An AI-powered Fake News Detection System")
    st.write(
        "Paste any news headline or article and SmartNews Shield will "
        "analyze its language patterns using Natural Language Processing "
        "and Machine Learning to estimate how likely it is to be **REAL** "
        "or **FAKE** news — along with a transparent explanation of *why*."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔍 Analyze News")
        st.write("Paste or upload an article and get an instant prediction.")
        if st.button("Go to Analyze News", use_container_width=True):
            go_to("Analyze News"); st.rerun()
    with col2:
        st.markdown("### 📊 Model Performance")
        st.write("See accuracy, precision, recall and the confusion matrix.")
        if st.button("Go to Model Performance", use_container_width=True):
            go_to("Model Performance"); st.rerun()
    with col3:
        st.markdown("### ℹ️ About the Project")
        st.write("Learn how the pipeline and architecture work.")
        if st.button("Go to About", use_container_width=True):
            go_to("About"); st.rerun()


# ---------------------------- ANALYZE NEWS ----------------------------
elif st.session_state.page == "Analyze News":
    st.title("🔍 Analyze a News Article")
    st.write("Paste a headline or full article below, or upload a `.txt` file.")

    uploaded_file = st.file_uploader("Upload a .txt file (optional)", type=["txt"])
    default_text = uploaded_file.read().decode("utf-8", errors="ignore") if uploaded_file else ""

    article_text = st.text_area("Article text", value=default_text, height=220,
                                 placeholder="Paste the news headline or article text here...")

    if st.button("🚀 Analyze News", type="primary"):
        if not article_text.strip():
            st.warning("Please paste some text or upload a file before analyzing.")
        else:
            result = predict_news(pipeline, article_text)
            stats = basic_stats(article_text)

            st.markdown("### Result")
            if result["label"] == "FAKE":
                st.error(f"🚨 Prediction: **FAKE NEWS**  ({result['confidence']}% confidence)")
            elif result["label"] == "REAL":
                st.success(f"✅ Prediction: **REAL NEWS**  ({result['confidence']}% confidence)")
            else:
                st.info("Could not confidently classify this text — try pasting more content.")

            if result["confidence"] < 60 and result["label"] in ("FAKE", "REAL"):
                st.warning("⚠️ Low confidence — treat this result as inconclusive. Try pasting a longer article for a more reliable prediction.")

            if result["label"] in ("FAKE", "REAL"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Confidence breakdown**")
                    fig, ax = plt.subplots(figsize=(4, 2.2))
                    bars = ax.barh(["FAKE", "REAL"], [result["prob_fake"], result["prob_real"]],
                                    color=["#e63946", "#2a9d8f"])
                    ax.set_xlim(0, 100)
                    ax.set_xlabel("Probability (%)")
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center")
                    st.pyplot(fig)
                with c2:
                    st.markdown("**Article snapshot**")
                    st.write(f"- Words: **{stats['words']}**")
                    st.write(f"- Sentences: **{stats['sentences']}**")
                    st.write(f"- Characters: **{stats['characters']}**")
                    st.write(f"- Avg. word length: **{stats['avg_word_length']}**")

                st.markdown("### Why this decision? (Explainability)")
                e1, e2 = st.columns(2)
                with e1:
                    st.markdown("🚩 **Words pushing toward FAKE**")
                    st.write(", ".join(result["top_fake_words"]) if result["top_fake_words"] else "_None detected._")
                with e2:
                    st.markdown("✅ **Words pushing toward REAL**")
                    st.write(", ".join(result["top_real_words"]) if result["top_real_words"] else "_None detected._")

                st.caption(
                    "Note: This is a demo educational tool trained on a limited "
                    "dataset. Always verify news through multiple trusted sources."
                )

# ---------------------------- MODEL PERFORMANCE ----------------------------
elif st.session_state.page == "Model Performance":
    st.title("📊 Model Performance")
    st.write(f"Training samples: **{metrics['train_size']}**  |  Test samples: **{metrics['test_size']}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']}%")
    m2.metric("Precision", f"{metrics['precision']}%")
    m3.metric("Recall", f"{metrics['recall']}%")
    m4.metric("F1 Score", f"{metrics['f1']}%")

    st.markdown("### Confusion Matrix")
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["FAKE", "REAL"]); ax.set_yticklabels(["FAKE", "REAL"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

    st.markdown("### Classification Report")
    st.text(metrics["report"])

    st.markdown("### Top words the model associates with each class")
    k1, k2 = st.columns(2)
    with k1:
        st.markdown("🚩 **Top FAKE indicator words**")
        st.write(", ".join(top_keywords_overall(pipeline, label=0)))
    with k2:
        st.markdown("✅ **Top REAL indicator words**")
        st.write(", ".join(top_keywords_overall(pipeline, label=1)))

# ---------------------------- ABOUT ----------------------------
elif st.session_state.page == "About":
    st.title("ℹ️ About SmartNews Shield")
    st.write(
        "**SmartNews Shield** is a web-based Fake News Detection system "
        "built with Streamlit and scikit-learn. It classifies news text as "
        "REAL or FAKE using a TF-IDF + Logistic Regression pipeline, and "
        "explains each prediction using the model's learned word weights."
    )
    st.markdown("### Tech Stack")
    st.write("Python · Streamlit · scikit-learn · pandas · NumPy · matplotlib")
    st.markdown("### Pipeline")
    st.code(
        "Text Cleaning -> TF-IDF Vectorization -> Logistic Regression "
        "-> Prediction + Confidence + Word-level Explanation",
        language="text",
    )

    st.markdown("### Installation")
    st.code("pip install streamlit scikit-learn pandas numpy matplotlib", language="bash")

    st.markdown("### Running the App")
    st.code("py -m streamlit run app.py", language="bash")