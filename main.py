import streamlit as st
import pickle
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection App")
st.write("Enter a headline or full article and analyze whether the content is Real or Fake.")

# -----------------------------
# MODEL PATHS
# -----------------------------
import os

MODEL_PATH = os.path.join(os.getcwd(), "models", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(os.getcwd(), "models", "vectorizer.pkl")


# -----------------------------
# CHECK IF MODEL FILES EXIST
# -----------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.error(
        "❌ Model files not found! Please make sure 'fake_news_model.pkl' and 'vectorizer.pkl' "
        "are inside the 'models/' folder."
    )
    st.stop()  # Stop execution if files are missing

# -----------------------------
# LOAD ML MODEL & VECTORIZER
# -----------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# -----------------------------
# ML CLASSIFIER FUNCTION
# -----------------------------
def classify_ml(text):
    tfidf = vectorizer.transform([text])
    result = model.predict(tfidf)[0]
    proba = model.predict_proba(tfidf)[0]
    confidence = max(proba) * 100  # Confidence %
    return "Real" if result == 1 else "Fake", confidence

# -----------------------------
# STREAMLIT INPUT AREA
# -----------------------------
st.header("📝 Select Input Type")
choice = st.radio("Select what you want to enter:", ["Headline Only", "Full Article"])

headline = ""
article = ""

if choice == "Headline Only":
    headline = st.text_input("Enter News Headline")
else:
    headline = st.text_input("Headline (optional)")
    article = st.text_area("Enter Full Article Text", height=180)

# -----------------------------
# ANALYZE BUTTON
# -----------------------------
if st.button("Analyze"):
    combined_text = (headline + " " + article).strip()

    if not combined_text:
        st.warning("⚠ Please enter some text first.")
    else:
        prediction, confidence = classify_ml(combined_text)

        # -----------------------------
        # DYNAMIC REASONING & ADVICE
        # -----------------------------
        if prediction == "Fake":
            reasoning = "The ML model predicts this content is likely fake based on patterns learned from real and fake news datasets."
            advice = """
            ❌ **Advice if Fake:**  
            - Verify using trusted fact-checking sites  
            - Do NOT share this content unless confirmed  
            - Look for official government or credible news sources  
            """
            sources = """
            - [Alt News](https://www.altnews.in/)  
            - [BOOM Fact Check](https://www.boomlive.in/)  
            - [Factly](https://factly.in/)  
            """
            st.error(f"❌ FAKE NEWS ({confidence:.2f}% confidence)")

        else:
            reasoning = "The ML model predicts this content is likely real based on patterns learned from real and fake news datasets."
            advice = """
            ✔ **Advice if Real:**  
            - Check the original source for any updates  
            - Share responsibly from official/reputed outlets  
            - Verify facts from authentic government or national agencies  
            """
            sources = """
            - [Reuters](https://www.reuters.com/)  
            - [BBC News](https://www.bbc.com/)  
            - [The Hindu](https://www.thehindu.com/)  
            """
            st.success(f"✔ REAL NEWS ({confidence:.2f}% confidence)")

        # -----------------------------
        # DISPLAY RESULTS
        # -----------------------------
        st.subheader("🧠 Reasoning")
        st.write(reasoning)

        st.subheader("💡 Smart Advice")
        st.write(advice)

        st.subheader("🔗 Trusted Verification Sources")
        st.markdown(sources)

        st.info("This ML model is trained on Kaggle True vs Fake news datasets. Always verify important news from multiple sources.")
