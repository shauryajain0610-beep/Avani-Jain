import streamlit as st


import streamlit as st

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection App")
st.write("Enter a headline and article text, and the app will predict whether it appears Real or Fake.")


# -----------------------------
# SIMPLE RULE-BASED CLASSIFIER
# -----------------------------
def classify_news(headline, article):
    text = (headline + " " + article).lower()

    fake_keywords = [
        "shocking", "secret", "breaking!!!", "miracle", 
        "unbelievable", "banned", "hidden truth", "exposed",
        "100% guarantee", "cure", "conspiracy"
    ]

    score = 0
    for word in fake_keywords:
        if word in text:
            score += 1

    if score >= 2:
        prediction = "Fake"
        reasoning = "The text contains multiple suspicious or sensational keywords commonly used in misleading content."
    elif score == 1:
        prediction = "Possibly Fake"
        reasoning = "The text contains at least one sensational keyword, which may indicate misinformation."
    else:
        prediction = "Real"
        reasoning = "No major signs of sensational or misleading keywords were detected in the text."

    return prediction, reasoning


# -----------------------------
# STREAMLIT INPUT AREA
# -----------------------------
st.header("📝 Enter News Content")

headline = st.text_input("News Headline")
article = st.text_area("Article Text", height=200)
‘
if st.button("Analyze"):
    if not headline.strip() or not article.strip():
        st.warning("Please enter both a headline and article text.")
    else:
        # run the classifier
        prediction, reasoning = classify_news(headline, article)

        # Display results
        st.subheader("🔍 Prediction")
        if prediction == "Fake":
            st.error("❌ FAKE")
        elif prediction == "Possibly Fake":
            st.warning("⚠️ POSSIBLY FAKE")
        else:
            st.success("✔️ REAL")

        st.subheader("🧠 Reasoning")
        st.write(reasoning)

        st.subheader("💡 Advice")
        st.write("""
        - Verify the claim using trusted fact-checking organizations  
        - Avoid sharing content without confirming accuracy  
        - Check if credible sources are reporting the same information  
        """)

        # External Sources (clickable)
        st.subheader("🔗 External Fact-Checking Sources")
        st.markdown("""
        - [Snopes](https://www.snopes.com/)  
        - [PolitiFact](https://www.politifact.com/)  
        - [Reuters Fact Check](https://www.reuters.com/fact-check/)  
        - [AFP Fact Check](https://factcheck.afp.com/)  
        - [Google Fact Check Explorer](https://toolbox.google.com/factcheck/explorer)  
        """)

        st.info("This is a simple rule-based model. For real accuracy, connect a trained ML model.")
