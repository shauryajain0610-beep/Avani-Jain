import streamlit as st

st.title("My First Streamlit AI App")
st.write("Streamlit is working perfectly!")
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
import urllib.parse

from Orange.data import Table, Domain, StringVariable
from Orange.preprocess.text import preprocess_strings
from Orange.classification import NaiveBayesLearner

# ------------------------------------------------------------
# ORANGE TEXT MINING SETUP
# ------------------------------------------------------------
# 1. Create tiny demo dataset
domain = Domain([StringVariable("text")], class_vars=StringVariable("class"))
training_data = Table.from_list(domain, [
    ["Breaking! Miracle cure found!", "FAKE"],
    ["Government releases new policy", "REAL"],
    ["Viral rumor about celebrity", "FAKE"],
    ["Stock market shows steady growth", "REAL"]
])

# 2. Preprocess text
training_data.X = preprocess_strings(training_data.X)

# 3. Train Naive Bayes model
nb_model = NaiveBayesLearner()(training_data)

# 4. Prediction function for user input
def orange_predict(user_text):
    processed = preprocess_strings([user_text])
    test_table = Table(domain, processed)
    pred = nb_model(test_table)
    return str(pred[0])

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# LOGO + TITLE
st.image("logo.png", width=120)  # Make sure logo.png is in the same folder
st.title("📰 Fake News Detector")
st.write("Analyze any headline or full article to detect if it's fake or real.")

# ------------------------------------------------------------
# Reasoning
# ------------------------------------------------------------
def generate_reasoning(prediction):
    if prediction == "FAKE":
        return (
            "This content shows several characteristics commonly seen in fake news:\n"
            "- The language appears overly dramatic or sensational.\n"
            "- Claims lack credible or verifiable sources.\n"
            "- Exaggerated or absolute terms are used.\n"
            "- The information lacks context or seems manipulated.\n"
            "These indicators collectively point toward misinformation."
        )
    else:
        return (
            "The content appears more balanced and credible:\n"
            "- Language is factual and not overly emotional.\n"
            "- Claims appear more grounded with possible context.\n"
            "- The style avoids unrealistic or exaggerated claims.\n"
            "Overall, it gives the impression of being authentic."
        )

# ------------------------------------------------------------
# Advice
# ------------------------------------------------------------
def generate_advice(prediction):
    if prediction == "FAKE":
        return (
            "Do not share this content immediately. Verify it through reliable "
            "fact-checkers like BBC Reality Check, AFP Fact Check, Alt News, or BOOMLive."
        )
    else:
        return (
            "Even though it looks real, always verify the source before forwarding it."
        )

# ------------------------------------------------------------
# External Sources
# ------------------------------------------------------------
def generate_links(query):
    encoded = urllib.parse.quote(query)
    return {
        "Google News Search": f"https://news.google.com/search?q={encoded}",
        "BBC Search": f"https://www.bbc.co.uk/search?q={encoded}",
        "Alt News Fact Check": f"https://www.altnews.in/?s={encoded}",
        "BOOMLive Fact Check": f"https://www.boomlive.in/search?query={encoded}"
    }

# ------------------------------------------------------------
# USER INPUT
# ------------------------------------------------------------
st.subheader("Choose Input Type")

choice = st.radio(
    "Select what you want to analyze:",
    ["News Headline", "Full Article"]
)

if choice == "News Headline":
    user_input = st.text_input("Enter your headline below:")
else:
    user_input = st.text_area("Enter your full article below:", height=180)

# ------------------------------------------------------------
# PREDICTION BUTTON
# ------------------------------------------------------------
if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text first.")
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
        # ✅ PREDICTION USING ORANGE
        prediction = orange_predict(user_input)

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
        # Show Prediction
        st.subheader("🧪 Prediction Result")
        if prediction == "FAKE":
            st.error("❌ This news appears to be FAKE.")
        else:
            st.success("✔️ REAL")
            st.success("✔ This news appears to be REAL.")

        st.subheader("🧠 Reasoning")
        st.write(reasoning)
        # Reasoning
        st.subheader("🧠 Detailed Reasoning")
        st.write(generate_reasoning(prediction))

        # Advice
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
        st.info(generate_advice(prediction))

        # External Links
        st.subheader("🌍 External Sources")
        links = generate_links(user_input)
        for name, url in links.items():
            st.write(f"- [{name}]({url})")

        # Thank you message
        st.success("🙏 THANK YOU FOR USING THIS APP! Stay aware, stay smart, stay safe ❤️")
