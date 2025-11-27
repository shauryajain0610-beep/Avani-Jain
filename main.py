import streamlit as st
import pandas as pd
import urllib.parse
from Orange.data import Table, Domain, StringVariable
from Orange.classification import NaiveBayesLearner
from Orange.preprocess.text import preprocess_strings

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detector")
st.write("Enter a headline or article to detect if it is Real or Fake.")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_data():
    fake = pd.read_csv("data/Fake.csv")
    true = pd.read_csv("data/True.csv")
    fake["class"] = "FAKE"
    true["class"] = "REAL"
    data = pd.concat([fake, true], ignore_index=True)
    data["text"] = data["text"].astype(str)
    return data

data = load_data()

# ------------------------------------------------------------
# CREATE ORANGE DOMAIN AND TABLE
# ------------------------------------------------------------
domain = Domain([StringVariable("text")], class_vars=StringVariable("class"))
training_data = Table.from_list(domain, data[["text", "class"]].values.tolist())

# Preprocess text
training_data.X = preprocess_strings(training_data.X)

# Train Naive Bayes
nb_model = NaiveBayesLearner()(training_data)

# ------------------------------------------------------------
# PREDICTION FUNCTION
# ------------------------------------------------------------
def orange_predict(text):
    processed = preprocess_strings([text])
    test_table = Table(domain, processed)
    pred = nb_model(test_table)
    return str(pred[0].value)

# ------------------------------------------------------------
# REASONING FUNCTION
# ------------------------------------------------------------
def generate_reasoning(prediction):
    if prediction == "FAKE":
        return (
            "- Overly dramatic or sensational language.\n"
            "- Claims lack credible or verifiable sources.\n"
            "- Exaggerated or absolute terms are used.\n"
            "- Information may be manipulated or lack context."
        )
    else:
        return (
            "- Balanced and factual language.\n"
            "- Claims seem grounded with context.\n"
            "- Avoids exaggerated terms.\n"
            "- Likely authentic information."
        )

# ------------------------------------------------------------
# ADVICE FUNCTION
# ------------------------------------------------------------
def generate_advice(prediction):
    if prediction == "FAKE":
        return "Do not share this content. Verify via BBC Reality Check, AFP Fact Check, Alt News, or BOOMLive."
    else:
        return "Even though it looks real, always verify the source before sharing."

# ------------------------------------------------------------
# EXTERNAL LINKS FUNCTION
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
st.subheader("📝 Enter News Content")
headline = st.text_input("News Headline")
article = st.text_area("Article Text", height=200)

# ------------------------------------------------------------
# ANALYZE BUTTON
# ------------------------------------------------------------
if st.button("🔍 Analyze"):
    if not headline.strip() and not article.strip():
        st.warning("⚠️ Please enter headline or article text.")
    else:
        full_text = headline + " " + article
        prediction = orange_predict(full_text)

        # Prediction result
        st.subheader("🧪 Prediction Result")
        if prediction == "FAKE":
            st.error("❌ This news appears to be FAKE.")
        else:
            st.success("✔ This news appears to be REAL.")

        # Reasoning
        st.subheader("🧠 Detailed Reasoning")
        st.write(generate_reasoning(prediction))

        # Advice
        st.subheader("💡 Advice")
        st.info(generate_advice(prediction))

        # External Links
        st.subheader("🌍 External Fact-Checking Sources")
        links = generate_links(full_text)
        for name, url in links.items():
            st.write(f"- [{name}]({url})")

        # Thank you message
        st.success("🙏 THANK YOU FOR USING THIS APP! Stay aware, stay smart, stay safe ❤️")
