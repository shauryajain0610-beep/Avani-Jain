import streamlit as st
import urllib.parse
import pandas as pd
from Orange.data import Table, Domain, StringVariable
from Orange.preprocess.text import preprocess_strings
from Orange.classification import NaiveBayesLearner

# ------------------------------------------------------------
# LOAD AND PREPARE DATA (KAGGLE DATASET)
# ------------------------------------------------------------
@st.cache_data
def load_orange_model():
    # Load CSVs
    fake = pd.read_csv("data/Fake.csv")
    true = pd.read_csv("data/True.csv")

    # Add c
