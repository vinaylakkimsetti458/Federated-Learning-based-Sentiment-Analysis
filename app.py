# streamlit_app.py
import streamlit as st
import torch
from transformers import DistilBertTokenizer
from model import DistilBERTSentiment

# ------------------------
# Configuration
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.set_page_config(page_title="Sentiment Analysis", page_icon="📝")

st.title("🎬 Movie Sentiment Analysis using Federated Learning")
st.write("Type a movie review and get the predicted sentiment!")

# ------------------------
# Load tokenizer and model
# ------------------------
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBERTSentiment()
    model.load_state_dict(torch.load("global_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# ------------------------
# User input
# ------------------------
user_input = st.text_area("Enter your review:", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a review text!")
    else:
        # Tokenize
        inputs = tokenizer(
            user_input,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

        predicted_class = torch.argmax(logits, dim=1).item()
        label_map = {0: "Negative", 1: "Positive"}
        predicted_label = label_map[predicted_class]

        st.success(f"Predicted Sentiment: **{predicted_label}**")


