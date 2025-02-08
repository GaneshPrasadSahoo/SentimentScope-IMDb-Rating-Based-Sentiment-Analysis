import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# Load the saved model
model = load_model("D:/All_DataSet2/sentiment analysis/sentiment.keras")

# Load the saved tokenizer from JSON file
with open('D:/All_DataSet2/sentiment analysis/tokenizer.json', 'r') as f:
    tokenizer_data = json.load(f)

# Check the structure of tokenizer_data
st.write(tokenizer_data)  # This will help you verify the structure in the streamlit app

# If tokenizer_data is a string, try to load it properly
if isinstance(tokenizer_data, str):
    tokenizer_data = json.loads(tokenizer_data)

# Recreate the tokenizer manually
tokenizer = Tokenizer()

# Set the tokenizer's attributes based on the saved data
tokenizer.word_index = tokenizer_data.get('word_index', {})
tokenizer.word_counts = tokenizer_data.get('word_counts', {})
tokenizer.document_count = tokenizer_data.get('document_count', 0)
tokenizer.index_docs = tokenizer_data.get('index_docs', {})
tokenizer.index_word = tokenizer_data.get('index_word', {})
tokenizer.oov_token = tokenizer_data.get('oov_token', None)

# Function to predict sentiment
def predict_sentiment(review):
    sequences = tokenizer.texts_to_sequences([review])
    st.write(f"Sequences: {sequences}")  # Debugging: Check sequences
    pad_sequence = pad_sequences(sequences, maxlen=200)
    st.write(f"Padded Sequence: {pad_sequence}")  # Debugging: Check padded sequences
    prediction = model.predict(pad_sequence)
    st.write(f"Prediction: {prediction}")  # Debugging: Check prediction output
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment

# Streamlit interface
st.title("Sentiment Analysis Web App")

st.write("Enter a review to check whether it's positive or negative:")

# Input for user review
user_review = st.text_area("Enter your review:")

# When the user submits the review
if st.button("Predict Sentiment"):
    if user_review:
        sentiment = predict_sentiment(user_review)
        st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter a review to predict the sentiment.")
