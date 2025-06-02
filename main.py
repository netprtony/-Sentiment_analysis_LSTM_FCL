import streamlit as st
import torch
import torch.nn as nn
from tokenizers import Tokenizer
import os
import re

# Định nghĩa mô hình với Unidirectional LSTM
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        _, (hidden, _) = self.lstm(x)  # hidden: (1, batch_size, hidden_dim)
        x = hidden[-1]  # Lấy hidden state cuối cùng: (batch_size, hidden_dim)
        x = self.fc(x)  # (batch_size, num_classes)
        return x

# Tiền xử lý văn bản (tiếng Anh)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# Vectorization
def vectorize(sentence, tokenizer, sequence_length=128):
    output = tokenizer.encode(sentence)
    return torch.tensor(output.ids, dtype=torch.long).unsqueeze(0)  # [1, sequence_length]

# Dự đoán cảm xúc
def predict_sentiment(text, model, tokenizer, sequence_length=128):
    model.eval()
    processed_text = preprocess_text(text)
    input_ids = vectorize(processed_text, tokenizer, sequence_length)
    with torch.no_grad():
        outputs = model(input_ids)
        _, pred = torch.max(outputs, dim=1)
    return "Positive" if pred.item() == 1 else "Negative"

# Tải mô hình và tokenizer từ máy cục bộ
vocab_size = 1000
embedding_dim = 128
hidden_dim = 256
num_classes = 2
sequence_length = 128

model_path = "./sentiment_model/model.pt"
tokenizer_path = "./sentiment_model/tokenizer.json"

if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
    st.error("Model file (model.pt) or tokenizer file (tokenizer.json) not found in sentiment_model directory. Please check!")
    st.stop()

model = SentimentClassifier(vocab_size, embedding_dim, hidden_dim, num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
tokenizer = Tokenizer.from_file(tokenizer_path)

# Giao diện Streamlit
st.title("English Sentiment Analysis")
st.write("Enter an English sentence to predict its sentiment (Positive or Negative):")

user_input = st.text_area("Your sentence:", height=100)
if st.button("Predict"):
    if user_input:
        prediction = predict_sentiment(user_input, model, tokenizer, sequence_length)
        st.success(f"Prediction result: **{prediction}**")
    else:
        st.error("Please enter a sentence to predict!")