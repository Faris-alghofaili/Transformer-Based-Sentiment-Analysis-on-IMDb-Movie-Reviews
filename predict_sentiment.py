# predict_sentiment.py

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load saved tokenizer and model
model_path = "bert_sentiment_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to predict sentiment
def predict_sentiment(review):
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    sentiment = "Positive 😊" if prediction == 1 else "Negative 😠"
    return sentiment

# Get review from user input
if __name__ == "__main__":
    print("📝 Enter a movie review below:")
    user_review = input(">> ")
    result = predict_sentiment(user_review)
    print(f"\n📣 Sentiment: {result}")
