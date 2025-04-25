# chatbot-ml-nlp
A simple Machine Learning + NLP-based chatbot built using Python.

# 🤖 Chatbot with NLP + Machine Learning

This is a simple yet powerful chatbot built using **Natural Language Processing (NLP)** and **Machine Learning**. It uses a Naive Bayes classifier to predict the intent behind user input and respond with appropriate pre-defined replies from an intents dataset.

---

## 📌 Features

- Intent recognition using ML (Naive Bayes)
- Bag-of-Words + Tokenization + Stemming (NLTK)
- Easy-to-edit training data (`intents.json`)
- Real-time command line interaction
- Fully customizable & extendable

---

## 🧠 Technologies Used

- Python 3.9+
- [NLTK](https://www.nltk.org/) (tokenization, stemming)
- [scikit-learn](https://scikit-learn.org/) (Naive Bayes, LabelEncoder)
- NumPy
- JSON + Pickle

---

## 📁 Project Structure

chatbot-ml-nlp/ ├── chat.py # Chat interface (user types, bot replies) ├── train_chatbot.py # Training script (model creation & saving) ├── intents.json # Dataset with user patterns and bot responses ├── chatbot_model.pkl # Trained model (saved) ├── label_encoder.pkl # Encoded intent labels ├── words.pkl # Tokenized vocabulary words ├── requirements.txt # Python dependencies └── README.md # You are here 😄

---

## Example Interactions

You: hello
Bot: Hi there!

You: thanks
Bot: You're welcome!

You: goodbye
Bot: See you later!

---
