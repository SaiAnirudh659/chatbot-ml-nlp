import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import PorterStemmer

# Load saved model and data
model = pickle.load(open("chatbot_model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
words = pickle.load(open("words.pkl", "rb"))

# Load intents
with open("intents.json") as file:
    intents = json.load(file)

stemmer = PorterStemmer()

# Preprocess user input
def bag_of_words(sentence, words):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

# Chat loop
print("🤖 Chatbot is ready! Type 'quit' to exit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Bot: Goodbye! 👋")
        break

    bow = bag_of_words(user_input, words).reshape(1, -1)
    prediction = model.predict(bow)[0]
    tag = le.inverse_transform([prediction])[0]

    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            print("Bot:", response)
