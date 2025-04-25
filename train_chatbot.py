import json
import numpy as np
import random
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
nltk.download('punkt')

# Download nltk resources (only the first time)
nltk.download('punkt')

# Load the intents file
with open('intents.json') as file:
    data = json.load(file)

# Initialize stemmer
stemmer = PorterStemmer()

# Prepare data
all_words = []
tags = []
xy = []

for intent in data['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        words = nltk.word_tokenize(pattern)
        all_words.extend(words)
        xy.append((words, tag))
    tags.append(tag)

# Stem and lowercase words, remove punctuation
ignore_words = ['?', '!', '.', ',']
all_words = [stemmer.stem(w.lower()) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create training data
X = []
y = []

for (pattern_sentence, tag) in xy:
    bag = []
    pattern_words = [stemmer.stem(w.lower()) for w in pattern_sentence if w not in ignore_words]
    for w in all_words:
        bag.append(1) if w in pattern_words else bag.append(0)

    X.append(bag)
    y.append(tag)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

X = np.array(X)
y = np.array(y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test accuracy
print("Model Accuracy:", model.score(X_test, y_test))

# Save files
pickle.dump(model, open('chatbot_model.pkl', 'wb'))
pickle.dump(le, open('label_encoder.pkl', 'wb'))
pickle.dump(all_words, open('words.pkl', 'wb'))
