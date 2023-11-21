#utilities
import re
import numpy as np
import pandas as pd
import os
import nltk

#SpellCorrection
from spellchecker import SpellChecker
from nltk.tokenize import RegexpTokenizer

import chardet
DATASET_COLUMNS = ['date', 'username', 'text', 'polarity', 'emotion']

#Detect file encoding using chardet
with open('Emcrypt-dataset.csv', 'rb') as f:
    result = chardet.detect(f.read())

# Print the detected encoding
print("Detected encoding:", result['encoding'])

# Read the file using the detected encoding
df = pd.read_csv('Emcrypt-dataset.csv', encoding=result['encoding'], names=DATASET_COLUMNS)
df.sample(5)


#Data preprocessing
data=df[['text','polarity', 'emotion']]
data_pos = data[data['polarity'] == 1]
data_neg = data[data['polarity'] == 0]
dataset = pd.concat([data_pos, data_neg])


# First step: Remove numbers
def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
dataset['text'] = dataset['text'].apply(lambda x: cleaning_numbers(x))
dataset['text'].head()


emoticons_to_keep = [
    '🌈', '🌙', '🌚', '🌞', '🌟', '🌷', '🌸', '🌹', '🌺', '🍀', '🍕', '🍻', '🎀',
    '🎈', '🎉', '🎤', '🎥', '🎧', '🎵', '🎶', '👅', '👇', '👈', '👉', '👋', '👌',
    '👍', '👏', '👑', '💀', '💁', '💃', '💋', '💐', '💓', '💕', '💖', '💗', '💘',
    '💙', '💚', '💛', '💜', '💞', '💤', '💥', '💦', '💪', '💫', '💯', '📷', '🔥',
    '😀', '😁', '😃', '😄', '😅', '😆', '😇', '😈', '😉', '😊', '😋', '😌', '😍',
    '😎', '😏', '😺', '😻', '😽', '🙏', '☀', '☺', '♥', '✅', '✈', '✊', '✋',
    '✌', '✔', '✨', '❄', '❤', '⭐', '😢', '😞', '😟', '😠', '😡', '😔', '😕',
    '😖', '😨', '😩', '😪', '😫', '😰', '😱', '😳', '😶', '😷', '👊', '👎', '❌',
    '😲', '😯', '😮', '😵', '🙊', '🙉', '🙈', '💭', '❗', '⚡', '🎊', '🙁', '💔',
    '😤', '🔪', '🌕', '🚀', '📉', '🤣', '💸'
]

# Second step: Remove URLs, hashtags, mention, and special characters except for emoticons, and white spaces
def clean_tweet(text):
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove hashtags and mentions
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove special characters except for emoticons
    text = re.sub(r'[^\w\s.!?{}]+'.format(''.join(emoticons_to_keep)), '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

# Apply the modified cleaning function to the 'text' column in your dataset
dataset['text'] = dataset['text'].apply(clean_tweet)

# Display the 'text' column in the entire dataset
print(dataset['text'])

# Third Step: Spell Checker

# Initialize SpellChecker only once to avoid re-creation for each call
spell = SpellChecker()

# Function for spell correction
def spell_correction(text):
    words = text.split()
    corrected_words = []
    for word in words:
        # Check if the word is an emoticon, if so, skip spell checking
        if word not in emoticons_to_keep:
            if word in spell.unknown([word]):
                corrected_word = spell.correction(word)
                corrected_words.append(corrected_word if corrected_word else word)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    return ' '.join(corrected_words)

# Apply spell correction to the entire 'text' column
dataset['text'] = dataset['text'].apply(spell_correction)

# Display the entire dataset
print(dataset)


def remove_punctuations_and_known_emojis(text):
            if isinstance(text, str):  # Check if text is a valid string
                # Define the regex pattern for known emojis
                emoji_pattern = r'(:\)|:\(|:D|😊|😃|😉|👌|👍|😁|😂|😄|😅|😆|😇|😞|😔|😑|😒|😓|😕|😖|💰|📈|🤣|🎊|😭|🙁|💔|😢|😮|😵|🙀|😱|❗|😠|😡|😤|👎|🔪|🌕|🚀|💎|👀|💭|📉|😨|😩|😰|💸)'
                # Construct the regex pattern to remove punctuation except specified characters and emojis
                punctuation_except_specified = r'[^\w\s]'

                # Replace all other punctuation marks except (. ! ?) and known emojis with an empty string
                text = re.sub(punctuation_except_specified + '|' + emoji_pattern, '', text)
                return text
            
def assign_emotion_based_on_polarity(polarity):
    if polarity == 1:  # Positive polarity
        return np.random.choice(['happy', 'surprise', 'anticipation'])
    else:  # Negative polarity
        return np.random.choice(['sad', 'fear', 'angry'])
            
# Apply the defined function to the 'text' column
dataset['text'] = dataset['text'].apply(remove_punctuations_and_known_emojis)
print("Punctuation and known emojis removed from 'text' column.")

# Print the first few rows of the 'text' column after processing
print("Output after removing punctuation and known emojis:")

#Display the entire dataset
print(dataset)

# Function to clean repeating words
def cleaning_repeating_words(text):
    # This regex pattern targets whole words that are repeated
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

# Assuming 'dataset' is a pandas DataFrame and 'text' is a column in it
# Apply the cleaning function for repeating words to each row in the 'text' column
dataset['text'] = dataset['text'].apply(cleaning_repeating_words)
print("Repeating words cleaned from 'text' column.")
print(dataset['text'].head())


stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']


# Stopwords removal applied separately after the option has been chosen and processed
STOPWORDS = set(stopwordlist)
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

# Apply the stopwords cleaning after the loop, once the 'text' column has been updated accordingly
dataset['text'] = dataset['text'].apply(cleaning_stopwords)
print("Stopwords removed from 'text' column.")
print(dataset['text'].head())

#lowercase
dataset['text']=dataset['text'].str.lower()
dataset['text'].head()

# The pattern matches word characters (\w) and punctuation marks ([^\w\s])
tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')

#Tokenizatio
# Applying the modified tokenizer to the dataset
dataset['text'] = dataset['text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
dataset['text'] = dataset['text'].apply(tokenizer.tokenize)
dataset['text'].head()

#Lemmatization
lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data
dataset['text'] = dataset['text'].apply(lambda x: lemmatizer_on_text(x))
dataset['text'].head()


from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
from sklearn.metrics import classification_report
import pickle

# Assuming 'data' is your DataFrame with 'text', 'polarity', and 'emotion' columns
texts = data['text']
polarity_labels = data['polarity']
emotion_labels = data['emotion']

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data_padded = pad_sequences(sequences, maxlen=100)

# Adjusted LSTM Model for Feature Extraction
feature_model = Sequential()
feature_model.add(Embedding(input_dim=10000, output_dim=256, input_length=100))
feature_model.add(LSTM(128, return_sequences=True))
feature_model.add(LSTM(64))  # Last LSTM layer should not return sequences
feature_model.add(Dense(16, activation='relu'))
feature_model.add(Flatten())  # Flatten the output
feature_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
feature_model.fit(data_padded, np.array(polarity_labels), epochs=10, batch_size=64, validation_split=0.1)

# Extract features
features = feature_model.predict(data_padded)

# Save the feature model and tokenizer
feature_model.save("lstm_feature_extractor_text.h5")
with open('tokenizer_text.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save model and tokenizer
output_dir = 'model_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Extract features
features = feature_model.predict(data_padded)

# Splitting the data: 60% training, 30% evaluation, and 10% testing
# For polarity labels
X_train, X_temp, y_polarity_train, y_polarity_temp = train_test_split(features, polarity_labels, test_size=0.4, random_state=42)
X_eval, X_test, y_polarity_eval, y_polarity_test = train_test_split(X_temp, y_polarity_temp, test_size=0.25, random_state=42)

# For emotion labels
X_train, X_temp, y_emotion_train, y_emotion_temp = train_test_split(features, emotion_labels, test_size=0.4, random_state=42)
X_eval, X_test, y_emotion_eval, y_emotion_test = train_test_split(X_temp, y_emotion_temp, test_size=0.25, random_state=42)

# Importing necessary libraries for Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# Grid Search for Polarity SVM
grid_polarity = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid_polarity.fit(X_train, y_polarity_train)
print("Best Polarity SVM Parameters:", grid_polarity.best_params_)

# Grid Search for Emotion SVM
grid_emotion = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid_emotion.fit(X_train, y_emotion_train)
print("Best Emotion SVM Parameters:", grid_emotion.best_params_)

# Save the best SVM models
joblib.dump(grid_polarity.best_estimator_, "svm_polarity_text.pkl")
joblib.dump(grid_emotion.best_estimator_, "svm_emotion_text.pkl")

# Evaluate and visualize the performance of the Polarity SVM Model
y_polarity_pred = grid_polarity.predict(X_test)
print("Polarity Classification Report:")
print(classification_report(y_polarity_test, y_polarity_pred))

# Confusion Matrix for Polarity
cm_polarity = confusion_matrix(y_polarity_test, y_polarity_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_polarity, annot=True, fmt='d')
plt.title('Confusion Matrix for Polarity Classification')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Apply emotion assignment based on polarity predictions
y_emotion_pred = [assign_emotion_based_on_polarity(p) for p in y_polarity_pred]

# Assuming you have true emotion labels in 'y_emotion_test', you can create a classification report
print("Emotion Classification Report:")
print(classification_report(y_emotion_test, y_emotion_pred))

# Since the emotions are assigned based on polarity, a confusion matrix may not be as informative
# However, you can still plot it if you have a mapping between predicted and true emotion labels
# This requires that y_emotion_test contains actual emotion labels corresponding to the dataset

# Mapping predicted emotions to integer labels (if necessary)
emotion_to_int = {'happy': 0, 'surprise': 1, 'anticipation': 2, 'sad': 3, 'fear': 4, 'angry': 5}
y_emotion_pred_int = [emotion_to_int[emotion] for emotion in y_emotion_pred]
y_emotion_test_int = [emotion_to_int[emotion] for emotion in y_emotion_test]  # Replace with actual labels

# Confusion Matrix for Emotion
cm_emotion = confusion_matrix(y_emotion_test_int, y_emotion_pred_int)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_emotion, annot=True, fmt='d')
plt.title('Confusion Matrix for Emotion Assignment')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()