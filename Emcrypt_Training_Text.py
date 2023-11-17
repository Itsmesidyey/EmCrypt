#!/usr/bin/env python
# coding: utf-8

# In[1]:


#utilities
import re
import numpy as np
import pandas as pd

#nltk
from nltk.stem import WordNetLemmatizer

#SpellCorrection
from spellchecker import SpellChecker

import string
import emoji


# In[2]:


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


# In[3]:


#Data preprocessing
data=df[['text','polarity', 'emotion']]


# In[4]:


data['polarity'].unique()


# In[5]:


data_pos = data[data['polarity'] == 1]
data_neg = data[data['polarity'] == 0]


# In[6]:


dataset = pd.concat([data_pos, data_neg])


# In[7]:


def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
dataset['text'] = dataset['text'].apply(lambda x: cleaning_numbers(x))
dataset['text'].head()


# In[8]:


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


# In[9]:


from spellchecker import SpellChecker

# Initialize SpellChecker only once to avoid re-creation for each call
spell = SpellChecker()

# List of emoticons to keep
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


# In[10]:


def remove_punctuations_and_known_emojis(text):
            if isinstance(text, str):  # Check if text is a valid string
                # Define the regex pattern for known emojis
                emoji_pattern = r'(:\)|:\(|:D|😊|😃|😉|👌|👍|😁|😂|😄|😅|😆|😇|😞|😔|😑|😒|😓|😕|😖|💰|📈|🤣|🎊|😭|🙁|💔|😢|😮|😵|🙀|😱|❗|😠|😡|😤|👎|🔪|🌕|🚀|💎|👀|💭|📉|😨|😩|😰|💸)'
                # Construct the regex pattern to remove punctuation except specified characters and emojis
                punctuation_except_specified = r'[^\w\s]'

                # Replace all other punctuation marks except (. ! ?) and known emojis with an empty string
                text = re.sub(punctuation_except_specified + '|' + emoji_pattern, '', text)
                return text
            
# Apply the defined function to the 'text' column
dataset['text'] = dataset['text'].apply(remove_punctuations_and_known_emojis)
print("Punctuation and known emojis removed from 'text' column.")

# Print the first few rows of the 'text' column after processing
print("Output after removing punctuation and known emojis:")

#Display the entire dataset
print(dataset)


# In[11]:


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


# In[12]:


# Stopwords removal applied separately after the option has been chosen and processed
STOPWORDS = set(stopwordlist)
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

# Apply the stopwords cleaning after the loop, once the 'text' column has been updated accordingly
dataset['text'] = dataset['text'].apply(cleaning_stopwords)
print("Stopwords removed from 'text' column.")
print(dataset['text'].head())


# In[13]:


# Function to clean repeating words
def cleaning_repeating_words(text):
    # This regex pattern targets whole words that are repeated
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

# Assuming 'dataset' is a pandas DataFrame and 'text' is a column in it
# Apply the cleaning function for repeating words to each row in the 'text' column
dataset['text'] = dataset['text'].apply(cleaning_repeating_words)
print("Repeating words cleaned from 'text' column.")
print(dataset['text'].head())


# In[14]:


dataset['text']=dataset['text'].str.lower()
dataset['text'].head()


# In[15]:


import pandas as pd

# Assuming 'dataset' is your DataFrame

# Replace 'output_file.xlsx' with the desired file name
output_file = 'Feature2_file.xlsx'

# Save the dataset to an Excel file
dataset.to_excel(output_file, index=False)

print(f'Dataset saved to {output_file}')


# In[54]:


from nltk.tokenize import RegexpTokenizer

# The pattern matches word characters (\w) and punctuation marks ([^\w\s])
tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')

# Applying the modified tokenizer to the dataset
dataset['text'] = dataset['text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
dataset['text'] = dataset['text'].apply(tokenizer.tokenize)
dataset['text'].head()


# In[55]:


import nltk
st = nltk.PorterStemmer()
def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return data
dataset['text']= dataset['text'].apply(lambda x: stemming_on_text(x))
dataset['text'].head()


# In[56]:


lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data
dataset['text'] = dataset['text'].apply(lambda x: lemmatizer_on_text(x))
dataset['text'].head()


# In[57]:
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
from sklearn.metrics import classification_report
import numpy as np
import pickle
from keras.layers import Dropout
from keras import regularizers


# Assuming 'data' is your DataFrame with 'text', 'polarity', and 'emotion' columns
texts = data['text']
polarity_labels = data['polarity']
emotion_labels = data['emotion']

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data_padded = pad_sequences(sequences, maxlen=100)

# Adjusting LSTM Model for Feature Extraction
feature_model = Sequential()
feature_model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))  # Increased output_dim
feature_model.add(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))  # Added dropout
feature_model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))  # Adjusted LSTM units
feature_model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))  # Added regularization
feature_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
feature_model.fit(data_padded, polarity_labels, epochs=15, batch_size=64, validation_split=0.1)  # Adjusted epochs, batch size, and added validation split


# Save the feature model and tokenizer
feature_model.save("lstm_feature_extractor.h5")
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Extract features
features = feature_model.predict(data_padded)

# Splitting the data: 60% training, 30% evaluation, and 10% testing
X_train, X_temp, y_polarity_train, y_polarity_temp = train_test_split(features, polarity_labels, test_size=0.4, random_state=42)
X_eval, X_test, y_polarity_eval, y_polarity_test = train_test_split(X_temp, y_polarity_temp, test_size=0.25, random_state=42)
X_train, X_temp, y_emotion_train, y_emotion_temp = train_test_split(features, emotion_labels, test_size=0.4, random_state=42)
X_eval, X_test, y_emotion_eval, y_emotion_test = train_test_split(X_temp, y_emotion_temp, test_size=0.25, random_state=42)

# Train SVM for Polarity
svm_polarity = SVC(kernel='linear')
svm_polarity.fit(X_train, y_polarity_train)

# Train SVM for Emotion
svm_emotion = SVC(kernel='linear')
svm_emotion.fit(X_train, y_emotion_train)

from sklearn.model_selection import GridSearchCV

# Tuning SVM for Polarity
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly']}
grid_polarity = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid_polarity.fit(X_train, y_polarity_train)

# Tuning SVM for Emotion
grid_emotion = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid_emotion.fit(X_train, y_emotion_train)

# Save the best SVM models
joblib.dump(grid_polarity.best_estimator_, "svm_polarity_text.pkl")
joblib.dump(grid_emotion.best_estimator_, "svm_emotion_text.pkl")




# In[ ]:
# Evaluate Polarity SVM Model
y_polarity_pred = grid_polarity.predict(X_test)
print("Polarity Classification Report:")
print(classification_report(y_polarity_test, y_polarity_pred))

# Evaluate Emotion SVM Model
y_emotion_pred = grid_emotion.predict(X_test)
print("Emotion Classification Report:")
print(classification_report(y_emotion_test, y_emotion_pred))