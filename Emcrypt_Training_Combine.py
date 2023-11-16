#!/usr/bin/env python
# coding: utf-8

# In[30]:


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


# In[31]:


import chardet
DATASET_COLUMNS = ['date', 'username', 'text', 'polarity', 'emotion']

#Detect file encoding using chardet
with open('Emcrypt-dataset.csv', 'rb') as f:
    result = chardet.detect(f.read())

# Print the detected encoding
print("Detected encoding:", result['encoding'])

# Read the file using the detected encoding
df = pd.read_csv('Emcrypt-dataset.csv', encoding=result['encoding'], names=DATASET_COLUMNS)
df.sample(10)


# In[32]:


#Data preprocessing
data=df[['text','polarity', 'emotion']]


# In[33]:


data['polarity'].unique()


# In[34]:


data_pos = data[data['polarity'] == 1]
data_neg = data[data['polarity'] == 0]


# In[35]:


dataset = pd.concat([data_pos, data_neg])


# In[36]:


def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
dataset['text'] = dataset['text'].apply(lambda x: cleaning_numbers(x))
dataset['text'].head()


# In[38]:


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


# In[39]:


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


# In[ ]:


#Define the emoticon dictionary outside the function for a wider scope
emoticon_dict = {
    "🌈": "Rainbow",
    "🌙": "Crescent Moon",
    "🌚": "New Moon Face",
    "🌞": "Sun with Face",
    "🌟": "Glowing Star",
    "🌷": "Tulip",
    "🌸": "Cherry Blossom",
    "🌹": "Rose",
    "🌺": "Hibiscus",
    "🍀": "Four Leaf Clover",
    "🍕": "Pizza",
    "🍻": "Clinking Beer Mugs",
    "🎀": "Ribbon",
    "🎈": "Balloon",
    "🎉": "Party Popper",
    "🎤": "Microphone",
    "🎥": "Movie Camera",
    "🎧": "Headphone",
    "🎵": "Musical Note",
    "🎶": "Musical Notes",
    "👀": "Eyes",
    "👅": "Tongue",
    "👇": "Backhand Index Pointing Down",
    "👈": "Backhand Index Pointing Left",
    "👉": "Backhand Index Pointing Right",
    "👋": "Waving Hand",
    "👌": "OK Hand",
    "👍": "Thumbs Up",
    "👏": "Clapping Hands",
    "👑": "Crown",
    "💀": "Skull",
    "💁": "Person Tipping Hand",
    "💃": "Woman Dancing",
    "💋": "Kiss Mark",
    "💎": "Gem Stone",
    "💐": "Bouquet",
    "💓": "Beating Heart",
    "💕": "Two Hearts",
    "💖": "Sparkling Heart",
    "💗": "Growing Heart",
    "💘": "Heart with Arrow",
    "💙": "Blue Heart",
    "💚": "Green Heart",
    "💛": "Yellow Heart",
    "💜": "Purple Heart",
    "💞": "Revolving Hearts",
    "💤": "Zzz",
    "💥": "Collision",
    "💦": "Sweat Droplets",
    "💪": "Flexed Biceps",
    "💫": "Dizzy",
    "💯": "Hundred Points",
    "💰": "Money Bag",
    "📷": "Camera",
    "🔥": "Fire",
    "😀": "Grinning Face",
    "😁": "Beaming Face with Smiling Eyes",
    "😂": "Face with Tears of Joy",
    "😃": "Grinning Face with Big Eyes",
    "😄": "Grinning Face with Smiling Eyes",
    "😅": "Grinning Face with Sweat",
    "😆": "Grinning Squinting Face",
    "😇": "Smiling Face with Halo",
    "😈": "Smiling Face with Horns",
    "😉": "Winking Face",
    "😊": "Smiling Face with Smiling Eyes",
    "😋": "Face Savoring Food",
    "😌": "Relieved Face",
    "😍": "Smiling Face with Heart-Eyes",
    "😎": "Smiling Face with Sunglasses",
    "😏": "Smirking Face",
    "😺": "Smiling Cat with Smiling Eyes",
    "😻": "Smiling Cat with Heart-Eyes",
    "😽": "Kissing Cat with Closed Eyes",
    "🙀": "Weary Cat",
    "🙏": "Folded Hands",
    "☀": "Sun",
    "☺": "Smiling Face",
    "♥": "Heart Suit",
    "✅": "Check Mark Button",
    "✈": "Airplane",
    "✊": "Raised Fist",
    "✋": "Raised Hand",
    "✌": "Victory Hand",
    "✔": "Check Mark",
    "✨": "Sparkles",
    "❄": "Snowflake",
    "❤": "Red Heart",
    "⭐": "Star",
    "😢": "Crying Face",
    "😭": "Loudly Crying Face",
    "😞": "Disappointed Face",
    "😟": "Worried Face",
    "😠": "Angry Face",
    "😡": "Pouting Face",
    "😔": "Pensive Face",
    "😕": "Confused Face",
    "😖": "Confounded Face",
    "😨": "Fearful Face",
    "😩": "Weary Face",
    "😪": "Sleepy Face",
    "😫": "Tired Face",
    "😰": "Anxious Face with Sweat",
    "😱": "Face Screaming in Fear",
    "😳": "Flushed Face",
    "😶": "Face Without Mouth",
    "😷": "Face with Medical Mask",
    "👊": "Oncoming Fist",
    "👎": "Thumbs Down",
    "❌": "Cross Mark",
    "😲": "Astonished Face",
    "😯": "Hushed Face",
    "😮": "Face with Open Mouth",
    "😵": "Dizzy Face",
    "🙊": "Speak-No-Evil Monkey",
    "🙉": "Hear-No-Evil Monkey",
    "🙈": "See-No-Evil Monkey",
    "💭": "Thought Balloon",
    "❗": "Exclamation Mark",
    "⚡": "High Voltage",
    "🎊": "Confetti Ball",
    "🙁": "Slightly frowning face",
    "💔": "Broken Heart",
    "😤": "Face with Steam from Nose",
    "🔪": "Hocho",
    "🌕": "Full Moon",
    "🚀": "Rocket",
    "📉": "Down Trend",
    "🤣": "Rolling on the Floor Laughing",
    "💸": "Money with Wings"
}

# Emoticon to word conversion function
def convert_emoticons_to_words(text):
    changed_emoticons = 0  # Variable to count the number of changed emoticons
    for emoticon, word in emoticon_dict.items():
        while emoticon in text:
            text = text.replace(emoticon, word + " ", 1)
            changed_emoticons += 1
    return text, changed_emoticons

# Apply the function and count emoticons for each row
def apply_conversion(text):
    converted_text, count = convert_emoticons_to_words(text)
    return pd.Series([converted_text, count], index=['converted_text', 'emoticons_count'])

conversion_results = dataset['text'].apply(apply_conversion)
dataset['converted_text'] = conversion_results['converted_text']
dataset['emoticons_count'] = conversion_results['emoticons_count']
print("Emoticons converted to words in 'converted_text' column.")
print(dataset[['converted_text', 'emoticons_count']].head())


# In[40]:


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


# In[42]:


# Stopwords removal applied separately after the option has been chosen and processed
STOPWORDS = set(stopwordlist)
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

# Apply the stopwords cleaning after the loop, once the 'text' column has been updated accordingly
dataset['text'] = dataset['text'].apply(cleaning_stopwords)
print("Stopwords removed from 'text' column.")
print(dataset['text'].head())


# In[44]:


# Function to clean repeating words
def cleaning_repeating_words(text):
    # This regex pattern targets whole words that are repeated
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

# Assuming 'dataset' is a pandas DataFrame and 'text' is a column in it
# Apply the cleaning function for repeating words to each row in the 'text' column
dataset['text'] = dataset['text'].apply(cleaning_repeating_words)
print("Repeating words cleaned from 'text' column.")
print(dataset['text'].head())


# In[45]:


dataset['text']=dataset['text'].str.lower()
dataset['text'].head()


# In[46]:


import pandas as pd

# Assuming 'dataset' is your DataFrame

# Replace 'output_file.xlsx' with the desired file name
output_file = 'Feature1_file.xlsx'

# Save the dataset to an Excel file
dataset.to_excel(output_file, index=False)

print(f'Dataset saved to {output_file}')


# In[25]:


from nltk.tokenize import RegexpTokenizer

# The pattern matches word characters (\w) and punctuation marks ([^\w\s])
tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')

# Applying the modified tokenizer to the dataset
dataset['text'] = dataset['text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
dataset['text'] = dataset['text'].apply(tokenizer.tokenize)
dataset['text'].head()


# In[26]:


import nltk
st = nltk.PorterStemmer()
def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return data
dataset['text']= dataset['text'].apply(lambda x: stemming_on_text(x))
dataset['text'].head()


# In[27]:


lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data
dataset['text'] = dataset['text'].apply(lambda x: lemmatizer_on_text(x))
dataset['text'].head()


# In[29]:


import pandas as pd

# Assuming 'dataset' is your DataFrame

# Replace 'output_file.xlsx' with the desired file name
output_file = 'Feature2_file.xlsx'

# Save the dataset to an Excel file
dataset.to_excel(output_file, index=False)

print(f'Dataset saved to {output_file}')


# In[19]:


from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# Assuming `data` is your dataset with 'text', 'polarity', and 'emotion' columns
texts = data['text']
polarity_labels = data['polarity']
emotion_labels = data['emotion']

# Splitting the dataset into training, testing, and evaluation sets
X_train, X_temp, y_polarity_train, y_polarity_temp = train_test_split(
    texts, polarity_labels, test_size=0.4, random_state=42)
_, _, y_emotion_train, y_emotion_temp = train_test_split(
    texts, emotion_labels, test_size=0.4, random_state=42)

# Further splitting the temporary sets into test and evaluation sets
X_test, X_eval, y_polarity_test, y_polarity_eval = train_test_split(
    X_temp, y_polarity_temp, test_size=0.25, random_state=42)  # 0.25 * 0.4 = 0.1
_, _, y_emotion_test, y_emotion_eval = train_test_split(
    X_temp, y_emotion_temp, test_size=0.25, random_state=42)



# Tokenize words
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_eval_seq = tokenizer.texts_to_sequences(X_eval)

# Pad sequences to ensure uniform input size
max_seq_length = max([len(x) for x in X_train_seq])  # Get the length of the longest sequence
X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_seq_length)
X_eval_pad = pad_sequences(X_eval_seq, maxlen=max_seq_length)

# LSTM model for polarity classification
def create_lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=64, input_length=max_seq_length))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(32))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification for polarity

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

lstm_model_polarity = KerasClassifier(build_fn=create_lstm_model, epochs=10, batch_size=32, verbose=1)

# SVM model for emotion classification
svm_model_emotion = SVC(probability=True, kernel='linear', verbose=1)

# Train the models
lstm_model_polarity.fit(X_train_pad, y_polarity_train)
svm_model_emotion.fit(X_train_pad, y_emotion_train)

# Save the models
joblib.dump(lstm_model_polarity, "lstm_model_polarity_combine.pkl")
joblib.dump(svm_model_emotion, "svm_model_emotion_combine.pkl")


# In[ ]:

# Evaluate the models on the test set
y_pred_polarity = lstm_model_polarity.predict(X_test_pad)
# If y_pred_polarity is in a continuous format (e.g., probabilities), you may need to convert it to discrete labels
y_pred_polarity_labels = (y_pred_polarity > 0.5).astype(int)

y_pred_emotion = svm_model_emotion.predict(X_test_pad)

print("Polarity Classification Report:")
print(classification_report(y_polarity_test, y_pred_polarity_labels))

print("Emotion Classification Report:")
print(classification_report(y_emotion_test, y_pred_emotion))

# Evaluation on the evaluation set
y_pred_eval_polarity = lstm_model_polarity.predict(X_eval_pad)
# Convert to discrete labels if necessary
y_pred_eval_polarity_labels = (y_pred_eval_polarity > 0.5).astype(int)

y_pred_eval_emotion = svm_model_emotion.predict(X_eval_pad)

print("Polarity Evaluation Set Classification Report:")
print(classification_report(y_polarity_eval, y_pred_eval_polarity_labels))

print("Emotion Evaluation Set Classification Report:")
print(classification_report(y_emotion_eval, y_pred_eval_emotion))


# In[ ]:




