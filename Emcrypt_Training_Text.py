#utilities
import re
import numpy as np
import pandas as pd
import nltk

#SpellCorrection
from spellchecker import SpellChecker
from nltk.tokenize import RegexpTokenizer

import chardet
DATASET_COLUMNS = ['text', 'polarity', 'emotion', 'intensity']

#Detect file encoding using chardet
with open('dataset_text.csv', 'rb') as f:
    result = chardet.detect(f.read())

# Print the detected encoding
print("Detected encoding:", result['encoding'])

# Read the file using the detected encoding
df = pd.read_csv('dataset_text.csv', encoding=result['encoding'], names=DATASET_COLUMNS)
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
                emoji_pattern = r'(🌈|🌙|🌚|🌞|🌟|🌷|🌸|🌹|🌺|🍀|🍕|🍻|🎀|🎈|🎉|🎤|🎥|🎧|🎵|🎶|👅|👇|👈|👉|👋|👌|👍|👏|👑|💀|💁|💃|💋|💐|💓|💕|💖|💗|💘|💙|💚|💛|💜|💞|💤|💥|💦|💪|💫|💯|📷|🔥|😀|😁|😃|😄|😅|😆|😇|😈|😉|😊|😋|😌|😍|😎|😏|😺|😻|😽|🙏|☀|☺|♥|✅|✈|✊|✋|✌|✔|✨|❄|❤|⭐|😢|😞|😟|😠|😡|😔|😕|😖|😨|😩|😪|😫|😰|😱|😳|😶|😷|👊|👎|❌|😲|😯|😮|😵|🙊|🙉|🙈|💭|❗|⚡|🎊|🙁|💔|😤|🔪|🌕|🚀|📉|🤣|💸)'
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

# Convert list to string for database insertion
dataset['text'] = dataset['text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

import mysql.connector
from mysql.connector import Error

# Function to connect to MySQL database and insert data
def insert_into_database(data, table_name):
    try:
        connection = mysql.connector.connect(
            host='localhost',  # usually 'localhost' or an IP address
            user='emcrypt',
            password='sentiment123*',
            database= 'emcrypt_database'
        )
        if connection.is_connected():
            cursor = connection.cursor()
            insert_query = f"INSERT INTO {table_name} (text, polarity, emotion, intensity) VALUES (%s, %s, %s, %s)"

            for i, row in data.iterrows():
                # Assuming 'text' column contains the preprocessed and lemmatized text
                text_value = row['text']
                # If 'text' is a list, convert it to string
                if isinstance(text_value, list):
                    text_value = ' '.join(text_value)
                intensity_value = row['intensity'] if 'intensity' in row else None

                cursor.execute(insert_query, (text_value, row['polarity'], row['emotion'], intensity_value))
            
            connection.commit()
            print("Data inserted successfully")
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

# Call the function to insert data into MySQL
insert_into_database(dataset, 'text')

#Sentiment Analysis Stage


#Sentiment Analysis Stage
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import resample
import gc


# Balancing and Shuffling the dataset
data_majority = data[data.polarity == 1]
data_minority = data[data.polarity == 0]

data_minority_upsampled = resample(data_minority, 
                                   replace=True, 
                                   n_samples=len(data_majority),
                                   random_state=123)

data_balanced = pd.concat([data_majority, data_minority_upsampled])
data_balanced = data_balanced.sample(frac=1).reset_index(drop=True)

# Assuming 'data' is your DataFrame with 'text', 'polarity', and 'emotion' columns
# Assuming 'data' is your DataFrame with 'text', 'polarity', and 'emotion' columns
# Preprocess the text data here (if needed)
texts = data['text']
polarity_labels = data['polarity']
emotion_labels = data['emotion']

# Tokenize and pad sequences
# Preprocess the text data
#texts_to_use = data_balanced['text']
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data_padded = pad_sequences(sequences, maxlen=100)

# Convert labels to NumPy arrays for better performance
#polarity_labels = np.array(data_balanced['polarity'])
#emotion_labels = np.array(data_balanced['emotion'])

# LSTM Model for Feature Extraction
model = Sequential()
model.add(Embedding(input_dim=20000, output_dim=256, input_length=100))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
num_emotions = data_balanced['emotion'].nunique()
model.add(Dense(num_emotions, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(data_padded, pd.get_dummies(emotion_labels).values, epochs=15, batch_size=32, validation_split=0.2)  # Adjusted training

# Add callbacks for efficient training
#early_stopping = EarlyStopping(monitor='val_loss', patience=3)
#model_checkpoint = ModelCheckpoint('best_lstm_model_text.h5', save_best_only=True)

#model.fit(data_padded, pd.get_dummies(emotion_labels).values, epochs=15, batch_size=32, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

# Save the LSTM model and tokenizer
model.save("lstm_model_text.h5")

# Save the tokenizer
with open('tokenizer_text.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Create a new model for feature extraction
feature_extraction_model = Model(inputs=model.input, outputs=model.layers[-2].output)
features = feature_extraction_model.predict(data_padded)
print(features.shape)

# Splitting the data for polarity and emotion
X_train, X_temp, y_train_polarity, y_temp_polarity = train_test_split(features, polarity_labels, test_size=0.4, random_state=42)
X_eval_polarity, X_test_polarity, y_eval_polarity, y_test_polarity = train_test_split(X_temp, y_temp_polarity, test_size=0.25, random_state=42)

X_train_emotion, X_temp_emotion, y_train_emotion, y_temp_emotion = train_test_split(features, emotion_labels, test_size=0.4, random_state=42)
X_eval_emotion, X_test_emotion, y_eval_emotion, y_test_emotion = train_test_split(X_temp_emotion, y_temp_emotion, test_size=0.25, random_state=42)

# Define the parameter grid for SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# Grid Search for Polarity and Emotion SVMs with parallel processing
grid_polarity = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, n_jobs=-1)
grid_polarity.fit(X_train, y_train_polarity)
print("Best Polarity SVM Parameters:", grid_polarity.best_params_)

# Grid Search for Emotion SVM
grid_emotion = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid_emotion.fit(X_train_emotion, y_train_emotion)  # Use X_train_emotion for emotion classification

# Save the best SVM models
joblib.dump(grid_polarity.best_estimator_, "svm_polarity_text.pkl")
joblib.dump(grid_emotion.best_estimator_, "svm_emotion_text.pkl")

# Evaluation functions
def evaluate_model(grid, X_test, y_test, title):
    y_pred = grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{title} Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"{title} Model Accuracy: {accuracy * 100:.2f}%\n")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'Confusion Matrix for {title} Classification')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Evaluate and visualize the performance of the Polarity and Emotion SVM Models
evaluate_model(grid_polarity, X_test_polarity, y_test_polarity, "Polarity")
evaluate_model(grid_emotion, X_test_emotion, y_test_emotion, "Emotion")