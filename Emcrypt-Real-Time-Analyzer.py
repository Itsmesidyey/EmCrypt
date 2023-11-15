# Utilities
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from textblob import TextBlob 
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from spellchecker import SpellChecker
import string
import emoji
import chardet
import nltk
from nltk.tokenize import RegexpTokenizer

# nltk
from nltk.stem import WordNetLemmatizer

# Your dataset columns and initial setup here
# ...

# Data preprocessing functions
def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)

def clean_tweet(text):
    # Remove URLs, hashtags, mentions, and special characters
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

def spell_correction(text):
    # Spell correction logic
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

def convert_emoticons_to_words(text):
    # Emoticon to word conversion logic
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

def apply_conversion(text):
    # Apply emoticon to word conversion for each row
    # Apply the function and count emoticons for each row
    def apply_conversion(text):
        converted_text, count = convert_emoticons_to_words(text)
        return pd.Series([converted_text, count], index=['converted_text', 'emoticons_count'])

    conversion_results = dataset['text'].apply(apply_conversion)
    dataset['converted_text'] = conversion_results['converted_text']
    dataset['emoticons_count'] = conversion_results['emoticons_count']
    print("Emoticons converted to words in 'converted_text' column.")
    print(dataset[['converted_text', 'emoticons_count']].head())


# Additional preprocessing functions
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

# Function to clean repeating words
def cleaning_repeating_words(text):
    # This regex pattern targets whole words that are repeated
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

# Assuming 'dataset' is a pandas DataFrame and 'text' is a column in it
# Apply the cleaning function for repeating words to each row in the 'text' column
dataset['text'] = dataset['text'].apply(cleaning_repeating_words)
print("Repeating words cleaned from 'text' column.")
print(dataset['text'].head())

dataset['text']=dataset['text'].str.lower()
dataset['text'].head()

# Sentiment analysis and intensity classification functions
def analyze_sentiment(text):
    # Use TextBlob for polarity analysis
    # ...

def classify_intensity(emoticons_count, text):
    # Classify intensity based on certain rules
    # ...

def classify_emotion(text):
    # Analyze emotion using TextBlob
    # ...

# PyQt5 GUI Class and its methods
class Ui_OtherWindow(object):
    # Your existing PyQt5 setup and methods here
    # ...

    def updateTextInTable(self):
        # Get the text from the plain text edit widget
        text = self.plainTextEdit.toPlainText()

        # Apply all preprocessing steps to the input text
        preprocessed_text = cleaning_numbers(text)
        preprocessed_text = clean_tweet(preprocessed_text)
        preprocessed_text = spell_correction(preprocessed_text)
        # Apply other preprocessing functions if necessary

        # Perform sentiment analysis and classify intensity
        sentiment = analyze_sentiment(preprocessed_text)
        emotion = classify_emotion(preprocessed_text)
        intensity = classify_intensity(0, preprocessed_text)  # Assuming no emoticon count

        # Insert processed data into the table
        # ...

# Main application
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    OtherWindow = QtWidgets.QMainWindow()
    ui = Ui_OtherWindow()
    ui.setupUi(OtherWindow)
    OtherWindow.show()
    sys.exit(app.exec_())
