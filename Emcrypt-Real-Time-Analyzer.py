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
    # ...

def spell_correction(text):
    # Spell correction logic
    # ...

def convert_emoticons_to_words(text):
    # Emoticon to word conversion logic
    # ...

def apply_conversion(text):
    # Apply emoticon to word conversion for each row
    # ...

# Additional preprocessing functions
# ...

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
