import os
import re
import pandas as pd
import pickle
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from spellchecker import SpellChecker
import joblib
import keras
import numpy as np

class Ui_OtherWindow(object):
    # Initialize class attributes
    def __init__(self):
        try:
            self.polarity_model_combine = joblib.load('svm_polarity_model_combine.pkl')
        except Exception as e:
            print(f"1Error loading models: {e}")
            self.polarity_model_combine = None

        try:
            self.emotion_model_combine = joblib.load('svm_emotion_model_combine.pkl')
        except Exception as e:
            print(f"2Error loading models: {e}")
            self.emotion_model_combine = None

        try:
            self.polarity_model_text = joblib.load('svm_polarity_model_text.pkl')
        except Exception as e:
            print(f"3Error loading models: {e}")
            self.polarity_model_text= None

        try:
            self.emotion_model_text = joblib.load('svm_emotion_model_text.pkl')
        except Exception as e:
            print(f"4Error loading models: {e}")
            self.emotion_model_text = None


        # Initialize SpellChecker
        self.spell = SpellChecker()

        self.emoticon_dict = {
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
    
    def setupUi(self, OtherWindow):
        OtherWindow.setObjectName("OtherWindow")
        OtherWindow.resize(1034, 1086)
        self.centralwidget = QtWidgets.QWidget(OtherWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 1051, 1061))
        self.label.setStyleSheet("background-image: url(:/bgapp/Frame.png);")
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(":/bgapp/Frame.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(140, 469, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("background-color: rgb(126,217,87);\n"
    "color: white;\n"
    "border-radius:10px\n"
    "")
        self.pushButton.setObjectName("pushButton")
        self.radioButton1 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton1.setGeometry(QtCore.QRect(170, 383, 21, 20))
        self.radioButton1.setText("")
        self.radioButton1.setObjectName("radioButton1")
        self.radioButton1.setChecked(True)

        self.radioButton2 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton2.setGeometry(QtCore.QRect(170, 415, 21, 21))
        self.radioButton2.setText("")
        self.radioButton2.setObjectName("radioButton2")

        self.pushButton.setObjectName("pushButton")

        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setGeometry(QtCore.QRect(120, 180, 811, 171))
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(11)
        self.plainTextEdit.setFont(font)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(330, 469, 141, 41))
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("background-color: rgb(249,107,107);\n"
    "color: white;\n"
    "border-radius:10px\n"
    "")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(730, 470, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setStyleSheet("background-color: black;\n"
    "color: white;\n"
    "border-radius:10px\n"
    "")
        self.pushButton_3.setObjectName("pushButton_3")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(120, 560, 811, 421))
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(14)
        self.tableWidget.setFont(font)
        self.tableWidget.setStyleSheet("QTableWidget{\n"
    "background-color: white;\n"
    "color: white;\n"
    "border-radius:10px\n"
    "\n"
    "}\n"
    "QHeaderView::section { background-color:rgb(126,217,87)}\");\n"
    "\n"
    "")
        self.tableWidget.setShowGrid(True)
        self.tableWidget.setGridStyle(QtCore.Qt.CustomDashLine)
            
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(4)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(10)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(10)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(10)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(10)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(3, item)
        self.tableWidget.horizontalHeader().setVisible(True)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(192)
        self.tableWidget.horizontalHeader().setHighlightSections(False)
        self.tableWidget.horizontalHeader().setSortIndicatorShown(False)
        self.tableWidget.verticalHeader().setDefaultSectionSize(40)
        OtherWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(OtherWindow)
        self.statusbar.setObjectName("statusbar")
        OtherWindow.setStatusBar(self.statusbar)

        self.retranslateUi(OtherWindow)
        QtCore.QMetaObject.connectSlotsByName(OtherWindow)

    # Utility methods for text processing
    @staticmethod
    def cleaning_numbers(original_text):
        cleaned_text = re.sub('[0-9]+', '', original_text)
        
        # Print the cleaned text
        print("Text after removing numbers:", cleaned_text)
        
        return cleaned_text
    
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
            '😤', '🔪', '🌕', '🚀', '📉', '🤣', '💸']

    @staticmethod
    def clean_tweet(original_text, emoticons_to_keep):
        text = re.sub(r'https?://\S+|www\.\S+', '', original_text)  # Remove URLs
        text = re.sub(r'@\w+|#\w+', '', text)  # Remove hashtags and mentions
        text = re.sub(r'[^\w\s.!?{}]+'.format(''.join(emoticons_to_keep)), '', text)  # Remove special characters
        cleaned_text = ' '.join(text.split())  # Remove extra whitespace
        
        # Print the cleaned text
        print("Cleaned Text:", cleaned_text)
        
        return cleaned_text
    

    def spell_correction(self, original_text, emoticons_to_keep):
        
        words = original_text.split()
        corrected_words = []
        for word in words:
            if word not in emoticons_to_keep:
                corrected_word = self.spell.correction(word)
                corrected_words.append(corrected_word if corrected_word else word)
            else:
                corrected_words.append(word)
        corrected_text = ' '.join(corrected_words)

        # Print the corrected text
        print("Corrected Text:", corrected_text)

        return corrected_text

    @staticmethod
    def cleaning_stopwords(original_text):
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

        cleaned_text = " ".join([word for word in str(original_text).split() if word not in stopwordlist])

        # Print the text after removing stopwords
        print("Text after removing stopwords:", cleaned_text)

        return cleaned_text

    @staticmethod
    def cleaning_repeating_words(original_text):
        cleaned_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', original_text)
        
        # Print the text after removing repeating words
        print("Text after removing repeating words:", cleaned_text)
        
        return cleaned_text

    @staticmethod
    def stemming_on_text(original_text):
        st = nltk.PorterStemmer()
        stemmed_text = [st.stem(word) for word in original_text.split()]
        
        # Print the text after stemming
        print("Text after stemming:", ' '.join(stemmed_text))
        
        return stemmed_text

    @staticmethod
    def lemmatizer_on_text(original_text):
        lm = nltk.WordNetLemmatizer()
        lemmatized_text = [lm.lemmatize(word) for word in original_text.split()]
        
        # Print the text after lemmatization
        print("Text after lemmatization:", ' '.join(lemmatized_text))
        
        return lemmatized_text

    def classify_intensity(self, emoticons_count, original_text):
        question_marks = original_text.count('?')
        periods = original_text.count('.')
        exclamation_marks = original_text.count('!')

        if exclamation_marks > 1 or question_marks > 1 or emoticons_count > 1:
            return 'High'
        elif periods == 1 or question_marks == 1 or emoticons_count == 1 or exclamation_marks == 1:
            return 'Medium'
        elif question_marks == 0 and emoticons_count == 0:
            return 'Low'
        else:
            return 'Undetermined'
    
    def convert_emoticons_to_words(self, text_no_stopwords):
        text = text_no_stopwords  # Initialize 'text' with 'original_text'
        emoticons_count = 0
        for emoticon, word in self.emoticon_dict.items():
            while emoticon in text:
                text = text.replace(emoticon, word + " ", 1)
                emoticons_count += 1
        return text, emoticons_count


    def remove_punctuations_and_known_emojis(self, original_text):
        if isinstance(original_text, str):  # Check if text is a valid string
            # Define the regex pattern for known emojis
            emoji_pattern = r'(:\)|:\(|:D|😊|😃|😉|👌|👍|😁|😂|😄|😅|😆|😇|😞|😔|😑|😒|😓|😕|😖|💰|📈|🤣|🎊|😭|🙁|💔|😢|😮|😵|🙀|😱|❗|😠|😡|😤|👎|🔪|🌕|🚀|💎|👀|💭|📉|😨|😩|😰|💸)'
            # Construct the regex pattern to remove punctuation except specified characters and emojis
            punctuation_except_specified = r'[^\w\s]'

            # Replace all other punctuation marks except (. ! ?) and known emojis with an empty string
            text = re.sub(punctuation_except_specified + '|' + emoji_pattern, '', original_text)
            return text
        else:
            return original_text
    
    def transform_text_to_features(self, text):
        # Load the tokenizer and LSTM feature extractor model
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        feature_model = keras.models.load_model('lstm_feature_extractor.h5')

        # Preprocess the text (tokenization and padding)
        # Assuming 'text' is a single string of input text
        seq = tokenizer.texts_to_sequences([text])
        padded_seq = keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)  # Use the same maxlen as used during training

        # Use the LSTM model to get the feature vector
        features = feature_model.predict(padded_seq)

        return features

    def updateTextInTable(self):
        original_text = self.plainTextEdit.toPlainText()
        
        # Additional preprocessing
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
            '😤', '🔪', '🌕', '🚀', '📉', '🤣', '💸']

        # Text processing steps
        text_no_numbers = self.cleaning_numbers(original_text)
        text_cleaned = self.clean_tweet(text_no_numbers, self.emoticons_to_keep)  # Use class attribute for emoticons_to_keep
        text_spell_checked = self.spell_correction(text_cleaned, self.emoticons_to_keep)  # Removed the emoticons_to_keep parameter if not used in spell_correction
        text_no_stopwords = self.cleaning_stopwords(text_spell_checked)

        # Check radio button selection
        if self.radioButton1.isChecked():
            print("Option 1")
            converted_text, emoticons_count = self.convert_emoticons_to_words(text_no_stopwords)  # Use the processed text
            # Print the text after coverting
            print("Combination of Keywords, Ending Punctuation Marks, and Emoticons :", ' '.join(converted_text))
        elif self.radioButton2.isChecked():
            print("Option 2")
            # Remove punctuations and known emojis and use the 'text' models
            converted_text = self.remove_punctuations_and_known_emojis(original_text)
            # Print the text after lemmatization
            print("Plain Text Only :", ' '.join(converted_text))

        text_no_repeating_words = self.cleaning_repeating_words(converted_text)
        text_lowercased = text_no_repeating_words.lower()
        
        tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')
        text_tokenized = tokenizer.tokenize(text_lowercased)
        print("Text after Tokenization:", ' '.join(text_tokenized))

        text_stemmed = self.stemming_on_text(' '.join(text_tokenized))
        text_lemmatized = self.lemmatizer_on_text(' '.join(text_stemmed))

        emoticons_count = 0

        # Check radio button selection
        if self.radioButton1.isChecked():
            print("Option 1")
            prepared_text = text_lemmatized    # Use the processed text
            features = self.transform_text_to_features(prepared_text)
            if self.polarity_model_combine is not None:
                polarity_result = self.polarity_model_combine.predict(features)
            else:
                polarity_result = "Model not loaded"

            if self.emotion_model_combine is not None:
                emotion_result = self.emotion_model_combine.predict(features)
            else:
                emotion_result = "Model not loaded"
            intensity_result = self.classify_intensity(emoticons_count, prepared_text) # Assuming classify_intensity requires emoticons_count and text

        elif self.radioButton2.isChecked():
            print("Option 2")
            # Similar processing for radioButton2, if different
            prepared_text = text_lemmatized     # Use the processed text
            features = self.transform_text_to_features(prepared_text)
            if self.polarity_model_text is not None:
                polarity_result = self.polarity_model_text.predict(features)
            else:
                polarity_result = "Model not loaded"

            if self.emotion_model_text is not None:
                emotion_result = self.emotion_model_text.predict(features)
            else:
                emotion_result = "Model not loaded"
            # Assuming classify_intensity requires emoticons_count and text
            intensity_result = self.classify_intensity(emoticons_count, prepared_text)  # Assuming classify_intensity requires emoticons_count and text

        # Convert the NumPy array to a string
        polarity_result_str = np.array2string(polarity_result)
        emotion_result_str = np.array2string(emotion_result)

        # Map polarity_result 0 to 'negative' and 1 to 'positive'
        polarity_result_str = 'negative' if polarity_result == 0 else 'positive'
        # Convert emotion_result to a string
        emotion_result_str = emotion_result[0] if emotion_result else 'unknown'

        # Define a dictionary for emotion mappings
        emotion_mappings = {
            'happy': 'happy',
            'sad': 'sad',
            'fear': 'fear',
            'anticipation': 'anticipation',
            'surprise': 'surprise',
            'angry': 'angry'
        }

        # Get the emotion from the dictionary with a default value of 'unknown'
        emotion_result_str = emotion_mappings.get(emotion_result_str, 'unknown')

        # Updating the table with analysis results
        current_row_count = self.tableWidget.rowCount()
        self.tableWidget.insertRow(current_row_count)

        # Set the original text in the table
        original_text_item = QtWidgets.QTableWidgetItem(original_text)
        original_text_item.setForeground(QtGui.QColor(0, 0, 0))
        font = QtGui.QFont()
        font.setPointSize(8)
        original_text_item.setFont(font)
        self.tableWidget.setItem(current_row_count, 0, original_text_item)

        # Set the analysis result in the table
        analysis_item = QtWidgets.QTableWidgetItem(polarity_result_str)
        analysis_item.setForeground(QtGui.QColor(0, 0, 0))
        analysis_item.setTextAlignment(QtCore.Qt.AlignCenter)
        analysis_item.setFont(font)
        self.tableWidget.setItem(current_row_count, 1, analysis_item)

        # Set the emotion result in the table
        emotion_item = QtWidgets.QTableWidgetItem(emotion_result_str)  # Replace with actual emotion
        emotion_item.setForeground(QtGui.QColor(0, 0, 0))
        emotion_item.setTextAlignment(QtCore.Qt.AlignCenter)
        emotion_item.setFont(font)
        self.tableWidget.setItem(current_row_count, 2, emotion_item)

        # Set the intensity result in the table
        intensity_item = QtWidgets.QTableWidgetItem(intensity_result)
        intensity_item.setForeground(QtGui.QColor(0, 0, 0))
        intensity_item.setTextAlignment(QtCore.Qt.AlignCenter)
        intensity_item.setFont(font)
        self.tableWidget.setItem(current_row_count, 3, intensity_item)

    def clearPlainText(self):
        self.plainTextEdit.setPlainText("")

    def uploadFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(None, "Upload File", "", "Excel Files (*.xlsx);;All Files (*)", options=options)
        if file_name:
            try:
                df = pd.read_excel(file_name)
                self.tableWidget.setRowCount(0)
                for index, row in df.iterrows():
                    text = row['Tweets']  # Assuming 'Tweets' is the column name with the text data
                pass
            except Exception as e:
                print("An error occurred:", e)

    def retranslateUi(self, OtherWindow):
        _translate = QtCore.QCoreApplication.translate
        OtherWindow.setWindowTitle(_translate("OtherWindow", "MainWindow"))
        self.pushButton.setText(_translate("OtherWindow", "Upload File"))
        self.plainTextEdit.setPlainText(_translate("OtherWindow", " Enter the Cryptocurrency related tweets here..."))
        self.pushButton_2.setText(_translate("OtherWindow", "Clear"))
        self.pushButton_3.setText(_translate("OtherWindow", "Evaluate"))
        self.tableWidget.setSortingEnabled(False)
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("OtherWindow", "Tweets"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("OtherWindow", "Polarity"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("OtherWindow", "Emotion"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("OtherWindow", "Intensity"))
                
        self.pushButton.clicked.connect(self.uploadFile)  # Connect the button to the function
        self.pushButton_2.clicked.connect(self.clearPlainText)
                
        self.pushButton_3.clicked.connect(self.updateTextInTable)

import design2

# Main application execution
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    OtherWindow = QtWidgets.QMainWindow()
    ui = Ui_OtherWindow()
    ui.setupUi(OtherWindow)
    OtherWindow.show()
    sys.exit(app.exec_())
