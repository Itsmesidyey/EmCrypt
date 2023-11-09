
# utilities
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from textblob import TextBlob 
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

# nltk
from nltk.stem import WordNetLemmatizer

#SpellCorrection
from spellchecker import SpellChecker

import string
import emoji
import chardet

DATASET_COLUMNS = ['date', 'username', 'text', 'polarity', 'emotion']

#Detect file encoding using chardet
with open('data.csv', 'rb') as f:
    result = chardet.detect(f.read())

# Print the detected encoding
print("Detected encoding:", result['encoding'])

# Read the file using the detected encoding
df = pd.read_csv('data.csv', encoding=result['encoding'], names=DATASET_COLUMNS)
df.sample(5)

#Data preprocessing
data=df[['text','polarity', 'emotion']]
data['polarity'].unique()
data_pos = data[data['polarity'] == 1]
data_neu = data[data['polarity'] == 0]
data_neg = data[data['polarity'] == -1]
data_pos = data_pos.iloc[:int(100)]
data_neu = data_neu.iloc[:int(100)]
data_neg = data_neg.iloc[:int(100)] 
dataset = pd.concat([data_pos, data_neu, data_neg])

def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
dataset['text'] = dataset['text'].apply(lambda x: cleaning_numbers(x))
dataset['text'].tail()

emoticons_to_keep = [
    '💰', '📈', '🤣', '🎊', '😂', '😭', '🙁', '😞', '💔', '😢', '😮', '😵', '🙀',
    '😱', '❗', '😠', '😡', '😤', '👎', '🔪', '🌕', '🚀', '💎', '👀', '💭', '📉',
    '😨', '😩', '😰', '💸'
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

# Initialize SpellChecker only once to avoid re-creation for each call
spell = SpellChecker()

# Function for spell correction
def spell_correction(text):
    words = text.split()
    misspelled = spell.unknown(words)
    corrected_words = []
    for word in words:
        if word in misspelled:
            corrected_word = spell.correction(word)
            # Check if the correction is not None, otherwise use the original word
            corrected_words.append(corrected_word if corrected_word is not None else word)
        else:
            corrected_words.append(word)
    return ' '.join(corrected_words)

# Apply spell correction to the entire 'text' column
dataset['text'] = dataset['text'].apply(spell_correction)

#Define the emoticon dictionary outside the function for a wider scope
emoticon_dict = {
    ":)": "smile ",
    ":(": "sad ",
    ":D": "laugh ",
    "😊": "smiling face with smiling eyes ",
    "😃": "grinning face with big eyes ",
    "😉": "winking face ",
    "👌": "OK hand ",
    "👍": "Thumbs up ",
    "😁": "beaming face with smiling eyes ",
    "😂": "face with tears of joy ",
    "😄": "grinning face with smiling eyes ",
    "😅": "grinning face with sweat ",
    "😆": "grinning squinting face ",
    "😇": "smiling face with halo ",
    "😞": "disappointed face ",
    "😔": "pensive face ",
    "😑": "expressionless face ",
    "😒": "unamused face ",
    "😓": "downcast face with sweat ",
    "😕": "confused face ",
    "😖": "confounded face ",
    "💰": "Money Bag ",
    "📈": "Up Trend ",
    "🤣": "Rolling on the Floor Laughing ",
    "🎊": "Confetti Ball ",
    "😭": "Loudly Crying ",
    "🙁": "Slightly frowning face ",
    "💔": "Broken Heart ",
    "😢": "Crying Face ",
    "😮": "Face with Open Mouth ",
    "😵": "Dizzy Face ",
    "🙀": "Weary Cat ",
    "😱": "Face Screaming in Fear ",
    "❗": "Exclamation Mark ",
    "😠": "Angry Face ",
    "😡": "Pouting Face ",
    "😤": "Face with Steam from Nose ",
    "👎": "Thumbs Down ",
    "🔪": "Hocho ",
    "🌕": "Moon ",
    "🚀": "Rocket ",
    "💎": "Diamond ",
    "👀": "Eyes ",
    "💭": "Thought Balloon ",
    "📉": "Down Trend ",
    "😨": "Fearful Face ",
    "😩": "Weary Face ",
    "😰": "Anxious Face with Fear ",
    "💸": "Money with Wings "
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



def analyze_sentiment(text):
    # Use TextBlob for polarity analysis
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"
    
def classify_intensity(sentiment_score):
    if sentiment_score > 0.5:
        return "High"
    elif sentiment_score > 0:
        return "Medium"
    else:
        return "Low"   

def classify_emotion(text):
    # Analyze emotion using TextBlob
    blob = TextBlob(text)
    
    # Extract sentiment polarity scores
    sentiment_score = blob.sentiment.polarity
    
    # Define emotion labels and their corresponding sentiment score ranges
    emotion_mapping = {
        "Happy": (0.5, 1.0),    # Positive sentiment
        "Sad": (-1.0, -0.5),   # Negative sentiment
        "Angry": (-0.5, 0.0),  # Negative sentiment
        "Anticipation": (0.0, 0.5),  # Neutral sentiment
        "Eagerness": (0.5, 1.0),    # Positive sentiment
        "Fear": (-1.0, -0.5),       # Negative sentiment
    }
    
    # Determine the emotion based on sentiment score
    detected_emotion = "Neutral"  # Default to neutral
    
    for emotion, (min_score, max_score) in emotion_mapping.items():
        if min_score <= sentiment_score <= max_score:
            detected_emotion = emotion
            break
    
    return detected_emotion

class Ui_OtherWindow(object):
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
        
    def clearPlainText(self):
        self.plainTextEdit.setPlainText("") 
   
    def uploadFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(None, "Upload File", "", "Excel Files (*.xlsx);;All Files (*)", options=options)

        if file_name:
            # Use pandas to read the Excel file
            try:
                df = pd.read_excel(file_name)
                # Clear any previous data in the tableWidget
                self.tableWidget.setRowCount(0)
                
                # Iterate over rows in the Excel file
                for index, row in df.iterrows():
                    text = row['Tweets']  # Assuming 'Tweets' is the column name with the text data

                    current_row_count = self.tableWidget.rowCount()
                    self.tableWidget.insertRow(current_row_count)

                    # Perform sentiment analysis using TextBlob
                    blob = TextBlob(text)
                    sentiment_score = blob.sentiment.polarity

                    # Classify the intensity level
                    intensity = classify_intensity(sentiment_score)
                    #Emotion
                    emotion = classify_emotion(text)

                    # Set the sentiment result in the current row of the table (column 1)
                    sentiment_item = QtWidgets.QTableWidgetItem(analyze_sentiment(text))
                    sentiment_item.setForeground(QtGui.QColor(0, 0, 0))
                    sentiment_item.setTextAlignment(QtCore.Qt.AlignCenter)  # Set text color to black
                    font = QtGui.QFont()
                    font.setPointSize(8)
                    sentiment_item.setFont(font)
                    self.tableWidget.setItem(current_row_count, 1, sentiment_item)

                    # Set the emotion result in the first row of the table (row 0, column 2)
                    emotion_item = QtWidgets.QTableWidgetItem(emotion)
                    emotion_item.setForeground(QtGui.QColor(0, 0, 0))
                    emotion_item.setTextAlignment(QtCore.Qt.AlignCenter)  # Set text color to black
                    emotion_item.setFont(font)
                    self.tableWidget.setItem(current_row_count, 2, emotion_item)

                    # Set the intensity result in the current row of the table (column 3)
                    intensity_item = QtWidgets.QTableWidgetItem(intensity)
                    intensity_item.setForeground(QtGui.QColor(0, 0, 0))
                    intensity_item.setTextAlignment(QtCore.Qt.AlignCenter)  # Set text color to black
                    intensity_item.setFont(font)
                    self.tableWidget.setItem(current_row_count, 3, intensity_item)

                    # Set the text in the current row of the table (column 0)
                    text_item = QtWidgets.QTableWidgetItem(text)
                    text_item.setForeground(QtGui.QColor(0, 0, 0))  # Set text color to black
                    text_item.setFont(font)
                    self.tableWidget.setItem(current_row_count, 0, text_item)

            except Exception as e:
                # Handle any errors that may occur while reading the Excel file
                QtWidgets.QMessageBox.critical(None, "Error", f"An error occurred: {str(e)}")

    def retranslateUi(self, OtherWindow):
        _translate = QtCore.QCoreApplication.translate
        OtherWindow.setWindowTitle(_translate("OtherWindow", "MainWindow"))
        self.pushButton.setText(_translate("OtherWindow", "Upload File"))
        self.plainTextEdit.setPlainText(_translate("OtherWindow", "   Enter the Cryptocurrency related tweets here..."))
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

        
    def updateTextInTable(self):
        # Get the text from the plain text edit widget
        text = self.plainTextEdit.toPlainText()

        current_row_count = self.tableWidget.rowCount()
        self.tableWidget.insertRow(current_row_count)

        # Perform sentiment analysis using TextBlob
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity

        # Classify the intensity level
        intensity = classify_intensity(sentiment_score)
        emotion = classify_emotion(text)


        # Set the sentiment result in the first row of the table (row 0, column 1)
        sentiment_item = QtWidgets.QTableWidgetItem(analyze_sentiment(text))
        sentiment_item.setForeground(QtGui.QColor(0, 0, 0))
        sentiment_item.setTextAlignment(QtCore.Qt.AlignCenter)  # Set text color to black
        font = QtGui.QFont()
        font.setPointSize(8)
        sentiment_item.setFont(font)
        self.tableWidget.setItem(current_row_count, 1, sentiment_item)

        # Set the emotion result in the first row of the table (row 0, column 2)
        emotion_item = QtWidgets.QTableWidgetItem(emotion)
        emotion_item.setForeground(QtGui.QColor(0, 0, 0))
        emotion_item.setTextAlignment(QtCore.Qt.AlignCenter)  # Set text color to black
        emotion_item.setFont(font)
        self.tableWidget.setItem(current_row_count, 2, emotion_item)


        # Set the intensity result in the first row of the table (row 0, column 3)
        intensity_item = QtWidgets.QTableWidgetItem(intensity)
        intensity_item.setForeground(QtGui.QColor(0, 0, 0))
        intensity_item.setTextAlignment(QtCore.Qt.AlignCenter)  # Set text color to black
        intensity_item.setFont(font)
        self.tableWidget.setItem(current_row_count, 3, intensity_item)

        # Set the text in the first row of the table (row 0, column 0)
        text_item = QtWidgets.QTableWidgetItem(text)
        text_item.setForeground(QtGui.QColor(0, 0, 0))  # Set text color to black
        text_item.setFont(font)
        self.tableWidget.setItem(current_row_count, 0, text_item)


import design2


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    OtherWindow = QtWidgets.QMainWindow()
    ui = Ui_OtherWindow()
    ui.setupUi(OtherWindow)
    OtherWindow.show()
    sys.exit(app.exec_())
