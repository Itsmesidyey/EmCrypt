import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from textblob import TextBlob 
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Function to clean and preprocess text
def preprocess_text(text):
    # Remove emojis
    text = re.sub(r"(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))", " ", text)
    
    # Remove special characters and punctuation
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Remove repeating characters (e.g., loooove -> love)
    text = re.sub(r"(.)\1{2,}", r"\1", " ".join(filtered_words))
    
    return text

def detect_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F4B0"  # Money Bag
        u"\U0001F4B5"  # Banknote with Dollar Sign
        u"\U0001F4B8"  # Money with Wings
        u"\U0001F4B9"  # Credit Card
        u"\U0001F4B1"  # Currency Exchange
        u"\U0001F4F1"  # Mobile Phone with Dollar Sign
        u"\U0001F4F2"  # Personal Computer with Dollar Sign
        u"\U0001F4F0"  # Money Mouth Face
        u"\U0001F3E6"  # Bank
        u"\U0001F3E8"  # Hotel
        u"\U0001F4F4"  # Currency Exchange
        u"\U0001F4F5"  # Heavy Dollar Sign
        u"\U0001F4F6"  # Credit Card
        u"\U0001F4F7"  # Banknote with Yen Sign
        u"\U0001F4F8"  # Banknote with Euro Sign
        u"\U0001F4F9"  # Banknote with Pound Sign
        u"\U0001F4B4"  # Banknote with Yen Sign
        u"\U0001F4B7"  # Banknote with Pound Sign
        u"\U0001F4B6"  # Banknote with Euro Sign
        u"\U0001F4C8"  # Chart Increasing with Yen Sign
        u"\U0001F4C9"  # Chart Decreasing with Yen Sign
        u"\U0001F4CA"  # Bar Chart with Yen Sign
        u"\U0001F4CB"  # Bar Chart with Dollar Sign
        u"\U0001F4CC"  # Bar Chart with Euro Sign
        u"\U0001F4CD"  # Bar Chart with Pound Sign
        u"\U0001F50E"  # Money-Mouth Face
        u"\U0001F52A"  # Dollar Banknote
        u"\U0001F52B"  # Euro Banknote
        u"\U0001F4BB"  # Dollar Banknote with Wings
        u"\U0001F4C1"  # Dollar Banknote with Coins
        u"\U0001F4C2"  # Yen Banknote
        u"\U0001F4C3"  # Dollar Banknote
        u"\U0001F4C4"  # Euro Banknote
        u"\U0001F4C5"  # Pound Banknote
        u"\U0001F4C6"  # Money Bag
        u"\U0001F4C7"  # Credit Card
        u"\U0001F4C9"  # Chart Increasing
        u"\U0001F4CA"  # Chart Decreasing
        u"\U0001F4E6"  # Package
        u"\U0001F4E7"  # E-Mail
        u"\U0001F4E8"  # Incoming Envelope
        u"\U0001F4E9"  # Envelope with Downwards Arrow Above
        u"\U0001F4EA"  # Closed Mailbox with Lowered Flag
        u"\U0001F4EB"  # Closed Mailbox with Raised Flag
        u"\U0001F4EC"  # Open Mailbox with Raised Flag
        u"\U0001F4ED"  # Open Mailbox with Lowered Flag
        u"\U0001F4EE"  # Postbox
        u"\U0001F4EF"  # Postal Horn
        u"\U0001F4DC"  # Scroll
        u"\U0001F4DD"  # Page with Curl
        u"\U0001F4D6"  # Money with Wings
        u"\U0001F4E2"  # Loudspeaker
        u"\U0001F4E3"  # Right Speaker
        u"\U0001F4E4"  # Right Speaker with One Sound Wave
        u"\U0001F4E5"  # Right Speaker with Three Sound Waves
        u"\U0001F4E6"  # Bullhorn
        u"\U0001F4E7"  # Megaphone
        u"\U0001F4E8"  # Postal Horn
        u"\U0001F4E9"  # Bell
        u"\U0001F4EA"  # Bell with Slash
        u"\U0001F4EB"  # Bookmark
        u"\U0001F4EC"  # Link Symbol
        u"\U0001F4ED"  # Paperclip
        u"\U0001F4EE"  # Linked Paperclips
        u"\U0001F4EF"  # Black Pushpin
        u"\U0001F4F0"  # White Pushpin
        u"\U0001F4F1"  # Round Pushpin
        u"\U0001F4F2"  # Triangular Pushpin
        u"\U0001F4F3"  # Bookmark Tabs
        u"\U0001F4F4"  # Ledger
        u"\U0001F4F5"  # Notebook
        u"\U0001F4F6"  # Notebook with Decorative Cover
        u"\U0001F4F7"  # Closed Book
        u"\U0001F4F8"  # Open Book
        u"\U0001F4F9"  # Green Book
        u"\U0001F4FA"  # Blue Book
        u"\U0001F4FB"  # Orange Book
        u"\U0001F4FC"  # Books
        u"\U0001F4FD"  # Name Badge
        u"\U0001F4FE"  # Scroll
        u"\U0001F4FF"  # Memo
        u"\U0001F52A"  # Closed Book
        u"\U0001F52B"  # Open Book
        u"\U0001F573"  # Newspaper
        u"\U0001F58A"  # Money Bag
        u"\U0001F58B"  # Currency Exchange
        u"\U0001F58C"  # Heavy Dollar Sign
        u"\U0001F58D"  # Credit Card
        u"\U0001F58E"  # Dollar Banknote
        "]+", flags=re.UNICODE)
    return emoji_pattern.findall(text)

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

        text = self.plainTextEdit.toPlainText()

        current_row_count = self.tableWidget.rowCount()
        self.tableWidget.insertRow(current_row_count)
        # Preprocess the text
        text = preprocess_text(text)
        # Detect emojis in the preprocessed text
        emojis = detect_emojis(text)
        # Append detected emojis to the preprocessed text
        text_with_emojis = f"{text} {' '.join(emojis)}"
    
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
