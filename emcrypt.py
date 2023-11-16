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

class Ui_OtherWindow(object):
    # Initialize class attributes
    def __init__(self):
        try:
            self.polarity_model_combine = pickle.load(open('lstm_model_polarity_combine.pkl', 'rb'))
            self.emotion_model_combine = pickle.load(open('svm_model_emotion_combine.pkl', 'rb'))
            self.polarity_model_text = pickle.load(open('lstm_model_polarity_text.pkl', 'rb'))
            self.emotion_model_text = pickle.load(open('svm_model_emotion_text.pkl', 'rb'))
        except Exception as e:
            print("Error loading models:", e)
        
        self.emoticon_dict = {
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
        font.setFamily("Arial")
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
        font.setFamily("Arial")
        font.setPointSize(11)
        self.plainTextEdit.setFont(font)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(330, 469, 141, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
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
        font.setFamily("Arial")
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
        font.setFamily("Arial")
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
        font.setFamily("Arial")
        font.setPointSize(10)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Arial")
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

    # Utility methods for text processing
    @staticmethod
    def cleaning_numbers(original_text):
        cleaned_text = re.sub('[0-9]+', '', original_text)
        
        # Print the cleaned text
        print("Text after removing numbers:", cleaned_text)
        
        return cleaned_text

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

    def analyze_text(self, original_text, emoticons_count):
        # Implement text analysis logic
                # Example:
        polarity_result = "Positive" # Placeholder
        emotion_result = "Happy" # Placeholder
        intensity_result = "High" # Placeholder
        return "Polarity: {}, Emotion: {}, Intensity: {}".format(polarity_result, emotion_result, intensity_result)

    def convert_emoticons_to_words(self, original_text):
        emoticons_count = 0
        for emoticon, word in self.emoticon_dict.items():
            while emoticon in text:
                        text = text.replace(emoticon, word + " ", 1)
                        emoticons_count += 1
        return text, emoticons_count

    def remove_punctuations_and_known_emojis(self, original_text):
        emoji_pattern = re.compile(r'(' + '|'.join(re.escape(key) for key in self.emoticon_dict.keys()) + r')')
        return re.sub(emoji_pattern, '', original_text)

    def updateTextInTable(self):
        original_text = self.plainTextEdit.toPlainText()
        # Additional preprocessing
        emoticons_to_keep = [
            '💰', '📈', '🤣', '🎊', '😂', '😭', '🙁', '😞', '💔', '😢', '😮', '😵', '🙀',
            '😱', '❗', '😠', '😡', '😤', '👎', '🔪', '🌕', '🚀', '💎', '👀', '💭', '📉',
            '😨', '😩', '😰', '💸']

        text_no_numbers = self.cleaning_numbers(original_text)
        text_cleaned = self.clean_tweet(text_no_numbers, emoticons_to_keep)
        text_spell_checked = self.spell_correction(text_cleaned, emoticons_to_keep)
        text_no_stopwords = self.cleaning_stopwords(text_spell_checked)
        text_no_repeating_words = self.cleaning_repeating_words(text_no_stopwords)
        text_lowercased = text_no_repeating_words.lower()
        tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')
        text_tokenized = tokenizer.tokenize(text_lowercased)
        print("Text after Tokenization:", ' '.join(text_tokenized))
        text_stemmed = self.stemming_on_text(' '.join(text_tokenized))
        text_lemmatized = self.lemmatizer_on_text(' '.join(text_stemmed))

        # Check which radio button is selected and process the text accordingly
        emoticons_count = 0
        if self.radioButton1.isChecked():
            # Convert emoticons to words
            converted_text, emoticons_count = self.convert_emoticons_to_words(original_text)
        elif self.radioButton2.isChecked():
            # Remove punctuations and known emojis
            converted_text = self.remove_punctuations_and_known_emojis(original_text)

        # Analyze the text
        analysis_result = self.analyze_text(converted_text, emoticons_count)

        # Calculate intensity
        intensity_result = self.classify_intensity(emoticons_count, original_text)

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
        analysis_item = QtWidgets.QTableWidgetItem(analysis_result)
        analysis_item.setForeground(QtGui.QColor(0, 0, 0))
        analysis_item.setTextAlignment(QtCore.Qt.AlignCenter)
        analysis_item.setFont(font)
        self.tableWidget.setItem(current_row_count, 1, analysis_item)

        # Set the emotion result in the table
        emotion_item = QtWidgets.QTableWidgetItem("Emotion Placeholder")  # Replace with actual emotion
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
                        
                self.retranslateUi(OtherWindow)
                QtCore.QMetaObject.connectSlotsByName(OtherWindow)

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

# Main application execution
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    OtherWindow = QtWidgets.QMainWindow()
    ui = Ui_OtherWindow()
    ui.setupUi(OtherWindow)
    OtherWindow.show()
    sys.exit(app.exec_())
