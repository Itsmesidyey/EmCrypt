import re
import numpy as np
import pandas as pd
# nltk
from nltk.stem import WordNetLemmatizer

# sklearn
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

class Ui_OtherWindow(object):
    def setupUi(self, OtherWindow):
        OtherWindow.setObjectName("OtherWindow")
        OtherWindow.resize(1054, 1086)
        self.centralwidget = QtWidgets.QWidget(OtherWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 1051, 1061))
        self.label.setStyleSheet("background-image: url(:/bgapp/Frame.png);")
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(":/bgapp/Frame.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(110, 580, 821, 411))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(10)
        self.tableWidget.setFont(font)
        self.tableWidget.setAutoFillBackground(False)
        self.tableWidget.setStyleSheet("QTableWidget{\n"
"background-color: white;\n"
"color: white;\n"
"border-radius:10px\n"
"\n"
"}\n"
"QHeaderView::section { background-color:rgb(126,217,87)}\");\n"
"\n"
"")
        self.tableWidget.setAutoScroll(True)
        self.tableWidget.setShowGrid(True)
        self.tableWidget.setRowCount(0)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(4)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(12)
        font.setKerning(True)
        item.setFont(font)
        item.setBackground(QtGui.QColor(255, 255, 255, 0))
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item.setForeground(brush)
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(12)
        item.setFont(font)
        item.setBackground(QtGui.QColor(255, 255, 255, 0))
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item.setForeground(brush)
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setText("Emotion")
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(12)
        item.setFont(font)
        item.setBackground(QtGui.QColor(255, 255, 255, 0))
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item.setForeground(brush)
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(12)
        item.setFont(font)
        item.setBackground(QtGui.QColor(126, 217, 87, 0))
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item.setForeground(brush)
        self.tableWidget.setHorizontalHeaderItem(3, item)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(True)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(201)
        self.tableWidget.horizontalHeader().setHighlightSections(True)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(17)
        self.tableWidget.horizontalHeader().setSortIndicatorShown(True)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.verticalHeader().setCascadingSectionResizes(True)
        self.tableWidget.verticalHeader().setDefaultSectionSize(100)
        self.tableWidget.verticalHeader().setHighlightSections(True)
        self.tableWidget.verticalHeader().setMinimumSectionSize(100)
        self.tableWidget.verticalHeader().setSortIndicatorShown(True)
        self.tableWidget.verticalHeader().setStretchLastSection(True)

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(140, 469, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(11)
        font.setBold(True)
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
        self.pushButton_3.setFont(font)
        self.pushButton_3.setStyleSheet("background-color: black;\n"
"color: white;\n"
"border-radius:10px\n"
"")
        self.pushButton_3.setObjectName("pushButton_3")
        OtherWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(OtherWindow)
        self.statusbar.setObjectName("statusbar")
        OtherWindow.setStatusBar(self.statusbar)

        self.retranslateUi(OtherWindow)
        QtCore.QMetaObject.connectSlotsByName(OtherWindow)

    def retranslateUi(self, OtherWindow):
        _translate = QtCore.QCoreApplication.translate
        OtherWindow.setWindowTitle(_translate("OtherWindow", "MainWindow"))
        self.tableWidget.setWhatsThis(_translate("OtherWindow", "<html><head/><body><p>dcgvdfvbf</p></body></html>"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("OtherWindow", "Tweets"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("OtherWindow", "Polarity"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("OtherWindow", "Intensity"))
        self.pushButton.setText(_translate("OtherWindow", "Upload File"))

        self.pushButton.clicked.connect(self.uploadFile)  # Connect the button to the function
       
        self.plainTextEdit.setPlainText(_translate("OtherWindow", "   Enter the Cryptocurrency related tweets here..."))
        self.pushButton_2.setText(_translate("OtherWindow", "Clear"))
        self.pushButton_3.setText(_translate("OtherWindow", "Evaluate"))
        self.pushButton_2.clicked.connect(self.clearPlainText)

    def clearPlainText(self):
        self.plainTextEdit.setPlainText("") 
    def uploadFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(None, "Upload File", "", "Text Files (*.txt);;All Files (*)", options=options)

        if file_name:
            with open(file_name, 'r') as file:
                content = file.read()
                self.plainTextEdit.setPlainText(content)

import design2



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    OtherWindow = QtWidgets.QMainWindow()
    ui = Ui_OtherWindow()
    ui.setupUi(OtherWindow)
    OtherWindow.show()
    sys.exit(app.exec_())
