import re
import pandas as pd
import pickle
import joblib
import keras
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from spellchecker import SpellChecker
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QLabel, QDialog, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QPlainTextEdit
from keras.models import Model

class ClearablePlainTextEdit(QPlainTextEdit):
    def __init__(self, parent=None):
        super(ClearablePlainTextEdit, self).__init__(parent)
        self.placeholder_text = " Enter the Cryptocurrency related tweets here..."
        self.setPlainText(self.placeholder_text)
        self.setMaximumLength(280)  # Set the maximum length to 280 characters

    def focusInEvent(self, event):
        if self.toPlainText() == self.placeholder_text:
            self.setPlainText("")
        super(ClearablePlainTextEdit, self).focusInEvent(event)

    def keyPressEvent(self, event):
        if len(self.toPlainText()) < self.maximumLength() or event.key() == QtCore.Qt.Key_Backspace:
            super(ClearablePlainTextEdit, self).keyPressEvent(event)
        else:
            event.ignore()  # Ignore the key press if it exceeds the limit

    def setMaximumLength(self, length):
        self._maximumLength = length

    def maximumLength(self):
        return self._maximumLength

class Ui_OtherWindow(object):
    # Initialize class attributes
    def __init__(self):
        try:
            self.polarity_model_combine = joblib.load('svm_polarity.pkl')
            self.emotion_model_combine = joblib.load('svm_emotion.pkl')
            self.polarity_model_text = joblib.load('svm_polarity_text.pkl')
            self.emotion_model_text = joblib.load('svm_emotion_text.pkl')
        except Exception as e:
            print(f"Error loading models: {e}")

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
        "🍃": "Leaf Fluttering in Wind",
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
        "👊": "Oncoming Fist",
        "👋": "Waving Hand",
        "👌": "OK Hand",
        "👍": "Thumbs Up",
        "👎": "Thumbs Down",
        "👏": "Clapping Hands",
        "👑": "Crown",
        "👻": "Ghost",
        "💀": "Skull",
        "💁": "Person Tipping Hand",
        "💃": "Woman Dancing",
        "💋": "Kiss Mark",
        "💎": "Gem Stone",
        "💐": "Bouquet",
        "💓": "Beating Heart",
        "💔": "Broken Heart",
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
        "💩": "Pile of Poo",
        "💪": "Flexed Biceps",
        "💫": "Dizzy",
        "💭": "Thought Balloon",
        "💯": "Hundred Points",
        "💰": "Money Bag",
        "📷": "Camera",
        "🔞": "No One Under Eighteen",
        "🔥": "Fire",
        "🔫": "Pistol",
        "🔴": "Red Circle",
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
        "😐": "Neutral Face",
        "😑": "Expressionless Face",
        "😒": "Unamused Face",
        "😓": "Downcast Face with Sweat",
        "😔": "Pensive Face",
        "😕": "Confused Face",
        "😖": "Confounded Face",
        "😘": "Face Blowing a Kiss",
        "😙": "Kissing Face with Smiling Eyes",
        "😚": "Kissing Face with Closed Eyes",
        "😛": "Face with Tongue",
        "😜": "Winking Face with Tongue",
        "😝": "Squinting Face with Tongue",
        "😞": "Disappointed Face",
        "😟": "Worried Face",
        "😠": "Angry Face",
        "😡": "Pouting Face",
        "😢": "Crying Face",
        "😣": "Persevering Face",
        "😤": "Face with Steam from Nose",
        "😥": "Sad but Relieved Face",
        "😨": "Fearful Face",
        "😩": "Weary Face",
        "😪": "Sleepy Face",
        "😫": "Tired Face",
        "😬": "Grimacing Face",
        "😭": "Loudly Crying Face",
        "😰": "Anxious Face with Sweat",
        "😱": "Face Screaming in Fear",
        "😳": "Flushed Face",
        "😴": "Sleeping Face",
        "😶": "Face Without Mouth",
        "😷": "Face with Medical Mask",
        "😹": "Cat with Tears of Joy",
        "😻": "Smiling Cat with Heart-Eyes",
        "🙅": "Person Gesturing NO",
        "🙆": "Person Gesturing OK",
        "🙈": "See-No-Evil Monkey",
        "🙉": "Hear-No-Evil Monkey",
        "🙊": "Speak-No-Evil Monkey",
        "🙋": "Person Raising Hand",
        "🙌": "Raising Hands",
        "🙏": "Folded Hands",
        "‼": "Double Exclamation Mark",
        "↩": "Right Arrow Curving Left",
        "↪": "Left Arrow Curving Right",
        "▶": "Play Button",
        "◀": "Reverse Button",
        "☀": "Sun",
        "☑": "Check Box with Check",
        "☝": "Index Pointing Up",
        "☺": "Smiling Face",
        "♥": "Heart Suit",
        "♻": "Recycling Symbol",
        "⚡": "High Voltage",
        "⚽": "Soccer Ball",
        "✅": "Check Mark Button",
        "✈": "Airplane",
        "✊": "Raised Fist",
        "✋": "Raised Hand",
        "✌": "Victory Hand",
        "✔": "Check Mark",
        "✨": "Sparkles",
        "❄": "Snowflake",
        "❌": "Cross Mark",
        "❗": "Exclamation Mark",
        "❤": "Red Heart",
        "⭐": "Star",
        "😲": "Astonished Face",
        "😯": "Hushed Face",
        "😮": "Face with Open Mouth",
        "😵": "Dizzy Face",
        "💭": "Thought Balloon",
        "❗": "Exclamation Mark",
        "⚡": "High Voltage",
        "🎊": "Confetti Ball",
        "🙁": "Slightly Frowning Face",
        "🔪": "Hocho",
        "🌕": "Full Moon",
        "🚀": "Rocket",
        "📉": "Down Trend",
        "🤣": "Rolling on the Floor Laughing",
        "💸": "Money with Wings"
}

        self.emoticon_weights = {
            '🌈': {'angry': 0.0, 'anticipation': 0.28, 'fear': 0.0, 'happy': 0.69, 'sad': 0.06, 'surprise': 0.22 },
            '🌙': {'angry': 0.0, 'anticipation': 0.31, 'fear': 0.0, 'happy': 0.25, 'sad': 0.0, 'surprise': 0.06},
            '🌚': {'angry': 0.06, 'anticipation': 0.08, 'fear': 0.06, 'happy': 0.42, 'sad': 0.19, 'surprise': 0.06},
            '🌞': {'angry': 0.0, 'anticipation': 0.22, 'fear': 0.0, 'happy': 0.78, 'sad': 0.0, 'surprise': 0.11},
            '🌟': {'angry': 0.0, 'anticipation': 0.28, 'fear': 0.0, 'happy': 0.53, 'sad': 0.0, 'surprise': 0.25},
            '🌷': {'angry': 0.0, 'anticipation': 0.31, 'fear': 0.0, 'happy': 0.44, 'sad': 0.0, 'surprise': 0.0},
            '🌸': {'angry': 0.0, 'anticipation': 0.22, 'fear': 0.0, 'happy': 0.56, 'sad': 0.0, 'surprise': 0.14},
            '🌹': {'angry': 0.0, 'anticipation': 0.36, 'fear': 0.0, 'happy': 0.56, 'sad': 0.0, 'surprise': 0.11},
            '🌺': {'angry': 0.0, 'anticipation': 0.11, 'fear': 0.0, 'happy': 0.39, 'sad': 0.0, 'surprise': 0.06},
            '🍀': {'angry': 0.0, 'anticipation': 0.39, 'fear': 0.0, 'happy': 0.47, 'sad': 0.0, 'surprise': 0.22},
            '🍃': {'angry': 0.0, 'anticipation': 0.31, 'fear': 0.0, 'happy': 0.11, 'sad': 0.17, 'surprise': 0.03},
            '🍕': {'angry': 0.06, 'anticipation': 0.39, 'fear': 0.06, 'happy': 0.47, 'sad': 0.06, 'surprise': 0.17},
            '🍻': {'angry': 0.0, 'anticipation': 0.44, 'fear': 0.0, 'happy': 0.72, 'sad': 0.0, 'surprise': 0.25},
            '🎀': {'angry': 0.0, 'anticipation': 0.42, 'fear': 0.0, 'happy': 0.44, 'sad': 0.0, 'surprise': 0.36},
            '🎈': {'angry': 0.06, 'anticipation': 0.25, 'fear': 0.06, 'happy': 0.47, 'sad': 0.06, 'surprise': 0.31},
            '🎉': {'angry': 0.0, 'anticipation': 0.33, 'fear': 0.0, 'happy': 0.92, 'sad': 0.0, 'surprise': 0.5},
            '🎤': {'angry': 0.0, 'anticipation': 0.39, 'fear': 0.06, 'happy': 0.39, 'sad': 0.08, 'surprise': 0.08},
            '🎥': {'angry': 0.0, 'anticipation': 0.28, 'fear': 0.0, 'happy': 0.19, 'sad': 0.0, 'surprise': 0.17},
            '🎧': {'angry': 0.0, 'anticipation': 0.08, 'fear': 0.0, 'happy': 0.44, 'sad': 0.0, 'surprise': 0.0},
            '🎵': {'angry': 0.0, 'anticipation': 0.25, 'fear': 0.0, 'happy': 0.47, 'sad': 0.08, 'surprise': 0.08},
            '🎶': {'angry': 0.0, 'anticipation': 0.22, 'fear': 0.0, 'happy': 0.47, 'sad': 0.0, 'surprise': 0.22},
            '👀': {'angry': 0.14, 'anticipation': 0.81, 'fear': 0.42, 'happy': 0.0, 'sad': 0.17, 'surprise': 0.64},
            '👅': {'angry': 0.0, 'anticipation': 0.17, 'fear': 0.0, 'happy': 0.36, 'sad': 0.0, 'surprise': 0.08},
            '👇': {'angry': 0.11, 'anticipation': 0.14, 'fear': 0.06, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.0},
            '👈': {'angry': 0.14, 'anticipation': 0.17, 'fear': 0.0, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.06},
            '👉': {'angry': 0.06, 'anticipation': 0.25, 'fear': 0.0, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.03},
            '👊': {'angry': 0.44, 'anticipation': 0.36, 'fear': 0.11, 'happy': 0.22, 'sad': 0.0, 'surprise': 0.0},
            '👋': {'angry': 0.08, 'anticipation': 0.28, 'fear': 0.0, 'happy': 0.22, 'sad': 0.0, 'surprise': 0.08},
            '👌': {'angry': 0.0, 'anticipation': 0.36, 'fear': 0.0, 'happy': 0.22, 'sad': 0.0, 'surprise': 0.22},
            '👍': {'angry': 0.11, 'anticipation': 0.39, 'fear': 0.08, 'happy': 0.39, 'sad': 0.06, 'surprise': 0.14},
            '👎': {'angry': 0.5, 'anticipation': 0.08, 'fear': 0.14, 'happy': 0.0, 'sad': 0.31, 'surprise': 0.14},
            '👏': {'angry': 0.08, 'anticipation': 0.39, 'fear': 0.0, 'happy': 0.64, 'sad': 0.0, 'surprise': 0.25},
            '👑': {'angry': 0.0, 'anticipation': 0.25, 'fear': 0.0, 'happy': 0.28, 'sad': 0.0, 'surprise': 0.11},
            '👻': {'angry': 0.11, 'anticipation': 0.08, 'fear': 0.69, 'happy': 0.0, 'sad': 0.11, 'surprise': 0.31},
            '💀': {'angry': 0.19, 'anticipation': 0.14, 'fear': 0.61, 'happy': 0.03, 'sad': 0.31, 'surprise': 0.06},
            '💁': {'angry': 0.08, 'anticipation': 0.33, 'fear': 0.06, 'happy': 0.14, 'sad': 0.06, 'surprise': 0.17},
            '💃': {'angry': 0.0, 'anticipation': 0.11, 'fear': 0.0, 'happy': 0.69, 'sad': 0.0, 'surprise': 0.17},
            '💋': {'angry': 0.0, 'anticipation': 0.28, 'fear': 0.0, 'happy': 0.78, 'sad': 0.0, 'surprise': 0.19},
            '💎': {'angry': 0.0, 'anticipation': 0.31, 'fear': 0.06, 'happy': 0.33, 'sad': 0.0, 'surprise': 0.25},
            '💐': {'angry': 0.0, 'anticipation': 0.39,  'fear': 0.0, 'happy': 0.69, 'sad': 0.11, 'surprise': 0.36},
            '💓': {'angry': 0.0, 'anticipation': 0.47,  'fear': 0.08, 'happy': 0.61, 'sad': 0.0, 'surprise': 0.19},
            '💔': {'angry': 0.39, 'anticipation': 0.19,  'fear': 0.14, 'happy': 0.0, 'sad': 0.94, 'surprise': 0.08},
            '💕': {'angry': 0.0, 'anticipation': 0.31, 'fear': 0.0, 'happy': 0.83, 'sad': 0.0, 'surprise': 0.11},
            '💖': {'angry': 0.0, 'anticipation': 0.33, 'fear': 0.0, 'happy': 0.89, 'sad': 0.0, 'surprise': 0.25},
            '💗': {'angry': 0.0, 'anticipation': 0.36, 'fear': 0.0, 'happy': 0.89, 'sad': 0.0, 'surprise': 0.22},
            '💘': {'angry': 0.03, 'anticipation': 0.31, 'fear': 0.06, 'happy': 0.67, 'sad': 0.14, 'surprise': 0.06},
            '💙': {'angry': 0.0, 'anticipation': 0.25, 'fear': 0.0, 'happy': 0.61, 'sad': 0.17, 'surprise': 0.17},
            '💚': {'angry': 0.0, 'anticipation': 0.11, 'fear': 0.0, 'happy': 0.58, 'sad': 0.03, 'surprise': 0.03},
            '💛': {'angry': 0.03, 'anticipation': 0.11, 'fear': 0.0, 'happy': 0.53, 'sad': 0.08, 'surprise': 0.08},
            '💜': {'angry': 0.0, 'anticipation': 0.11, 'fear': 0.06, 'happy': 0.47, 'sad': 0.11, 'surprise': 0.08},
            '💞': {'angry': 0.0, 'anticipation': 0.25,'fear': 0.0, 'happy': 0.83, 'sad': 0.0, 'surprise': 0.22},
            '💤': {'angry': 0.06, 'anticipation': 0.36, 'fear': 0.06, 'happy': 0.11, 'sad': 0.14, 'surprise': 0.06},
            '💥': {'angry': 0.44, 'anticipation': 0.19, 'fear': 0.31, 'happy': 0.11, 'sad': 0.14, 'surprise': 0.31},
            '💦': {'angry': 0.0, 'anticipation': 0.11, 'fear': 0.06, 'happy': 0.0, 'sad': 0.14, 'surprise': 0.0},
            '💩': {'angry': 0.14, 'anticipation': 0.08, 'fear': 0.0, 'happy': 0.25, 'sad': 0.03, 'surprise': 0.19},
            '💪': {'angry': 0.03, 'anticipation': 0.31, 'fear': 0.0, 'happy': 0.42, 'sad': 0.0, 'surprise': 0.08},
            '💫': {'angry': 0.0, 'anticipation': 0.19, 'fear': 0.06, 'happy': 0.44, 'sad': 0.0, 'surprise': 0.19},
            '💭': {'angry': 0.11, 'anticipation': 0.64, 'fear': 0.11, 'happy': 0.17, 'sad': 0.11, 'surprise': 0.17},
            '💯': {'angry': 0.06, 'anticipation': 0.28, 'fear': 0.06, 'happy': 0.64, 'sad': 0.06, 'surprise': 0.19},
            '💰': {'angry': 0.0, 'anticipation': 0.58,  'fear': 0.06, 'happy': 0.47, 'sad': 0.06, 'surprise': 0.25},
            '📷': {'angry': 0.0, 'anticipation': 0.19, 'fear': 0.0, 'happy': 0.14, 'sad': 0.0, 'surprise': 0.08},
            '🔞': {'angry': 0.11, 'anticipation': 0.11,  'fear': 0.03, 'happy': 0.08, 'sad': 0.11, 'surprise': 0.0},
            '🔥': {'angry': 0.47, 'anticipation': 0.22, 'fear': 0.17, 'happy': 0.25, 'sad': 0.11, 'surprise': 0.39},
            '🔫': {'angry': 0.44, 'anticipation': 0.14, 'fear': 0.14, 'happy': 0.03, 'sad': 0.14, 'surprise': 0.0},
            '🔴': {'angry': 0.08, 'anticipation': 0.06, 'fear': 0.11, 'happy': 0.0, 'sad': 0.03, 'surprise': 0.19},
            '😀': {'angry': 0.06, 'anticipation': 0.22, 'fear': 0.06, 'happy': 0.69, 'sad': 0.06, 'surprise': 0.14},
            '😁': {'angry': 0.06, 'anticipation': 0.25,  'fear': 0.08, 'happy': 0.89, 'sad': 0.06, 'surprise': 0.33},
            '😂': {'angry': 0.0, 'anticipation': 0.17, 'fear': 0.06, 'happy': 0.94, 'sad': 0.0, 'surprise': 0.33},
            '😃': {'angry': 0.0, 'anticipation': 0.31, 'fear': 0.06, 'happy': 0.83, 'sad': 0.0, 'surprise': 0.33},
            '😄': {'angry': 0.0, 'anticipation': 0.36, 'fear': 0.0, 'happy': 0.86, 'sad': 0.0, 'surprise': 0.28},
            '😅': {'angry': 0.08, 'anticipation': 0.44, 'fear': 0.28, 'happy': 0.42, 'sad': 0.06, 'surprise': 0.36},
            '😆': {'angry': 0.06, 'anticipation': 0.19, 'fear': 0.06, 'happy': 0.94, 'sad': 0.06, 'surprise': 0.25},
            '😇': {'angry': 0.0, 'anticipation': 0.31, 'fear': 0.0, 'happy': 0.72, 'sad': 0.0, 'surprise': 0.17},
            '😈': {'angry': 0.14, 'anticipation': 0.44, 'fear': 0.19, 'happy': 0.33, 'sad': 0.08, 'surprise': 0.03},
            '😉': {'angry': 0.0, 'anticipation': 0.42, 'fear': 0.0, 'happy': 0.44, 'sad': 0.08, 'surprise': 0.28},
            '😊': {'angry': 0.0, 'anticipation': 0.42, 'fear': 0.0, 'happy': 0.92, 'sad': 0.0, 'surprise': 0.33},
            '😋': {'angry': 0.0, 'anticipation': 0.47, 'fear': 0.0, 'happy': 0.78, 'sad': 0.0, 'surprise': 0.19},
            '😌': {'angry': 0.0, 'anticipation': 0.33, 'fear': 0.11, 'happy': 0.81, 'sad': 0.0, 'surprise': 0.22},
            '😍': {'angry': 0.0, 'anticipation': 0.31, 'fear': 0.0, 'happy': 0.83, 'sad': 0.0, 'surprise': 0.5},
            '😎': {'angry': 0.0, 'anticipation': 0.22, 'fear': 0.0, 'happy': 0.75, 'sad': 0.0, 'surprise': 0.06},
            '😏': {'angry': 0.22, 'anticipation': 0.33,  'fear': 0.14, 'happy': 0.22, 'sad': 0.22, 'surprise': 0.11},
            '😐': {'angry': 0.14, 'anticipation': 0.33,  'fear': 0.17, 'happy': 0.06, 'sad': 0.25, 'surprise': 0.31},
            '😑': {'angry': 0.28, 'anticipation': 0.22, 'fear': 0.14, 'happy': 0.0, 'sad': 0.33, 'surprise': 0.19},
            '😒': {'angry': 0.58, 'anticipation': 0.14, 'fear': 0.17, 'happy': 0.0, 'sad': 0.42, 'surprise': 0.11},
            '😓': {'angry': 0.19, 'anticipation': 0.44, 'fear': 0.64, 'happy': 0.0, 'sad': 0.36, 'surprise': 0.17},
            '😔': {'angry': 0.25, 'anticipation': 0.22, 'fear': 0.28, 'happy': 0.0, 'sad': 0.72, 'surprise': 0.19},
            '😕': {'angry': 0.19, 'anticipation': 0.42, 'fear': 0.36, 'happy': 0.0, 'sad': 0.39, 'surprise': 0.28},
            '😖': {'angry': 0.22, 'anticipation': 0.36, 'fear': 0.5, 'happy': 0.08, 'sad': 0.53, 'surprise': 0.11},
            '😘': {'angry': 0.0, 'anticipation': 0.33, 'fear': 0.0, 'happy': 0.72, 'sad': 0.0, 'surprise': 0.17},
            '😙': {'angry': 0.0, 'anticipation': 0.47, 'fear': 0.0, 'happy': 0.83, 'sad': 0.0, 'surprise': 0.17},
            '😚': {'angry': 0.0, 'anticipation': 0.44, 'fear': 0.0, 'happy': 0.86, 'sad': 0.0, 'surprise': 0.22},
            '😛': {'angry': 0.0, 'anticipation': 0.31, 'fear': 0.03, 'happy': 0.69, 'sad': 0.0, 'surprise': 0.28},
            '😜': {'angry': 0.0, 'anticipation': 0.42, 'fear': 0.06, 'happy': 0.64, 'sad': 0.0, 'surprise': 0.28},
            '😝': {'angry': 0.0, 'anticipation': 0.22, 'fear': 0.08, 'happy': 0.83, 'sad': 0.0, 'surprise': 0.22},
            '😞': {'angry': 0.39, 'anticipation': 0.19, 'fear': 0.33, 'happy': 0.0, 'sad': 0.92, 'surprise': 0.06},
            '😟': {'angry': 0.25, 'anticipation': 0.44, 'fear': 0.72, 'happy': 0.0, 'sad': 0.69, 'surprise': 0.17},
            '😠': {'angry': 1.0, 'anticipation': 0.17, 'fear': 0.17, 'happy': 0.0, 'sad': 0.25, 'surprise': 0.11},
            '😡': {'angry': 1.0, 'anticipation': 0.11, 'fear': 0.11, 'happy': 0.0, 'sad': 0.36, 'surprise': 0.08},
            '😢': {'angry': 0.25, 'anticipation': 0.08, 'fear': 0.5, 'happy': 0.0, 'sad': 1.0, 'surprise': 0.08},
            '😣': {'angry': 0.31, 'anticipation': 0.28, 'fear': 0.47, 'happy': 0.0, 'sad': 0.64, 'surprise': 0.0},
            '😤': {'angry': 0.75, 'anticipation': 0.11, 'fear': 0.14, 'happy': 0.0, 'sad': 0.25, 'surprise': 0.03},
            '😥': {'angry': 0.14, 'anticipation': 0.19, 'fear': 0.33, 'happy': 0.03, 'sad': 0.81, 'surprise': 0.08},
            '😨': {'angry': 0.17, 'anticipation': 0.39, 'fear': 0.97, 'happy': 0.0, 'sad': 0.56, 'surprise': 0.39},
            '😩': {'angry': 0.33, 'anticipation': 0.25, 'fear': 0.47, 'happy': 0.0, 'sad': 0.75, 'surprise': 0.14},
            '😪': {'angry': 0.11, 'anticipation': 0.08, 'fear': 0.28, 'happy': 0.0, 'sad': 0.64, 'surprise': 0.06},
            '😫': {'angry': 0.36, 'anticipation': 0.14, 'fear': 0.17, 'happy': 0.11, 'sad': 0.72, 'surprise': 0.06},
            '😬': {'angry': 0.14, 'anticipation': 0.53, 'fear': 0.44, 'happy': 0.17, 'sad': 0.11, 'surprise': 0.25},
            '😭': {'angry': 0.22, 'anticipation': 0.08, 'fear': 0.33, 'happy': 0.0, 'sad': 1.0, 'surprise': 0.08},
            '😰': {'angry': 0.22, 'anticipation': 0.31, 'fear': 0.83, 'happy': 0.0, 'sad': 0.69, 'surprise': 0.08},
            '😱': {'angry': 0.28, 'anticipation': 0.42,  'fear': 0.92, 'happy': 0.06, 'sad': 0.25, 'surprise': 0.69},
            '😳': {'angry': 0.06, 'anticipation': 0.36, 'fear': 0.5, 'happy': 0.14, 'sad': 0.19, 'surprise': 0.44},
            '😴': {'angry': 0.0, 'anticipation': 0.06, 'fear': 0.0, 'happy': 0.03, 'sad': 0.03, 'surprise': 0.0},
            '😶': {'angry': 0.06, 'anticipation': 0.22, 'fear': 0.36, 'happy': 0.0, 'sad': 0.14, 'surprise': 0.19},
            '😷': {'angry': 0.03, 'anticipation': 0.17, 'fear': 0.5, 'happy': 0.0, 'sad': 0.22, 'surprise': 0.03},
            '😹': {'angry': 0.0, 'anticipation': 0.17, 'fear': 0.0, 'happy': 0.94, 'sad': 0.0, 'surprise': 0.14},
            '😻': {'angry': 0.0, 'anticipation': 0.42, 'fear': 0.0, 'happy': 0.75, 'sad': 0.06, 'surprise': 0.33},
            '🙅': {'angry': 0.47, 'anticipation': 0.25, 'fear': 0.33, 'happy': 0.06, 'sad': 0.33, 'surprise': 0.11},
            '🙆': {'angry': 0.03, 'anticipation': 0.33, 'fear': 0.0, 'happy': 0.39, 'sad': 0.0, 'surprise': 0.03},
            '🙈': {'angry': 0.0, 'anticipation': 0.39, 'fear': 0.17, 'happy': 0.28, 'sad': 0.03, 'surprise': 0.5},
            '🙊': {'angry': 0.06, 'anticipation': 0.44, 'fear': 0.47, 'happy': 0.14, 'sad': 0.08, 'surprise': 0.42},
            '🙋': {'angry': 0.0, 'anticipation': 0.53, 'fear': 0.0, 'happy': 0.44, 'sad': 0.0, 'surprise': 0.19},
            '🙌': {'angry': 0.0, 'anticipation': 0.33, 'fear': 0.0, 'happy': 0.72, 'sad': 0.0, 'surprise': 0.39},
            '🙏': {'angry': 0.06, 'anticipation': 0.44, 'fear': 0.11, 'happy': 0.25, 'sad': 0.11, 'surprise': 0.17},
            '‼': {'angry': 0.44, 'anticipation': 0.42, 'fear': 0.06, 'happy': 0.14, 'sad': 0.0, 'surprise': 0.89},
            '↩': {'angry': 0.0, 'anticipation': 0.06, 'fear': 0.0, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.0},
            '↪': {'angry': 0.06, 'anticipation': 0.19, 'fear': 0.0, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.0},
            '▶': {'angry': 0.0, 'anticipation': 0.08, 'fear': 0.0, 'happy': 0.03, 'sad': 0.0, 'surprise': 0.0},
            '◀': {'angry': 0.0, 'anticipation': 0.06, 'fear': 0.0, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.06},
            '☀': {'angry': 0.0, 'anticipation': 0.22, 'fear': 0.0, 'happy': 0.44, 'sad': 0.0, 'surprise': 0.06},
            '☑': {'angry': 0.0, 'anticipation': 0.22, 'fear': 0.0, 'happy': 0.25, 'sad': 0.0, 'surprise': 0.0},
            '☝': {'angry': 0.11, 'anticipation': 0.31, 'fear': 0.11, 'happy': 0.03, 'sad': 0.11, 'surprise': 0.0},
            '☺': {'angry': 0.0, 'anticipation': 0.42, 'fear': 0.0, 'happy': 1.0, 'sad': 0.0, 'surprise': 0.39},
            '♥': {'angry': 0.0, 'anticipation': 0.28, 'fear': 0.0, 'happy': 0.72, 'sad': 0.0, 'surprise': 0.11},
            '♻': {'angry': 0.03, 'anticipation': 0.19, 'fear': 0.0, 'happy': 0.03, 'sad': 0.03, 'surprise': 0.0},
            '⚡': {'angry': 0.28, 'anticipation': 0.31, 'fear': 0.25, 'happy': 0.08, 'sad': 0.0, 'surprise': 0.36},
            '⚽': {'angry': 0.0, 'anticipation': 0.33, 'fear': 0.06, 'happy': 0.25, 'sad': 0.0, 'surprise': 0.0},
            '✅': {'angry': 0.0, 'anticipation': 0.31, 'fear': 0.0, 'happy': 0.19, 'sad': 0.0, 'surprise': 0.0},
            '✈': {'angry': 0.0, 'anticipation': 0.44, 'fear': 0.11, 'happy': 0.28, 'sad': 0.11, 'surprise': 0.19},
            '✊': {'angry': 0.25, 'anticipation': 0.5, 'fear': 0.11, 'happy': 0.03, 'sad': 0.11, 'surprise': 0.08},
            '✋': {'angry': 0.22, 'anticipation': 0.25, 'fear': 0.11, 'happy': 0.06, 'sad': 0.06, 'surprise': 0.08},
            '✌': {'angry': 0.0, 'anticipation': 0.42, 'fear': 0.0, 'happy': 0.61, 'sad': 0.0, 'surprise': 0.17},
            '✔': {'angry': 0.0, 'anticipation': 0.25, 'fear': 0.0, 'happy': 0.14, 'sad': 0.0, 'surprise': 0.0,},
            '✨': {'angry': 0.0, 'anticipation': 0.36, 'fear': 0.06, 'happy': 0.53, 'sad': 0.0, 'surprise': 0.44,},
            '❄': {'angry': 0.11, 'anticipation': 0.33, 'fear': 0.17, 'happy': 0.28, 'sad': 0.14, 'surprise': 0.22,},
            '❌': {'angry': 0.5, 'anticipation': 0.14, 'fear': 0.25, 'happy': 0.0, 'sad': 0.31, 'surprise': 0.08,},
            '❗': {'angry': 0.44, 'anticipation': 0.42, 'fear': 0.42, 'happy': 0.08, 'sad': 0.17, 'surprise': 0.81,},
            '❤': {'angry': 0.0, 'anticipation': 0.36, 'fear': 0.0, 'happy': 0.69, 'sad': 0.0, 'surprise': 0.14,},
            '➡': {'angry': 0.0, 'anticipation': 0.06, 'fear': 0.0, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.0,},
            '⬅': {'angry' : 0.17, 'anticipation' : 0.14, 'fear' : 0.14, 'happy' :0.0, 'sad': 0.14, 'surprise': 0.03},
            '⭐': { 'angry': 0.0, 'anticipation' : 0.17, 'fear' : 0.0, 'happy' :	0.39, 'sad' : 0.0, 'surprise' :	0.17},
            "😲": { 'angry': 0.0, 'anticipation': 0.33, 'fear': 0.33, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.67 },
            "😯": { 'angry': 0.0, 'anticipation': 0.25, 'fear': 0.25, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.50 },
            "😮": { 'angry': 0.0, 'anticipation': 0.40, 'fear': 0.20, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.60 },
            "😵": { 'angry': 0.0, 'anticipation': 0.0, 'fear': 0.50, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.50 },
            "❗": { 'angry': 0.25, 'anticipation': 0.50, 'fear': 0.25, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.75 },
            "⚡": { 'angry': 0.2, 'anticipation': 0.4, 'fear': 0.3, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.5 },
            "🎊": { 'angry': 0.0, 'anticipation': 0.6, 'fear': 0.0, 'happy': 0.8, 'sad': 0.0, 'surprise': 0.7 },
            "🙁": { 'angry': 0.2, 'anticipation': 0.0, 'fear': 0.1, 'happy': 0.0, 'sad': 0.7, 'surprise': 0.1 },
            "🔪": { 'angry': 0.4, 'anticipation': 0.2, 'fear': 0.6, 'happy': 0.0, 'sad': 0.1, 'surprise': 0.2 },
            "🌕": { 'angry': 0.0, 'anticipation': 0.3, 'fear': 0.0, 'happy': 0.4, 'sad': 0.0, 'surprise': 0.3 },
            "🚀": { 'angry': 0.0, 'anticipation': 0.7, 'fear': 0.1, 'happy': 0.6, 'sad': 0.0, 'surprise': 0.5 },
            "📉": { 'angry': 0.3, 'anticipation': 0.1, 'fear': 0.4, 'happy': 0.0, 'sad': 0.7, 'surprise': 0.2 },
            "🤣": { 'angry': 0.0, 'anticipation': 0.2, 'fear': 0.0, 'happy': 1.0, 'sad': 0.0, 'surprise': 0.3 },
            "💸": { 'angry': 0.2, 'anticipation': 0.5, 'fear': 0.1, 'happy': 0.3, 'sad': 0.4, 'surprise': 0.4 }
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

        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(120, 560, 811, 421)) #Set the geometry as needed
        self.scrollArea.setWidgetResizable(True)

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

        self.plainTextEdit = ClearablePlainTextEdit(self.centralwidget)
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

        # Create and set up the QScrollArea
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(120, 560, 811, 421))  # Adjust size and position as needed
        self.scrollArea.setWidgetResizable(True)

        # Create the QTableWidget
        self.tableWidget = QtWidgets.QTableWidget()
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setObjectName("tableWidget")
        
        # Set the headers
        headers = ["Tweets", "Polarity", "Emotion", "Intensity"]
        self.tableWidget.setHorizontalHeaderLabels(headers)

        # Change font size of headers
        font = QFont()
        font.setPointSize(12)  # Set your desired font size here
        self.tableWidget.horizontalHeader().setFont(font)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(203)
        self.tableWidget.verticalHeader().setDefaultSectionSize(50)

        # Set header font color to white
        header_stylesheet = "QHeaderView::section { bbackground-color:rgb(126,217,87); color: #000000; }"
        self.tableWidget.horizontalHeader().setStyleSheet(header_stylesheet)

        # Table Widget font and style settings
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(16)
        self.tableWidget.setFont(font)
        self.tableWidget.setStyleSheet("QTableWidget{\n"
                                    "background-color: white;\n"
                                    "color: black;\n"  # Changed color to black for visibility
                                    "border-radius:20px\n"
                                    "}\n"
                                    "QHeaderView::section { background-color:rgb(126,217,87)}\");\n")

        self.tableWidget.setShowGrid(True)
        self.tableWidget.setGridStyle(QtCore.Qt.CustomDashLine)


        # Add the QTableWidget to the QScrollArea
        self.scrollArea.setWidget(self.tableWidget)


        self.iconButton = QtWidgets.QPushButton(self.centralwidget)
        self.iconButton.setGeometry(QtCore.QRect(980, 10, 40, 40))  # Position in the upper right corner
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("/Users/cjcasinsinan/Documents/GitHub/EmCrypt/assets/manual-icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.iconButton.setIcon(icon)
        self.iconButton.setIconSize(QtCore.QSize(30, 30))  # Set size of the icon
        self.iconButton.setObjectName("iconButton")

        # Connect the icon button to the function to show the pop-up
        self.iconButton.clicked.connect(self.showPopup)

        OtherWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(OtherWindow)
        self.statusbar.setObjectName("statusbar")
        OtherWindow.setStatusBar(self.statusbar)

        self.retranslateUi(OtherWindow)
        QtCore.QMetaObject.connectSlotsByName(OtherWindow)

    # Utility methods for text pre-processing

    @staticmethod
    def cleaning_numbers(original_text):
        cleaned_text = re.sub('[0-9]+', '', original_text)
        
        # Print the cleaned text
        print("Text after removing numbers: ", cleaned_text) 
        print("\n")
        
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
        print("\n")

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
        print("\n")
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
        print("\n")
        return cleaned_text

    @staticmethod
    def cleaning_repeating_words(original_text):
        cleaned_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', original_text)
        
        # Print the text after removing repeating words
        print("Text after removing repeating words:", cleaned_text)
        print("\n")

        return cleaned_text

    @staticmethod
    def lemmatizer_on_text(original_text):
        lm = nltk.WordNetLemmatizer()
        lemmatized_text = [lm.lemmatize(word) for word in original_text.split()]
        
        # Print the text after lemmatization
        print("Text after lemmatization:", ' '.join(lemmatized_text))
        print("\n")
        return lemmatized_text

    def classify_intensity(self, emotion_result_str, emoticons_count, prepared_text):
        question_marks = prepared_text.count('?')
        periods = prepared_text.count('.')
        exclamation_marks = prepared_text.count('!')

        # Happy
        if emotion_result_str == 'Happy':
            if exclamation_marks >= 1 or emoticons_count > 2:
                return 'High'
            elif periods == 1 and emoticons_count == 1:
                return 'Medium'
            elif question_marks == 0:
                return 'Low'

        # Sadness
        elif emotion_result_str == 'Sad':
            if exclamation_marks >= 1 and question_marks > 1 or emoticons_count > 1:
                return 'High'
            elif periods == 1 and question_marks == 1 and emoticons_count <= 1:
                return 'Medium'
            elif question_marks == 0:
                return 'Low'

        # Surprise
        elif emotion_result_str == 'Surprise':
            if exclamation_marks >= 1 and question_marks > 1 or emoticons_count > 2:
                return 'High'
            elif periods == 1 and question_marks == 1 and exclamation_marks == 1 and emoticons_count <= 2:
                return 'Medium'
            elif question_marks == 0:
                return 'Low'

        # Anger
        elif emotion_result_str == 'Angry':
            if exclamation_marks >= 1 and question_marks > 1 or emoticons_count > 2:
                return 'High'
            elif periods == 1 and question_marks == 1 and exclamation_marks == 1 and emoticons_count <= 2:
                return 'Medium'
            elif question_marks == 0:
                return 'Low'

        # Anticipation
        elif emotion_result_str == 'Anticipation':
            if exclamation_marks >= 1 and question_marks > 1 or emoticons_count > 2:
                return 'High'
            elif periods == 1 and question_marks == 1 and exclamation_marks == 1 and emoticons_count <= 2:
                return 'Medium'
            elif question_marks == 0:
                return 'Low'

        # Fear
        elif emotion_result_str == 'Fear':
            if exclamation_marks >= 1 and question_marks > 1 or emoticons_count > 2:
                return 'High'
            elif periods == 1 and question_marks == 1 and exclamation_marks == 1 and emoticons_count <= 2:
                return 'Medium'
            elif question_marks == 0:
                return 'Low'

        return 'Low'

    
    def convert_emoticons_to_words(self, text_no_stopwords):
        text = text_no_stopwords  # Initialize 'text' with 'original_text'
        emoticons_count = 0
        for emoticon, word in self.emoticon_dict.items():
            while emoticon in text:
                text = text.replace(emoticon, word + " ", 1)
                emoticons_count += 1
        return text, emoticons_count
    

    def remove_punctuations_and_known_emojis(self, text_no_stopwords):
        if isinstance(text_no_stopwords, str):  # Check if text is a valid string
            # Define the regex pattern for known emojis
            emoji_pattern = r'(🌈|🌙|🌚|🌞|🌟|🌷|🌸|🌹|🌺|🍀|🍕|🍻|🎀|🎈|🎉|🎤|🎥|🎧|🎵|🎶|👅|👇|👈|👉|👋|👌|👍|👏|👑|💀|💁|💃|💋|💐|💓|💕|💖|💗|💘|💙|💚|💛|💜|💞|💤|💥|💦|💪|💫|💯|📷|🔥|😀|😁|😃|😄|😅|😆|😇|😈|😉|😊|😋|😌|😍|😎|😏|😺|😻|😽|🙏|☀|☺|♥|✅|✈|✊|✋|✌|✔|✨|❄|❤|⭐|😢|😞|😟|😠|😡|😔|😕|😖|😨|😩|😪|😫|😰|😱|😳|😶|😷|👊|👎|❌|😲|😯|😮|😵|🙊|🙉|🙈|💭|❗|⚡|🎊|🙁|💔|😤|🔪|🌕|🚀|📉|🤣|💸)'
            # Construct the regex pattern to remove punctuation except specified characters and emojis
            punctuation_except_specified = r'[^\w\s]'

            # Replace all other punctuation marks except (. ! ?) and known emojis with an empty string
            text = re.sub(punctuation_except_specified + '|' + emoji_pattern, '', text_no_stopwords)
            return text
        else:
            return text
        
    def assign_emotion_based_on_polarity(self, polarity):
        if polarity == 1:  # Positive polarity
            return np.random.choice(['Happy', 'Surprise', 'Anticipation'])
        else:  # Negative polarity
            return np.random.choice(['Sad', 'Fear', 'Angry'])
    
    def transform_text_to_features(self, text):
    # Check radio button selection
        if self.radioButton1.isChecked():
            print("The Feature that you use is the Proposed System")
            print("\n")
            features = self.extract_features_from_lstm(text, 'lstm_model.h5', 'tokenizer.pkl')
        elif self.radioButton2.isChecked():
            print("The Feature that you use is using the Plain-Text Only")
            print("\n")
            features = self.extract_features_from_lstm(text, 'lstm_model_text.h5', 'tokenizer_text.pkl')

        # Transform features to 32 dimensions
        transformed_features = self.adjust_features_to_expected_dim(features, 7)
        return transformed_features

    def extract_features_from_lstm(self, text, lstm_model_path, tokenizer_path):
        # Load the tokenizer and LSTM model
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        lstm_model = keras.models.load_model(lstm_model_path)

        # Preprocess the text
        seq = tokenizer.texts_to_sequences([text])
        data_padded = keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)

        # Create a feature extraction model
        feature_extraction_model = Model(inputs=lstm_model.input, outputs=lstm_model.layers[-2].output)

        # Extract features
        features = feature_extraction_model.predict(data_padded)
        return features

    def adjust_features_to_expected_dim(self, features, expected_dim):
        # Transform the features to match the expected dimensionality
        if features.shape[1] > expected_dim:
            return features[:, :expected_dim]
        elif features.shape[1] < expected_dim:
            additional_features = np.zeros((features.shape[0], expected_dim - features.shape[1]))
            return np.concatenate((features, additional_features), axis=1)
        else:
            return features

    def validate_svm_model(self, svm_model_path):
        svm_model = joblib.load(svm_model_path)
        print(f"{svm_model_path} expects {svm_model.n_features_in_} features.")
        return svm_model
    
    def showInputWarning(self):
        # Create and set up the pop-up dialog
        msgBox = QtWidgets.QMessageBox()
        msgBox.setIcon(QtWidgets.QMessageBox.Warning)
        msgBox.setText("Please input a tweet")
        msgBox.setWindowTitle("Input Required")
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msgBox.exec_()

    def updateTextInTable(self):
        original_text = self.plainTextEdit.toPlainText()

        # Text processing steps
        print("<---------- Pre-processing Stage ---------->")
        print("\n")
        text_no_numbers = self.cleaning_numbers(original_text)
        text_cleaned = self.clean_tweet(text_no_numbers, self.emoticons_to_keep)  # Use class attribute for emoticons_to_keep
        text_spell_checked = self.spell_correction(text_cleaned, self.emoticons_to_keep)  # Removed the emoticons_to_keep parameter if not used in spell_correction

        # Check radio button selection
        if self.radioButton1.isChecked():
            converted_text, emoticons_count = self.convert_emoticons_to_words(text_spell_checked)  # Use the processed text
             # Convert and Calculate
            # Assuming 'convert_and_calculate' is a method in your class and 'text_lemmatized' is the final processed text
        

            # Print the text after coverting
            print("Combination of Keywords, Ending Punctuation Marks, and Emoticons :", ' '.join(converted_text))
            print("\n")

        elif self.radioButton2.isChecked():
            # Remove punctuations and known emojis and use the 'text' models
            converted_text = self.remove_punctuations_and_known_emojis(text_spell_checked)

            # Print the text after lemmatization
            print("Plain Text Only :", ' '.join(converted_text))
            print("\n")

        text_no_repeating_words = self.cleaning_repeating_words(converted_text)
        text_no_stopwords = self.cleaning_stopwords(text_no_repeating_words)
        text_lowercased = text_no_stopwords.lower()
        tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')
        text_tokenized = tokenizer.tokenize(text_lowercased)
        print("Text after Tokenization: ", ' '.join(text_tokenized))
        print("\n")

        text_lemmatized = self.lemmatizer_on_text(' '.join(text_tokenized))

        emoticons_count = 0

        print("<---------- Sentiment Analysis Stage ---------->")
        print("\n")
        
        # Import necessary libraries
        import mysql.connector
        from mysql.connector import Error

        # Prepare a DataFrame for insertion into the database
        # Assuming that text_lemmatized is a list of preprocessed text strings
        data = pd.DataFrame({
            'text': text_lemmatized,
        })

        # Function to connect to MySQL database and insert data
        def insert_into_database(lemmatized_text, table_name):
            try:
                # Establish connection to the MySQL database
                connection = mysql.connector.connect(
                    host='localhost',
                    user='emcrypt',
                    password='sentiment123*',
                    database='emcrypt_database'
                )

                if connection.is_connected():
                    cursor = connection.cursor()
                    # Prepare the insert query
                    insert_query = f"INSERT INTO {table_name} (text) VALUES (%s)"
                    # Execute the insert query with the lemmatized text
                    cursor.execute(insert_query, (lemmatized_text,))
                    # Commit the transaction
                    connection.commit()
                    print("Data inserted successfully in the EmCrypt Database")
                    print("\n")

            except Error as e:
                print("Error while connecting to MySQL", e)

            finally:
                # Close the cursor and the connection
                if connection.is_connected():
                    cursor.close()
                    connection.close()
                    #print("MySQL connection is closed")

        # Use this function to insert the lemmatized text into your database
        lemmatized_text_string = ' '.join(text_lemmatized)  # Assuming text_lemmatized is your list of lemmatized words
        insert_into_database(lemmatized_text_string, 'emcrypt')

        # Check radio button selection
        if self.radioButton1.isChecked():
            prepared_text = text_lemmatized    # Use the processed text
            features = self.transform_text_to_features(prepared_text)
            print("\n")
            print("Features Extracting...")
            print("\n")
            if self.polarity_model_combine is not None:
                polarity_result = self.polarity_model_combine.predict(features)
                print("Applying Polarity Classifier Model - LSTM-SVM...")
                print("\n")
            else:
                polarity_result = "Model not loaded"

            if self.emotion_model_combine is not None:
                emotion_result = self.emotion_model_combine.predict(features)
                print("Applying Emotion Recognition Model - LSTM-SVM...")
                print("\n")
            else:
                emotion_result = "Model not loaded"
            
        elif self.radioButton2.isChecked():
            # Similar processing for radioButton2, if different
            prepared_text = text_lemmatized    # Use the processed text
            features = self.transform_text_to_features(prepared_text)
            print("\n")
            print("Features Extracting...")
            print("\n")
            if self.polarity_model_text is not None:
                polarity_result = self.polarity_model_text.predict(features)
                print("Applying Polarity Classifier Model - LSTM-SVM...")
                print("\n")
            else:
                polarity_result = "Model not loaded"

            if self.emotion_model_text is not None:
                emotion_result = self.emotion_model_text.predict(features)
                print("Applying Emotion Recognition Model - LSTM-SVM...")
                print("\n")
            else:
                emotion_result = "Model not loaded"
                
        # Convert the NumPy array to a string
        polarity_result_str = np.array2string(polarity_result)
        emotion_result_str = np.array2string(emotion_result)

        # Map polarity_result 0 to 'negative' and 1 to 'positive'
        polarity_result_str = 'Negative' if polarity_result == 0 else 'Positive'
        # Convert emotion_result to a string
        emotion_result_str = emotion_result[0] if emotion_result else 'unknown'


        if self.polarity_model_text is not None:
            polarity_result = self.polarity_model_text.predict(features)
            # Convert the NumPy array to a string
            polarity_result_str = np.array2string(polarity_result)
            # Map polarity_result 0 to 'negative' and 1 to 'positive'
            polarity_result_str = 'Negative' if polarity_result == 0 else 'Positive'

            # Assign emotion based on the predicted polarity
            emotion_result_str = self.assign_emotion_based_on_polarity(polarity_result)
        else:
            polarity_result_str = "Model not loaded"
            emotion_result_str = "Model not loaded"

        # Define a dictionary for emotion mappings
        emotion_mappings = {
            'Happy': 'Happy',
            'Sad': 'Sad',
            'Fear': 'Fear',
            'Anticipation': 'Anticipation',
            'Surprise': 'Surprise',
            'Angry': 'Angry'
        }

        # Check if the original text is empty and show the popup
        if not original_text or original_text == self.plainTextEdit.placeholder_text:
            self.showInputWarning()
            return
        
        # Get the emotion from the dictionary with a default value of 'unknown'
        emotion_result_str = emotion_mappings.get(emotion_result_str, 'unknown')

        intensity_result = self.classify_intensity(emotion_result_str, emoticons_count, prepared_text) # Assuming classify_intensity requires emoticons_count and text

        print("The Polarity is: ", polarity_result_str)
        print("\n")
        print("The Emotion is: ", emotion_result_str)
        print("\n")
        print("The Intensity Level is: ", intensity_result)
        print("\n")
        print("===============================================================================================================================================================================")
        print("\n")

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
        # Create and set up the pop-up dialog
        dialog = QDialog()
        dialog.setWindowTitle("Upload File Instructions")
        dialog.setWindowModality(Qt.ApplicationModal)  # Make the dialog modal

        layout = QVBoxLayout(dialog)

        # Create and set the image label
        label = QLabel(dialog)
        pixmap = QPixmap("/Users/cjcasinsinan/Documents/GitHub/EmCrypt/assets/upload-file-instruction.png")
        label.setPixmap(pixmap)
        layout.addWidget(label)

        dialog.exec_()  # Show the dialog

        # Proceed to the file upload process after closing the dialog
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(None, "Upload File", "", "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            try:
                if file_name.endswith('.xlsx'):
                    df = pd.read_excel(file_name)
                elif file_name.endswith('.csv'):
                    df = pd.read_csv(file_name)
                else:
                    raise ValueError("Unsupported file format")

                self.tableWidget.setRowCount(0)
                for index, row in df.iterrows():
                    self.plainTextEdit.setPlainText(row['Tweets'])  # Set the text in the plainTextEdit widget
                    self.updateTextInTable()  # Process and update the table
            except Exception as e:
                print("An error occurred:", e)

        
    def showPopup(self):
        # Create and set up the pop-up dialog
        dialog = QDialog()
        dialog.setWindowTitle("User Manual")
        dialog.setWindowModality(Qt.ApplicationModal)
        layout = QVBoxLayout(dialog)

        # Create and set the image label
        label = QLabel(dialog)
        pixmap = QPixmap("/Users/cjcasinsinan/Documents/GitHub/EmCrypt/assets/user-manual.png")
        label.setPixmap(pixmap)
        layout.addWidget(label)
        dialog.exec_()  # Show the dialog

    def retranslateUi(self, OtherWindow):
        _translate = QtCore.QCoreApplication.translate
        OtherWindow.setWindowTitle(_translate("OtherWindow", "Emcrypt"))
        self.pushButton.setText(_translate("OtherWindow", "Upload File"))
        self.plainTextEdit.setPlainText(_translate("OtherWindow", " Enter the Cryptocurrency related tweets here..."))
        self.pushButton_2.setText(_translate("OtherWindow", "Clear"))
        self.pushButton_3.setText(_translate("OtherWindow", "Evaluate"))
        self.tableWidget.setSortingEnabled(True)
        self.tableWidget.setHorizontalHeaderLabels([
        _translate("OtherWindow", "Tweets"),
        _translate("OtherWindow", "Polarity"),
        _translate("OtherWindow", "Emotion"),
        _translate("OtherWindow", "Intensity")
        ])

        self.pushButton.clicked.connect(self.uploadFile)  # Connect the button to the function
        self.pushButton_2.clicked.connect(self.clearPlainText)  
        self.pushButton_3.clicked.connect(self.updateTextInTable)

import dsg2

# Main application execution
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    OtherWindow = QtWidgets.QMainWindow()
    ui = Ui_OtherWindow()
    ui.setupUi(OtherWindow)
    OtherWindow.show()
    sys.exit(app.exec_())
