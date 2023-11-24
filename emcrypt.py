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
from PyQt5.QtWidgets import QLabel, QDialog, QVBoxLayout, QPushButton
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

    def focusInEvent(self, event):
        if self.toPlainText() == self.placeholder_text:
            self.setPlainText("")
        super(ClearablePlainTextEdit, self).focusInEvent(event)

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
            '🌈': {'Angry': 0.0, 'Anticipation': 0.28, 'Fear': 0.0, 'Happy': 0.69, 'Sad': 0.06, 'Surprise': 0.22 },
            '🌙': {'Angry': 0.0, 'Anticipation': 0.31, 'Fear': 0.0, 'Happy': 0.25, 'Sad': 0.0, 'Surprise': 0.06},
            '🌚': {'Angry': 0.06, 'Anticipation': 0.08, 'Fear': 0.06, 'Happy': 0.42, 'Sad': 0.19, 'Surprise': 0.06},
            '🌞': {'Angry': 0.0, 'Anticipation': 0.22, 'Fear': 0.0, 'Happy': 0.78, 'Sad': 0.0, 'Surprise': 0.11},
            '🌟': {'Angry': 0.0, 'Anticipation': 0.28, 'Fear': 0.0, 'Happy': 0.53, 'Sad': 0.0, 'Surprise': 0.25},
            '🌷': {'Angry': 0.0, 'Anticipation': 0.31, 'Fear': 0.0, 'Happy': 0.44, 'Sad': 0.0, 'Surprise': 0.0},
            '🌸': {'Angry': 0.0, 'Anticipation': 0.22, 'Fear': 0.0, 'Happy': 0.56, 'Sad': 0.0, 'Surprise': 0.14},
            '🌹': {'Angry': 0.0, 'Anticipation': 0.36, 'Fear': 0.0, 'Happy': 0.56, 'Sad': 0.0, 'Surprise': 0.11},
            '🌺': {'Angry': 0.0, 'Anticipation': 0.11, 'Fear': 0.0, 'Happy': 0.39, 'Sad': 0.0, 'Surprise': 0.06},
            '🍀': {'Angry': 0.0, 'Anticipation': 0.39, 'Fear': 0.0, 'Happy': 0.47, 'Sad': 0.0, 'Surprise': 0.22},
            '🍃': {'Angry': 0.0, 'Anticipation': 0.31, 'Fear': 0.0, 'Happy': 0.11, 'Sad': 0.17, 'Surprise': 0.03},
            '🍕': {'Angry': 0.06, 'Anticipation': 0.39, 'Fear': 0.06, 'Happy': 0.47, 'Sad': 0.06, 'Surprise': 0.17},
            '🍻': {'Angry': 0.0, 'Anticipation': 0.44, 'Fear': 0.0, 'Happy': 0.72, 'Sad': 0.0, 'Surprise': 0.25},
            '🎀': {'Angry': 0.0, 'Anticipation': 0.42, 'Fear': 0.0, 'Happy': 0.44, 'Sad': 0.0, 'Surprise': 0.36},
            '🎈': {'Angry': 0.06, 'Anticipation': 0.25, 'Fear': 0.06, 'Happy': 0.47, 'Sad': 0.06, 'Surprise': 0.31},
            '🎉': {'Angry': 0.0, 'Anticipation': 0.33, 'Fear': 0.0, 'Happy': 0.92, 'Sad': 0.0, 'Surprise': 0.5},
            '🎤': {'Angry': 0.0, 'Anticipation': 0.39, 'Fear': 0.06, 'Happy': 0.39, 'Sad': 0.08, 'Surprise': 0.08},
            '🎥': {'Angry': 0.0, 'Anticipation': 0.28, 'Fear': 0.0, 'Happy': 0.19, 'Sad': 0.0, 'Surprise': 0.17},
            '🎧': {'Angry': 0.0, 'Anticipation': 0.08, 'Fear': 0.0, 'Happy': 0.44, 'Sad': 0.0, 'Surprise': 0.0},
            '🎵': {'Angry': 0.0, 'Anticipation': 0.25, 'Fear': 0.0, 'Happy': 0.47, 'Sad': 0.08, 'Surprise': 0.08},
            '🎶': {'Angry': 0.0, 'Anticipation': 0.22, 'Fear': 0.0, 'Happy': 0.47, 'Sad': 0.0, 'Surprise': 0.22},
            '👀': {'Angry': 0.14, 'Anticipation': 0.81, 'Fear': 0.42, 'Happy': 0.0, 'Sad': 0.17, 'Surprise': 0.64},
            '👅': {'Angry': 0.0, 'Anticipation': 0.17, 'Fear': 0.0, 'Happy': 0.36, 'Sad': 0.0, 'Surprise': 0.08},
            '👇': {'Angry': 0.11, 'Anticipation': 0.14, 'Fear': 0.06, 'Happy': 0.0, 'Sad': 0.0, 'Surprise': 0.0},
            '👈': {'Angry': 0.14, 'Anticipation': 0.17, 'Fear': 0.0, 'Happy': 0.0, 'Sad': 0.0, 'Surprise': 0.06},
            '👉': {'Angry': 0.06, 'Anticipation': 0.25, 'Fear': 0.0, 'Happy': 0.0, 'Sad': 0.0, 'Surprise': 0.03},
            '👊': {'Angry': 0.44, 'Anticipation': 0.36, 'Fear': 0.11, 'Happy': 0.22, 'Sad': 0.0, 'Surprise': 0.0},
            '👋': {'Angry': 0.08, 'Anticipation': 0.28, 'Fear': 0.0, 'Happy': 0.22, 'Sad': 0.0, 'Surprise': 0.08},
            '👌': {'Angry': 0.0, 'Anticipation': 0.36, 'Fear': 0.0, 'Happy': 0.22, 'Sad': 0.0, 'Surprise': 0.22},
            '👍': {'Angry': 0.11, 'Anticipation': 0.39, 'Fear': 0.08, 'Happy': 0.39, 'Sad': 0.06, 'Surprise': 0.14},
            '👎': {'Angry': 0.5, 'Anticipation': 0.08, 'Fear': 0.14, 'Happy': 0.0, 'Sad': 0.31, 'Surprise': 0.14},
            '👏': {'Angry': 0.08, 'Anticipation': 0.39, 'Fear': 0.0, 'Happy': 0.64, 'Sad': 0.0, 'Surprise': 0.25},
            '👑': {'Angry': 0.0, 'Anticipation': 0.25, 'Fear': 0.0, 'Happy': 0.28, 'Sad': 0.0, 'Surprise': 0.11},
            '👻': {'Angry': 0.11, 'Anticipation': 0.08, 'Fear': 0.69, 'Happy': 0.0, 'Sad': 0.11, 'Surprise': 0.31},
            '💀': {'Angry': 0.19, 'Anticipation': 0.14, 'Fear': 0.61, 'Happy': 0.03, 'Sad': 0.31, 'Surprise': 0.06},
            '💁': {'Angry': 0.08, 'Anticipation': 0.33, 'Fear': 0.06, 'Happy': 0.14, 'Sad': 0.06, 'Surprise': 0.17},
            '💃': {'Angry': 0.0, 'Anticipation': 0.11, 'Fear': 0.0, 'Happy': 0.69, 'Sad': 0.0, 'Surprise': 0.17},
            '💋': {'Angry': 0.0, 'Anticipation': 0.28, 'Fear': 0.0, 'Happy': 0.78, 'Sad': 0.0, 'Surprise': 0.19},
            '💎': {'Angry': 0.0, 'Anticipation': 0.31, 'Fear': 0.06, 'Happy': 0.33, 'Sad': 0.0, 'Surprise': 0.25},
            '💐': {'Angry': 0.0, 'Anticipation': 0.39,  'Fear': 0.0, 'Happy': 0.69, 'Sad': 0.11, 'Surprise': 0.36},
            '💓': {'Angry': 0.0, 'Anticipation': 0.47,  'Fear': 0.08, 'Happy': 0.61, 'Sad': 0.0, 'Surprise': 0.19},
            '💔': {'Angry': 0.39, 'Anticipation': 0.19,  'Fear': 0.14, 'Happy': 0.0, 'Sad': 0.94, 'Surprise': 0.08},
            '💕': {'Angry': 0.0, 'Anticipation': 0.31, 'Fear': 0.0, 'Happy': 0.83, 'Sad': 0.0, 'Surprise': 0.11},
            '💖': {'Angry': 0.0, 'Anticipation': 0.33, 'Fear': 0.0, 'Happy': 0.89, 'Sad': 0.0, 'Surprise': 0.25},
            '💗': {'Angry': 0.0, 'Anticipation': 0.36, 'Fear': 0.0, 'Happy': 0.89, 'Sad': 0.0, 'Surprise': 0.22},
            '💘': {'Angry': 0.03, 'Anticipation': 0.31, 'Fear': 0.06, 'Happy': 0.67, 'Sad': 0.14, 'Surprise': 0.06},
            '💙': {'Angry': 0.0, 'Anticipation': 0.25, 'Fear': 0.0, 'Happy': 0.61, 'Sad': 0.17, 'Surprise': 0.17},
            '💚': {'Angry': 0.0, 'Anticipation': 0.11, 'Fear': 0.0, 'Happy': 0.58, 'Sad': 0.03, 'Surprise': 0.03},
            '💛': {'Angry': 0.03, 'Anticipation': 0.11, 'Fear': 0.0, 'Happy': 0.53, 'Sad': 0.08, 'Surprise': 0.08},
            '💜': {'Angry': 0.0, 'Anticipation': 0.11, 'Fear': 0.06, 'Happy': 0.47, 'Sad': 0.11, 'Surprise': 0.08},
            '💞': {'Angry': 0.0, 'Anticipation': 0.25,'Fear': 0.0, 'Happy': 0.83, 'Sad': 0.0, 'Surprise': 0.22},
            '💤': {'Angry': 0.06, 'Anticipation': 0.36, 'Fear': 0.06, 'Happy': 0.11, 'Sad': 0.14, 'Surprise': 0.06},
            '💥': {'Angry': 0.44, 'Anticipation': 0.19, 'Fear': 0.31, 'Happy': 0.11, 'Sad': 0.14, 'Surprise': 0.31},
            '💦': {'Angry': 0.0, 'Anticipation': 0.11, 'Fear': 0.06, 'Happy': 0.0, 'Sad': 0.14, 'Surprise': 0.0},
            '💩': {'Angry': 0.14, 'Anticipation': 0.08, 'Fear': 0.0, 'Happy': 0.25, 'Sad': 0.03, 'Surprise': 0.19},
            '💪': {'Angry': 0.03, 'Anticipation': 0.31, 'Fear': 0.0, 'Happy': 0.42, 'Sad': 0.0, 'Surprise': 0.08},
            '💫': {'Angry': 0.0, 'Anticipation': 0.19, 'Fear': 0.06, 'Happy': 0.44, 'Sad': 0.0, 'Surprise': 0.19},
            '💭': {'Angry': 0.11, 'Anticipation': 0.64, 'Fear': 0.11, 'Happy': 0.17, 'Sad': 0.11, 'Surprise': 0.17},
            '💯': {'Angry': 0.06, 'Anticipation': 0.28, 'Fear': 0.06, 'Happy': 0.64, 'Sad': 0.06, 'Surprise': 0.19},
            '💰': {'Angry': 0.0, 'Anticipation': 0.58,  'Fear': 0.06, 'Happy': 0.47, 'Sad': 0.06, 'Surprise': 0.25},
            '📷': {'Angry': 0.0, 'Anticipation': 0.19, 'Fear': 0.0, 'Happy': 0.14, 'Sad': 0.0, 'Surprise': 0.08},
            '🔞': {'Angry': 0.11, 'Anticipation': 0.11,  'Fear': 0.03, 'Happy': 0.08, 'Sad': 0.11, 'Surprise': 0.0},
            '🔥': {'Angry': 0.47, 'Anticipation': 0.22, 'Fear': 0.17, 'Happy': 0.25, 'Sad': 0.11, 'Surprise': 0.39},
            '🔫': {'Angry': 0.44, 'Anticipation': 0.14, 'Fear': 0.14, 'Happy': 0.03, 'Sad': 0.14, 'Surprise': 0.0},
            '🔴': {'Angry': 0.08, 'Anticipation': 0.06, 'Fear': 0.11, 'Happy': 0.0, 'Sad': 0.03, 'Surprise': 0.19},
            '😀': {'Angry': 0.06, 'Anticipation': 0.22, 'Fear': 0.06, 'Happy': 0.69, 'Sad': 0.06, 'Surprise': 0.14},
            '😁': {'Angry': 0.06, 'Anticipation': 0.25,  'Fear': 0.08, 'Happy': 0.89, 'Sad': 0.06, 'Surprise': 0.33},
            '😂': {'Angry': 0.0, 'Anticipation': 0.17, 'Fear': 0.06, 'Happy': 0.94, 'Sad': 0.0, 'Surprise': 0.33},
            '😃': {'Angry': 0.0, 'Anticipation': 0.31, 'Fear': 0.06, 'Happy': 0.83, 'Sad': 0.0, 'Surprise': 0.33},
            '😄': {'Angry': 0.0, 'Anticipation': 0.36, 'Fear': 0.0, 'Happy': 0.86, 'Sad': 0.0, 'Surprise': 0.28},
            '😅': {'Angry': 0.08, 'Anticipation': 0.44, 'Fear': 0.28, 'Happy': 0.42, 'Sad': 0.06, 'Surprise': 0.36},
            '😆': {'Angry': 0.06, 'Anticipation': 0.19, 'Fear': 0.06, 'Happy': 0.94, 'Sad': 0.06, 'Surprise': 0.25},
            '😇': {'Angry': 0.0, 'Anticipation': 0.31, 'Fear': 0.0, 'Happy': 0.72, 'Sad': 0.0, 'Surprise': 0.17},
            '😈': {'Angry': 0.14, 'Anticipation': 0.44, 'Fear': 0.19, 'Happy': 0.33, 'Sad': 0.08, 'Surprise': 0.03},
            '😉': {'Angry': 0.0, 'Anticipation': 0.42, 'Fear': 0.0, 'Happy': 0.44, 'Sad': 0.08, 'Surprise': 0.28},
            '😊': {'Angry': 0.0, 'Anticipation': 0.42, 'Fear': 0.0, 'Happy': 0.92, 'Sad': 0.0, 'Surprise': 0.33},
            '😋': {'Angry': 0.0, 'Anticipation': 0.47, 'Fear': 0.0, 'Happy': 0.78, 'Sad': 0.0, 'Surprise': 0.19},
            '😌': {'Angry': 0.0, 'Anticipation': 0.33, 'Fear': 0.11, 'Happy': 0.81, 'Sad': 0.0, 'Surprise': 0.22},
            '😍': {'Angry': 0.0, 'Anticipation': 0.31, 'Fear': 0.0, 'Happy': 0.83, 'Sad': 0.0, 'Surprise': 0.5},
            '😎': {'Angry': 0.0, 'Anticipation': 0.22, 'Fear': 0.0, 'Happy': 0.75, 'Sad': 0.0, 'Surprise': 0.06},
            '😏': {'Angry': 0.22, 'Anticipation': 0.33,  'Fear': 0.14, 'Happy': 0.22, 'Sad': 0.22, 'Surprise': 0.11},
            '😐': {'Angry': 0.14, 'Anticipation': 0.33,  'Fear': 0.17, 'Happy': 0.06, 'Sad': 0.25, 'Surprise': 0.31},
            '😑': {'Angry': 0.28, 'Anticipation': 0.22, 'Fear': 0.14, 'Happy': 0.0, 'Sad': 0.33, 'Surprise': 0.19},
            '😒': {'Angry': 0.58, 'Anticipation': 0.14, 'Fear': 0.17, 'Happy': 0.0, 'Sad': 0.42, 'Surprise': 0.11},
            '😓': {'Angry': 0.19, 'Anticipation': 0.44, 'Fear': 0.64, 'Happy': 0.0, 'Sad': 0.36, 'Surprise': 0.17},
            '😔': {'Angry': 0.25, 'Anticipation': 0.22, 'Fear': 0.28, 'Happy': 0.0, 'Sad': 0.72, 'Surprise': 0.19},
            '😕': {'Angry': 0.19, 'Anticipation': 0.42, 'Fear': 0.36, 'Happy': 0.0, 'Sad': 0.39, 'Surprise': 0.28},
            '😖': {'Angry': 0.22, 'Anticipation': 0.36, 'Fear': 0.5, 'Happy': 0.08, 'Sad': 0.53, 'Surprise': 0.11},
            '😘': {'Angry': 0.0, 'Anticipation': 0.33, 'Fear': 0.0, 'Happy': 0.72, 'Sad': 0.0, 'Surprise': 0.17},
            '😙': {'Angry': 0.0, 'Anticipation': 0.47, 'Fear': 0.0, 'Happy': 0.83, 'Sad': 0.0, 'Surprise': 0.17},
            '😚': {'Angry': 0.0, 'Anticipation': 0.44, 'Fear': 0.0, 'Happy': 0.86, 'Sad': 0.0, 'Surprise': 0.22},
            '😛': {'Angry': 0.0, 'Anticipation': 0.31, 'Fear': 0.03, 'Happy': 0.69, 'Sad': 0.0, 'Surprise': 0.28},
            '😜': {'Angry': 0.0, 'Anticipation': 0.42, 'Fear': 0.06, 'Happy': 0.64, 'Sad': 0.0, 'Surprise': 0.28},
            '😝': {'Angry': 0.0, 'Anticipation': 0.22, 'Fear': 0.08, 'Happy': 0.83, 'Sad': 0.0, 'Surprise': 0.22},
            '😞': {'Angry': 0.39, 'Anticipation': 0.19, 'Fear': 0.33, 'Happy': 0.0, 'Sad': 0.92, 'Surprise': 0.06},
            '😟': {'Angry': 0.25, 'Anticipation': 0.44, 'Fear': 0.72, 'Happy': 0.0, 'Sad': 0.69, 'Surprise': 0.17},
            '😠': {'Angry': 1.0, 'Anticipation': 0.17, 'Fear': 0.17, 'Happy': 0.0, 'Sad': 0.25, 'Surprise': 0.11},
            '😡': {'Angry': 1.0, 'Anticipation': 0.11, 'Fear': 0.11, 'Happy': 0.0, 'Sad': 0.36, 'Surprise': 0.08},
            '😢': {'Angry': 0.25, 'Anticipation': 0.08, 'Fear': 0.5, 'Happy': 0.0, 'Sad': 1.0, 'Surprise': 0.08},
            '😣': {'Angry': 0.31, 'Anticipation': 0.28, 'Fear': 0.47, 'Happy': 0.0, 'Sad': 0.64, 'Surprise': 0.0},
            '😤': {'Angry': 0.75, 'Anticipation': 0.11, 'Fear': 0.14, 'Happy': 0.0, 'Sad': 0.25, 'Surprise': 0.03},
            '😥': {'Angry': 0.14, 'Anticipation': 0.19, 'Fear': 0.33, 'Happy': 0.03, 'Sad': 0.81, 'Surprise': 0.08},
            '😨': {'Angry': 0.17, 'Anticipation': 0.39, 'Fear': 0.97, 'Happy': 0.0, 'Sad': 0.56, 'Surprise': 0.39},
            '😩': {'Angry': 0.33, 'Anticipation': 0.25, 'Fear': 0.47, 'Happy': 0.0, 'Sad': 0.75, 'Surprise': 0.14},
            '😪': {'Angry': 0.11, 'Anticipation': 0.08, 'Fear': 0.28, 'Happy': 0.0, 'Sad': 0.64, 'Surprise': 0.06},
            '😫': {'Angry': 0.36, 'Anticipation': 0.14, 'Fear': 0.17, 'Happy': 0.11, 'Sad': 0.72, 'Surprise': 0.06},
            '😬': {'Angry': 0.14, 'Anticipation': 0.53, 'Fear': 0.44, 'Happy': 0.17, 'Sad': 0.11, 'Surprise': 0.25},
            '😭': {'Angry': 0.22, 'Anticipation': 0.08, 'Fear': 0.33, 'Happy': 0.0, 'Sad': 1.0, 'Surprise': 0.08},
            '😰': {'Angry': 0.22, 'Anticipation': 0.31, 'Fear': 0.83, 'Happy': 0.0, 'Sad': 0.69, 'Surprise': 0.08},
            '😱': {'Angry': 0.28, 'Anticipation': 0.42,  'Fear': 0.92, 'Happy': 0.06, 'Sad': 0.25, 'Surprise': 0.69},
            '😳': {'Angry': 0.06, 'Anticipation': 0.36, 'Fear': 0.5, 'Happy': 0.14, 'Sad': 0.19, 'Surprise': 0.44},
            '😴': {'Angry': 0.0, 'Anticipation': 0.06, 'Fear': 0.0, 'Happy': 0.03, 'Sad': 0.03, 'Surprise': 0.0},
            '😶': {'Angry': 0.06, 'Anticipation': 0.22, 'Fear': 0.36, 'Happy': 0.0, 'Sad': 0.14, 'Surprise': 0.19},
            '😷': {'Angry': 0.03, 'Anticipation': 0.17, 'Fear': 0.5, 'Happy': 0.0, 'Sad': 0.22, 'Surprise': 0.03},
            '😹': {'Angry': 0.0, 'Anticipation': 0.17, 'Fear': 0.0, 'Happy': 0.94, 'Sad': 0.0, 'Surprise': 0.14},
            '😻': {'Angry': 0.0, 'Anticipation': 0.42, 'Fear': 0.0, 'Happy': 0.75, 'Sad': 0.06, 'Surprise': 0.33},
            '🙅': {'Angry': 0.47, 'Anticipation': 0.25, 'Fear': 0.33, 'Happy': 0.06, 'Sad': 0.33, 'Surprise': 0.11},
            '🙆': {'Angry': 0.03, 'Anticipation': 0.33, 'Fear': 0.0, 'Happy': 0.39, 'Sad': 0.0, 'Surprise': 0.03},
            '🙈': {'Angry': 0.0, 'Anticipation': 0.39, 'Fear': 0.17, 'Happy': 0.28, 'Sad': 0.03, 'Surprise': 0.5},
            '🙊': {'Angry': 0.06, 'Anticipation': 0.44, 'Fear': 0.47, 'Happy': 0.14, 'Sad': 0.08, 'Surprise': 0.42},
            '🙋': {'Angry': 0.0, 'Anticipation': 0.53, 'Fear': 0.0, 'Happy': 0.44, 'Sad': 0.0, 'Surprise': 0.19},
            '🙌': {'Angry': 0.0, 'Anticipation': 0.33, 'Fear': 0.0, 'Happy': 0.72, 'Sad': 0.0, 'Surprise': 0.39},
            '🙏': {'Angry': 0.06, 'Anticipation': 0.44, 'Fear': 0.11, 'Happy': 0.25, 'Sad': 0.11, 'Surprise': 0.17},
            '‼': {'Angry': 0.44, 'Anticipation': 0.42, 'Fear': 0.06, 'Happy': 0.14, 'Sad': 0.0, 'Surprise': 0.89},
            '↩': {'Angry': 0.0, 'Anticipation': 0.06, 'Fear': 0.0, 'Happy': 0.0, 'Sad': 0.0, 'Surprise': 0.0},
            '↪': {'Angry': 0.06, 'Anticipation': 0.19, 'Fear': 0.0, 'Happy': 0.0, 'Sad': 0.0, 'Surprise': 0.0},
            '▶': {'Angry': 0.0, 'Anticipation': 0.08, 'Fear': 0.0, 'Happy': 0.03, 'Sad': 0.0, 'Surprise': 0.0},
            '◀': {'Angry': 0.0, 'Anticipation': 0.06, 'Fear': 0.0, 'Happy': 0.0, 'Sad': 0.0, 'Surprise': 0.06},
            '☀': {'Angry': 0.0, 'Anticipation': 0.22, 'Fear': 0.0, 'Happy': 0.44, 'Sad': 0.0, 'Surprise': 0.06},
            '☑': {'Angry': 0.0, 'Anticipation': 0.22, 'Fear': 0.0, 'Happy': 0.25, 'Sad': 0.0, 'Surprise': 0.0},
            '☝': {'Angry': 0.11, 'Anticipation': 0.31, 'Fear': 0.11, 'Happy': 0.03, 'Sad': 0.11, 'Surprise': 0.0},
            '☺': {'Angry': 0.0, 'Anticipation': 0.42, 'Fear': 0.0, 'Happy': 1.0, 'Sad': 0.0, 'Surprise': 0.39},
            '♥': {'Angry': 0.0, 'Anticipation': 0.28, 'Fear': 0.0, 'Happy': 0.72, 'Sad': 0.0, 'Surprise': 0.11},
            '♻': {'Angry': 0.03, 'Anticipation': 0.19, 'Fear': 0.0, 'Happy': 0.03, 'Sad': 0.03, 'Surprise': 0.0},
            '⚡': {'Angry': 0.28, 'Anticipation': 0.31, 'Fear': 0.25, 'Happy': 0.08, 'Sad': 0.0, 'Surprise': 0.36},
            '⚽': {'Angry': 0.0, 'Anticipation': 0.33, 'Fear': 0.06, 'Happy': 0.25, 'Sad': 0.0, 'Surprise': 0.0},
            '✅': {'Angry': 0.0, 'Anticipation': 0.31, 'Fear': 0.0, 'Happy': 0.19, 'Sad': 0.0, 'Surprise': 0.0},
            '✈': {'Angry': 0.0, 'Anticipation': 0.44, 'Fear': 0.11, 'Happy': 0.28, 'Sad': 0.11, 'Surprise': 0.19},
            '✊': {'Angry': 0.25, 'Anticipation': 0.5, 'Fear': 0.11, 'Happy': 0.03, 'Sad': 0.11, 'Surprise': 0.08},
            '✋': {'Angry': 0.22, 'Anticipation': 0.25, 'Fear': 0.11, 'Happy': 0.06, 'Sad': 0.06, 'Surprise': 0.08},
            '✌': {'Angry': 0.0, 'Anticipation': 0.42, 'Fear': 0.0, 'Happy': 0.61, 'Sad': 0.0, 'Surprise': 0.17},
            '✔': {'Angry': 0.0, 'Anticipation': 0.25, 'Fear': 0.0, 'Happy': 0.14, 'Sad': 0.0, 'Surprise': 0.0,},
            '✨': {'Angry': 0.0, 'Anticipation': 0.36, 'Fear': 0.06, 'Happy': 0.53, 'Sad': 0.0, 'Surprise': 0.44,},
            '❄': {'Angry': 0.11, 'Anticipation': 0.33, 'Fear': 0.17, 'Happy': 0.28, 'Sad': 0.14, 'Surprise': 0.22,},
            '❌': {'Angry': 0.5, 'Anticipation': 0.14, 'Fear': 0.25, 'Happy': 0.0, 'Sad': 0.31, 'Surprise': 0.08,},
            '❗': {'Angry': 0.44, 'Anticipation': 0.42, 'Fear': 0.42, 'Happy': 0.08, 'Sad': 0.17, 'Surprise': 0.81,},
            '❤': {'Angry': 0.0, 'Anticipation': 0.36, 'Fear': 0.0, 'Happy': 0.69, 'Sad': 0.0, 'Surprise': 0.14,},
            '➡': {'Angry': 0.0, 'Anticipation': 0.06, 'Fear': 0.0, 'Happy': 0.0, 'Sad': 0.0, 'Surprise': 0.0,},
            '⬅': {'Angry' : 0.17, 'Anticipation' : 0.14, 'Fear' : 0.14, 'Happy' :0.0, 'Sad': 0.14, 'Surprise': 0.03},
            '⭐': { 'Angry': 0.0, 'Anticipation' : 0.17, 'Fear' : 0.0, 'Happy' :	0.39, 'Sad' : 0.0, 'Surprise' :	0.17},
            "😲": { 'Angry': 0.0, 'Anticipation': 0.33, 'Fear': 0.33, 'Happy': 0.0, 'Sad': 0.0, 'Surprise': 0.67 },
            "😯": { 'Angry': 0.0, 'Anticipation': 0.25, 'Fear': 0.25, 'Happy': 0.0, 'Sad': 0.0, 'Surprise': 0.50 },
            "😮": { 'Angry': 0.0, 'Anticipation': 0.40, 'Fear': 0.20, 'Happy': 0.0, 'Sad': 0.0, 'Surprise': 0.60 },
            "😵": { 'Angry': 0.0, 'Anticipation': 0.0, 'Fear': 0.50, 'Happy': 0.0, 'Sad': 0.0, 'Surprise': 0.50 },
            "❗": { 'Angry': 0.25, 'Anticipation': 0.50, 'Fear': 0.25, 'Happy': 0.0, 'Sad': 0.0, 'Surprise': 0.75 },
            "⚡": { 'Angry': 0.2, 'Anticipation': 0.4, 'Fear': 0.3, 'Happy': 0.0, 'Sad': 0.0, 'Surprise': 0.5 },
            "🎊": { 'Angry': 0.0, 'Anticipation': 0.6, 'Fear': 0.0, 'Happy': 0.8, 'Sad': 0.0, 'Surprise': 0.7 },
            "🙁": { 'Angry': 0.2, 'Anticipation': 0.0, 'Fear': 0.1, 'Happy': 0.0, 'Sad': 0.7, 'Surprise': 0.1 },
            "🔪": { 'Angry': 0.4, 'Anticipation': 0.2, 'Fear': 0.6, 'Happy': 0.0, 'Sad': 0.1, 'Surprise': 0.2 },
            "🌕": { 'Angry': 0.0, 'Anticipation': 0.3, 'Fear': 0.0, 'Happy': 0.4, 'Sad': 0.0, 'Surprise': 0.3 },
            "🚀": { 'Angry': 0.0, 'Anticipation': 0.7, 'Fear': 0.1, 'Happy': 0.6, 'Sad': 0.0, 'Surprise': 0.5 },
            "📉": { 'Angry': 0.3, 'Anticipation': 0.1, 'Fear': 0.4, 'Happy': 0.0, 'Sad': 0.7, 'Surprise': 0.2 },
            "🤣": { 'Angry': 0.0, 'Anticipation': 0.2, 'Fear': 0.0, 'Happy': 1.0, 'Sad': 0.0, 'Surprise': 0.3 },
            "💸": { 'Angry': 0.2, 'Anticipation': 0.5, 'Fear': 0.1, 'Happy': 0.3, 'Sad': 0.4, 'Surprise': 0.4 }
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
        self.scrollArea.setGeometry(QtCore.QRect(120, 560, 811, 421))  # Set the geometry as needed
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
                print("Option 1")

                # Load the tokenizer
                with open('tokenizer.pkl', 'rb') as handle:
                    tokenizer = pickle.load(handle)

                # Load the original LSTM model
                lstm_model = keras.models.load_model('lstm_model.h5')

                # Preprocess the text (tokenization and padding)
                seq = tokenizer.texts_to_sequences([text])
                data_padded = keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)  # Use the same maxlen as used during training

                # Create a new model for feature extraction
                # This model uses the original model's input and the output of the second-to-last Dense layer
                feature_extraction_model = Model(inputs=lstm_model.input, outputs=lstm_model.layers[-2].output)

                # Use the new model to extract features
                features = feature_extraction_model.predict(data_padded)

                return features
        
        elif self.radioButton2.isChecked():
            print("Option 2")
            # Load the tokenizer and LSTM feature extractor model
            with open('tokenizer_text.pkl', 'rb') as handle:
                tokenizer = pickle.load(handle)
                            # Load the original LSTM model
                lstm_model = keras.models.load_model('lstm_model_text.h5')

                # Preprocess the text (tokenization and padding)
                seq = tokenizer.texts_to_sequences([text])
                data_padded = keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)  # Use the same maxlen as used during training

                # Create a new model for feature extraction
                # This model uses the original model's input and the output of the second-to-last Dense layer
                feature_extraction_model = Model(inputs=lstm_model.input, outputs=lstm_model.layers[-2].output)

                # Use the new model to extract features
                features = feature_extraction_model.predict(data_padded)

            return features
        
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

        # Check radio button selection
        if self.radioButton1.isChecked():

            print("Combination of Keywords, Ending Puctuation Marks, and Emoticons")
            converted_text, emoticons_count = self.convert_emoticons_to_words(text_spell_checked)  # Use the processed text
             # Convert and Calculate
            # Assuming 'convert_and_calculate' is a method in your class and 'text_lemmatized' is the final processed text
        

            # Print the text after coverting
            print("Combination of Keywords, Ending Punctuation Marks, and Emoticons :", ' '.join(converted_text))

        elif self.radioButton2.isChecked():
            print("Plain-text Only")

            # Remove punctuations and known emojis and use the 'text' models
            converted_text = self.remove_punctuations_and_known_emojis(text_spell_checked)

            # Print the text after lemmatization
            print("Plain Text Only :", ' '.join(converted_text))

        text_no_repeating_words = self.cleaning_repeating_words(converted_text)
        text_no_stopwords = self.cleaning_stopwords(text_no_repeating_words)
        text_lowercased = text_no_stopwords.lower()
        tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')
        text_tokenized = tokenizer.tokenize(text_lowercased)
        print("Text after Tokenization:", ' '.join(text_tokenized))

        text_lemmatized = self.lemmatizer_on_text(' '.join(text_tokenized))
        #text_stemmed = self.stemming_on_text(' '.join(text_lemmatized))
        text_lemmatized = self.lemmatizer_on_text(' '.join(text_tokenized))
        emoticons_count = 0
        
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
                    print("Data inserted successfully")

            except Error as e:
                print("Error while connecting to MySQL", e)

            finally:
                # Close the cursor and the connection
                if connection.is_connected():
                    cursor.close()
                    connection.close()
                    print("MySQL connection is closed")

        # Use this function to insert the lemmatized text into your database
        lemmatized_text_string = ' '.join(text_lemmatized)  # Assuming text_lemmatized is your list of lemmatized words
        insert_into_database(lemmatized_text_string, 'emcrypt')

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
            prepared_text = text_lemmatized    # Use the processed text
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
