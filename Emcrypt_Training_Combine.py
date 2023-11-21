#utilities
import re
import numpy as np
import pandas as pd
import os
import nltk

#SpellCorrection
from spellchecker import SpellChecker

import chardet
DATASET_COLUMNS = ['date', 'username', 'text', 'polarity', 'emotion']

#Detect file encoding using chardet
with open('Emcrypt-dataset.csv', 'rb') as f:
    result = chardet.detect(f.read())

# Print the detected encoding
print("Detected encoding:", result['encoding'])

# Read the file using the detected encoding
df = pd.read_csv('Emcrypt-dataset.csv', encoding=result['encoding'], names=DATASET_COLUMNS)
df.sample(10)


#Data preprocessing
data=df[['text','polarity', 'emotion']]

data_pos = data[data['polarity'] == 1]
data_neg = data[data['polarity'] == 0]

dataset = pd.concat([data_pos, data_neg])

# First step: Cleaning numbers
def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
dataset['text'] = dataset['text'].apply(lambda x: cleaning_numbers(x))
dataset['text'].head()


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

# Second Step: Remove URL, hastags, mentions, special characters, and extra whitespace
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

# Third step: Spell Checker
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

# Combination of Keywords, Ending Punctuation Marks, and Emojis : Emoticon Convertion
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

emoticon_weights = {
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

#Emoticon Convertion
def convert_and_calculate(text):
    emotional_scores = {emotion: 0.0 for emotion in ['angry', 'anticipation', 'fear', 'happy', 'sad', 'surprise']}
    changed_emoticons = 0
    for emoticon, word in emoticon_dict.items():
        while emoticon in text:
            text = text.replace(emoticon, word + " ", 1)
            changed_emoticons += 1
            scores = emoticon_weights.get(emoticon, {'angry': 0.0, 'anticipation': 0.0, 'fear': 0.0, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.0})
            for emotion, score in scores.items():
                emotional_scores[emotion] += score
    return text, changed_emoticons, emotional_scores

# Apply the combined function
result = dataset['text'].apply(convert_and_calculate)
dataset['converted_text'] = result.apply(lambda x: x[0])
dataset['emoticons_count'] = result.apply(lambda x: x[1])
dataset['emotional_scores'] = result.apply(lambda x: x[2])

# Function to clean repeating words
def cleaning_repeating_words(text):
    # This regex pattern targets whole words that are repeated
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

# Assuming 'dataset' is a pandas DataFrame and 'text' is a column in it
# Apply the cleaning function for repeating words to each row in the 'text' column
dataset['text'] = dataset['text'].apply(cleaning_repeating_words)
print("Repeating words cleaned from 'text' column.")
print(dataset['text'].head())

#Remove Stopwords
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


dataset['text']=dataset['text'].str.lower()
dataset['text'].head()


#Tokenization
from nltk.tokenize import RegexpTokenizer

# The pattern matches word characters (\w) and punctuation marks ([^\w\s])
tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')

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



#Sentiment Analysis Stage
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
from sklearn.metrics import classification_report
import pickle

# Assuming 'data' is your DataFrame with 'text', 'polarity', and 'emotion' columns
texts = data['text']
polarity_labels = data['polarity']
emotion_labels = data['emotion']

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data_padded = pad_sequences(sequences, maxlen=100)

# Adjusted LSTM Model for Feature Extraction
feature_model = Sequential()
feature_model.add(Embedding(input_dim=10000, output_dim=256, input_length=100))
feature_model.add(LSTM(128, return_sequences=True))
feature_model.add(LSTM(64))  # Last LSTM layer should not return sequences
feature_model.add(Dense(16, activation='relu'))
feature_model.add(Flatten())  # Flatten the output
feature_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
feature_model.fit(data_padded, np.array(polarity_labels), epochs=10, batch_size=64, validation_split=0.1)

# Extract features
features = feature_model.predict(data_padded)

# Save the feature model and tokenizer
feature_model.save("lstm_feature_extractor.h5")
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save model and tokenizer
output_dir = 'model_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Extract features
features = feature_model.predict(data_padded)

# Splitting the data: 60% training, 30% evaluation, and 10% testing
# For polarity labels
X_train, X_temp, y_polarity_train, y_polarity_temp = train_test_split(features, polarity_labels, test_size=0.4, random_state=42)
X_eval, X_test, y_polarity_eval, y_polarity_test = train_test_split(X_temp, y_polarity_temp, test_size=0.25, random_state=42)

# For emotion labels
X_train, X_temp, y_emotion_train, y_emotion_temp = train_test_split(features, emotion_labels, test_size=0.4, random_state=42)
X_eval, X_test, y_emotion_eval, y_emotion_test = train_test_split(X_temp, y_emotion_temp, test_size=0.25, random_state=42)

# Importing necessary libraries for Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# Grid Search for Polarity SVM
grid_polarity = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid_polarity.fit(X_train, y_polarity_train)
print("Best Polarity SVM Parameters:", grid_polarity.best_params_)

# Grid Search for Emotion SVM
grid_emotion = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid_emotion.fit(X_train, y_emotion_train)
print("Best Emotion SVM Parameters:", grid_emotion.best_params_)

# Save the best SVM models
joblib.dump(grid_polarity.best_estimator_, "svm_polarity_combine.pkl")
joblib.dump(grid_emotion.best_estimator_, "svm_emotion_combine.pkl")

# Evaluate and visualize the performance of the Polarity SVM Model
y_polarity_pred = grid_polarity.predict(X_test)
print("Polarity Classification Report:")
print(classification_report(y_polarity_test, y_polarity_pred))

# Confusion Matrix for Polarity
cm_polarity = confusion_matrix(y_polarity_test, y_polarity_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_polarity, annot=True, fmt='d')
plt.title('Confusion Matrix for Polarity Classification')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Evaluate and visualize the performance of the Emotion SVM Model
y_emotion_pred = grid_emotion.predict(X_test)
print("Emotion Classification Report:")
print(classification_report(y_emotion_test, y_emotion_pred))

# Confusion Matrix for Emotion
cm_emotion = confusion_matrix(y_emotion_test, y_emotion_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_emotion, annot=True, fmt='d')
plt.title('Confusion Matrix for Emotion Classification')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()




