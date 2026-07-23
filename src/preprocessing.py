import re
import nltk
import string

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

_factory = StemmerFactory()
_stemmer = _factory.create_stemmer()

_stop_words = set(stopwords.words('indonesian'))
_extra_stopwords = {
    'yg', 'dg', 'rt', 'dgn', 'ny', 'amp', 'gt', 'lt', 'ga', 'gak',
    'kalo', 'udah', 'aja', 'ya', 'si', 'tp', 'jg', 'lg', 'utk', 'utuk',
    'tapi', 'juga', 'buat', 'bisa', 'sama', 'gitu', 'gimana', 'emang',
    'sih', 'nih', 'dong', 'lah', 'kan', 'kok', 'toh', 'pun', 'itu', 'ini'
}
_stop_words = _stop_words.union(_extra_stopwords)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text):
    return word_tokenize(text)


def remove_stopwords(tokens):
    return [t for t in tokens if t not in _stop_words and len(t) > 1]


def stem_tokens(tokens):
    return [_stemmer.stem(t) for t in tokens]


def preprocess_pipeline(text):
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return ' '.join(tokens)
