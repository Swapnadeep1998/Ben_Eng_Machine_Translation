import unicodedata
import re
import io
import tensorflow as tf
import sys
sys.path.append(".")
from Utils import config

PADDING = config.PADDING

def unicode_to_ascii(s:str):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                 if unicodedata.category(c) != 'Mn')


def preprocess_sentence(sentence:str):
    w = unicode_to_ascii(sentence.lower().strip())

    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    #w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    w = '<start> '+ w + ' <end>'
    return w


def create_language_pairs(path):
  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
  
  word_pairs = [[preprocess_sentence(w) for w in line.split('\t')[:2]]
                for line in lines[:]]

  return zip(*word_pairs)


def tokenize(language):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(language)

    tensor = lang_tokenizer.texts_to_sequences(language)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
    padding=PADDING)

    return tensor, lang_tokenizer

