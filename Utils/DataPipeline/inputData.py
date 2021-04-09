import tensorflow as tf
import sys
import io
import json
sys.path.append(".")
from Utils.Preprocess.preprocessor import tokenize, create_language_pairs
from Utils import config
from sklearn.model_selection import train_test_split

INPUT_TOKENIZER_PATH = config.INPUT_TOKENIZER_PATH
TARGET_TOKENIZER_PATH = config.TARGET_TOKENIZER_PATH


class LoadDataset():
    def __init__(self, path:str):
        self.path = path
        self.inp_lang, self.targ_lang = create_language_pairs(self.path)
        self.input_tensor, self.input_lang_tokenizer = tokenize(self.inp_lang)
        self.target_tensor, self.target_lang_tokenizer = tokenize(self.targ_lang)
        self.input_tensor_train, self.input_tensor_val, self.target_tensor_train, self.target_tensor_val = train_test_split(self.input_tensor, self.target_tensor, test_size=0.2)
        
        inp_tokenizer_json = self.input_lang_tokenizer.to_json()
        targ_tokenizer_json = self.target_lang_tokenizer.to_json()
        
        with io.open(INPUT_TOKENIZER_PATH, 'w', encoding='utf-8') as f:
            f.write(json.dumps(inp_tokenizer_json, ensure_ascii=False))

        with io.open(TARGET_TOKENIZER_PATH, 'w', encoding='utf-8') as f:
            f.write(json.dumps(targ_tokenizer_json, ensure_ascii=False))

    def load_train_dataset(self, batch_size:int):
        buffer_size = len(self.input_tensor)
        dataset = tf.data.Dataset.from_tensor_slices((self.input_tensor_train, self.target_tensor_train)).shuffle(buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset

    def load_val_dataset(self, batch_size:int):
        dataset = tf.data.Dataset.from_tensor_slices((self.input_tensor_val,self.target_tensor_val))
        dataset = dataset.batch(batch_size)
        return dataset