import tensorflow as tf 
from tensorflow.keras import Model, layers
import sys
sys.path.append(".")
from Utils.Models.layers.attention import BahdanauAttention
import json

class AttentionDecoder(Model):
    def __init__(self, vocab_size:int,
        embedding_dim:int, dec_units:int, 
        batch_size:int):

        super(AttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units
        self.batch_size = batch_size

        self.embedding = layers.Embedding(self.vocab_size,
                                self.embedding_dim)
        
        self.gru = layers.GRU(self.dec_units,
                                return_sequences=True,
                                return_state=True,
                                recurrent_initializer='glorot_uniform')
        
        self.dense = layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    
    def call(self, x, hidden, enc_output):

        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector,1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.dense(output)

        return x, state, attention_weights

    
