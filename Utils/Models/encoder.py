from tensorflow.keras import Model, layers
import tensorflow as tf


class Encoder(Model):
    def __init__(self, vocab_size:int, embedding_dim:int,
        enc_units:int, batch_size:int):

        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_units = enc_units
        self.batch_size = batch_size

        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.gru = layers.GRU(self.enc_units,
                            return_sequences=True,
                            return_state=True,
                            recurrent_initializer='glorot_uniform')

        
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state


    def initialize_hidden_state(self):
        return tf.zeros([self.batch_size, self.enc_units])
