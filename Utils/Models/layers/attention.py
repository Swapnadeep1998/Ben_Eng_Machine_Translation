import tensorflow as tf
from tensorflow.keras import layers 


class BahdanauAttention(layers.Layer):
    def __init__(self, units:int):

        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    
    def call(self, query, values):

        query_with_time_axis = tf.expand_dims(query,1)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)
            ))
        
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights*values, axis=1)

        return context_vector, attention_weights

