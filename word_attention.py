import tensorflow as tf 
from tensorflow.keras import initializers
import numpy as np 
from attention import Attn

def get_model(vocab_size=1000,emb_dim = 100, input_len = 20, output_size = 2, num_cells = 64):
    """
    Returns a model with word attention based on iputs.
    param name: vocab_size
    param type: int
    param name: input_len
    param type: 
    """
    embedding_layer = tf.keras.layers.Embedding(vocab_size, emb_dim,
                                                input_length = input_len)
    
    sequence_input = tf.keras.layers.Input(shape = (input_len, ))
    embedding_seq = embedding_layer(sequence_input)

    gru_layer_pre = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_cells, return_sequences = True))(embedding_seq)

    td_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_cells*2,activation = 'tanh')) (gru_layer_pre)

    att = Attn(num_cells)(td_dense)

    att = tf.keras.layers.Flatten()(att)

    word_att_softmax = tf.keras.layers.Activation("softmax", name = 'word_attn') (att)

    word_att_softmax_reshape = tf.keras.layers.Reshape((input_len,1))(word_att_softmax)

    apply_attn_to_gru = tf.keras.layers.Multiply()([word_att_softmax_reshape, gru_layer_pre]) 

    attn_out = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis = 1)) (apply_attn_to_gru)

    inter_dense = tf.keras.layers.Dense(num_cells*4, activation = 'relu') (attn_out)

    inter_dense_do = tf.keras.layers.Dropout(0.2)(inter_dense)

    output = tf.keras.layers.Dense(output_size, activation = 'softmax') (inter_dense_do)

    model = tf.keras.models.Model(inputs = sequence_input, outputs = output)

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

    return model


if __name__ == "__main__":
    model = get_model()
    print(model.summary())
    