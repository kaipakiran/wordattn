import tensorflow as tf 
from tensorflow.keras import initializers
import numpy as np 
from attention import Attn,MyFlatten, MyReshape,NonMasking
def get_attn_layer(embedding_seq,num_cells=64, input_len=20):
    gru_layer_pre = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_cells, return_sequences = True))(embedding_seq)

    td_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_cells*2,activation = 'tanh')) (gru_layer_pre)

    att = Attn(num_cells)(td_dense)
    print("Attn Shape before flatten",tf.keras.backend.int_shape(att))
    # att = tf.keras.layers.Flatten()(att)
    # att = MyFlatten()(att)
    #model.add(Lambda(lambda x: K.batch_flatten(x)))
    att = NonMasking()(att)
    #att = MyReshape((input_len,))(att)
    att = tf.keras.layers.Reshape((input_len,))(att)
    print("Attn Shape after flatten",tf.keras.backend.int_shape(att))
    #print("Attn Shape after reshape",tf.keras.backend.int_shape(test))
    
    word_att_softmax = tf.keras.layers.Activation("softmax") (att)

    #word_att_softmax_reshape = tf.keras.layers.Reshape((input_len,1))(word_att_softmax)
    word_att_softmax_reshape = MyReshape((input_len,1))(word_att_softmax)
    apply_attn_to_gru = tf.keras.layers.Multiply()([word_att_softmax_reshape, gru_layer_pre]) 

    attn_out = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis = 1)) (apply_attn_to_gru)
    return attn_out

def get_model(vocab_size=1000,emb_dim = 100, input_len = (4,20), output_size = 2, num_cells = 64):
    """
    Returns a model with word attention based on iputs.
    param name: vocab_size
    param type: int
    param name: input_len
    param type: 
    """
    embedding_layer = tf.keras.layers.Embedding(vocab_size, emb_dim,
                                                input_length = input_len,mask_zero=True)
    
    sequence_inputs_org = tf.keras.layers.Input(shape = (input_len[0],input_len[1]))
    print(tf.keras.backend.shape(sequence_inputs_org))
    sequence_inputs = tf.keras.layers.Lambda(lambda x: tf.unstack(x,axis=1))(sequence_inputs_org)
    print(tf.keras.backend.shape(sequence_inputs))
    sentence_vectors = []
    for sequence_input in sequence_inputs:
        embedding_seq = embedding_layer(sequence_input)
        attn_out = get_attn_layer(num_cells=num_cells,input_len=input_len[1],embedding_seq=embedding_seq)
        sentence_vectors.append(attn_out)
    print(sentence_vectors)
    sentence_vectors = tf.keras.layers.Lambda(lambda x: tf.stack(x,axis=1))(sentence_vectors)
    print(tf.keras.backend.int_shape(sentence_vectors[0]))
    sent_attn_out = get_attn_layer(num_cells=num_cells,input_len=input_len[0],embedding_seq=sentence_vectors)
    print(sent_attn_out)
    inter_dense = tf.keras.layers.Dense(num_cells*4, activation = 'relu') (sent_attn_out)
    print(tf.keras.backend.int_shape(inter_dense))
    inter_dense_do = tf.keras.layers.Dropout(0.2)(inter_dense)

    output = tf.keras.layers.Dense(output_size, activation = 'softmax') (inter_dense_do)

    print(tf.keras.backend.int_shape(output))
    model = tf.keras.models.Model(inputs = sequence_inputs_org, outputs = output)

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam',metrics=['acc'])

    return model


if __name__ == "__main__":
    #model = get_model()
    model= get_model(vocab_size=10000,emb_dim = 100, input_len = (11,50), output_size = 8, num_cells = 64)
    print(model.summary())
    