import tensorflow as tf

def build_lstm(lstm_size, num_layers, n_seq, keep_prob):
    #get lstm layer
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    #get dropout layer
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob = keep_prob)
    
    #stack up multiple layers
    cell = tf.contrib.rnn.MultiRNNCell([drop]*num_layers)
    initial_state = cell.zero_state(n_seq, tf.float32)
    
    return  cell, initial_state