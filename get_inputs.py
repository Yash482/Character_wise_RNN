import tensorflow as tf

def build_inputs(n_seq, seq_len):
    # define the placeholders for the inputs and targets
    # define the keep prob for the dropout layer
    inputs = tf.placeholder(tf.int32, [n_seq, seq_len], name = 'inputs')
    targets = tf.placeholder(tf.int32, [n_seq, seq_len], name = 'targets')
    
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    
    return inputs, targets, keep_prob
