import numpy as np
import tensorflow as tf
import time
from collections import namedtuple

#get dataset
with open('text.txt', 'r') as f:
    text = f.read()
 
vocab = set(text)
vocab_to_int = {c:i for i,c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded_text = np.array([vocab_to_int[c] for c in text], dtype = np.int32)


#encoded_text = encoded_text[:100000]
#encoded_text = encoded_text.reshape((10, -1))

#generator fn to yield next batches every time
def get_batches(arr, n_seq, seq_length):
    batch_size = n_seq * seq_length
    n_batches = len(arr)//batch_size
    arr = arr[:n_batches*batch_size]
    arr = arr.reshape((n_seq, -1))
    for i in range(0, arr.shape[1], seq_length):
        #input of RNN
        x= arr[:, i:i+seq_length]
        #target of RNN
        y = np.zeros_like(x)
        
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, i+seq_length]
        except IndexError:
            #for the last set
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y
        
batches = get_batches(encoded_text, n_seq = 8, seq_length =50)
x, y = next(batches)
#now batches are ready

from get_inputs import build_inputs
from get_lstm import build_lstm
from get_output_and_loss import build_loss, build_optimizer, build_output

#Building our model
class CharRNN:
    
    def __init__(self, num_classes, n_seq=64, seq_len=50, 
                       lstm_size=128, num_layers=2, lr=0.001, 
                       grad_clip=5, sampling=False):
    
        # When we're using this network for sampling later, we'll be passing in
        # one character at a time, so providing an option for that
        if sampling == True:
            n_seq, seq_len = 1, 1
        else:
            n_seq, seq_len = n_seq, seq_len

        tf.reset_default_graph()
        
        # Build the input placeholder tensors
        self.inputs, self.targets, self.keep_prob = build_inputs(n_seq, seq_len)

        # Build the LSTM cell
        cell, self.initial_state = build_lstm(lstm_size, num_layers, n_seq, self.keep_prob)

        ### Run the data through the RNN layers
        # First, one-hot encode the input tokens
        x_one_hot = tf.one_hot(self.inputs, num_classes)
        
        # Run each sequence step through the RNN and collect the outputs
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state
        
        # Get softmax predictions and logits
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
        
        # Loss and optimizer (with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, lr, grad_clip)



    



        