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




    











        