# Character_wise_RNN
RNN model which learns to make words and predicts next character.

# Get Batches
The text is transformed to batches of sequence and the targets are the next letter as our model has to predict the next character.
We enoded the vocab 1st to integers.

# Get inputs, lstm and output
These contains the funcion which returns the placeholders for the future inputs, our lstm model and the processed output with loss.

# Char RNN
It is the main file which imports different files and where training is done.
