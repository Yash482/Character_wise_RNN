import tensorflow as tf

def build_optimizer(loss, lr, grad_clip):
    #gradient vanishing problem is tackeled by lstm cells
    #here we tackel gradient exploding problem by dedining a threshold(grad_clip)
    #when gradient increases threshold, it get equals to that only
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op= tf.train.AdamOptimizer(lr)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    return optimizer


def build_output(lstm_output, in_size, out_size):
    seq_output = tf.concat(lstm_output, axis = 1)
    x = tf.reshape(seq_output, [-1, in_size])
    
    #connect softmax
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev = 0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))

    logits = tf.matmul(x, softmax_w) + softmax_b
    output = tf.nn.softmax(logits, name = 'predictions')
    
    return output, logits


def build_loss(logits, targets, lstm_size, num_classes):
    #calculate loss for targets and logits
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    
    #softmax cross_entropy loss
    loss= tf.nn.softmax_cross_entropy_with_logits(logits, labels = y_reshaped)
    loss = tf.reduce_mean(loss)
    return loss    