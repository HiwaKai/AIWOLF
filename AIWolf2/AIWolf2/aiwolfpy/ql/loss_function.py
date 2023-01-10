from keras import backend as K
import tensorflow as tf

#損失関数
def huberloss(y_true,y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0 #もともとはsrr
    L2 = 0.5 * K.square(err)
    L1 = K.abs(err) - 0.5
    loss = tf.where(cond,L2,L1)

    return K.mean(loss)
