import tensorflow as tf


def _softmax_weighted_loss(pred, label, num_cls):
    '''
    calculate weighted cross-entropy loss, the weight is dynamic dependent on the data
    '''

    for i in xrange(num_cls):
        labeli = label[:,:,:,:,i]
        predi = pred[:,:,:,:,i]
        weighted = 1.0-(tf.reduce_sum(labeli) / tf.reduce_sum(label))
        if i == 0:
            raw_loss = -1.0 * weighted * labeli * tf.log(tf.clip_by_value(predi, 0.005, 1))
        else:
            raw_loss += -1.0 * weighted * labeli * tf.log(tf.clip_by_value(predi, 0.005, 1))

    loss = tf.reduce_mean(raw_loss)

    return loss


def _dice_loss_fun(pred, label, num_cls):
    '''
    calculate dice loss, - 2*interesction/union, with relaxed for gradients backpropagation
    '''
    dice = 0.0
    eps = 1e-7

    for i in xrange(num_cls):
        inse = tf.reduce_sum(pred[:, :, :, :, i] * label[:, :, :, :, i])
        l = tf.reduce_sum(pred[:, :, :, :, i] * pred[:, :, :, :, i])
        r = tf.reduce_sum(label[:, :, :, :, i])
        dice += 2.0 * inse/(l+r+eps)

    return 1.0 - 1.0 * dice / num_cls


def task_loss(pred, label, num_cls):

    ce_loss = _softmax_weighted_loss(pred, label, num_cls)
    dice_loss = _dice_loss_fun(pred, label, num_cls)

    return ce_loss, dice_loss


################################################################################
# KL DIVERGENCY LOSS weight 0.01?
################################################################################
def kl_loss(mu, logvar) :
    loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(logvar) - 1 - logvar, axis=-1)
    loss = tf.reduce_mean(loss)


    return loss