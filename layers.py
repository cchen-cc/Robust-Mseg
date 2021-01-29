import tensorflow as tf


def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):

    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak * x)


def instance_norm(x):

    with tf.variable_scope("instance_norm"):
        out = tf.contrib.layers.instance_norm(x)

        return out


def batch_norm(x, is_training = True):

    with tf.variable_scope("batch_norm"):
        return tf.contrib.layers.batch_normalization(x, decay=0.90, is_training=is_training,
                                            updates_collections=None)


def group_norm(x):

    with tf.variable_scope("group_norm"):
        return tf.contrib.layers.group_norm(x, groups=32, channels_axis=-1, reduction_axes=(-4, -3, -2))



def general_conv3d(input, o_d, k_size=3, s=1,
                   pad_type="zero", stddev = 0.01, name="conv3d",
                   drop_rate=0.0, norm_type='Ins', is_training=True, act_type='lrelu', relufactor=0.2):

    with tf.variable_scope(name):
        pad = (k_size-1)/2
        if pad_type == 'zero':
            input = tf.pad(input, [[0, 0], [pad, pad], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect':
            input = tf.pad(input, [[0, 0], [pad, pad], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        conv = tf.layers.conv3d(
            input, o_d, k_size, s,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
            bias_initializer=None,
        )

        conv = tf.layers.dropout(conv, drop_rate, training=is_training)

        if norm_type is not None:
            if norm_type=='Ins':
                conv = instance_norm(conv)
            elif norm_type=='Batch':
                conv = batch_norm(conv, is_training)
            elif norm_type=='Group':
                conv = group_norm(conv)

        if act_type is not None:
            if act_type=='relu':
                conv = tf.nn.relu(conv, "relu")
            elif act_type=='lrelu':
                conv = tf.nn.leaky_relu(conv, relufactor, 'lrelu')

        return conv


def dilate_conv2d(inputconv, i_d=64, o_d=64, f_h=7, f_w=7, rate=2, stddev=0.01,
                   padding="VALID", name="dilate_conv2d", do_norm=True, do_relu=True, keep_rate=None,
                   relufactor=0, norm_type=None, is_training=True):
    with tf.variable_scope(name):
        f_1 = tf.get_variable('weights', [f_h, f_w, i_d, o_d], initializer=tf.truncated_normal_initializer(stddev=stddev))
        b_1 = tf.get_variable('biases', [o_d], initializer=tf.constant_initializer(0.0, tf.float32))
        di_conv_2d = tf.nn.atrous_conv2d(inputconv, f_1, rate=rate, padding=padding)

        if not keep_rate is None:
            di_conv_2d = tf.nn.dropout(di_conv_2d, keep_rate)

        if do_norm:
            if norm_type is None:
                print "normalization type is not specified!"
                quit()
            elif norm_type=='Ins':
                di_conv_2d = instance_norm(di_conv_2d)
            elif norm_type=='Batch':
                di_conv_2d = batch_norm(di_conv_2d, is_training)

        if do_relu:
            if(relufactor == 0):
                di_conv_2d = tf.nn.relu(di_conv_2d, "relu")
            else:
                di_conv_2d = lrelu(di_conv_2d, relufactor, "lrelu")

        return di_conv_2d


def general_deconv2d(inputconv, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1,
                     stddev=0.02, padding="VALID", name="deconv2d",
                     do_norm=True, do_relu=True, relufactor=0, norm_type=None, is_training=True):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d_transpose(
            inputconv, o_d, [f_h, f_w],
            [s_h, s_w], padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.constant_initializer(0.0)
        )

        if do_norm:
            if norm_type is None:
                print "normalization type is not specified!"
                quit()
            elif norm_type=='Ins':
                conv = instance_norm(conv)
            elif norm_type=='Batch':
                conv = batch_norm(conv, is_training)


        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def linear(x, units, name='linear'):
    with tf.variable_scope(name):
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), use_bias=True)

        return x


def adaptive_resblock(x_init, channels, mu, sigma, scope='adaptive_resblock') :
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = general_conv3d(x_init, channels, k_size=3, s=1, pad_type='reflect', norm_type=None, act_type=None)
            x = adaptive_instance_norm(x, mu, sigma)
            x = tf.nn.relu(x)

        with tf.variable_scope('res2'):
            x = general_conv3d(x, channels, k_size=3, s=1, pad_type='reflect', norm_type=None, act_type=None)
            x = adaptive_instance_norm(x, mu, sigma)

        return x + x_init


def adaptive_instance_norm(content, gamma, beta, epsilon=1e-5):

    c_mean, c_var = tf.nn.moments(content, axes=[1, 2, 3], keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)

    return gamma * ((content - c_mean) / c_std) + beta


def up_sample(x, scale_factor=2):
    _, h, w, d, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor, d * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)


def layer_norm(x, scope='layer_norm') :
    return tf.contrib.layers.layer_norm(x, center=True, scale=True, scope=scope)