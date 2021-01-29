"""Code for constructing the model and get the outputs from the model."""

import tensorflow as tf
import layers
from main import NUM_CHANNEL
import numpy as np

n_base_filters = 16
n_base_ch_se = 32
mlp_ch = 128
img_ch = 1
scale = 4


def get_outputs(inputs, num_cls, is_training=True, drop_rate=0.3,
                modality_config_s1=None, modality_config_s2=None, modality_config_s3=None, modality_config_s4=None,
                modality_num=4.0,
                modality_config=None):

        input_flair = inputs['input_flair']
        input_t1c__ = inputs['input_t1c']
        input_t1___ = inputs['input_t1']
        input_t2___ = inputs['input_t2']
        input_brainmask = inputs['input_brainmask']
        input_label = inputs['input_label']

        with tf.variable_scope("Model", reuse=tf.AUTO_REUSE) as scope:
            style_flair = style_encoder(input_flair, name='se_flair')
            style_t1c__ = style_encoder(input_t1c__, name='se_t1c__')
            style_t1___ = style_encoder(input_t1___, name='se_t1___')
            style_t2___ = style_encoder(input_t2___, name='se_t2___')

            content_flair = content_encoder(input_flair, is_training=is_training, drop_rate=drop_rate, name='ce_flair')
            content_t1c__ = content_encoder(input_t1c__, is_training=is_training, drop_rate=drop_rate, name='ce_t1c')
            content_t1___ = content_encoder(input_t1___, is_training=is_training, drop_rate=drop_rate, name='ce_t1')
            content_t2___ = content_encoder(input_t2___, is_training=is_training, drop_rate=drop_rate, name='ce_t2')

            content_flair_s1 = tf.multiply(content_flair['s1'],tf.expand_dims(modality_config_s1[0, :, :, :, :], axis=0))
            content_flair_s2 = tf.multiply(content_flair['s2'],tf.expand_dims(modality_config_s2[0, :, :, :, :], axis=0))
            content_flair_s3 = tf.multiply(content_flair['s3'],tf.expand_dims(modality_config_s3[0, :, :, :, :], axis=0))
            content_flair_s4 = tf.multiply(content_flair['s4'],tf.expand_dims(modality_config_s4[0, :, :, :, :], axis=0))

            content_t1c___s1 = tf.multiply(content_t1c__['s1'],tf.expand_dims(modality_config_s1[1, :, :, :, :], axis=0))
            content_t1c___s2 = tf.multiply(content_t1c__['s2'],tf.expand_dims(modality_config_s2[1, :, :, :, :], axis=0))
            content_t1c___s3 = tf.multiply(content_t1c__['s3'],tf.expand_dims(modality_config_s3[1, :, :, :, :], axis=0))
            content_t1c___s4 = tf.multiply(content_t1c__['s4'],tf.expand_dims(modality_config_s4[1, :, :, :, :], axis=0))

            content_t1____s1 = tf.multiply(content_t1___['s1'],tf.expand_dims(modality_config_s1[2, :, :, :, :], axis=0))
            content_t1____s2 = tf.multiply(content_t1___['s2'],tf.expand_dims(modality_config_s2[2, :, :, :, :], axis=0))
            content_t1____s3 = tf.multiply(content_t1___['s3'],tf.expand_dims(modality_config_s3[2, :, :, :, :], axis=0))
            content_t1____s4 = tf.multiply(content_t1___['s4'],tf.expand_dims(modality_config_s4[2, :, :, :, :], axis=0))

            content_t2____s1 = tf.multiply(content_t2___['s1'],tf.expand_dims(modality_config_s1[3, :, :, :, :], axis=0))
            content_t2____s2 = tf.multiply(content_t2___['s2'],tf.expand_dims(modality_config_s2[3, :, :, :, :], axis=0))
            content_t2____s3 = tf.multiply(content_t2___['s3'],tf.expand_dims(modality_config_s3[3, :, :, :, :], axis=0))
            content_t2____s4 = tf.multiply(content_t2___['s4'],tf.expand_dims(modality_config_s4[3, :, :, :, :], axis=0))

            content_share_c1_concat = tf.concat([content_flair_s1, content_t1c___s1, content_t1____s1, content_t2____s1], axis=-1, name='concat_c1')
            content_share_c1_attmap = layers.general_conv3d(content_share_c1_concat, 4, pad_type='reflect', name="att_c1", act_type=None)
            content_share_c1_attmap = tf.nn.sigmoid(content_share_c1_attmap)
            content_share_c1 = tf.concat([tf.multiply(content_flair_s1, tf.tile(tf.expand_dims(content_share_c1_attmap[:, :, :, :, 0], axis=-1), tf.constant([1,1,1,1,16]))),
                                          tf.multiply(content_t1c___s1, tf.tile(tf.expand_dims(content_share_c1_attmap[:, :, :, :, 1], axis=-1), tf.constant([1,1,1,1,16]))),
                                          tf.multiply(content_t1____s1, tf.tile(tf.expand_dims(content_share_c1_attmap[:, :, :, :, 2], axis=-1), tf.constant([1,1,1,1,16]))),
                                          tf.multiply(content_t2____s1, tf.tile(tf.expand_dims(content_share_c1_attmap[:, :, :, :, 3], axis=-1), tf.constant([1,1,1,1,16])))], axis=-1)

            content_share_c2_concat = tf.concat([content_flair_s2, content_t1c___s2, content_t1____s2, content_t2____s2], axis=-1, name='concat_c2')
            content_share_c2_attmap = layers.general_conv3d(content_share_c2_concat, 4, pad_type='reflect',name="att_c2", act_type=None)
            content_share_c2_attmap = tf.nn.sigmoid(content_share_c2_attmap)
            content_share_c2 = tf.concat([tf.multiply(content_flair_s2, tf.tile(tf.expand_dims(content_share_c2_attmap[:, :, :, :, 0], axis=-1), tf.constant([1,1,1,1,32]))),
                                          tf.multiply(content_t1c___s2, tf.tile(tf.expand_dims(content_share_c2_attmap[:, :, :, :, 1], axis=-1), tf.constant([1,1,1,1,32]))),
                                          tf.multiply(content_t1____s2, tf.tile(tf.expand_dims(content_share_c2_attmap[:, :, :, :, 2], axis=-1), tf.constant([1,1,1,1,32]))),
                                          tf.multiply(content_t2____s2, tf.tile(tf.expand_dims(content_share_c2_attmap[:, :, :, :, 3], axis=-1), tf.constant([1,1,1,1,32])))], axis=-1)

            content_share_c3_concat = tf.concat([content_flair_s3, content_t1c___s3, content_t1____s3, content_t2____s3], axis=-1, name='concat_c3')
            content_share_c3_attmap = layers.general_conv3d(content_share_c3_concat, 4, pad_type='reflect',name="att_c3", act_type=None)
            content_share_c3_attmap = tf.nn.sigmoid(content_share_c3_attmap)
            content_share_c3 = tf.concat([tf.multiply(content_flair_s3, tf.tile(tf.expand_dims(content_share_c3_attmap[:, :, :, :, 0], axis=-1), tf.constant([1,1,1,1,64]))),
                                          tf.multiply(content_t1c___s3, tf.tile(tf.expand_dims(content_share_c3_attmap[:, :, :, :, 1], axis=-1), tf.constant([1,1,1,1,64]))),
                                          tf.multiply(content_t1____s3, tf.tile(tf.expand_dims(content_share_c3_attmap[:, :, :, :, 2], axis=-1), tf.constant([1,1,1,1,64]))),
                                          tf.multiply(content_t2____s3, tf.tile(tf.expand_dims(content_share_c3_attmap[:, :, :, :, 3], axis=-1), tf.constant([1,1,1,1,64])))], axis=-1)

            content_share_c4_concat = tf.concat([content_flair_s4, content_t1c___s4, content_t1____s4, content_t2____s4], axis=-1, name='concat_c4')
            content_share_c4_attmap = layers.general_conv3d(content_share_c4_concat, 4, pad_type='reflect',name="att_c4", act_type=None)
            content_share_c4_attmap = tf.nn.sigmoid(content_share_c4_attmap)
            content_share_c4 = tf.concat([tf.multiply(content_flair_s4, tf.tile(tf.expand_dims(content_share_c4_attmap[:, :, :, :, 0], axis=-1), tf.constant([1,1,1,1,128]))),
                                          tf.multiply(content_t1c___s4, tf.tile(tf.expand_dims(content_share_c4_attmap[:, :, :, :, 1], axis=-1), tf.constant([1,1,1,1,128]))),
                                          tf.multiply(content_t1____s4, tf.tile(tf.expand_dims(content_share_c4_attmap[:, :, :, :, 2], axis=-1), tf.constant([1,1,1,1,128]))),
                                          tf.multiply(content_t2____s4, tf.tile(tf.expand_dims(content_share_c4_attmap[:, :, :, :, 3], axis=-1), tf.constant([1,1,1,1,128])))], axis=-1)

            content_share_c1 = layers.general_conv3d(content_share_c1, n_base_filters, k_size=1, pad_type='reflect', name='fusion_c1')
            content_share_c2 = layers.general_conv3d(content_share_c2, n_base_filters * 2, k_size=1, pad_type='reflect', name='fusion_c2')
            content_share_c3 = layers.general_conv3d(content_share_c3, n_base_filters * 4, k_size=1, pad_type='reflect', name='fusion_c3')
            content_share_c4 = layers.general_conv3d(content_share_c4, n_base_filters * 8, k_size=1, pad_type='reflect', name='fusion_c4')

            reconstruct_flair, mu_flair, sigma_flair = image_decoder(style_flair, content_share_c4, name='image_de_flair')
            reconstruct_t1c__, mu_t1c__, sigma_t1c__ = image_decoder(style_t1c__, content_share_c4, name='image_de_t1c__')
            reconstruct_t1___, mu_t1___, sigma_t1___ = image_decoder(style_t1___, content_share_c4, name='image_de_t1___')
            reconstruct_t2___, mu_t2___, sigma_t2___ = image_decoder(style_t2___, content_share_c4, name='image_de_t2___')

            mask_de_input = {
                'e1_out': content_share_c1,
                'e2_out': content_share_c2,
                'e3_out': content_share_c3,
                'e4_out': content_share_c4,
            }

            seg_pred, seg_logit = mask_decoder(mask_de_input, num_cls, name='mask_de')

            return {
                'style_flair': style_flair,
                'style_t1c__': style_t1c__,
                'style_t1___': style_t1___,
                'style_t2___': style_t2___,
                'content_flair': content_flair,
                'content_t1c__': content_t1c__,
                'content_t1___': content_t1___,
                'content_t2___': content_t2___,
                'mu_flair': mu_flair,
                'mu_t1c__': mu_t1c__,
                'mu_t1___': mu_t1___,
                'mu_t2___': mu_t2___,
                'sigma_flair': sigma_flair,
                'sigma_t1c__': sigma_t1c__,
                'sigma_t1___': sigma_t1___,
                'sigma_t2___': sigma_t2___,
                'reconstruct_flair': reconstruct_flair,
                'reconstruct_t1c__': reconstruct_t1c__,
                'reconstruct_t1___': reconstruct_t1___,
                'reconstruct_t2___': reconstruct_t2___,
                'seg_pred': seg_pred,
                'seg_logit': seg_logit,
            }


def style_encoder(input, name='style_encoder'):
    with tf.variable_scope(name):
        x = layers.general_conv3d(input, n_base_ch_se, k_size=7, s=1, pad_type="reflect", name='c_0', norm_type=None, act_type='relu')

        x = layers.general_conv3d(x, n_base_ch_se*2, k_size=4, s=2, pad_type="reflect", name='c_1', norm_type=None, act_type='relu')

        x = layers.general_conv3d(x, n_base_ch_se*4, k_size=4, s=2, pad_type="reflect", name='c_2', norm_type=None, act_type='relu')

        x = layers.general_conv3d(x, n_base_ch_se*4, k_size=4, s=2, pad_type="reflect", name='c_3', norm_type=None, act_type='relu')
        x = layers.general_conv3d(x, n_base_ch_se*4, k_size=4, s=2, pad_type="reflect", name='c_4', norm_type=None, act_type='relu')

        x = tf.reduce_mean(x, axis=[1, 2, 3], keep_dims=True, name='gap')

        x = layers.general_conv3d(x, 8, k_size=1, s=1, pad_type="reflect", name='se_logit', norm_type=None, act_type=None)

        return x


def content_encoder(input, is_training=True, drop_rate=0.3, name='content_encoder'):
    with tf.variable_scope(name):
        e1_c1 = layers.general_conv3d(input, n_base_filters, pad_type='reflect', name="e1_c1")
        e1_c2 = layers.general_conv3d(e1_c1, n_base_filters, pad_type='reflect', name="e1_c2", drop_rate=drop_rate, is_training=is_training)
        e1_c3 = layers.general_conv3d(e1_c2, n_base_filters, pad_type='reflect', name="e1_c3")
        e1_out = e1_c1 + e1_c3

        e2_c1 = layers.general_conv3d(e1_out, n_base_filters * 2, s=2, pad_type='reflect', name="e2_c1")
        e2_c2 = layers.general_conv3d(e2_c1, n_base_filters * 2, pad_type='reflect', name="e2_c2", drop_rate=drop_rate, is_training=is_training)
        e2_c3 = layers.general_conv3d(e2_c2, n_base_filters * 2, pad_type='reflect', name="e2_c3")
        e2_out = e2_c1 + e2_c3

        e3_c1 = layers.general_conv3d(e2_out, n_base_filters * 4, s=2, pad_type='reflect', name="e3_c1")
        e3_c2 = layers.general_conv3d(e3_c1, n_base_filters * 4, pad_type='reflect', name="e3_c2", drop_rate=drop_rate, is_training=is_training)
        e3_c3 = layers.general_conv3d(e3_c2, n_base_filters * 4, pad_type='reflect', name="e3_c3")
        e3_out = e3_c1 + e3_c3

        e4_c1 = layers.general_conv3d(e3_out, n_base_filters * 8, s=2, pad_type='reflect', name="e4_c1")
        e4_c2 = layers.general_conv3d(e4_c1, n_base_filters * 8, pad_type='reflect', name="e4_c2", drop_rate=drop_rate, is_training=is_training)
        e4_c3 = layers.general_conv3d(e4_c2, n_base_filters * 8, pad_type='reflect', name="e4_c3")
        e4_out = e4_c1 + e4_c3

        return {
            's1':e1_out,
            's2':e2_out,
            's3':e3_out,
            's4':e4_out,
        }


def image_decoder(style, content, name='image_decoder'):
    channel = mlp_ch
    with tf.variable_scope(name):
        mu, sigma = mlp(style)
        x = content

        for i in range(4):
            x = layers.adaptive_resblock(x, channel, mu, sigma, scope='adaptive_resblock' + str(i))

        for i in range(scale-1):
            # # IN removes the original feature mean and variance that represent important style information
            x = tf.keras.layers.UpSampling3D(size=(2, 2, 2), data_format='channels_last')(x)
            x = layers.general_conv3d(x, channel // 2, k_size=5, s=1, pad_type='reflect', name='conv_' + str(i), norm_type=None, act_type=None)
            x = layers.layer_norm(x, scope='layer_norm_' + str(i))
            x = tf.nn.relu(x)

            channel = channel // 2

    x = layers.general_conv3d(x, img_ch, k_size=7, s=1, pad_type='reflect', name='G_logit', norm_type = None, act_type = None)

    return x, mu, sigma


def mask_decoder(input, num_cls, name='mask_decoder'):
    e4_out = input['e4_out']
    e3_out = input['e3_out']
    e2_out = input['e2_out']
    e1_out = input['e1_out']

    with tf.variable_scope(name):
        d3 = tf.keras.layers.UpSampling3D(size=(2, 2, 2), data_format='channels_last')(e4_out)
        d3_c1 = layers.general_conv3d(d3, n_base_filters * 4, pad_type='reflect', name="d3_c1")
        d3_cat = tf.concat([d3_c1, e3_out], axis=-1)
        d3_c2 = layers.general_conv3d(d3_cat, n_base_filters * 4, pad_type='reflect', name="d3_c2")
        d3_out = layers.general_conv3d(d3_c2, n_base_filters * 4, k_size=1, pad_type='reflect', name="d3_out")

        d2 = tf.keras.layers.UpSampling3D(size=(2, 2, 2), data_format='channels_last')(d3_out)
        d2_c1 = layers.general_conv3d(d2, n_base_filters * 2, pad_type='reflect', name="d2_c1")
        d2_cat = tf.concat([d2_c1, e2_out], axis=-1)
        d2_c2 = layers.general_conv3d(d2_cat, n_base_filters * 2, pad_type='reflect', name="d2_c2")
        d2_out = layers.general_conv3d(d2_c2, n_base_filters * 2, k_size=1, pad_type='reflect', name="d2_out")

        d1 = tf.keras.layers.UpSampling3D(size=(2, 2, 2), data_format='channels_last')(d2_out)
        d1_c1 = layers.general_conv3d(d1, n_base_filters, pad_type='reflect', name="d1_c1")
        d1_cat = tf.concat([d1_c1, e1_out], axis=-1)
        d1_c2 = layers.general_conv3d(d1_cat, n_base_filters, pad_type='reflect', name="d1_c2")
        d1_out = layers.general_conv3d(d1_c2, n_base_filters, k_size=1, pad_type='reflect', name="d1_out")

        seg_logit = layers.general_conv3d(d1_out, num_cls, k_size=1, pad_type='reflect', name='seg_logit', norm_type=None, act_type=None)
        seg_pred = tf.nn.softmax(seg_logit, name='seg_pred')

        return seg_pred, seg_logit


def mlp(style, name='MLP'):
    channel = mlp_ch
    with tf.variable_scope(name):
        x = layers.linear(style, channel, name='linear_0')
        x = tf.nn.relu(x)

        x = layers.linear(x, channel, name='linear_1')
        x = tf.nn.relu(x)

        mu = layers.linear(x, channel, name='mu')
        sigma = layers.linear(x, channel, name='sigma')

        mu = tf.reshape(mu, shape=[-1, 1, 1, 1, channel])
        sigma = tf.reshape(sigma, shape=[-1, 1, 1, 1, channel])

        return mu, sigma