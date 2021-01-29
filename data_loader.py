import tensorflow as tf
import numpy as np


def _decode_samples(data_list):

    decomp_feature = {
        'data_vol': tf.FixedLenFeature([], tf.string),
        'dsize_dim0': tf.FixedLenFeature([], tf.int64),
        'dsize_dim1': tf.FixedLenFeature([], tf.int64),
        'dsize_dim2': tf.FixedLenFeature([], tf.int64),
    }

    data_queue = tf.train.string_input_producer(data_list, shuffle=False)
    reader = tf.TFRecordReader()
    fid, serialized_example = reader.read(data_queue)
    parser = tf.parse_single_example(serialized_example, features=decomp_feature)

    data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
    dsize_dim0 = tf.cast(parser['dsize_dim0'], tf.int64)
    dsize_dim1 = tf.cast(parser['dsize_dim1'], tf.int64)
    dsize_dim2 = tf.cast(parser['dsize_dim2'], tf.int64)
    data_shape = tf.stack((dsize_dim0, dsize_dim1, dsize_dim2))

    data_vol = tf.reshape(data_vol, data_shape)

    return data_vol


def load_data(data_list_pth, data_pth, modality_list, batch_size, gt_flag, crop_size = 80, num_cls=5):

    with open(data_list_pth, 'r') as fp:
        rows = fp.readlines()
    pid_list = [row[:-1] for row in rows]

    cnt = 0
    for modality in modality_list:
        cnt += 1
        data_list = [data_pth+'/'+pid+'/'+modality+'_subtrMeanDivStd.tfrecords' for pid in pid_list]
        if cnt == 1:
            data_vol = tf.expand_dims(_decode_samples(data_list), axis=3)
        else:
            data_vol = tf.concat((data_vol, tf.expand_dims(_decode_samples(data_list), axis=3)), axis=3)

    data_list = [data_pth + '/' + pid + '/' + 'brainmask.tfrecords' for pid in pid_list]
    brainmask = _decode_samples(data_list)

    if gt_flag:
        data_list = [data_pth + '/' + pid + '/' + 'OTMultiClass.tfrecords' for pid in pid_list]
        label = _decode_samples(data_list)
        combine_all = tf.stack((brainmask, label), axis=3)
        combine_all = tf.concat((data_vol, combine_all), axis=3)
        combine_all = tf.random_crop(combine_all, [crop_size, crop_size, crop_size, len(modality_list)+2])
        data_vol = combine_all[:, :, :, 0]
        for m in range(1, len(modality_list)):
            data_vol = tf.concat((data_vol, combine_all[:, :, :, m]), axis=2)
        brainmask = combine_all[:, :, :, len(modality_list)]
        label = combine_all[:, :, :, len(modality_list)+1]

        data_vol = tf.expand_dims(data_vol, axis=3)
        brainmask = tf.expand_dims(brainmask, axis=3)
        label = tf.one_hot(tf.cast(label, tf.int64), depth = num_cls, axis=-1)

        data_vol_batch, brainmask_batch, label_batch = tf.train.shuffle_batch([data_vol, brainmask, label], batch_size, 50, 10, num_threads=1)
        return data_vol_batch, brainmask_batch, label_batch

    else:
        combine_all = tf.concat((data_vol, tf.expand_dims(brainmask, axis=3)), axis=3)
        combine_all = tf.random_crop(combine_all, [crop_size, crop_size, crop_size, len(modality_list)+1])
        data_vol = combine_all[:, :, :, 0]
        for m in range(1, len(modality_list)):
            data_vol = tf.concat((data_vol, combine_all[:, :, :, m]), axis=2)
        brainmask = combine_all[:, :, :, len(modality_list)]

        data_vol = tf.expand_dims(data_vol, axis=3)
        brainmask = tf.expand_dims(brainmask, axis=3)

        data_vol_batch, brainmask_batch = tf.train.shuffle_batch([data_vol, brainmask], batch_size, 50, 5, num_threads=5)
        return data_vol_batch, brainmask_batch



