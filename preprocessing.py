import numpy as np
import os
import medpy.io as medio
import tensorflow as tf
import shutil


def nii2tfrecord():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    raw_data_pth = ''
    save_data_pth = ''

    pid_all = os.listdir(raw_data_pth)
    pid_all.sort()

    with open('../Data/train.txt', 'r') as fp:
        rows = fp.readlines()

    pid_all = [row[:-1] for row in rows]
    pid_all.sort()

    cnt = 0
    for pid_indx, pid in enumerate(pid_all):
        cnt +=1
        modality_all = os.listdir(raw_data_pth+'/'+pid)
        modality_all.sort()
        for modality in modality_all:
            data_arr, data_header = medio.load(raw_data_pth + '/' + pid + '/' + modality)
            data_arr = np.float32(data_arr)
            dsize_dim0_val = data_arr.shape[0]
            dsize_dim1_val = data_arr.shape[1]
            dsize_dim2_val = data_arr.shape[2]

            if not os.path.exists(save_data_pth+'/'+pid):
                os.makedirs(save_data_pth+'/'+pid)

            writer = tf.python_io.TFRecordWriter(save_data_pth+'/'+pid+'/'+modality.split('.')[0]+'.tfrecords')

            feature = {'data_vol': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(data_arr.tostring())])),
                'dsize_dim0': tf.train.Feature(int64_list=tf.train.Int64List(value=[dsize_dim0_val])),
                'dsize_dim1': tf.train.Feature(int64_list=tf.train.Int64List(value=[dsize_dim1_val])),
                'dsize_dim2': tf.train.Feature(int64_list=tf.train.Int64List(value=[dsize_dim2_val])),
                'data_indx': tf.train.Feature(int64_list=tf.train.Int64List(value=[pid_indx])),
                }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            writer.close()
            print (pid, modality, cnt)



if __name__=='__main__':
    nii2tfrecord()