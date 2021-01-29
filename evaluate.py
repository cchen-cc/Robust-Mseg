
from datetime import datetime
import numpy as np
import os

import tensorflow as tf

import data_loader, losses
import model
import shutil
import csv
# import pymedimage.niftiio as nio
from stats_func import *
import argparse
import medpy.io as medio
import time

model_list = [89999]

base_fd = ''
raw_data_pth = ''
test_list_pth = './datalist/test.txt'

CHECKPOINT_PATH_LIST = ['./output/multimodal-' + str(i) for i in model_list]
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DROP_RATE = 0.0
IS_TRAINING = False

BATCH_SIZE = 1
HEIGHT = 80
WIDTH = 80
DEPTH = 80
overlap_perc = 0.5
NUM_CLS = 5
NUM_CHANNEL = 1
LABEL_FLAG = True

modality_missing_list = [
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [1, 0, 1, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 1],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 1],
    [1, 1, 1, 1],
]

dict_pth = '../../Data/ID_dict.npy'

modality_list = [
    'Flair',
    'T1c',
    'T1',
    'T2',
]


class MultiModal:

    def __init__(self):

        test_model = 1

        self.modality_config = tf.placeholder(tf.uint8, shape=(4,))

        self.modality_config_s1 = tf.placeholder(tf.float32, [4, 80, 80, 80, 16], name="modality_config_s1")
        self.modality_config_s2 = tf.placeholder(tf.float32, [4, 40, 40, 40, 32], name="modality_config_s2")
        self.modality_config_s3 = tf.placeholder(tf.float32, [4, 20, 20, 20, 64], name="modality_config_s3")
        self.modality_config_s4 = tf.placeholder(tf.float32, [4, 10, 10, 10, 128], name="modality_config_s4")
        self.modality_num = tf.placeholder(tf.float32, shape=(), name="modality_num")

    def model_setup(self):
        self.input_flair = tf.placeholder(
            tf.float32,
            shape=[BATCH_SIZE, HEIGHT, WIDTH, DEPTH, NUM_CHANNEL],
            name="input_flair")
        self.input_t1c = tf.placeholder(
            tf.float32,
            shape=[BATCH_SIZE, HEIGHT, WIDTH, DEPTH, NUM_CHANNEL],
            name="input_t1c")
        self.input_t1 = tf.placeholder(
            tf.float32,
            shape=[BATCH_SIZE, HEIGHT, WIDTH, DEPTH, NUM_CHANNEL],
            name="input_t1")
        self.input_t2 = tf.placeholder(
            tf.float32,
            shape=[BATCH_SIZE, HEIGHT, WIDTH, DEPTH, NUM_CHANNEL],
            name="input_t2")
        self.input_brainmask = tf.placeholder(
            tf.float32,
            shape=None,
            name="input_brainmask")
        self.input_label = tf.placeholder(
            tf.float32,
            shape=[BATCH_SIZE, HEIGHT, WIDTH, DEPTH, NUM_CLS],
            name="label")

        inputs = {
            'input_flair': self.input_flair,
            'input_t1c': self.input_t1c,
            'input_t1': self.input_t1,
            'input_t2': self.input_t2,
            'input_brainmask': self.input_brainmask,
            'input_label': self.input_label,
        }

        self.global_step = tf.train.get_or_create_global_step

        outputs = model.get_outputs(inputs, NUM_CLS, is_training=IS_TRAINING, drop_rate=DROP_RATE,
                                    modality_config_s1=self.modality_config_s1, modality_config_s2=self.modality_config_s2, modality_config_s3=self.modality_config_s3,
                                    modality_config_s4=self.modality_config_s4,
                                    modality_num=self.modality_num,
                                    modality_config=self.modality_config
                                    )

        self.seg_logit = outputs['seg_logit']
        self.seg_pred = outputs['seg_pred']
        self.seg_pred_compact = tf.argmax(self.seg_pred, axis=4)
        self.input_label_compact = tf.argmax(tf.cast(self.input_label, tf.int64), axis=4)

    def test(self):
        """Test Function."""

        with open(test_list_pth, 'r') as fp:
            test_fids = fp.readlines()

        test_fids = [base_fd+'/'+test_fid[:-1] for test_fid in test_fids]

        self.model_setup()
        saver = tf.train.Saver()
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())

        gpu_options = tf.GPUOptions(allow_growth=False)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)

            for CHECKPOINT_PATH in CHECKPOINT_PATH_LIST:
                for modality_missing_conf in modality_missing_list:

                    self._model_dir = '/'.join(CHECKPOINT_PATH.split('/')[:-1])
                    self._pred_dir = os.path.join(self._model_dir, 'test_pred',
                                                  'model-' + CHECKPOINT_PATH.split('/')[-1].split('-')[-1] +
                                                  '-modality-' + str(modality_missing_conf[0]) + str(
                                                      modality_missing_conf[1]) + str(modality_missing_conf[2]) + str(
                                                      modality_missing_conf[3]))
                    self._csv_pth = os.path.join(self._model_dir, 'test_pred/results.csv')
                    if not os.path.exists(self._pred_dir):
                        os.makedirs(self._pred_dir)

                    saver.restore(sess, CHECKPOINT_PATH)

                    modality_num_val = np.sum(modality_missing_conf, dtype=np.float32)
                    modality_config_s1_val = np.zeros((4, 80, 80, 80, 16), dtype=np.float32)
                    modality_config_s2_val = np.zeros((4, 40, 40, 40, 32), dtype=np.float32)
                    modality_config_s3_val = np.zeros((4, 20, 20, 20, 64), dtype=np.float32)
                    modality_config_s4_val = np.zeros((4, 10, 10, 10, 128), dtype=np.float32)
                    for mm in range(0, 4):
                        if modality_missing_conf[mm] == 1:
                            modality_config_s1_val[mm, :, :, :, :] = np.ones((1, 80, 80, 80, 16), dtype=np.float32)
                            modality_config_s2_val[mm, :, :, :, :] = np.ones((1, 40, 40, 40, 32), dtype=np.float32)
                            modality_config_s3_val[mm, :, :, :, :] = np.ones((1, 20, 20, 20, 64), dtype=np.float32)
                            modality_config_s4_val[mm, :, :, :, :] = np.ones((1, 10, 10, 10, 128), dtype=np.float32)
                        else:
                            modality_config_s1_val[mm, :, :, :, :] = np.zeros((1, 80, 80, 80, 16), dtype=np.float32)
                            modality_config_s2_val[mm, :, :, :, :] = np.zeros((1, 40, 40, 40, 32), dtype=np.float32)
                            modality_config_s3_val[mm, :, :, :, :] = np.zeros((1, 20, 20, 20, 64), dtype=np.float32)
                            modality_config_s4_val[mm, :, :, :, :] = np.zeros((1, 10, 10, 10, 128), dtype=np.float32)



                    dice_list_wt = []
                    dice_list_co = []
                    dice_list_ec = []
                    for f_idx, fid in enumerate(test_fids):
                        data_list = []
                        starttime = time.time()
                        for modality in modality_list:
                            image_arr, image_header = medio.load(fid + '/' + modality + '_subtrMeanDivStd.nii.gz')
                            data_list.append(image_arr)

                        h, w, d = image_arr.shape

                        h_cnt = np.int(np.ceil((h - HEIGHT) / (HEIGHT * (1 - overlap_perc))))
                        h_idx_list = range(0, h_cnt)
                        h_idx_list = [h_idx * np.int(HEIGHT * (1 - overlap_perc)) for h_idx in h_idx_list]
                        h_idx_list.append(h - HEIGHT)

                        w_cnt = np.int(np.ceil((w - WIDTH) / (WIDTH * (1 - overlap_perc))))
                        w_idx_list = range(0, w_cnt)
                        w_idx_list = [w_idx * np.int(WIDTH * (1 - overlap_perc)) for w_idx in w_idx_list]
                        w_idx_list.append(w - WIDTH)

                        d_cnt = np.int(np.ceil((d - DEPTH) / (DEPTH * (1 - overlap_perc))))
                        d_idx_list = range(0, d_cnt)
                        d_idx_list = [d_idx * np.int(DEPTH * (1 - overlap_perc)) for d_idx in d_idx_list]
                        d_idx_list.append(d - DEPTH)

                        pred_whole = np.zeros((h, w, d, NUM_CLS))
                        avg_whole = np.zeros((h, w, d, NUM_CLS))
                        avg_block = np.ones((HEIGHT, WIDTH, DEPTH, NUM_CLS))
                        for d_idx in d_idx_list:
                            for w_idx in w_idx_list:
                                for h_idx in h_idx_list:
                                    data_list_crop = []
                                    for d_iter in range(0, len(data_list)):
                                        data_list_crop.append(data_list[d_iter][h_idx:h_idx + HEIGHT, w_idx:w_idx + WIDTH,
                                                              d_idx:d_idx + DEPTH])


                                    if modality_missing_conf[0] == 1:
                                        flair_feed = np.expand_dims(np.expand_dims(data_list_crop[0], axis=0), axis=4)
                                    else:
                                        flair_feed = np.float32(np.zeros((1,HEIGHT,WIDTH,DEPTH,1)))

                                    if modality_missing_conf[1] == 1:
                                        t1c_feed = np.expand_dims(np.expand_dims(data_list_crop[1], axis=0), axis=4)
                                    else:
                                        t1c_feed = np.float32(np.zeros((1,HEIGHT,WIDTH,DEPTH,1)))

                                    if modality_missing_conf[2] == 1:
                                        t1_feed = np.expand_dims(np.expand_dims(data_list_crop[2], axis=0), axis=4)
                                    else:
                                        t1_feed = np.float32(np.zeros((1,HEIGHT,WIDTH,DEPTH,1)))

                                    if modality_missing_conf[3] == 1:
                                        t2_feed = np.expand_dims(np.expand_dims(data_list_crop[3], axis=0), axis=4)
                                    else:
                                        t2_feed = np.float32(np.zeros((1,HEIGHT,WIDTH,DEPTH,1)))

                                    seg_pred_val = sess.run(
                                        self.seg_pred,
                                        feed_dict={
                                            self.input_flair: flair_feed,
                                            self.input_t1c: t1c_feed,
                                            self.input_t1: t1_feed,
                                            self.input_t2: t2_feed,
                                            self.modality_config_s1: modality_config_s1_val,
                                            self.modality_config_s2: modality_config_s2_val,
                                            self.modality_config_s3: modality_config_s3_val,
                                            self.modality_config_s4: modality_config_s4_val,
                                            self.modality_num: modality_num_val,
                                        }
                                    )
                                    seg_pred_val = np.squeeze(seg_pred_val)
                                    pred_whole[h_idx:h_idx + HEIGHT, w_idx:w_idx + WIDTH, d_idx:d_idx + DEPTH,
                                    :] = pred_whole[h_idx:h_idx + HEIGHT, w_idx:w_idx + WIDTH, d_idx:d_idx + DEPTH,
                                         :] + seg_pred_val
                                    avg_whole[h_idx:h_idx + HEIGHT, w_idx:w_idx + WIDTH, d_idx:d_idx + DEPTH,
                                    :] = avg_whole[h_idx:h_idx + HEIGHT, w_idx:w_idx + WIDTH, d_idx:d_idx + DEPTH,
                                         :] + avg_block

                        pred_whole = pred_whole / avg_whole

                        seg_pred_compact_val = np.argmax(pred_whole, axis=-1)
                        seg_pred_compact_val = np.int16(seg_pred_compact_val)

                        brainmask_arr, brainmask_header = medio.load(raw_data_pth + '/' + fid.split('/')[-1] + '/brainmask.nii.gz')
                        roi_ind = np.where(brainmask_arr > 0)
                        roi_bbx = [roi_ind[0].min(), roi_ind[0].max(), roi_ind[1].min(), roi_ind[1].max(), roi_ind[2].min(),
                                   roi_ind[2].max()]
                        pred_complete_arr = np.zeros(brainmask_arr.shape, dtype=np.int16)
                        pred_complete_arr[roi_bbx[0]:roi_bbx[1] + 1, roi_bbx[2]:roi_bbx[3] + 1, roi_bbx[4]:roi_bbx[5] + 1] = seg_pred_compact_val

                        if LABEL_FLAG:
                            image_arr, image_header = medio.load(raw_data_pth + '/' + fid.split('/')[-1] + '/OTMultiClass.nii.gz')
                            dice_wt, dice_co, dice_ec = dice_stat(pred_complete_arr, image_arr)
                            dice_list_wt.append(dice_wt)
                            dice_list_co.append(dice_co)
                            dice_list_ec.append(dice_ec)

                        print (fid.split('/')[-1], time.time()-starttime)

                    if LABEL_FLAG:
                        if os.path.exists(self._csv_pth):
                            with open(self._csv_pth, 'r') as fp:
                                csv_reader = csv.reader(fp, delimiter=';')
                                rows = [row for row in csv_reader]
                        else:
                            rows = [
                                [' ', ' ', ' ', 'wt', 'co', 'ec', 'avg', ' ', 'wt', 'co', 'ec', 'avg', ' ', 'wt', 'co',
                                 'ec', 'avg',
                                 ' ', 'wt', 'co', 'ec', ' ', 'wt', 'co', 'ec', ' ', 'wt', 'co', 'ec']]
                        dice_arr_wt = np.asarray(dice_list_wt)
                        dice_arr_co = np.asarray(dice_list_co)
                        dice_arr_ec = np.asarray(dice_list_ec)

                        dice_arr_wt_mean = np.mean(dice_arr_wt, axis=0)
                        dice_arr_co_mean = np.mean(dice_arr_co, axis=0)
                        dice_arr_ec_mean = np.mean(dice_arr_ec, axis=0)

                        dice_arr_wt_std = np.std(dice_arr_wt, axis=0)
                        dice_arr_co_std = np.std(dice_arr_co, axis=0)
                        dice_arr_ec_std = np.std(dice_arr_ec, axis=0)

                        print CHECKPOINT_PATH + ' Dice:'
                        print (modality_missing_conf, modality_num_val)
                        for i in range(len(dice_arr_wt_mean)):
                            print 'wt:%.2f(%.2f)' % (100*dice_arr_wt_mean[i], 100*dice_arr_wt_std[i])
                            print 'co:%.2f(%.2f)' % (100*dice_arr_co_mean[i], 100*dice_arr_co_std[i])
                            print 'ec:%.2f(%.2f)' % (100*dice_arr_ec_mean[i], 100*dice_arr_ec_std[i])

                        row = [[' ',
                                'Modality-' + str(modality_missing_conf[0]) + str(modality_missing_conf[1]) + str(
                                    modality_missing_conf[2]) + str(modality_missing_conf[3]) +
                                '-filling0-' + str(filling_zero) +
                                '-replace-' + str(replace),
                                'Model-' + CHECKPOINT_PATH.split('/')[-1].split('-')[-1],
                                "{0:.2f}".format(100*dice_arr_wt_mean[0]),
                                "{0:.2f}".format(100*dice_arr_co_mean[0]),
                                "{0:.2f}".format(100*dice_arr_ec_mean[0]),
                                "{0:.2f}".format(100*(dice_arr_wt_mean[0] + dice_arr_co_mean[0] + dice_arr_ec_mean[0]) / 3.0),
                                ' ',
                                "{0:.2f}".format(100*dice_arr_wt_mean[1]),
                                "{0:.2f}".format(100*dice_arr_co_mean[1]),
                                "{0:.2f}".format(100*dice_arr_ec_mean[1]),
                                "{0:.2f}".format(100*(dice_arr_wt_mean[1] + dice_arr_co_mean[1] + dice_arr_ec_mean[1]) / 3.0),
                                ' ',
                                "{0:.2f}".format(100*dice_arr_wt_mean[2]),
                                "{0:.2f}".format(100*dice_arr_co_mean[2]),
                                "{0:.2f}".format(100*dice_arr_ec_mean[2]),
                                "{0:.2f}".format(100*(dice_arr_wt_mean[2] + dice_arr_co_mean[2] + dice_arr_ec_mean[2]) / 3.0),
                                ' ',
                                "{0:.2f}".format(100*dice_arr_wt_std[0]),
                                "{0:.2f}".format(100*dice_arr_co_std[0]),
                                "{0:.2f}".format(100*dice_arr_ec_std[0]),
                                ' ',
                                "{0:.2f}".format(100*dice_arr_wt_std[1]),
                                "{0:.2f}".format(100*dice_arr_co_std[1]),
                                "{0:.2f}".format(100*dice_arr_ec_std[1]),
                                ' ',
                                "{0:.2f}".format(100*dice_arr_wt_std[2]),
                                "{0:.2f}".format(100*dice_arr_co_std[2]),
                                "{0:.2f}".format(100*dice_arr_ec_std[2]),
                                ]]
                        rows = rows + row

                        with open(self._csv_pth, 'wb') as fp:
                            csv_writer = csv.writer(fp, delimiter=';')
                            csv_writer.writerows(rows)


def main():

    multimodal_model = MultiModal()
    multimodal_model.test()


if __name__ == '__main__':
    main()
