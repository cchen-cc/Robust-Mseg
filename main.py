
from datetime import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import time

import tensorflow as tf

import data_loader, losses, model, modalitydrop_generator
from utils import *

from stats_func import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

train_data_pth = ''
train_list_pth = './datalist/train.txt'
valid_list_pth = './datalist/test.txt'
evaluation_interval = 10
visual_interval = 300
save_interval = 500
drop_rate_value=0.0
is_training_value=True
output_root_dir = './output'
modality_list = [
    'Flair',
    'T1c',
    'T1',
    'T2',
]

BATCH_SIZE = 1
CROP_SIZE = 80
NUM_CHANNEL = 1
NUM_CLS = 5
max_inter = 300
max_epoch = 300
BASE_LEARNING_RATE = 0.0001
save_num_images = 2

class MultiModal:

    def __init__(self):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.drop_rate=tf.placeholder(tf.float32, shape=())
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.modality_config = tf.placeholder(tf.uint8, shape=(4,))

        self.modality_config_s1 = tf.placeholder(tf.float32, [4, 80, 80, 80, 16], name="modality_config_s1")
        self.modality_config_s2 = tf.placeholder(tf.float32, [4, 40, 40, 40, 32], name="modality_config_s2")
        self.modality_config_s3 = tf.placeholder(tf.float32, [4, 20, 20, 20, 64], name="modality_config_s3")
        self.modality_config_s4 = tf.placeholder(tf.float32, [4, 10, 10, 10, 128], name="modality_config_s4")
        self.modality_num = tf.placeholder(tf.float32, shape=(), name="modality_num")

        self._output_dir = os.path.join(output_root_dir, current_time)

    def model_setup(self):

        self.input_flair = tf.placeholder(
            tf.float32, [
                BATCH_SIZE,
                CROP_SIZE,
                CROP_SIZE,
                CROP_SIZE,
                NUM_CHANNEL,
            ], name="input_flair")
        self.input_t1c = tf.placeholder(
            tf.float32, [
                BATCH_SIZE,
                CROP_SIZE,
                CROP_SIZE,
                CROP_SIZE,
                NUM_CHANNEL,
            ], name="input_t1c")
        self.input_t1 = tf.placeholder(
            tf.float32, [
                BATCH_SIZE,
                CROP_SIZE,
                CROP_SIZE,
                CROP_SIZE,
                NUM_CHANNEL,
            ], name="input_t1")
        self.input_t2 = tf.placeholder(
            tf.float32, [
                BATCH_SIZE,
                CROP_SIZE,
                CROP_SIZE,
                CROP_SIZE,
                NUM_CHANNEL,
            ], name="input_t2")
        self.input_brainmask = tf.placeholder(
            tf.float32, [
                BATCH_SIZE,
                CROP_SIZE,
                CROP_SIZE,
                CROP_SIZE,
                NUM_CHANNEL,
            ], name="input_brainmask")
        self.input_label = tf.placeholder(
            tf.float32, [
                BATCH_SIZE,
                CROP_SIZE,
                CROP_SIZE,
                CROP_SIZE,
                NUM_CLS,
            ], name="label")

        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")
        self.lr_summ = tf.summary.scalar("lr", self.learning_rate)

        inputs = {
            'input_flair': self.input_flair,
            'input_t1c': self.input_t1c,
            'input_t1': self.input_t1,
            'input_t2': self.input_t2,
            'input_brainmask': self.input_brainmask,
            'input_label': self.input_label,
        }

        outputs = model.get_outputs(inputs, NUM_CLS, is_training=self.is_training, drop_rate=self.drop_rate,
                                    modality_config_s1=self.modality_config_s1,
                                    modality_config_s2=self.modality_config_s2,
                                    modality_config_s3=self.modality_config_s3,
                                    modality_config_s4=self.modality_config_s4,
                                    modality_num=self.modality_num,
                                    modality_config=self.modality_config)

        self.style_flair = outputs['style_flair']
        self.style_t1c__ = outputs['style_t1c__']
        self.style_t1___ = outputs['style_t1___']
        self.style_t2___ = outputs['style_t2___']

        self.content_flair = outputs['content_flair']
        self.content_t1c__ = outputs['content_t1c__']
        self.content_t1___ = outputs['content_t1___']
        self.content_t2___ = outputs['content_t2___']

        self.mu_flair = outputs['mu_flair']
        self.mu_t1c__ = outputs['mu_t1c__']
        self.mu_t1___ = outputs['mu_t1___']
        self.mu_t2___ = outputs['mu_t2___']

        self.sigma_flair = outputs['sigma_flair']
        self.sigma_t1c__ = outputs['sigma_t1c__']
        self.sigma_t1___ = outputs['sigma_t1___']
        self.sigma_t2___ = outputs['sigma_t2___']

        self.reconstruct_flair = outputs['reconstruct_flair']
        self.reconstruct_t1c__ = outputs['reconstruct_t1c__']
        self.reconstruct_t1___ = outputs['reconstruct_t1___']
        self.reconstruct_t2___ = outputs['reconstruct_t2___']

        self.seg_logit = outputs['seg_logit']
        self.seg_pred = outputs['seg_pred']
        self.seg_pred_compact = tf.argmax(self.seg_pred, axis=4)
        self.input_label_compact = tf.argmax(tf.cast(self.input_label, tf.int64), axis=4)

        self.dice_arr = dice_eval(self.seg_pred_compact, self.input_label_compact, NUM_CLS)
        self.dice_c1 = self.dice_arr[0]
        self.dice_c2 = self.dice_arr[1]
        self.dice_c3 = self.dice_arr[2]
        self.dice_c4 = self.dice_arr[3]

        self.dice_c1_summ = tf.summary.scalar("dice_wt", self.dice_c1)
        self.dice_c2_summ = tf.summary.scalar("dice_co", self.dice_c2)
        self.dice_c3_summ = tf.summary.scalar("dice_ec", self.dice_c3)
        self.dice_c4_summ = tf.summary.scalar("dice_bg", self.dice_c4)

        # Image visualization
        images_summary = tf.py_func(decode_images, [self.input_flair, save_num_images], tf.uint8)
        labels_summary = tf.py_func(decode_labels, [self.input_label_compact, save_num_images, NUM_CLS], tf.uint8)
        preds_summary = tf.py_func(decode_labels, [self.seg_pred_compact, save_num_images, NUM_CLS], tf.uint8)
        self.visual_summary = tf.summary.image('images',
                                               tf.concat(axis=2,
                                                         values=[images_summary, labels_summary, preds_summary]),
                                               max_outputs=save_num_images)  # Concatenate row-wise.

    def compute_losses(self):
        """
        In this function we are defining the variables for loss calculations
        and training model.
        """

        ce_loss, dice_loss = losses.task_loss(self.seg_pred, self.input_label, NUM_CLS)

        l2_loss_content = tf.add_n([0.0001 * tf.nn.l2_loss(v) for v in tf.trainable_variables() if '/kernel' in v.name and ('ce' in v.name or '/mask_de/' in v.name)])
        l2_loss_style_flair = tf.add_n([0.0001 * tf.nn.l2_loss(v) for v in tf.trainable_variables() if '/kernel' in v.name and ('/se_flair/' in v.name or '/image_de_flair/' in v.name)])
        l2_loss_style_t1c__ = tf.add_n([0.0001 * tf.nn.l2_loss(v) for v in tf.trainable_variables() if '/kernel' in v.name and ('/se_t1c__/' in v.name or '/image_de_t1c__/' in v.name)])
        l2_loss_style_t1___ = tf.add_n([0.0001 * tf.nn.l2_loss(v) for v in tf.trainable_variables() if '/kernel' in v.name and ('/se_t1___/' in v.name or '/image_de_t1___/' in v.name)])
        l2_loss_style_t2___ = tf.add_n([0.0001 * tf.nn.l2_loss(v) for v in tf.trainable_variables() if '/kernel' in v.name and ('/se_t2___/' in v.name or '/image_de_t2___/' in v.name)])

        reconstruction_loss_flair = tf.reduce_mean(tf.abs(self.input_flair - self.reconstruct_flair))
        reconstruction_loss_t1c__ = tf.reduce_mean(tf.abs(self.input_t1c - self.reconstruct_t1c__))
        reconstruction_loss_t1___ = tf.reduce_mean(tf.abs(self.input_t1 - self.reconstruct_t1___))
        reconstruction_loss_t2___ = tf.reduce_mean(tf.abs(self.input_t2 - self.reconstruct_t2___))

        kl_loss_flair = losses.kl_loss(self.mu_flair, tf.log(tf.square(self.sigma_flair)))
        kl_loss_t1c__ = losses.kl_loss(self.mu_t1c__, tf.log(tf.square(self.sigma_t1c__)))
        kl_loss_t1___ = losses.kl_loss(self.mu_t1___, tf.log(tf.square(self.sigma_t1___)))
        kl_loss_t2___ = losses.kl_loss(self.mu_t2___, tf.log(tf.square(self.sigma_t2___)))

        seg_loss = ce_loss + dice_loss + l2_loss_content + \
                   l2_loss_style_flair + l2_loss_style_t1c__ + l2_loss_style_t1___ + l2_loss_style_t2___ + \
                   0.1 * (reconstruction_loss_flair + reconstruction_loss_t1c__ + reconstruction_loss_t1___ + reconstruction_loss_t2___) + \
                   0.1 * (kl_loss_flair + kl_loss_t1c__ + kl_loss_t1___ + kl_loss_t2___)


        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.model_vars = tf.trainable_variables()

        s_vars = [var for var in self.model_vars]
        self.s_trainer = optimizer.minimize(seg_loss, var_list=s_vars)

        for var in self.model_vars:
            print(var.name)

        # Summary variables for tensorboard
        self.ce_loss_summ = tf.summary.scalar("ce_loss", ce_loss)
        self.dice_loss_summ = tf.summary.scalar("dice_loss", dice_loss)
        self.l2_loss_content_summ = tf.summary.scalar("l2_loss_content", l2_loss_content)
        self.l2_loss_style_flair_summ = tf.summary.scalar("l2_loss_style_flair", l2_loss_style_flair)
        self.l2_loss_style_t1c___summ = tf.summary.scalar("l2_loss_style_t1c__", l2_loss_style_t1c__)
        self.l2_loss_style_t1____summ = tf.summary.scalar("l2_loss_style_t1___", l2_loss_style_t1___)
        self.l2_loss_style_t2____summ = tf.summary.scalar("l2_loss_style_t2___", l2_loss_style_t2___)
        self.reconstruction_loss_flair_summ = tf.summary.scalar("reconstruction_loss_flair", reconstruction_loss_flair)
        self.reconstruction_loss_t1c___summ = tf.summary.scalar("reconstruction_loss_t1c__", reconstruction_loss_t1c__)
        self.reconstruction_loss_t1____summ = tf.summary.scalar("reconstruction_loss_t1___", reconstruction_loss_t1___)
        self.reconstruction_loss_t2____summ = tf.summary.scalar("reconstruction_loss_t2___", reconstruction_loss_t2___)
        self.kl_loss_flair_summ = tf.summary.scalar("kl_loss_flair", kl_loss_flair)
        self.kl_loss_t1c___summ = tf.summary.scalar("kl_loss_t1c__", kl_loss_t1c__)
        self.kl_loss_t1____summ = tf.summary.scalar("kl_loss_t1___", kl_loss_t1___)
        self.kl_loss_t2____summ = tf.summary.scalar("kl_loss_t2___", kl_loss_t2___)
        self.s_loss_summ = tf.summary.scalar("s_loss", seg_loss)
        self.s_loss_merge_summ = tf.summary.merge([self.ce_loss_summ, self.dice_loss_summ,
                                                   self.l2_loss_content_summ,
                                                   self.l2_loss_style_flair_summ, self.l2_loss_style_t1c___summ,
                                                   self.l2_loss_style_t1____summ, self.l2_loss_style_t2____summ,
                                                   self.reconstruction_loss_flair_summ,
                                                   self.reconstruction_loss_t1c___summ,
                                                   self.reconstruction_loss_t1____summ,
                                                   self.reconstruction_loss_t2____summ,
                                                   self.kl_loss_flair_summ,
                                                   self.kl_loss_t1c___summ,
                                                   self.kl_loss_t1____summ,
                                                   self.kl_loss_t2____summ,
                                                   self.s_loss_summ])

    def train(self):
        """Training Function."""
        # Load Dataset from the dataset folder

        data, brainmask, label = data_loader.load_data(train_list_pth, train_data_pth, modality_list, BATCH_SIZE, gt_flag=True, crop_size=CROP_SIZE, num_cls=NUM_CLS)
        data_valid, brainmask_valid, label_valid = data_loader.load_data(valid_list_pth, train_data_pth, modality_list, BATCH_SIZE,
                                                       gt_flag=True, crop_size=CROP_SIZE, num_cls=NUM_CLS)
        data_modality_config = modalitydrop_generator.generator(train_list_pth)


        # Build the network
        self.model_setup()

        # Loss function calculations
        self.compute_losses()

        # Initializing the global variables
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        save_var_list = [v for v in tf.trainable_variables()]
        saver = tf.train.Saver(var_list=save_var_list, max_to_keep=10000)

        with open(train_list_pth, 'r') as fp:
            rows = fp.readlines()

        max_images = len(rows)
        time_st = time.time()
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)

            writer = tf.summary.FileWriter(self._output_dir)
            writer_val = tf.summary.FileWriter(self._output_dir+'/val')

            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Training Loop
            for epoch in range(0, max_epoch):
                print("In the epoch ", epoch)

                curr_lr = BASE_LEARNING_RATE*(1.0-np.float32(epoch)/np.float32(max_epoch))**(0.9)

                for i in range(0, max_inter):
                    starttime = time.time()
                    data_feed, label_feed = sess.run([data, label])
                    data_feed_valid, label_feed_valid = sess.run([data_valid, label_valid])
                    modality_config_s1_val, modality_config_s2_val, modality_config_s3_val, modality_config_s4_val, modality_num_val = sess.run(
                        data_modality_config)

                    _, summary_str = sess.run(
                        [self.s_trainer, self.s_loss_merge_summ],
                        feed_dict={
                            self.input_flair: data_feed[:,:,:,:CROP_SIZE,:],
                            self.input_t1c: data_feed[:, :, :, CROP_SIZE:2 * CROP_SIZE, :],
                            self.input_t1: data_feed[:, :, :, 2 * CROP_SIZE:3 * CROP_SIZE, :],
                            self.input_t2: data_feed[:, :, :, 3 * CROP_SIZE:, :],
                            self.input_label: label_feed,
                            self.learning_rate: curr_lr,
                            self.drop_rate: drop_rate_value,
                            self.is_training: is_training_value,
                            self.modality_config_s1: modality_config_s1_val,
                            self.modality_config_s2: modality_config_s2_val,
                            self.modality_config_s3: modality_config_s3_val,
                            self.modality_config_s4: modality_config_s4_val,
                            self.modality_num: modality_num_val,
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_inter + i)

                    summary_str = sess.run(self.lr_summ,
                             feed_dict={
                                 self.learning_rate: curr_lr,
                             })
                    writer.add_summary(summary_str, epoch * max_inter + i)
                    writer.flush()

                    if (i+1) % evaluation_interval==0:
                        summary_str1, summary_str2, summary_str3, summary_str4 = sess.run(
                            [self.dice_c1_summ, self.dice_c2_summ, self.dice_c3_summ, self.dice_c4_summ], feed_dict={
                            self.input_flair: data_feed[:, :, :, :CROP_SIZE, :],
                            self.input_t1c: data_feed[:, :, :, CROP_SIZE:2 * CROP_SIZE, :],
                            self.input_t1: data_feed[:, :, :, 2 * CROP_SIZE:3 * CROP_SIZE, :],
                            self.input_t2: data_feed[:, :, :, 3 * CROP_SIZE:, :],
                            self.input_label: label_feed,
                            self.is_training: False,
                            self.drop_rate: 0.0,
                            self.modality_config_s1: np.ones((4, 80, 80, 80, 16), dtype=np.float32),
                            self.modality_config_s2: np.ones((4, 40, 40, 40, 32), dtype=np.float32),
                            self.modality_config_s3: np.ones((4, 20, 20, 20, 64), dtype=np.float32),
                            self.modality_config_s4: np.ones((4, 10, 10, 10, 128), dtype=np.float32),
                            self.modality_num: 4.0,
                        })
                        writer.add_summary(summary_str1, epoch * max_inter + i)
                        writer.add_summary(summary_str2, epoch * max_inter + i)
                        writer.add_summary(summary_str3, epoch * max_inter + i)
                        writer.add_summary(summary_str4, epoch * max_inter + i)
                        writer.flush()

                        summary_str1, summary_str2, summary_str3, summary_str4, summary_str5 = sess.run([self.dice_c1_summ, self.dice_c2_summ, self.dice_c3_summ, self.dice_c4_summ, self.s_loss_merge_summ], feed_dict={
                            self.input_flair: data_feed_valid[:, :, :, :CROP_SIZE, :],
                            self.input_t1c: data_feed_valid[:, :, :, CROP_SIZE:2 * CROP_SIZE, :],
                            self.input_t1: data_feed_valid[:, :, :, 2 * CROP_SIZE:3 * CROP_SIZE, :],
                            self.input_t2: data_feed_valid[:, :, :, 3 * CROP_SIZE:, :],
                            self.input_label: label_feed_valid,
                            self.is_training: False,
                            self.drop_rate: 0.0,
                            self.modality_config_s1: np.ones((4, 80, 80, 80, 16), dtype=np.float32),
                            self.modality_config_s2: np.ones((4, 40, 40, 40, 32), dtype=np.float32),
                            self.modality_config_s3: np.ones((4, 20, 20, 20, 64), dtype=np.float32),
                            self.modality_config_s4: np.ones((4, 10, 10, 10, 128), dtype=np.float32),
                            self.modality_num: 4.0,
                        })
                        writer_val.add_summary(summary_str1, epoch * max_inter + i)
                        writer_val.add_summary(summary_str2, epoch * max_inter + i)
                        writer_val.add_summary(summary_str3, epoch * max_inter + i)
                        writer_val.add_summary(summary_str4, epoch * max_inter + i)
                        writer_val.add_summary(summary_str5, epoch * max_inter + i)
                        writer_val.flush()

                    if (i + 1) % visual_interval == 0:
                        summary_img = sess.run(self.visual_summary, feed_dict={
                                self.input_flair: data_feed[:, :, :, :CROP_SIZE, :],
                                self.input_t1c: data_feed[:, :, :, CROP_SIZE:2 * CROP_SIZE, :],
                                self.input_t1: data_feed[:, :, :, 2 * CROP_SIZE:3 * CROP_SIZE, :],
                                self.input_t2: data_feed[:, :, :, 3 * CROP_SIZE:, :],
                                self.input_label: label_feed,
                                self.is_training: False,
                                self.drop_rate: 0.0,
                                self.modality_config_s1: np.ones((4, 80, 80, 80, 16), dtype=np.float32),
                                self.modality_config_s2: np.ones((4, 40, 40, 40, 32), dtype=np.float32),
                                self.modality_config_s3: np.ones((4, 20, 20, 20, 64), dtype=np.float32),
                                self.modality_config_s4: np.ones((4, 10, 10, 10, 128), dtype=np.float32),
                                self.modality_num: 4.0,
                            })
                        writer.add_summary(summary_img, epoch * max_inter + i)
                        writer.flush()

                        summary_img = sess.run(self.visual_summary, feed_dict={
                            self.input_flair: data_feed_valid[:, :, :, :CROP_SIZE, :],
                            self.input_t1c: data_feed_valid[:, :, :, CROP_SIZE:2 * CROP_SIZE, :],
                            self.input_t1: data_feed_valid[:, :, :, 2 * CROP_SIZE:3 * CROP_SIZE, :],
                            self.input_t2: data_feed_valid[:, :, :, 3 * CROP_SIZE:, :],
                            self.input_label: label_feed_valid,
                            self.is_training: False,
                            self.drop_rate: 0.0,
                            self.modality_config_s1: np.ones((4, 80, 80, 80, 16), dtype=np.float32),
                            self.modality_config_s2: np.ones((4, 40, 40, 40, 32), dtype=np.float32),
                            self.modality_config_s3: np.ones((4, 20, 20, 20, 64), dtype=np.float32),
                            self.modality_config_s4: np.ones((4, 10, 10, 10, 128), dtype=np.float32),
                            self.modality_num: 4.0,
                        })
                        writer_val.add_summary(summary_img, epoch * max_inter + i)
                        writer_val.flush()


                    if (epoch * max_inter + i + 1) % save_interval ==0:
                        saver.save(sess, os.path.join(
                            self._output_dir, "multimodal"), global_step=epoch * max_inter + i)

                    print("Processed batch {}/{}  {}".format(i, max_inter, time.time() - starttime))

            coord.request_stop()
            coord.join(threads)


def main():

    multimodal_model = MultiModal()
    multimodal_model.train()


if __name__ == '__main__':
    main()
