import tensorflow as tf
import numpy as np
import medpy.io as medio
import os
import matplotlib.pyplot as plt


def _generate_sequence(filename):
    modality_config_val = np.random.randint(0, 2, 4)
    if not (modality_config_val == 1).any():
        modality_config_val[np.random.randint(0, 4, 1)] = 1

    modality_num_val = np.sum(modality_config_val, dtype=np.float32)

    modality_config_s1_val = np.zeros((4, 80, 80, 80, 16), dtype=np.float32)
    modality_config_s2_val = np.zeros((4, 40, 40, 40, 32), dtype=np.float32)
    modality_config_s3_val = np.zeros((4, 20, 20, 20, 64), dtype=np.float32)
    modality_config_s4_val = np.zeros((4, 10, 10, 10, 128), dtype=np.float32)

    for mm in range(0, 4):
        if modality_config_val[mm] == 1:
            modality_config_s1_val[mm, :, :, :, :] = np.ones((1, 80, 80, 80, 16), dtype=np.float32)
            modality_config_s2_val[mm, :, :, :, :] = np.ones((1, 40, 40, 40, 32), dtype=np.float32)
            modality_config_s3_val[mm, :, :, :, :] = np.ones((1, 20, 20, 20, 64), dtype=np.float32)
            modality_config_s4_val[mm, :, :, :, :] = np.ones((1, 10, 10, 10, 128), dtype=np.float32)
        else:
            modality_config_s1_val[mm, :, :, :, :] = np.zeros((1, 80, 80, 80, 16), dtype=np.float32)
            modality_config_s2_val[mm, :, :, :, :] = np.zeros((1, 40, 40, 40, 32), dtype=np.float32)
            modality_config_s3_val[mm, :, :, :, :] = np.zeros((1, 20, 20, 20, 64), dtype=np.float32)
            modality_config_s4_val[mm, :, :, :, :] = np.zeros((1, 10, 10, 10, 128), dtype=np.float32)

    return modality_config_s1_val, modality_config_s2_val, modality_config_s3_val, modality_config_s4_val, modality_num_val


def generator(data_list_pth):

    with open(data_list_pth, 'r') as fp:
        rows = fp.readlines()
    pid_list = [row[:-1] for row in rows]

    get_sequence_fn = lambda filename: tf.py_func(_generate_sequence, [filename], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])

    dataset = (tf.data.Dataset.from_tensor_slices(pid_list)
               .map(get_sequence_fn, num_parallel_calls=1)  # 3. return the patches
               .repeat()  # repeat indefinitely (train.py will count the epochs)
               .prefetch(20)  # 7. always have one batch ready to go
               )
    iterator = dataset.make_one_shot_iterator()
    samples = iterator.get_next()

    return samples




