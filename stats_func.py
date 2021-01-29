import tensorflow as tf
import numpy as np
import medpy.metric.binary as mmb


def dice_eval(compact_pred, compact_label, num_cls):
    """
    calculate standard dice for evaluation, here uses the class prediction, not the probability
    """
    dice_arr = []
    eps = 1e-7
    pred = tf.one_hot(compact_pred, depth=num_cls, axis=-1)
    label = tf.one_hot(compact_label, depth=num_cls, axis=-1)

    for i in xrange(1, num_cls):
        if i==1:
            inse_wt = tf.reduce_sum(pred[:, :, :, :, i] * label[:, :, :, :, i], axis=[1,2,3])
            union_wt = tf.reduce_sum(pred[:, :, :, :, i], axis=[1,2,3]) + tf.reduce_sum(label[:, :, :, :, i], axis=[1,2,3])
            inse_co = tf.reduce_sum(pred[:, :, :, :, i] * label[:, :, :, :, i], axis=[1,2,3])
            union_co = tf.reduce_sum(pred[:, :, :, :, i], axis=[1,2,3]) + tf.reduce_sum(label[:, :, :, :, i], axis=[1,2,3])

        else:
            inse_wt = inse_wt + tf.reduce_sum(pred[:, :, :, :, i] * label[:, :, :, :, i], axis=[1,2,3])
            union_wt = union_wt + tf.reduce_sum(pred[:, :, :, :, i], axis=[1, 2, 3]) + tf.reduce_sum(label[:, :, :, :, i], axis=[1, 2, 3])
            if i!=2:
                inse_co = inse_co + tf.reduce_sum(pred[:, :, :, :, i] * label[:, :, :, :, i], axis=[1, 2, 3])
                union_co = union_co + tf.reduce_sum(pred[:, :, :, :, i], axis=[1, 2, 3]) + tf.reduce_sum(label[:, :, :, :, i], axis=[1, 2, 3])
            if i==4:
                inse_ec = tf.reduce_sum(pred[:, :, :, :, i] * label[:, :, :, :, i], axis=[1, 2, 3])
                union_ec = tf.reduce_sum(pred[:, :, :, :, i], axis=[1, 2, 3]) + tf.reduce_sum(label[:, :, :, :, i], axis=[1, 2, 3])

    inse_bg = tf.reduce_sum(pred[:, :, :, :, 0] * label[:, :, :, :, 0], axis=[1, 2, 3])
    union_bg = tf.reduce_sum(pred[:, :, :, :, 0], axis=[1, 2, 3]) + tf.reduce_sum(label[:, :, :, :, 0], axis=[1, 2, 3])
    dice_arr.append(tf.reduce_mean(2.0 * inse_wt / (union_wt + eps)))
    dice_arr.append(tf.reduce_mean(2.0 * inse_co / (union_co + eps)))
    dice_arr.append(tf.reduce_mean(2.0 * inse_ec / (union_ec + eps)))
    dice_arr.append(tf.reduce_mean(2.0 * inse_bg / (union_bg + eps)))

    return dice_arr

def dice_stat(pred_data, gt_data):
    pred_data_wt = pred_data.copy()
    pred_data_wt[pred_data_wt > 0] = 1

    pred_data_co = pred_data.copy()
    pred_data_co[pred_data_co == 2] = 0
    pred_data_co[pred_data_co > 0] = 1

    pred_data_ec = pred_data.copy()
    pred_data_ec[pred_data_ec != 4] = 0
    pred_data_ec[pred_data_ec > 0] = 1

    gt_data_wt = gt_data.copy()
    gt_data_wt[gt_data_wt > 0] = 1

    gt_data_co = gt_data.copy()
    gt_data_co[gt_data_co == 2] = 0
    gt_data_co[gt_data_co > 0] = 1

    gt_data_ec = gt_data.copy()
    gt_data_ec[gt_data_ec != 4] = 0
    gt_data_ec[gt_data_ec > 0] = 1

    return [mmb.dc(pred_data_wt, gt_data_wt), mmb.precision(pred_data_wt, gt_data_wt), mmb.sensitivity(pred_data_wt, gt_data_wt)],\
           [mmb.dc(pred_data_co, gt_data_co), mmb.precision(pred_data_co, gt_data_co), mmb.sensitivity(pred_data_co, gt_data_co)],\
           [mmb.dc(pred_data_ec, gt_data_ec), mmb.precision(pred_data_ec, gt_data_ec), mmb.sensitivity(pred_data_ec, gt_data_ec)]