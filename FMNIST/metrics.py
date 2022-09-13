import tensorflow as tf
import numpy as np
from sklearn.metrics import auc


def calculate_kl(ori_predictions, transformed_predictions):
    kl_list = []
    for ori, transformed in zip(ori_predictions, transformed_predictions):
        kl_divergence = tf.keras.metrics.kullback_leibler_divergence(ori, transformed).numpy()
        kl_list.append(kl_divergence)
    return kl_list


def calculate_crossentropy(ori_predictions, transformed_predictions):
    crossentropy_list = []
    for ori, transformed in zip(ori_predictions, transformed_predictions):
        crossentropy = tf.keras.metrics.categorical_crossentropy(ori, transformed)
        crossentropy_list.append(crossentropy)
    return crossentropy_list


def calculate_l1_distance(ori_predictions, transformed_predictions):
    distance_list = []
    for ori, transformed in zip(ori_predictions, transformed_predictions):
        distance = 0
        for i in range(ori.shape[0]):
            distance += abs(ori[i] - transformed[i])
        distance_list.append(distance)
    return distance_list


def calculate_prediction_changes(ori_predictions, transformed_predictions):
    changes_list = []
    ori_prediction = np.argmax(ori_predictions, axis=1)
    transformed_prediction = np.argmax(transformed_predictions, axis=1)
    for ori, transformed in zip(ori_prediction, transformed_prediction):
        changes_list.append(ori != transformed)
    return changes_list


def calculate_auroc(distance_dict, length):
    ori_kl = distance_dict['ori']
    adv_kl = distance_dict['adv']
    tpr_list = [1.0, ]
    fpr_list = [1.0, ]
    for i in range(301):
        boundary = 0.01 * i
        tp, fn, tn, fp = 0, 0, 0, 0
        for j in range(length):
            if ori_kl[j] > boundary:
                fp += 1
            else:
                tn += 1
            if adv_kl[j] > boundary:
                tp += 1
            else:
                fn += 1
        tpr_list.append(tp / (tp + fn))
        fpr_list.append(fp / (fp + tn))

    tpr_list.append(0)
    fpr_list.append(0)
    fpr_list.reverse()
    tpr_list.reverse()
    auroc = auc(np.array(fpr_list), np.array(tpr_list))
    print('AUROC: {}'.format(auroc))

    return auroc
