import torch
import numpy as np
from itertools import product
import os, sys
import glob
import rasterio

import logging

logger = logging.getLogger(__name__)


def calculate_accuracy(predicted_labels, true_labels, mode, fmask=None):
    mat = (predicted_labels == true_labels) * 1.0
    deno = torch.numel(mat)
    acc = torch.sum(mat) / deno
    fmask_acc = 0
    if mode != 'train':
        non_zero_lables = true_labels != 0
        mat = mat * non_zero_lables
        deno = torch.sum(non_zero_lables)

        fmask_flag = (predicted_labels == fmask) * mat
        fmask_acc = fmask_flag.sum()
        acc = torch.sum(mat)

    return acc, deno, fmask_acc


def _flatten_matrix(torch_tensor):
    return torch_tensor.detach().cpu().numpy().reshape(-1, )


def calculate_confusion_matrix(predicted_labels, true_labels, mode):
    labels = [0, 1, 2, 3, 4, 5]
    conf_mat_t = torch.zeros((len(labels), len(labels)), dtype=int).to(
        predicted_labels.device)

    if mode != 'train':
        predicted_labels = predicted_labels[true_labels != 0]
        true_labels = true_labels[true_labels != 0]

    for (i, j) in product(labels, repeat=2):
        conf_mat_t[i, j] = ((predicted_labels == j) * (true_labels == i)).sum()

    # del true_labels_np
    # del predicted_labels_np
    # gc.collect()

    return conf_mat_t


def class_distribution(labels):
    labels_np = labels.cpu().numpy()
    class_freq = np.unique(labels_np, return_counts=True)
    total = np.sum(class_freq[1])
    for i in range(class_freq[0].size):
        print(
            '\nClass:{} Count:{}'.format(class_freq[0][i], class_freq[1][i] / total))


def confusion_matrix_percent(confusion_matrix):
    class_sum = np.sum(confusion_matrix, axis=0)  # check this if it is axis=1
    class_sum[class_sum == 0] = -1
    confusion_matrix = confusion_matrix / class_sum
    confusion_matrix[confusion_matrix < 0] = 0
    return confusion_matrix


def conf_mat_class_accuracy(conf_mat):
    """Returns the class accuracy for each class from the confidence matrix"""
    class_sum = torch.sum(conf_mat, 1).float()

    class_acc = torch.where(class_sum != 0, conf_mat.diagonal() / class_sum,
                            torch.zeros_like(class_sum))

    return class_acc, class_acc[1:].mean()


def conf_mat_class_MIOU(conf_mat):
    """Returns the class MIOU for each class from the confidence matrix"""
    intersection = conf_mat.diagonal().float()
    union = torch.sum(conf_mat, 0).float() + torch.sum(conf_mat,
                                                       1).float() - intersection
    class_miou = intersection / union
    return class_miou, class_miou[1:].mean()


def read_file(path):
    band = rasterio.open(path)
    data = band.read(1).astype(np.uint8)
    return torch.tensor(data)


def get_full_stats(safe_folder, exp_name):
    fmask_path = glob.glob(os.path.join(safe_folder, '**/*FMASK.tif'),
                           recursive=True)
    assert len(fmask_path) == 1
    fmask_path = fmask_path[0]
    fmask = read_file(fmask_path)

    sen2cor_path = glob.glob(os.path.join(safe_folder, '**/*SEN2COR.tif'),
                             recursive=True)
    assert len(sen2cor_path) == 1
    sen2cor_path = sen2cor_path[0]
    sen2cor = read_file(sen2cor_path)

    label_path = glob.glob(os.path.join(safe_folder, '**/*LABELS.tif'),
                           recursive=True)
    assert len(label_path) == 1
    label_path = label_path[0]
    labels = read_file(label_path)

    predict_path = sorted(
        glob.glob(os.path.join(safe_folder, '**/*LABELS_{}_*.tif'.format(exp_name)),
                  recursive=True))
    assert len(predict_path) != 0
    predict_path = predict_path[-1]
    prediction = read_file(predict_path)

    conf_fmask = calculate_confusion_matrix(fmask, labels, mode='predict')
    conf_sen2cor = calculate_confusion_matrix(sen2cor, labels, mode='predict')
    conf_labels = calculate_confusion_matrix(prediction, labels, mode='predict')

    return conf_fmask, conf_sen2cor, conf_labels


def get_metrics(conf_matrix):
    metric = {}
    metric['acc'] = conf_matrix.diagonal().sum().float() / conf_matrix.sum()
    CLASS_DICT = {'Clear Land': 1, 'Cloud': 2, 'Cloud Shadow': 3, 'Snow': 4,
                  'Water': 5}
    sum_IOU = 0
    for class_name in CLASS_DICT:
        idx = CLASS_DICT[class_name]
        metric['{}_precision'.format(class_name)] = conf_matrix[
                                                        idx, idx].float() / conf_matrix[
                                                                            :,
                                                                            idx].sum()
        metric['{}_recall'.format(class_name)] = conf_matrix[
                                                     idx, idx].float() / conf_matrix[
                                                                         idx,
                                                                         :].sum()
        metric['{}_f1'.format(class_name)] = 2 * metric[
            '{}_precision'.format(class_name)] * metric[
                                                 '{}_recall'.format(class_name)] / (
                                                         metric[
                                                             '{}_precision'.format(
                                                                 class_name)] +
                                                         metric['{}_recall'.format(
                                                             class_name)])
        metric['{}_IOU'.format(class_name)] = conf_matrix[idx, idx].float() / (
                    conf_matrix[:, idx].sum() + conf_matrix[idx, :].sum() -
                    conf_matrix[idx, idx])
        sum_IOU += metric['{}_IOU'.format(class_name)]
    metric['mIOU'] = sum_IOU / 5
    return metric
