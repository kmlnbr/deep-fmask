import torch
import numpy as np
from itertools import product
import os, sys
import glob
import rasterio

import logging
from utils.script_utils import print_epoch_info, print_conf_matrix

logger = logging.getLogger(__name__)

CLASS_DICT = {'clear': 1, 'cloud': 2, 'shadow': 3, 'snow': 4, 'water': 5}


def read_tif(safe_folder, name, height, width):
    """
    Returns the data from the tif file as torch tensor. If the file is not found, return
    a zero valued tensor of the given height and width.
    """
    tif_path = glob.glob(os.path.join(safe_folder, '**/*{}.tif'.format(name)),
                         recursive=True)
    if len(tif_path) == 1:
        tif_path = tif_path[0]
        tif_data = read_file(tif_path)
    else:
        logger.info(
            '{} file not found for {}. Using all-zero file instead'.format(name,
                                                                           safe_folder))
        tif_data = torch.zeros((height, width), dtype=torch.uint8)

    return tif_data


def calculate_accuracy(predicted_labels, true_labels, mode):
    mat = (predicted_labels == true_labels) * 1.0
    deno = torch.numel(mat)
    acc = torch.sum(mat) / deno
    # When mode is not train, we exclude class 0 from the accuracy calculation
    # because in validation/test mode, we use sparsely labelled data where
    # unlabelled data is given class 0.
    if mode != 'train':
        non_zero_lables = true_labels != 0
        mat = mat * non_zero_lables
        deno = torch.sum(non_zero_lables)

        acc = torch.sum(mat)

    return acc


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

    return conf_mat_t


def conf_mat_class_accuracy(conf_mat):
    """Returns the class accuracy for each class from the confidence matrix"""
    class_sum = torch.sum(conf_mat, 1).float()

    class_acc = torch.where(class_sum != 0, conf_mat.diagonal() / class_sum,
                            torch.zeros_like(class_sum))

    overall_acc = (1.0 * conf_mat.diagonal().sum()) / conf_mat.sum()

    return class_acc, overall_acc


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
    """Read fmask, sen2cor, predicted labels and true labels and generate
    confusion matrix for each against true labels """

    predict_path = sorted(
        glob.glob(os.path.join(safe_folder, '**/*LABELS_{}_*.tif'.format(exp_name)),
                  recursive=True))
    assert len(predict_path) != 0
    predict_path = predict_path[-1]
    prediction = read_file(predict_path)

    fmask = read_tif(safe_folder, 'FMASK', *prediction.shape)
    sen2cor = read_tif(safe_folder, 'SEN2COR', *prediction.shape)
    labels = read_tif(safe_folder, 'LABELS', *prediction.shape)

    conf_fmask = calculate_confusion_matrix(fmask, labels, mode='predict')
    conf_sen2cor = calculate_confusion_matrix(sen2cor, labels, mode='predict')
    conf_labels = calculate_confusion_matrix(prediction, labels, mode='predict')

    return conf_fmask, conf_sen2cor, conf_labels


def get_metrics(conf_matrix):
    """Calculate different metrics from the given confusion matrix"""
    metric = {}
    metric['acc'] = conf_matrix.diagonal().sum().float() / conf_matrix.sum()

    sum_IOU = 0
    for class_name in CLASS_DICT:
        idx = CLASS_DICT[class_name]
        metric['{}_Precision'.format(class_name)] = conf_matrix[
                                                        idx, idx].float() / conf_matrix[
                                                                            :,
                                                                            idx].sum()
        metric['{}_Recall'.format(class_name)] = conf_matrix[
                                                     idx, idx].float() / conf_matrix[
                                                                         idx,
                                                                         :].sum()
        metric['{}_f1'.format(class_name)] = 2 * metric[
            '{}_Precision'.format(class_name)] * metric[
                                                 '{}_Recall'.format(class_name)] / (
                                                         metric[
                                                             '{}_Precision'.format(
                                                                 class_name)] +
                                                         metric['{}_Recall'.format(
                                                             class_name)])
        metric['{}_iou'.format(class_name)] = conf_matrix[idx, idx].float() / (
                    conf_matrix[:, idx].sum() + conf_matrix[idx, :].sum() -
                    conf_matrix[idx, idx])
        sum_IOU += metric['{}_iou'.format(class_name)]
    metric['mIOU'] = sum_IOU / 5
    return metric


class Metrics:
    def __init__(self, device):
        self.device = device
        self.val_mIoU_history = []
        self.reset_metrics()

    def aggregate_metrics(self, epoch):
        # train_metrics[0] is avg loss
        train_metrics = torch.tensor(self.train).mean(dim=0)

        valid_metrics = torch.tensor(self.valid).mean(dim=0)  # Accuracy

        class_acc, overall_acc = conf_mat_class_accuracy(self.val_confusion_matrix)
        class_MIOU, avg_MIOU = conf_mat_class_MIOU(self.val_confusion_matrix)
        class_metrics_dict = get_metrics(self.val_confusion_matrix)

        valid_metrics = torch.tensor([valid_metrics[0], overall_acc, avg_MIOU]).to(
            self.device)
        self.val_mIoU_history.append(avg_MIOU.item())

        print_epoch_info(epoch, *train_metrics, *valid_metrics)
        print_conf_matrix(self.val_confusion_matrix)

        return train_metrics, valid_metrics, class_metrics_dict

    def reset_metrics(self):

        self.train = []
        self.valid = []
        self.val_confusion_matrix = torch.zeros((6, 6), dtype=torch.long).to(
            self.device)

    def add_step_info(self, mode, loss, acc=0):
        if mode == 'train':
            lst = self.train
        else:
            lst = self.valid

        lst.append([loss, acc])
