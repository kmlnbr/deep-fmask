"""This script is used to generate predictions from the trained model"""

import argparse
import logging
import os
import glob

from tqdm import tqdm
import numpy as np
import torch

from dataset.patch_dataset import setup_data
from network.model import Model
from utils.experiment import Experiment
from utils.dir_paths import PRED_SAFE_PATH
from utils.make_network_data_pred import make_patch
from utils.metrics import get_full_stats, get_metrics
# from utils.visualizer import get_latex_tables
from utils.join_h5_pred import join_files

# Logging
logger = logging.getLogger('predict_script')


def get_args(argv=None):
    """Parses the arguments entered by the user."""

    parser = argparse.ArgumentParser(description="Prediction Script")
    parser.add_argument('-e', '--exp_name', help='Name of experiment')
    # Network
    parser.add_argument('-ep', '--model_epoch', type=int, default=0,
                        help='Epoch of the trained model (Starting from 1). '
                             'Defaults to best model')
    parser.add_argument('-st', '--stage', type=int, default=3,
                        help='Training stage')
    parser.add_argument('-p', '--pred_path', help='folder containing safe file',
                        default=PRED_SAFE_PATH)

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = get_args()
    exp = Experiment(args, mode='predict')
    model = Model(exp)
    model.network.eval()

    # Check if h5 files are already generated. If not, generate h5 files to use as
    # input for the model.

    args.pred_path = os.path.abspath(args.pred_path)

    if args.pred_path.endswith('.SAFE'):
        safe_files = [args.pred_path]
    else:
        safe_files = glob.glob(os.path.join(args.pred_path, '*.SAFE'))
    H5_folder_name = [i + '_H5' for i in safe_files]
    new_safe_folders = [i[:-3] for i in H5_folder_name if not os.path.exists(i)]
    if new_safe_folders:
        make_patch(new_safe_folders, mode='predict')


    file_count = len(H5_folder_name)
    for file_idx, H5_folder in enumerate(H5_folder_name):
        logger.info("Predicting {}".format(H5_folder))
        test_loader = setup_data(path=H5_folder, mode='predict')
        loader_itr = tqdm(test_loader,
                          total=int(len(test_loader)),
                          leave=False,
                          desc='Predict {}/{}'.format(file_idx+1, file_count))

        # for batch, network_input in enumerate(loader_itr):
        #     model.valid_step(network_input, mode='predict')
        #
        # # Join output for each patch into a single image and save it as tif file in
        # # the same directory as the bands.
        output_folder = glob.glob(os.path.join(H5_folder[:-3], '**/*B02.jp2'),
                                  recursive=True)
        #
        img_path = os.path.dirname(output_folder[0]) if output_folder else None
        #
        join_files(H5_folder, output=img_path, exp=args.exp_name)


    # If Sen2Cor, Fmask results and true lables are available: Model comparison
    # can be done by commenting the below code.
    #
    #
    # CONF_FMASK_FULL = torch.zeros((6, 6), dtype=torch.long)
    # CONF_SEN2COR_FULL = torch.zeros((6, 6), dtype=torch.long)
    # CONF_PRED_FULL = torch.zeros((6, 6), dtype=torch.long)
    # logger.info('Prediction Complete')
    # for safe_file in safe_files:
    #     conf_fmask, conf_sen2cor, conf_labels = get_full_stats(safe_file,
    #                                                            args.exp_name)
    #     acc = conf_labels.diagonal().sum().float() / conf_labels.sum()
    #     fmask_acc = conf_fmask.diagonal().sum().float() / conf_fmask.sum()
    #     sen_acc = conf_sen2cor.diagonal().sum().float() / conf_sen2cor.sum()
    #     print('Accuracy {} => {:.2%} [Fmask: {:.2%}, Sen2Cor: {:.2%}]'.format(
    #         os.path.basename(safe_file), acc, fmask_acc, sen_acc))
    #     CONF_FMASK_FULL += conf_fmask
    #     CONF_SEN2COR_FULL += conf_sen2cor
    #     CONF_PRED_FULL += conf_labels
    #
    # metrics = []
    # for matrix in [CONF_FMASK_FULL, CONF_SEN2COR_FULL, CONF_PRED_FULL]:
    #     metrics.append(get_metrics(matrix))
    #
    # print('Accuracy\t\t FMask {:.2}\t Sen2Cor {:.2}\t OurModel {:.2}'.format(
    #     metrics[0]['acc'], metrics[1]['acc'], metrics[2]['acc']))
