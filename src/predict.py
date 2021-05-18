"""This script is used to make prediction from the trained model. The script
processes multiple sub-scenes stored as h5 files and the predictions from each
sub-scene (h5 file) is stitched together to generate the full scene prediction in
20m resolution. """

import argparse
import logging
import os
import glob
import sys

from tqdm import tqdm
import numpy as np
import torch

from dataset.patch_dataset import setup_data
from network.model import Model
from utils.experiment import Experiment
from utils.dir_paths import PRED_PATH
from utils.make_network_data_pred import make_patch
from utils.metrics import get_full_stats, get_metrics
from utils.join_h5_pred import join_files
from utils.csv_logger import pred_csv

# Logging
logger = logging.getLogger('predict_script')


def get_args(argv=None):
    """Parses the arguments entered by the user."""

    parser = argparse.ArgumentParser(description="Prediction Script")
    parser.add_argument('-e', '--exp_name', help='Name of experiment')
    # Network
    parser.add_argument('-bs', '--batch_size', type=int, default=10,
                        help='Batch size for each step of training')
    parser.add_argument('-ep', '--model_epoch', type=int, default=0,
                        help='Epoch of the trained model (Starting from 1). '
                             'Defaults to best model')
    parser.add_argument('-p', '--pred_path', help='folder containing safe file',
                        default=PRED_PATH)

    # Hardware
    parser.add_argument('-gpu', '--gpu_id', type=int, nargs='+', default=[0],
                        help='ID of the GPUs used for training. Ex: -gpu 0 3')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = get_args()
    exp = Experiment(args, mode='predict')
    trained_model = exp.get_trained_model_info()
    if "inp_mode" in trained_model:
        inp_mode = trained_model["inp_mode"]
        if inp_mode == "novegndwi": inp_mode = "nonvegndwi"
    else:
        inp_mode = 'all'
    if "full" in trained_model:
        full = trained_model["full"]
    else:
        full = args.full

    model = Model(exp, gpu_id=args.gpu_id)
    model.network.eval()
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
                          desc='Predict {}/{}'.format(file_idx + 1, file_count))

        for batch, network_input in enumerate(loader_itr):
            model.valid_step(network_input, mode='predict')
        output_folder = glob.glob(os.path.join(H5_folder[:-3], '**/*B02.jp2'),
                                  recursive=True)

        img_path = os.path.dirname(output_folder[0]) if output_folder else None

        join_files(H5_folder, output=img_path, exp=args.exp_name)

    logger.info('Prediction Complete')
    #################################################################################

    # Printing metrics when labels are available
    np.set_printoptions(suppress=True, precision=3)

    CONF_FMASK_FULL = torch.zeros((6, 6), dtype=torch.long)
    CONF_SEN2COR_FULL = torch.zeros((6, 6), dtype=torch.long)
    CONF_PRED_FULL = torch.zeros((6, 6), dtype=torch.long)

    for safe_file in safe_files:
        conf_fmask, conf_sen2cor, conf_labels = get_full_stats(safe_file,
                                                               args.exp_name)
        CONF_FMASK_FULL += conf_fmask
        CONF_SEN2COR_FULL += conf_sen2cor
        CONF_PRED_FULL += conf_labels

    if torch.all(CONF_PRED_FULL == 0):
        logger.info('Metrics not calculated because true label file not found.')
        sys.exit(0)

    metrics = []
    for matrix in [CONF_FMASK_FULL, CONF_SEN2COR_FULL, CONF_PRED_FULL]:
        metrics.append(get_metrics(matrix))

    pred_csv(metrics)
