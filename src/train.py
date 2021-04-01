"""This script is used to train each stage of the self-learning workflow"""

import argparse
import logging
import os
import sys

from tqdm import tqdm

from dataset.patch_dataset import setup_data, get_MFB_weights, shuffle_train_list
from network.model import Model
from utils.experiment import Experiment

from utils.dir_paths import TRAIN_PATH, VALID_PATH

# Logging
logger = logging.getLogger('train_script')


def get_args(argv=None):
    """Parses the arguments entered by the user."""

    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('-e', '--exp_name', help='Name of experiment')
    # Network
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help='Batch size for each step of training')
    parser.add_argument('-ep', '--num_epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('-st', '--stage', type=int, default=0,
                        help='Training stage')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='Learning Rate')
    parser.add_argument('-fc', '--file_count', type=int, default=sys.maxsize,
                        help='File Count in each stage')
    parser.add_argument('--full', dest='full', action='store_true', default=False,
                        help='File Count in each stage')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = get_args()
    exp = Experiment(args)

    files_per_stage = args.file_count
    train_loader = setup_data(files_per_stage, args.batch_size, 'train', exp,
                              stage=args.stage, path=TRAIN_PATH, full=args.full)

    test_loader = setup_data(1, 1, 'test', path=VALID_PATH)  # Max 498

    model = Model(exp, full=args.full)

    for epoch in range(args.num_epochs):
        model.network.train()
        shuffle_train_list(train_loader)
        exp.weights = get_MFB_weights(train_loader)
        loader_itr_train = tqdm(train_loader,
                                total=int(len(train_loader)),
                                leave=False,
                                desc='Train Epoch {}'.format(epoch + 1))

        for batch, network_input in enumerate(loader_itr_train):
            model.train_step(network_input)

        loader_itr = tqdm(test_loader,
                          total=int(len(test_loader)),
                          leave=False,
                          desc='Valid Epoch {}'.format(epoch + 1))
        model.network.eval()

        for batch, network_input in enumerate(loader_itr):
            model.valid_step(network_input)
        logger.info('Epoch - {} Training'.format(epoch + 1))
        early_stop_flag = model.refresh_stats()
        if early_stop_flag:
            break
    model.save_best_model()
