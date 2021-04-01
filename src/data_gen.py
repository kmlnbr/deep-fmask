"""This script is used to perform the pseudo label generation from a trained model"""

import argparse
import logging
import os

from tqdm import tqdm
# Logging
logger = logging.getLogger('Data Labelling Script')

from dataset.patch_dataset import setup_data
from network.model import Model
from utils.experiment import Experiment
from utils.dir_paths import TRAIN_PATH


def get_args(argv=None):
    """Parses the arguments entered by the user."""

    parser = argparse.ArgumentParser(description="Data Labelling Script")
    parser.add_argument('-e', '--exp_name', help='Name of experiment')
    # Network
    parser.add_argument('-bs', '--batch_size', type=int, default=1,
                        help='Batch size for each step of training')
    parser.add_argument('-st', '--stage', type=int, default=1,
                        help='Training stage for which labelling is done')
    parser.add_argument('-ep', '--model_epoch', type=int, default=0,
                        help='Epoch of the trained model (Starting from 1). '
                             'Defaults to best model')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = get_args()
    exp = Experiment(args,mode='data_gen')

    print('\n')
    test_loader = setup_data(1, args.batch_size, 'data_gen',
                             exp=exp,stage=args.stage,
                             path = TRAIN_PATH)

    model = Model(exp)


    model.network.eval()
    loader_itr = tqdm(test_loader,
                      total=int(len(test_loader)),
                      leave=False,
                      desc='New Label Generation')
    for batch, network_input in enumerate(loader_itr):
        model.valid_step(network_input,mode='data_gen')
    model.write_stage_stats()
    logger.info('New Data generation Complete')



