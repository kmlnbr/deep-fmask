"""
This script is used to train the network at each stage of the self-training framework.
"""

import argparse
import logging

from tqdm import tqdm

from dataset.patch_dataset import setup_data, set_seed
from network.model import Model
from utils.MFB import get_MFB_weights
from utils.dir_paths import TRAIN_PATH, VALID_PATH
from utils.experiment import Experiment

# Logging
logger = logging.getLogger('train_script')


def get_args(argv=None):
    """Parses the arguments entered by the user."""

    parser = argparse.ArgumentParser(description="Training Script")

    parser.add_argument('-e', '--exp_name', help='Name of experiment')
    # Network
    parser.add_argument('-st', '--stage', type=int, default=0,
                        help='Training stage')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='Learning Rate')
    parser.add_argument('--full', dest='full', action='store_true', default=False,
                        help='Train the largest network in the pipeline using the whole data on '
                             'the F-Mask labels, i.e., supervised learning using F-Mask labels')
    parser.add_argument('--no_dropout', dest='dropout', action='store_false', default=True,
                        help='Flag used to avoid dropout usage during training')
    parser.add_argument('-ep', '--num_epochs', type=int, default=400,
                        help='Number of training epochs')
    # Data
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help='Batch size for each step of training')
    parser.add_argument('--no_aug', dest='aug', action='store_false', default=True,
                        help='Flag used to avoid augmentation during training')
    parser.add_argument('--reset_stage_data', dest='reset_stage_data', action='store_true',
                        default=False, help='Flag used reshuffle and split stage data')
    parser.add_argument('-ip', '--inp_mode', default='all',
                        help='Bands used as input to the network')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    # Hardware
    parser.add_argument('-gpu', '--gpu_id', type=int,nargs='+', default=[0],
                        help='ID of the GPUs used for training. Ex: -gpu 0 3')
    return parser.parse_args(argv)


if __name__ == '__main__':

    # Read arguments and initialize the training experiment
    args = get_args()
    exp = Experiment(args)
    set_seed(args.seed)

    # Initialize the dataloader for the train and validation step.
    train_loader = setup_data(args.batch_size, mode='train',
                              stage=args.stage, path=TRAIN_PATH,
                              full=args.full, aug=args.aug,
                              reset=args.reset_stage_data)

    test_loader = setup_data(mode='test', path=VALID_PATH)

    # Initial the model
    model = Model(exp,gpu_id=args.gpu_id)

    # Get the weights for the loss functions using the median frequency balancing method
    exp.weights = get_MFB_weights(train_loader)

    for epoch in range(args.num_epochs):
        logger.info('Epoch - {}'.format(epoch + 1))
        ## Train step
        model.network.train()
        loader_itr_train = tqdm(train_loader,
                                total=int(len(train_loader)),
                                leave=False,
                                desc='Train Epoch {}'.format(epoch + 1))  # Progress bars

        for batch, network_input in enumerate(loader_itr_train):
            model.train_step(network_input)

        ## Validation step
        loader_itr = tqdm(test_loader,
                          total=int(len(test_loader)),
                          leave=False,
                          desc='Valid Epoch {}'.format(epoch + 1))  # Progress bars
        model.network.eval()

        for batch, network_input in enumerate(loader_itr):
            model.valid_step(network_input)

        # Check if validation metrics satisfy early stopping condition and reset
        # metrics for next epoch
        early_stop_flag = model.refresh_stats()
        if early_stop_flag:
            break

    model.save_best_model()
