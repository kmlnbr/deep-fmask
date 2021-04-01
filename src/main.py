import argparse
import logging
import os
import sys

from tqdm import tqdm

from dataset.patch_dataset import setup_data,split_data
from network.model import Model
from utils.experiment import Experiment

logger = logging.getLogger('train_script')


def get_args(argv=None):
    """Parses the arguments entered by the user."""

    parser = argparse.ArgumentParser(description="Cloud Detection Training Script")
    parser.add_argument('-e', '--exp_name', help='Name of experiment')
    # Network
    parser.add_argument('-bs', '--batch_size', type=int, default=4,
                        help='Batch size for each step of training')
    parser.add_argument('-ep', '--num_epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001,
                        help='Learning Rate')

    return parser.parse_args(argv)

if __name__ == '__main__':
    split_data("utils")