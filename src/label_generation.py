"""This script is used to perform the pseudo label generation from a trained
model. These labels will be used for training the model in the subsequent
stage of the self-training pipeline. """

import argparse
import logging

from tqdm import tqdm

from dataset.patch_dataset import setup_data
from network.model import Model
from utils.experiment import Experiment
from utils.dir_paths import TRAIN_PATH

# Logging
logger = logging.getLogger('Data Labelling Script')


def get_args(argv=None):
    """Parses the arguments entered by the user."""

    parser = argparse.ArgumentParser(description="Data Labelling Script")
    parser.add_argument('-e', '--exp_name', help='Name of experiment')
    # Network
    parser.add_argument('-bs', '--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('-st', '--stage', type=int, default=1,
                        help='Training stage for which labelling is done')
    parser.add_argument('-ep', '--model_epoch', type=int, default=0,
                        help='Epoch of the trained model to be used for label '
                             'generation. Note: Epochs indexing starts from 1 '
                             'and the default value (0) corresponds to the '
                             'best model')

    # Hardware
    parser.add_argument('-gpu', '--gpu_id', type=int, nargs='+', default=[0],
                        help='ID of the GPUs used for training. Ex: -gpu 0 3')

    return parser.parse_args(argv)


if __name__ == '__main__':

    # Read arguments and initialize the training experiment
    args = get_args()
    exp = Experiment(args, mode='label_gen')

    # Initialize the dataloader for the label generation step.
    test_loader = setup_data(args.batch_size, 'label_gen',
                             stage=args.stage,
                             path=TRAIN_PATH)

    # Initialize model for evaluation
    model = Model(exp, gpu_id=args.gpu_id)
    model.network.eval()

    loader_itr = tqdm(test_loader,
                      total=int(len(test_loader)),
                      leave=False,
                      desc='New Label Generation')  # Progress bars

    # Label generation step
    for batch, network_input in enumerate(loader_itr):
        model.valid_step(network_input, mode='label_gen')

    # Store the frequency of each class for calculating loss function weights in
    # next stage.
    model.write_stage_stats()
    logger.info('New Data generation Complete')
