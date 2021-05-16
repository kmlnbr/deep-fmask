"""
This script is used to train the network at each stage of the self-training framework.
"""

import argparse
import logging
import os

from tqdm import tqdm

from dataset.patch_dataset import setup_data,get_MFB_weights,set_seed
from network.model import Model
from utils.experiment import Experiment

from utils.dir_paths import TRAIN_PATH,VALID_PATH

# Logging
logger = logging.getLogger('train_script')


def get_args(argv=None):
    """Parses the arguments entered by the user."""

    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('-e', '--exp_name', help='Name of experiment')
    # Network
    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help='Batch size for each step of training')
    parser.add_argument('-ep', '--num_epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('-st', '--stage', type=int, default=0,
                        help='Training stage')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='Learning Rate')
    parser.add_argument('-fc', '--file_count', type=int, default=5000,
                        help='File Count in each stage')
    parser.add_argument('--full',  dest='full', action='store_true', default=False,
                        help='File Count in each stage')

    parser.add_argument('--no_dropout',  dest='dropout', action='store_false', default=True,
                        help='Flag used to avoid dropout usage during training')

    parser.add_argument('--no_aug',  dest='aug', action='store_false', default=True,
                        help='Flag used to avoid augmentation during training')

    parser.add_argument('--reset_stage_data', dest='reset_stage_data', action='store_true', default=False,
                        help='Flag used reshuffle and split stage data')

    parser.add_argument('--stage_0_ratio', type=float, default=0.25,
                        help='Percent of data in stage 0')

    parser.add_argument('-ip', '--inp_mode', default='all',
                        help='Input mode')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed.')
    return parser.parse_args(argv)




if __name__ == '__main__':
    args = get_args()
    exp = Experiment(args)
    set_seed(args.seed)
    # split_train_data(n_iterations=2, initial_size=1500,experiment=exp)


    files_per_stage = int(5e8)
    train_loader = setup_data(files_per_stage, args.batch_size, 'train',
                              exp,stage=args.stage,path = TRAIN_PATH,
                              full = args.full,aug=args.aug,
                              reset = args.reset_stage_data,
                              stage_0_ratio=args.stage_0_ratio)

    if 'lms37-22' in os.uname()[1]:
        test_batch = 4
    else:
        test_batch = 1

    test_loader = setup_data(1, test_batch, 'test',path = VALID_PATH) # Max 498

    model = Model(exp, full = args.full, dropout=args.dropout, inp_mode=args.inp_mode)
    exp.weights = get_MFB_weights(train_loader)



    for epoch in range(args.num_epochs):
        logger.info('Epoch - {}'.format(epoch + 1))
        model.network.train()
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
        early_stop_flag = model.refresh_stats()
        if early_stop_flag:
            break
    model.save_best_model()

