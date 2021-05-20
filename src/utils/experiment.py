import os
import shutil
import logging
import torch
from collections import OrderedDict

from utils.script_utils import set_logfile_path

logger = logging.getLogger(__name__)


def fix_multi_gpu_model(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove module.
        new_state_dict[name] = v
    return new_state_dict


class Experiment:
    def __init__(self, args, mode='train', overwrite=True):

        self.name = args.exp_name
        logger.info("Exp name = {}".format(self.name))
        self.mode = mode
        folder_names = ['model', 'log', ]
        self.folder_paths = {
            folder_name: '../exp_data/{}/{}'.format(self.name, folder_name) for
            folder_name in folder_names
        }

        if mode == 'train':

            self.weights = [1] * 6
            if overwrite:
                self._overwrite_folders()
            else:
                self._make_folders()

            self.config = {
                'name': args.exp_name,
                'lr': args.learning_rate,
                'n_epoch': args.num_epochs,
                'stage': args.stage,
                'full': args.full,
                'inp_mode': args.inp_mode,
                'dropout':args.dropout,
            }
            set_logfile_path(self.log_path, mode)
            logger.info('Train experiment stage {}'.format(args.stage))
            logger.info('Learning Rate {}'.format(args.learning_rate))
            logger.info('Input data {} channels'.format(args.inp_mode))
        elif mode == 'predict':
            self.config = {
                'name': args.exp_name,
                'model_epoch': args.model_epoch,
            }
            set_logfile_path(self.log_path, mode)
            logger.info('Prediction Experiment {}'.format(args.exp_name))
            self.get_trained_model_info()

        elif mode == 'label_gen':
            self.config = {
                'name': args.exp_name,
                'model_epoch': args.model_epoch,
                'stage': args.stage,
            }
            set_logfile_path(self.log_path, 'data-gen')
            logger.info('Data Relabelling experiment')
            self.get_trained_model_info()

    def _make_folders(self):
        """Creates new experiment folders if they don't already exist."""
        for folder_path in self.folder_paths.values():
            os.makedirs(folder_path, exist_ok=True)

    def _overwrite_folders(self):
        """ Deletes old experiments folders and creates new folders"""
        for folder_path in self.folder_paths.values():
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
        self._make_folders()

    def get_trained_model_info(self):
        """Load the model using the model_epoch argument"""
        epoch = self.config['model_epoch']
        trained_model_path = os.path.join(self.model_folder,
                                          'model_{}.pth'.format(epoch))
        if not os.path.exists(trained_model_path):
            trained_model_path = trained_model_path.replace(
                'model_{}.pth'.format(epoch),
                'model_best.pth')

        trained_model = torch.load(trained_model_path)
        if self.mode != 'train':
            self.config['stage'] = trained_model['stage']
            self.config['inp_mode'] = trained_model['inp_mode']
            self.config['full'] = trained_model['full']
            self.config['dropout'] = False

        logger.info('Model loaded:{}'.format(trained_model_path))
        return trained_model

    @property
    def model_folder(self):
        return self.folder_paths['model']

    @property
    def log_path(self):
        return self.folder_paths['log']

    @property
    def lr(self):
        return self.config['lr']

    @property
    def stage(self):
        return self.config['stage']

    @property
    def full(self):
        return self.config['full']

    @property
    def inp_mode(self):
        return self.config['inp_mode']

    @property
    def dropout(self):
        return self.config['dropout']

