import glob
import logging
import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import Compose

from utils.dir_paths import TRAIN_PATH
from utils.MFB import calculate_MFB_stage
from dataset.transforms import VerticalFlip, HorizontalFlip, Rotate90, CutOut, ZoomIn

logger = logging.getLogger(__name__)
np.set_printoptions(precision=4, suppress=True)

# Set number of workers for parallel dataloading based on the capacity of the CPU
if 'lms' in os.uname()[1]:
    WORKERS = 30
else:
    WORKERS = 0


def set_seed(user_seed):
    # To ensure reproducibility
    if user_seed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(user_seed)
        torch.manual_seed(user_seed)
        np.random.seed(user_seed)
        logger.info(
            'Training using fixed seed {} with {}'.format(user_seed, WORKERS))
    else:
        logger.info('Training without fixed seed')


def check_data_split(train_path, reset=False):
    file_list = glob.glob(os.path.join(train_path, 'stage_*.txt'))
    if file_list:
        if reset:
            for stage_file in file_list: os.remove(stage_file)
        else:
            return True

    return False


def setup_data(n_files=1, batch_size=1, mode='train', exp=None, stage=0, path=None,
               full=False, aug=False,
               reset=False, stage_0_ratio=0.1):
    files_list = []
    datasets = []
    shuffle = False

    if mode == 'train':
        if stage != 0 and reset:
            logger.warning(
                "Stage data reset can be done only in stage 0. Setting reset_stage_data flag to False")
            reset = False
        if not check_data_split(path, reset=reset):
            split_data(path, stage_0_ratio)
        shuffle = True
        if full:
            file_path = os.path.join(path, 'stage_full.txt')
            with open(file_path, 'r') as fl:
                files_list = [line.rstrip() for line in fl.readlines()]
                datasets.append(
                    PatchDataset(n_files, mode, file_list=files_list, stage=0,
                                 aug=aug))
            logger.info(
                "Total stage full train set size: {}".format(len(files_list)))
        else:
            for i in range(stage + 1):
                file_path = os.path.join(path, 'stage_{}.txt'.format(i))
                with open(file_path, 'r') as fl:
                    files_list = [line.rstrip() for line in fl.readlines()]

                    # No augmentation for stage 0 data even if aug parameter is True.
                    if aug:
                        stage_aug = bool(stage)
                    else:
                        stage_aug = aug

                    datasets.append(
                        PatchDataset(n_files, mode, file_list=files_list, stage=i,
                                     aug=stage_aug))
                logger.info(
                    "Total stage {} train set size: {}".format(i, len(files_list)))

    elif mode == 'data_gen':
        for i in range(1, stage + 1):
            file_path = os.path.join(path, 'stage_{}.txt'.format(i))
            with open(file_path, 'r') as fl:
                files_list = [line.rstrip() for line in fl.readlines()]
                datasets.append(
                    PatchDataset(len(files_list), mode, file_list=files_list,
                                 stage=i, aug=aug))

    else:
        files_list = glob.glob(os.path.join(path, '*.h5'))
        if not len(files_list):
            raise FileNotFoundError('H5 files not found')
        datasets.append(
            PatchDataset(len(files_list), mode, file_list=files_list, stage=stage,
                         aug=aug))
    concat_dataset = ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(dataset=concat_dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle, num_workers=WORKERS,
                                             )
    return dataloader


def split_data(h5_folder, stage_0_ratio, stages=4):
    h5_folder = os.path.abspath(h5_folder)
    file_list = glob.glob(os.path.join(h5_folder, '*.h5'))
    n_files = len(file_list)
    if not n_files:
        raise FileNotFoundError('No h5 files found in {}'.format(h5_folder))

    train_list_filename = os.path.join(h5_folder, 'stage_full.txt')
    with open(train_list_filename, 'w') as f:
        for train_file in file_list:
            f.write("{}\n".format(train_file))
    unlabelled_ratio = 1 - stage_0_ratio
    unlabelled_size = int(n_files * unlabelled_ratio / (stages - 1))
    labeled_size = n_files - (stages - 1) * unlabelled_size

    random.shuffle(file_list)

    start = 0
    for count, stage in enumerate(range(stages)):
        stage_size = labeled_size if stage == 0 else unlabelled_size

        end = min(n_files, start + stage_size)
        stage_list = file_list[start:end]
        # save stage file list to text file
        train_list_filename = os.path.join(h5_folder, 'stage_{}.txt'.format(count))
        with open(train_list_filename, 'w') as f:
            for train_file in stage_list:
                f.write("{}\n".format(train_file))
        start = end


def get_MFB_weights(trainloader):
    freq = np.zeros((6))

    file_count = 0
    for dataset in trainloader.dataset.datasets:
        file_list = dataset.file_list[:dataset.size]
        stage = dataset.stage

        stage_freq, stage_counter = calculate_MFB_stage(file_list, stage)

        file_count += stage_counter
        logger.info(
            'Stage {} freq {}'.format(stage, 100 * stage_freq / stage_counter))
        freq += stage_freq

    freq = freq / file_count

    logger.info('Total freq {}'.format(100 * freq))

    freq_median = np.median(freq)
    weight = np.divide(freq_median, freq, out=np.zeros_like(freq), where=freq > 1e-5)
    weight[0] = 0  # For the none class
    # logger.info('MFB Weights {}'.format(weight))
    logger.info('Class Weights {}'.format(weight))
    return weight


class PatchDataset(Dataset):
    """
    Loads the saved sub-scenes of satellite images.
    """

    def __init__(self, size, mode, file_list, stage=0, aug=False):

        self.size = size
        self.mode = mode

        self.stage = stage
        self.file_list = file_list

        if mode == 'train':
            if stage == 0:
                # multiplier = 2
                multiplier = 1
                if aug:
                    multiplier = 1
            else:
                multiplier = 1

            self.file_list = self.file_list * multiplier

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.size = min(len(self.file_list), self.size)

        self.transforms = None
        if mode == 'train':
            if aug:
                self.transforms = Compose([HorizontalFlip(),
                                           VerticalFlip(),
                                           ZoomIn(),
                                           Rotate90(),
                                           CutOut(),

                                           ])
            logger.info("Stage {} augmentation flag set to {}".format(stage, aug))

    def __len__(self):
        """Returns length of the dataset"""
        return self.size

    def __getitem__(self, idx):
        """
        Returns the data sample corresponding to the given index
        """
        with h5py.File(self.file_list[idx], 'r') as hf:
            spectral_image = hf.get('data')[:]

        labels = spectral_image[:, :, 13:].astype(np.uint8)
        labels[labels < 0] = 0

        spectral_image = spectral_image[:, :, :13].astype(np.float32)

        if self.mode == 'train':
            if self.stage == 0:
                # fmask is at index 13 in spectral_data
                # i.e index 0 in labels which started from 13
                idx = 0
            else:
                idx = 2  # stage labels at index 15
                if idx >= labels.shape[-1]:
                    raise NotImplementedError(
                        'Stage labels not found for stage {}'.format(self.stage))

            labels = labels[:, :, idx][:, :, None]

        if self.mode == 'train':
            if self.transforms is not None:
                transform_input = [spectral_image, labels]
                transform_out = self.transforms(transform_input)
                spectral_image, labels = transform_out[0], transform_out[1]

        # Padding around the image
        p2d = ((1, 1), (1, 1), (0, 0))  # pad dim of height and width by (1, 1)
        spectral_image = np.pad(spectral_image, p2d, 'constant', constant_values=0)
        cl10 = spectral_image[:, :, 9] / 1000
        spectral_image = spectral_image / 10000
        spectral_image[:, :, 9] = cl10
        labels = np.pad(labels, p2d, 'constant', constant_values=0)

        # Convert to HWC format to CHW
        # H=height, W=width, C=channel
        spectral_image = np.transpose(spectral_image, (2, 0, 1))
        labels = np.transpose(labels, (2, 0, 1)).astype(np.long)

        return spectral_image, labels, self.file_list[idx]


if __name__ == '__main__':
    pass
