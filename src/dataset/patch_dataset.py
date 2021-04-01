import glob
import logging
import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import Compose
import torch.nn.functional as F

from utils.dir_paths import TRAIN_PATH
from utils.MFB import calculate_MFB_stage
from dataset.transforms import VerticalFlip, HorizontalFlip, Rotate90

logger = logging.getLogger(__name__)
np.set_printoptions(precision=4, suppress=True)

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


if 'lms' in os.uname()[1]:
    WORKERS = 30
else:
    WORKERS = 0


def check_data_split(train_path):
    file_list = glob.glob(os.path.join(train_path, 'stage_*.txt'))
    if file_list:
        return True
    else:
        return False


def setup_data(n_files=1, batch_size=1, mode='train', exp=None, stage=0, path=None,full=False):
    files_list = []
    datasets = []
    shuffle = False

    if mode == 'train':
        if not check_data_split(path):
            split_data(path)
        shuffle = True
        if full:
            file_path = os.path.join(path,'stage_full.txt')
            with open(file_path, 'r') as fl:
                files_list = [line.rstrip() for line in fl.readlines()]
                datasets.append(PatchDataset(n_files, mode, file_list=files_list, stage=0))
            logger.info("Total stage full train set size: {}".format(len(files_list)))
        else:
            for i in range(stage + 1):
                file_path = os.path.join(path, 'stage_{}.txt'.format(i))
                with open(file_path, 'r') as fl:
                    files_list = [line.rstrip() for line in fl.readlines()]
                    datasets.append(PatchDataset(n_files, mode, file_list=files_list, stage=i))
                logger.info("Total stage {} train set size: {}".format(stage, len(files_list)))

    elif mode == 'data_gen':
        for i in range(1, stage + 1):
            file_path = os.path.join(path, 'stage_{}.txt'.format(i))
            with open(file_path, 'r') as fl:
                files_list = [line.rstrip() for line in fl.readlines()]
                datasets.append(PatchDataset(len(files_list), mode, file_list=files_list, stage=i))

    else:
        files_list = glob.glob(os.path.join(path, '*.h5'))
        if not len(files_list):
            raise FileNotFoundError('H5 files not found')
        datasets.append(PatchDataset(len(files_list), mode, file_list=files_list, stage=stage))
    concat_dataset = ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(dataset=concat_dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle, num_workers=WORKERS)
    return dataloader


def split_data(h5_folder, stages=4):
    h5_folder = os.path.abspath(h5_folder)
    file_list = glob.glob(os.path.join(h5_folder, '*.h5'))
    n_files = len(file_list)
    if not n_files:
        raise FileNotFoundError('No h5 files found in {}'.format(h5_folder))
    stage_size = np.ceil(n_files / stages).astype(int)
    random.shuffle(file_list)

    for count, start in enumerate(range(0, n_files, stage_size)):

        end = min(n_files, start + stage_size)
        stage_list = file_list[start:end]
        # save stage file list to text file
        train_list_filename = os.path.join(h5_folder, 'stage_{}.txt'.format(count))
        with open(train_list_filename, 'w') as f:
            for train_file in stage_list:
                f.write("{}\n".format(train_file))


def get_MFB_weights(trainloader):
    freq = np.zeros((6))

    file_count = 0
    for dataset in trainloader.dataset.datasets:
        file_list = dataset.file_list[:dataset.size]
        stage = dataset.stage

        stage_freq, stage_counter = calculate_MFB_stage(file_list, stage)

        file_count += stage_counter
        logger.info('Stage {} freq {}'.format(stage, 100 * stage_freq / stage_counter))
        freq += stage_freq

    freq = freq / file_count

    logger.info('Total freq {}'.format(100 * freq))

    freq_median = np.median(freq)
    weight = np.divide(freq_median, freq, out=np.zeros_like(freq), where=freq > 1e-5)
    weight[0] = 1  # For the none class
    logger.info('MFB Weights {}'.format(weight))
    return weight


def shuffle_train_list(train_loader):
    for dt in train_loader.dataset.datasets:
        random.shuffle(dt.file_list)


class PatchDataset(Dataset):
    """
    Loads the saved patches of satellite images.
    """

    def __init__(self, size, mode, file_list, stage=0):
        """
        Inputs:
            dataset_name (string): Folder name of the dataset.
            size (int): Nr of files used in the dataset
            mode (string):
                Nature of operation to be done with the data.

            dataset_parent_path (string):
                Path of the folder where the dataset folder is present
                Default: DEFAULT_DATA_PATH as per config.json
            cfg_path (string):
                Config file path of the experiment

            seed
                Seed used for random functions
                Default:1
            batch_size: Batch size used for loader. For point cloud dataset, use 1
        """


        self.mode = mode

        # Initialize Database in order to get list of point clouds
        self.stage = stage
        self.file_list = file_list
        # Since we are randomly clipping the image, we process the same larger image multiple times in an epoch
        if mode == 'train':
            multiplier = 8
            self.file_list = self.file_list * multiplier

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.size = min(size,len(self.file_list))

        if mode == 'train':
            self.transforms = Compose([HorizontalFlip(),
                                       VerticalFlip(),
                                       Rotate90()])
        else:
            self.transforms = None

    def __len__(self):
        """Returns length of the dataset"""
        return self.size

    # @profile
    def __getitem__(self, idx):
        """
        Returns the data sample corresponding to the given index
        """
        with h5py.File(self.file_list[idx], 'r') as hf:
            spectral_image = hf.get('data')[:]
            # spectral_image = np.hstack((spectral_image, np.zeros((spectral_image.shape[0], 1,spectral_image.shape[2]), dtype=spectral_image.dtype)))
            # spectral_image = np.vstack((spectral_image, np.zeros((1, spectral_image.shape[1],spectral_image.shape[2]), dtype=spectral_image.dtype)))

        # with np.load(self.file_list[idx]) as data:
        #     # Normalize to range 0 to 1 from range 5000 to 12000
        #     spectral_image = data[data.files[0]]
        labels = spectral_image[:, :, 13:].astype(np.uint8)
        labels[labels < 0] = 0

        # band10 = spectral_image[:, :, 9] / 200
        spectral_image = spectral_image[:, :, :13].astype(np.float32)

        if self.transforms is not None:
            transform_input = [spectral_image, labels]
            transform_out = self.transforms(transform_input)
            spectral_image, labels = transform_out[0], transform_out[1]

        if self.mode == 'train':
            if self.stage == 0:
                # fmask is at index 13 in spectral_data
                # i.e index 0 in labels which started from 13
                idx = 0
            else:
                idx = 2  # stage labels at index 15
                if idx >= labels.shape[-1]:
                    raise NotImplementedError('Stage labels not found for stage {}'.format(self.stage))

            labels = labels[:, :, idx][:, :, None]

        # sp_max = spectral_image.max(axis=(0,1)).reshape(1,1,13)
        # sp_min = spectral_image.min(axis=(0,1)).reshape(1,1,13)
        sp_mean = spectral_image.mean(axis=(0, 1)).reshape(1, 1, 13)
        sp_std = spectral_image.std(axis=(0, 1)).reshape(1, 1, 13)
        # sp_max[sp_max==sp_min]=1
        sp_std[sp_std == 0] = 1
        # spectral_image = (spectral_image - sp_min)/(sp_max-sp_min)
        # spectral_image = (spectral_image - sp_mean) / (sp_std)

        if self.mode == 'train':
            # if True:
            x_start = random.randint(0, 549 - 256)
            y_start = random.randint(0, 549 - 256)
            labels = labels[x_start:x_start + 256, y_start:y_start + 256, :]
            spectral_image = spectral_image[x_start:x_start + 256, y_start:y_start + 256, :]

        spectral_image = torch.tensor(spectral_image, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long, )

        if self.mode == 'train' or self.mode == 'predict':
            p2d = (0,0,1, 1, 1, 1)  # pad last dim by (1, 1) and 2nd to last by (2, 2)
            spectral_image = F.pad(spectral_image, p2d, "constant", 0)
            labels = F.pad(labels, p2d, "constant", 0)

        return spectral_image, labels, self.file_list[idx]



if __name__ == '__main__':
    p = setup_data(12, 3, 'train')
    for n, i in enumerate(p):
        print(n)
