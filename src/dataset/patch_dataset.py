import glob
import logging
import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import Compose


from dataset.transforms import VerticalFlip, HorizontalFlip, Rotate90, CutOut, ZoomIn

logger = logging.getLogger(__name__)
np.set_printoptions(precision=4, suppress=True)

# Set number of workers for parallel dataloading based on the capacity of the CPU
WORKERS = 16


def set_seed(user_seed):
    # Set the seed used in random number generators to ensure reproducibility
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
    """Checks if h5 files have been already assigned to different stages. This
    is done by checking if a file named stage_1.txt, stage_2.txt, etc exists.
    If reset flag is true, then these files are deleted so that they can be
    generated again resulting in a new set of files for each stage.
    """
    file_list = glob.glob(os.path.join(train_path, 'stage_*.txt'))
    if file_list:
        if reset:
            for stage_file in file_list: os.remove(stage_file)
        else:
            return True

    return False


def setup_data(batch_size=1, mode='train', stage=0, path=None,
               full=False, aug=False,
               reset=False):
    """Setups up the dataloaders for each stage of the pipeline"""
    datasets = []
    shuffle = True if mode == 'train' else False

    if mode == 'train':
        if stage != 0 and reset:
            logger.warning(
                "Stage data reset can be done only in stage 0. Setting "
                "reset_stage_data flag to False")
            reset = False
        # Check if h5 have been already assigned to each stage
        if not check_data_split(path, reset=reset):
            split_data(path)

        if full:
            # For fully supervised mode, the text file named stage_full.txt with
            # all the h5 files is used.
            file_path = os.path.join(path, 'stage_full.txt')
            with open(file_path, 'r') as fl:
                # Read file names and store to list
                files_list = [line.rstrip() for line in fl.readlines()]
                datasets.append(
                    PatchDataset(mode, file_list=files_list, stage=0,
                                 aug=aug))
            logger.info(
                "Total stage full train set size: {}".format(len(files_list)))
        else:
            for i in range(stage + 1):
                # For self-trained mode, the text file named stage_0.txt, etc with
                # all the h5 files for stage 0 is used. Similarly for stages 1,
                # 2 and 3.
                file_path = os.path.join(path, 'stage_{}.txt'.format(i))
                with open(file_path, 'r') as fl:
                    files_list = [line.rstrip() for line in fl.readlines()]

                    # No augmentation for stage 0 data even if aug parameter is True.
                    if aug:
                        stage_aug = bool(stage)
                    else:
                        stage_aug = aug

                    datasets.append(
                        PatchDataset(mode, file_list=files_list, stage=i,
                                     aug=stage_aug))
                logger.info(
                    "Total stage {} train set size: {}".format(i, len(files_list)))

    elif mode == 'label_gen':
        for i in range(1, stage + 1):
            # For label generation, we use previous h5 files expect from stage 0.
            file_path = os.path.join(path, 'stage_{}.txt'.format(i))
            with open(file_path, 'r') as fl:
                files_list = [line.rstrip() for line in fl.readlines()]
                datasets.append(
                    PatchDataset(mode, file_list=files_list,
                                 stage=i, aug=aug))

    else:
        files_list = glob.glob(os.path.join(path, '*.h5'))
        if not len(files_list):
            raise FileNotFoundError('H5 files not found')
        datasets.append(
            PatchDataset(mode, file_list=files_list, stage=stage,
                         aug=aug))
    concat_dataset = ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(dataset=concat_dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle, num_workers=WORKERS,
                                             )
    return dataloader


def split_data(h5_folder, stage_0_ratio=0.25, stages=4):
    """Assigns a set of h5 files that are to be used in each stage of the pipeline.
    These filenames are stored in the train path in a text file named as stage_1.txt,
    stage_2.txt, etc."""

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





class PatchDataset(Dataset):
    """
    Loads the saved sub-scenes of satellite images from the h5 files.
    """

    def __init__(self, mode, file_list, stage=0, aug=False):

        self.mode = mode

        self.stage = stage
        self.file_list = file_list
        self.size = len(self.file_list)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
