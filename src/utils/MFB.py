"""This script implements the mean frequency balancing used to calculate the weights
for the loss function"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PIXELS_PER_IMAGE = 549 ** 2


def calculate_file_freq(labels):
    """Calculate the class frequency from the labels numpy array"""

    file_freq = np.zeros(6)
    # Labels are numbered from 0 to 5. Apply equality condition for each label and
    # count.
    for i in range(6):
        file_freq[i] = np.count_nonzero(labels == i)

    return file_freq


def calculate_MFB_stage(file_list, stage):
    """Read the class frequency for the files that belong to the given stage (
    provided in the file_list argument). The frequencies are stored in the label
    stats csv file. Return the total class frequency for each class. """

    filename_list = [os.path.basename(i) for i in file_list]
    parent_folder = os.path.dirname(file_list[0])

    # Select the label stats csv file depending on the stage
    if stage == 0:
        # fmask is at index 13 in spectral_data
        label_stats_file = os.path.join(parent_folder, 'label_stats.csv')
    else:
        label_stats_file = os.path.join(parent_folder,
                                        'label_stats_stage{}.csv'.format(stage))

    # Read csv as pandas dataframe
    label_stats_df = pd.read_csv(label_stats_file)

    # Obtain frequencies for files from filename_list and store the sum of
    # frequencies for each class.
    stage_stats = label_stats_df[label_stats_df['FILENAME'].isin(filename_list)]
    stage_freq = np.array([stage_stats['NONE_F'].sum(),
                           stage_stats['CLEAR_F'].sum(),
                           stage_stats['CLOUD_F'].sum(),
                           stage_stats['SHADOW_F'].sum(),
                           stage_stats['ICE_F'].sum(),
                           stage_stats['WATER_F'].sum()],
                          dtype=float) / PIXELS_PER_IMAGE

    return stage_freq, stage_stats.shape[0]


def get_MFB_weights(trainloader):
    freq = np.zeros((6))

    file_count = 0
    for dataset in trainloader.dataset.datasets:
        file_list = dataset.file_list[:dataset.size]

        stage_freq, stage_counter = calculate_MFB_stage(file_list, dataset.stage)

        file_count += stage_counter
        logger.info(
            'Stage {} freq {}'.format(dataset.stage, 100 * stage_freq / stage_counter))
        freq += stage_freq

    freq = freq / file_count

    logger.info('Total freq {}'.format(100 * freq))

    freq_median = np.median(freq)
    weight = np.divide(freq_median, freq, out=np.zeros_like(freq), where=freq >= 1)
    weight[0] = 0  # For the no-data class
    logger.info('Class Weights {}'.format(weight))
    return weight