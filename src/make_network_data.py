"""
This script is used to split the Sentinel-2 scene into multiple sub-scenes stored as h5 files
containing the 13 spectral bands as well as the labels (when available).
"""

import argparse
import glob
import logging
import os
from itertools import product

import cv2
import h5py
import numpy as np
import rasterio

from utils.dir_paths import TRAIN_SAFE_PATH, VALID_SAFE_PATH
from utils.dataset_stats import save_stats

# Logging
logger = logging.getLogger('__name__')


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
    )


# When we have tiles_per_row=n, we will have total tiles as n*n
PATCHES_PER_ROW = 10
WINDOW_SIZE = np.ceil(10980 / PATCHES_PER_ROW).astype(int)

RESCALE_SIZE = 549  # Set to 0 if rescaling is not to be done.

if not RESCALE_SIZE:
    RESCALE_SIZE = WINDOW_SIZE


def get_args(argv=None):
    """Parses the arguments entered by the user."""

    parser = argparse.ArgumentParser(description="Splits SAFE files into h5 files ")
    parser.add_argument('-p', '--path', help='Path of containing safe folders',
                        default=None)
    parser.add_argument('-m', '--mode', default='train',
                        help='mode for data use')

    return parser.parse_args(argv)


def get_img_paths(safe_path):
    """Returns a sorted list of image files for a given safe folder path. The
    FMASK will be included and the true color image will be excluded from the
    list.
    """

    img_paths = sorted(glob.glob(os.path.join(safe_path, '**', 'IMG_DATA', '*.jp2'),
                                 recursive=True))
    img_paths.extend(
        glob.glob(os.path.join(safe_path, '**', 'IMG_DATA', '*FMASK.tif'),
                  recursive=True))
    img_paths.extend(
        glob.glob(os.path.join(safe_path, '**', 'IMG_DATA', '*LABELS.tif'),
                  recursive=True))
    img_paths = list(filter(lambda x: not x.endswith('TCI.jp2'), img_paths))

    return img_paths


def resize_window(new_size, window_data, window_transform,
                  label_interpolation=False):
    """Resizes the windows from different bands. Since bands come in multiple
    resolutions they need to be resized. Cubic interpolation is used for resizing
    the bands. Nearest neighbour interpolation is used for labels because
    cubic interpolation will lead to floating point labels which cannot be used
    for training or validation.
    """

    old = window_data.shape[0]
    if label_interpolation:
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_CUBIC
    window_data = cv2.resize(window_data, dsize=(new_size, new_size),
                             interpolation=interpolation)

    # scale image transform
    window_transform = window_transform * window_transform.scale(
        (old / new_size),
        (old / new_size)
    )
    return window_data, window_transform


def read_window(img_path, vertical_index=None, horizontal_index=None):
    """Extract band data from the image path using a window whose bounds are
    defined by vertical and horizontal indices and the window size. By passing the
    vertical indices and horizontal indices as None, the whole image can be read.
    The images are then rescaled to the size defined by RESCALE_SIZE global
    parameter.
    """

    band = rasterio.open(img_path)
    window_length = WINDOW_SIZE * 10

    top_bound = band.bounds.top - window_length
    right_bound = band.bounds.right - window_length
    if vertical_index is not None:
        left = min(band.bounds.left + (horizontal_index * window_length),
                   right_bound)
        bottom = min(band.bounds.bottom + (vertical_index * window_length),
                     top_bound)
        top = bottom + window_length
        right = left + window_length
    else:
        left = band.bounds.left
        bottom = band.bounds.bottom
        top = band.bounds.top
        right = band.bounds.right

    polygon_window = rasterio.windows.from_bounds(left=left,
                                                  bottom=bottom,
                                                  right=right,
                                                  top=top,
                                                  transform=band.transform)

    window_data = band.read(1, window=polygon_window)
    window_transform = rasterio.windows.transform(polygon_window, band.transform)
    metadata = band.meta.copy()

    metadata['width'] = window_data.shape[1]
    metadata['height'] = window_data.shape[0]
    metadata['bounds'] = (left, bottom, right, top)

    if vertical_index is not None:
        if 'LABEL' in img_path or 'FMASK' in img_path:
            label_interpolation = True
        else:
            label_interpolation = False

        if window_data.shape[0] != RESCALE_SIZE:
            window_data, window_transform = resize_window(RESCALE_SIZE, window_data,
                                                          window_transform,
                                                          label_interpolation)

    return window_data, metadata




def make_patch(safe_file_list, mode='train'):
    """Generates patch files in h5 format for each safe file in the safe_file_list"""

    for file_n, safe_path in enumerate(safe_file_list):
        logger.info('{} {}'.format(file_n + 1, safe_path))
        img_paths = get_img_paths(safe_path)

        # create directory to save h5 files
        if mode == 'predict':
            h5_folder_path = safe_path + '_H5'
            safe_filename = os.path.basename(safe_path)
        else:
            [parent_path, safe_filename] = safe_path.rsplit('/', 1)
            h5_folder_path = parent_path + '_H5'
        os.makedirs(h5_folder_path, exist_ok=True)

        # Store metadata of the parent file  and each patch (done inside loop) to
        # use for joining later if required.
        metadata_dict = {}

        B05_path = [pth for pth in img_paths if 'B05' in pth][0]
        B05_band = rasterio.open(B05_path)
        metadata = B05_band.meta.copy()
        metadata.update({
            'driver': 'GTiff',
            'dtype': 'uint8'
        })

        metadata_dict['parent'] = metadata
        del B05_band

        for n_patch, (a, b) in enumerate(product(range(PATCHES_PER_ROW), repeat=2)):

            spectral_data = np.zeros((RESCALE_SIZE, RESCALE_SIZE, len(img_paths)),
                                     dtype=np.uint16)

            out_file_name = safe_filename.replace('.SAFE',
                                                  '_PATCH{}.h5'.format(n_patch))
            for n_i, img_path in enumerate(img_paths):

                window_data, metadata = read_window(img_path, a, b)
                spectral_data[:, :, n_i] = window_data

                if mode == 'validation' and 'LABELS.tif' in img_path:
                    if np.all(window_data == 0):
                        out_file_name = safe_filename.replace('.SAFE',
                                                              '_PATCH{}.h5.NIL'.format(
                                                                  n_patch))

                if 'B05' in img_path:
                    metadata_dict[n_patch] = metadata

            out_file_path = os.path.join(h5_folder_path, out_file_name)

            # Prevent saving patch file that don't have any validation labels.
            # This check does not apply for the train mode.
            if not out_file_name.endswith('.NIL'):
                with h5py.File(out_file_path, "w") as hf:
                    hf.create_dataset('data', data=spectral_data, )

        # save the metadata dictionary as a numpy data file
        out_file_name = safe_filename.replace('.SAFE', '_META.npy')
        out_file_path = os.path.join(h5_folder_path, out_file_name)
        np.save(out_file_path, metadata_dict)
    if mode == 'train':
        logger.info('Generating dataset stats from {}'.format(h5_folder_path))
        save_stats(h5_folder_path)





if __name__ == '__main__':

    setup_logger()
    args = get_args()

    # When path is not given, use default path from dir_paths.py
    if not args.path:
        if args.mode == 'train':
            args.path = TRAIN_SAFE_PATH
        elif args.mode == 'validation':
            args.path = VALID_SAFE_PATH
        else:
            NotImplementedError()

    safe_parent_path = os.path.abspath(args.path)
    logger.info('Processing files from {}'.format(safe_parent_path))
    safe_file_list = glob.glob(os.path.join(safe_parent_path, '*.SAFE'))

    if len(safe_file_list) == 0:
        # When no safe file is found
        raise FileNotFoundError('Provide path of folder containing safe files '
                                'using --path argument.')

    make_patch(safe_file_list, args.mode)
