"""This script contains various functions that are used for splitting a given
Sentinel-2 input from the SAFE folder into sub-scenes stored as h5 files containing
the 13 spectral bands as well as the labels (when available). """

import argparse
import glob
import logging
import os
import tempfile
from itertools import product

import cv2
import h5py
import numpy as np
import rasterio

from utils.dataset_stats import main_csv
from utils.dir_paths import TRAIN_SAFE_PATH, VALID_SAFE_PATH

logger = logging.getLogger('__name__')


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
    )


# When we have tiles_per_row=n, we will have total tiles as n*n
# window size is written in terms of pixels in 10m resolution
# Network processes input at 20m resolution. The input is zero-padded.


NETWORK_INPUT_SIZE = 256
border_width = 1

# Effective sub-scene size in 20m resolution by subtracting the border width on
# both sides.
SIZE_20M = NETWORK_INPUT_SIZE - (border_width * 2)

SIZE_10M = 2 * SIZE_20M

# There are 10980 pixels in the 10m resolution image.
TILES_PER_ROW = np.ceil(10980 / SIZE_10M).astype(int)

# To rescale to 20m resolution set Rescale size to half of window size.
RESCALE_SIZE = int(SIZE_10M / 2)


def get_args(argv=None):
    """Parses the arguments entered by the user."""

    parser = argparse.ArgumentParser(description="Splits SAFE files into h5 files ")
    parser.add_argument('-p', '--path', help='Path of containing safe folders',
                        default=None)
    parser.add_argument('-m', '--mode', default='train',
                        help='mode for data use')

    return parser.parse_args(argv)


def generate_out_filename(img_path, mode, file_format='tif'):
    parent_path, image = img_path.replace('jp2', file_format).replace('raw', mode).split(
        '.SAFE')
    out_filename = '{}_{}'.format(parent_path, image.split('_')[-1])
    return out_filename


def get_img_paths(safe_path, mode):
    """Returns a sorted list of image files for a given safe folder path. The
    FMASK will be included and the true color image will be excluded from the
    list. """

    # get all Sentinel2 filenames
    img_paths = sorted(glob.glob(os.path.join(safe_path, '**', 'IMG_DATA', '*.jp2'),
                                 recursive=True))
    # remove true color image
    img_paths = list(filter(lambda x: not x.endswith('TCI.jp2'), img_paths))
    # get fmask
    fmask_path = glob.glob(os.path.join(safe_path, '**', 'IMG_DATA', '*FMASK.tif'),
                           recursive=True)  # F4MASK
    # raise error if fmask not found in train mode
    if len(fmask_path) == 1:
        img_paths.extend(fmask_path)
    elif mode == 'train':
        raise FileNotFoundError('FMask file not found in format *FMASK.tif')
    else:
        # if fmask is not available in test or predict, we create a zero array
        # instead
        logger.warning(
            'No F4MASK file found for {}'.format(os.path.basename(safe_path)))
        nofile_path = 'NOFILE' + os.path.basename(img_paths[0]).replace('B01.jp2',
                                                                        'FMASK.tif')
        img_paths.append(nofile_path)

    # For validation mode check if true label is available and get its path
    if mode == 'test':
        label_path = glob.glob(
            os.path.join(safe_path, '**', 'IMG_DATA', '*LABELS.tif'),
            recursive=True)
        # raise error if fmask not found in train mode
        if len(label_path) == 1:
            img_paths.extend(label_path)
        else:
            raise FileNotFoundError('Label file not found in format *LABELS.tif')

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


def save_as_gtiff(window_data, metadata, out_filename):
    metadata.update({'driver': 'GTiff'})

    with rasterio.open(out_filename, 'w', **metadata) as dst:
        dst.write(window_data, 1)


def fix_window_size(window_data, window_transform, label_interpolation=False):
    if window_data.shape[0] != RESCALE_SIZE:
        window_data, window_transform = resize_window(RESCALE_SIZE, window_data,
                                                      window_transform,
                                                      label_interpolation)

    return window_data, window_transform


def split_img(img_path, temp_dirpath, overlap):
    if img_path.startswith('NOFILE'):
        window_data = np.zeros((RESCALE_SIZE, RESCALE_SIZE), dtype=np.uint16)
        for n_patch in range(TILES_PER_ROW ** 2):
            out_path = os.path.join(temp_dirpath,
                                    img_path.replace('NOFILE',
                                                     '') + "_PATCH{}".format(
                                        n_patch))
            np.save(out_path, window_data)

    else:
        band = rasterio.open(img_path)
        window_length = SIZE_10M * 10

        top_bound = band.bounds.top - window_length
        right_bound = band.bounds.right - window_length

        for n_patch, (a, b) in enumerate(product(range(TILES_PER_ROW), repeat=2)):
            left = min(band.bounds.left + (b * window_length), right_bound)
            bottom = min(band.bounds.bottom + (a * window_length), top_bound)
            top = bottom + window_length
            right = left + window_length

            polygon_window = rasterio.windows.from_bounds(left=left,
                                                          bottom=bottom,
                                                          right=right,
                                                          top=top,
                                                          transform=band.transform)

            window_data = band.read(1, window=polygon_window)
            window_transform = rasterio.windows.transform(polygon_window,
                                                          band.transform)

            if 'LABEL' in img_path or 'FMASK' in img_path:
                label_interpolation = True
            else:
                label_interpolation = False
            window_data, window_transform = fix_window_size(window_data,
                                                            window_transform,
                                                            label_interpolation)

            out_path = os.path.join(temp_dirpath,
                                    os.path.basename(img_path) + "_PATCH{}".format(
                                        n_patch))
            np.save(out_path, window_data)


def make_patch(safe_file_list, mode):
    total_files = len(safe_file_list)
    for file_n, safe_path in enumerate(safe_file_list):

        logger.info('[{}/{}] {}'.format(file_n + 1,
                                        total_files,
                                        os.path.basename(safe_path)))

        img_paths = get_img_paths(safe_path, mode=mode)
        if mode == 'predict':
            out_folder_path = safe_path + '_H5'
            safe_filename = os.path.basename(safe_path)
            overlap = True
        else:
            [parent_path, safe_filename] = safe_path.rsplit('/', 1)
            out_folder_path = parent_path + '_H5'
            overlap = False
        os.makedirs(out_folder_path, exist_ok=True)

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
        with tempfile.TemporaryDirectory() as td:
            for n_i, img_path in enumerate(img_paths):
                split_img(img_path, td, overlap)
            logger.debug("Completed split")
            for n_patch in range(TILES_PER_ROW ** 2):
                spectral_data = np.zeros(
                    (RESCALE_SIZE, RESCALE_SIZE, len(img_paths)), dtype=np.uint16)
                out_file_name = safe_filename.replace('.SAFE',
                                                      '_PATCH{}.h5'.format(n_patch))

                split_img_paths = sorted(
                    glob.glob(os.path.join(td, '*_PATCH{}.npy'.format(n_patch))))
                for idx, split_img_path in enumerate(split_img_paths):
                    window = np.load(split_img_path)
                    spectral_data[:, :, idx] = window

                    labels_file = {'test': 'LABELS.tif', 'train': 'FMASK.tif'}

                    if mode in labels_file and labels_file[mode] in split_img_path:
                        if np.all(window == 0):
                            out_file_name = safe_filename.replace('.SAFE',
                                                                  '_PATCH{}.h5.NIL'.format(
                                                                      n_patch))
                if not out_file_name.endswith('h5.NIL'):
                    out_file_path = os.path.join(out_folder_path, out_file_name)
                    with h5py.File(out_file_path, "w") as hf:
                        hf.create_dataset('data', data=spectral_data, )
    main_csv(mode=mode, path=out_folder_path)


if __name__ == '__main__':

    setup_logger()
    args = get_args()

    # When path is not given, use default path from dir_paths.py
    if not args.path:
        if args.mode == 'train':
            args.path = TRAIN_SAFE_PATH
        elif args.mode == 'test':
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
