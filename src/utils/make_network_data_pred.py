import random
import glob
import os
import numpy as np
import rasterio
import cv2
import argparse
from itertools import product
import logging

logger = logging.getLogger('__name__')
import h5py

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
    )

# When we have tiles_per_row=n, we will have totol tiles as n*n
TILES_PER_ROW = 10
WINDOW_SIZE = np.ceil(10980 / TILES_PER_ROW).astype(int)

RESCALE_SIZE = 549  # 512 Set to 0 if rescaling is not to be done.
if not RESCALE_SIZE:
    RESCALE_SIZE = WINDOW_SIZE


def get_args(argv=None):
    """Parses the arguments entered by the user."""

    parser = argparse.ArgumentParser(description="Splits SAFE files into h5 files ")
    parser.add_argument('-p', '--path', help='Path of containing safe folders', default=None)
    parser.add_argument('-m', '--mode', default='train',
                        help='mode for data use')

    return parser.parse_args(argv)


def generate_out_filename(img_path, mode, format='tif'):
    parent_path, image = img_path.replace('jp2', format).replace('raw', mode).split('.SAFE')
    out_filename = '{}_{}'.format(parent_path, image.split('_')[-1])
    return out_filename


def save_band(img_path, mode='train'):
    window_data, metadata = read_window(img_path)
    out_filename = generate_out_filename(img_path, mode)
    save_as_gtiff(window_data, metadata, out_filename)


def get_img_paths(safe_path):
    """Returns a sorted list of image files for a given safe folder path. The FMASK will be included and the true
    color image will be excluded from the list."""

    img_paths = sorted(glob.glob(os.path.join(safe_path, '**', 'IMG_DATA', '*.jp2'), recursive=True))
    img_paths.extend(glob.glob(os.path.join(safe_path, '**', 'IMG_DATA', '*FMASK.img'), recursive=True))
    img_paths.extend(glob.glob(os.path.join(safe_path, '**', 'IMG_DATA', '*LABELS.tif'), recursive=True))
    img_paths = list(filter(lambda x: not x.endswith('TCI.jp2'), img_paths))
    # img path should contain 13 spectral images and fmask output
    # assert len(img_paths) == 14
    return img_paths


def resize_window(new_size, window_data, window_transform, label_interpolation=False):
    old = window_data.shape[0]
    if label_interpolation:
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_CUBIC
    window_data = cv2.resize(window_data, dsize=(new_size, new_size), interpolation=interpolation)

    # scale image transform
    window_transform = window_transform * window_transform.scale(
        (old / new_size),
        (old / new_size)
    )
    return window_data, window_transform


def save_as_gtiff(window_data, metadata, out_filename):
    metadata.update({
        'driver': 'GTiff'})

    with rasterio.open(out_filename, 'w', **metadata) as dst:
        dst.write(window_data, 1)


def read_window(img_path, bottom_offset=None, left_offset=None):
    band = rasterio.open(img_path)
    window_length = WINDOW_SIZE * 10
    window_length1 = WINDOW_SIZE * 8

    top_bound = band.bounds.top - window_length
    right_bound = band.bounds.right - window_length
    if not bottom_offset is None:
        left = min(band.bounds.left + (left_offset * window_length1), right_bound)
        bottom = min(band.bounds.bottom + (bottom_offset * window_length1), top_bound)
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

    if not bottom_offset is None:
        if 'LABEL' in img_path or 'FMASK' in img_path:
            label_interpolation = True
        else:
            label_interpolation = False
        window_data, window_transform = fix_window_size(window_data, window_transform, label_interpolation)

    return window_data, metadata


def fix_window_size(window_data, window_transform, label_interpolation=False):
    if window_data.shape[0] != RESCALE_SIZE:
        window_data, window_transform = resize_window(RESCALE_SIZE, window_data, window_transform, label_interpolation)

    return window_data, window_transform


# def get_true_color_img(img_paths,mode='train'):
#     img_size = 2048
#     channels = np.zeros((img_size,img_size,3))
#     for i in range(1,4):
#         # cv2 also does BGR like sentinel 2
#         channels[:,:,i-1]= cv2.resize(read_window(img_paths[i])[0], dsize=(img_size, img_size))
#     channels =cv2.normalize(channels, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     out_filename = generate_out_filename(img_paths[0], mode,'jpg').replace('B01.','RGB.')
#     cv2.imwrite(out_filename, channels)

def make_patch(safe_file_list, mode='train'):


    for file_n, safe_path in enumerate(safe_file_list):
        logger.info('{} {}'.format(file_n + 1, safe_path))
        img_paths = get_img_paths(safe_path)
        if mode=='predict':
            out_folder_path = safe_path+'_H5'
            safe_filename = os.path.basename(safe_path)
        else:
            [parent_path,safe_filename] = safe_path.rsplit('/', 1)
            out_folder_path = parent_path + '_H5'
        os.makedirs(out_folder_path, exist_ok=True)
        # get_true_color_img(img_paths,mode)

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

        for n_patch, (a, b) in enumerate(product(range(13), repeat=2)):

            spectral_data = np.zeros((RESCALE_SIZE, RESCALE_SIZE, len(img_paths)), dtype=np.uint16)

            out_file_name = safe_filename.replace('.SAFE', '_PATCH{}.h5'.format(n_patch))
            for n_i, img_path in enumerate(img_paths):
                # if not 'FMASK' in img_path:
                #     continue
                window_data, metadata = read_window(img_path, a, b)
                spectral_data[:, :, n_i] = window_data

                if mode == 'test' and 'LABELS.tif' in img_path:
                    if np.all(window_data==0):
                        out_file_name = safe_filename.replace('.SAFE', '_PATCH{}.h5.NIL'.format(n_patch))

                if 'B05' in img_path:
                    metadata_dict[n_patch] = metadata

            out_file_path = os.path.join(out_folder_path, out_file_name)
            with h5py.File(out_file_path, "w") as hf:
                hf.create_dataset('data', data=spectral_data, )

        out_file_name = safe_filename.replace('.SAFE', '_META.npy')
        out_file_path = os.path.join(out_folder_path, out_file_name)
        np.save(out_file_path, metadata_dict)



if __name__ == '__main__':

    # Set a constant seed for reproducibility
    random.seed(42)
    setup_logger()

    args = get_args()

    # Setup path containing SAFE folder
    if args.path:
        safe_parent_path = os.path.abspath(args.path)

    else:
        # When path is not given as argument
        raise NotImplementedError('Provide path of folder containing safe files')

    logger.info('Processing files from {}'.format(safe_parent_path))
    safe_file_list = glob.glob(os.path.join(safe_parent_path, '*.SAFE'))
    make_patch(safe_file_list, args.mode)
