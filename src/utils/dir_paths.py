"""Contains the path of the training, validation and test files"""
import os

# Define path of EXP_DATA
PARENT_PATH = os.path.abspath(__file__).rsplit('/', 3)[0]
EXP_DATA_PATH = os.path.join(PARENT_PATH, 'exp_data')

TRAIN_SAFE_PATH = os.path.join(EXP_DATA_PATH, 'TRAIN')
TRAIN_PATH = os.path.join(EXP_DATA_PATH, 'TRAIN_H5')

VALID_SAFE_PATH = os.path.join(EXP_DATA_PATH, 'VALIDATION')
VALID_PATH = os.path.join(EXP_DATA_PATH, 'VALIDATION_H5')

PRED_SAFE_PATH = os.path.join(EXP_DATA_PATH, 'TEST')
