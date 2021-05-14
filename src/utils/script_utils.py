import datetime
import os
import shutil
from time import time
import logging

import numpy as np

CURSOR_UP_ONE = '\x1b[1A'
CURSOR_DOWN_ONE = '\x1b[1B'
ERASE_LINE = '\x1b[2K'

PREV = []
LABELS = ['None', 'Rock', 'Cloud','Shadow','Ice','Water']
# Logging
logger = logging.getLogger(__name__)


def get_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def delete_folder_contents(folder_path):
    """
    Deletes the contents of a folder but retains the parent folder.
    """
    shutil.rmtree(folder_path)
    os.makedirs(folder_path)


class Timer:
    def __init__(self, start_now=True):
        self.time_taken = 0
        if start_now:
            self.start_time = time()
            self.pause_flag = False
        else:
            self.pause_flag = True

    def reset(self, start_now=True):
        self.__init__(start_now)

    def get_elapsed_time(self):
        if not self.pause_flag:
            self.time_taken += (time() - self.start_time)

        hours, rem = divmod(self.time_taken, 3600)
        minutes, seconds = divmod(rem, 60)
        return hours, minutes, seconds

    def get_elapsed_string(self):
        hours, minutes, seconds = self.get_elapsed_time()
        return '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)

    def pause(self):
        self.pause_flag = True
        self.time_taken += (time() - self.start_time)

    def continue_timer(self):
        self.pause_flag = False
        self.start_time = time()


class time_code:
    def __init__(self, timer):
        self.timer = timer

    def __enter__(self):
        self.timer.continue_timer()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.timer.pause()

def print_epoch_info(epoch, train_loss, train_acc, val_loss, val_acc, val_miou):
    """
    Print the epoch wise training and validation stats on the console.
    """
    if epoch == 0:
        print('      {: ^18}\t{: ^26}'.format('Train','Validation'))
        print('{:^6} {: ^6}  {: ^10}\t{: ^6}  {: ^10}  {: ^6}'.format('Epoch', 'Loss','Accuracy','Loss','Accuracy','mIoU'))

    msg = '{:^6} ' \
          '{:^6.4f}  {:^10.2%}\t' \
          '{:^6.4f}  {:^10.2%}  {:^6.2%}'.format(
        epoch + 1, train_loss, train_acc,val_loss, val_acc,val_miou)

    print(msg)


def print_conf_matrix(confusion_matrix):
    """Prints the confusion matrix into the log file"""
    confusion_matrix = confusion_matrix.cpu().numpy()
    conf_str=np.array2string(confusion_matrix.astype(int), separator=' ')
    conf_str = conf_str.replace(']',' ').replace('[',' ')

    logger.info('Confidence Matrix \n{}'.format(conf_str))


def set_logfile_path(log_path, mode='train'):
    log_filename = os.path.join(log_path, '{}_experiment_{}.log'.format(mode, get_timestamp()))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        filename=log_filename,
        filemode='w'

    )
