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


def print_valinfo(epoch, avg_loss, avg_acc,avg_miou,avg_fmask_acc):
    """Print the epoch wise validation stats in the log file. """
    if epoch == 0:
        msg = 'Epoch {} Validation: \tLoss {:.4f} [{:+.4f}] \t\tAccuracy {:6.2%} [{:+.2f}] ' \
              '\t\tMIOU {:6.2%} [{:+.2f}] \t\tFMask_Acc {:6.2%} [{:+.2f}]'.format(epoch + 1,avg_loss, 0,
                                                     avg_acc, 0,avg_miou,0,avg_fmask_acc,0)
    else:
        msg = 'Epoch {} Validation: \tLoss {:.4f} [{:+.4f}] \t\tAccuracy {:6.2%} [{:+.2f}] ' \
              '\t\tMIOU {:6.2%} [{:+.2f}]\t\tFMask_Acc {:6.2%} [{:+.2f}]'.format(
            epoch + 1,
            avg_loss,
            avg_loss - PREV[3],
            avg_acc, 100 * (avg_acc -PREV[4]),
            avg_miou, 100*(avg_miou-PREV[5]),
            avg_fmask_acc, 100*(avg_fmask_acc-PREV[6]))

    PREV[3:7] = avg_loss, avg_acc,avg_miou,avg_fmask_acc


    logger.info(msg)


def print_traininfo(epoch, avg_loss, avg_acc,avg_miou):
    """Print the epoch wise training stats in the log file """

    if epoch == 0:
        msg = 'Epoch {} Training  : \tLoss {:.4f} [{:+.4f}] \t\tAccuracy {:6.2%} [{:+.2f}] ' \
              '\t\tMIOU {:6.2%} [{:+.2f}]'.format(epoch + 1,avg_loss, 0,
                                                     avg_acc, 0,avg_miou,0)
    else:
        msg = 'Epoch {} Training  : \tLoss {:.4f} [{:+.4f}] \t\tAccuracy {:6.2%} [{:+.2f}] ' \
              '\t\tMIOU {:6.2%} [{:+.2f}]'.format(
            epoch + 1,
            avg_loss,
            avg_loss - PREV[0],
            avg_acc, 100 * (avg_acc -PREV[1]),
            avg_miou, 100*(avg_miou-PREV[2]))

    PREV[0:3] = avg_loss, avg_acc,avg_miou


    logger.info(msg)

def print_class_accuracy(class_acc):
    """Prints the Class accuracies into the log file"""

    if len(PREV) == 7:
        msg = 'Val Class Accuracy: \t{} {:6.2%} [{:+.2f}] \t{} {:6.2%} [{:+.2f}]' \
              '\t{} {:6.2%} [{:+.2f}] \t{} {:6.2%} [{:+.2f}] \t{} {:6.2%} [{:+.2f}] \t{} {:6.2%} [{:+.2f}] ' .format(
                                                     LABELS[0],class_acc[0], 0,LABELS[1],class_acc[1], 0,LABELS[2],class_acc[2], 0,LABELS[3],class_acc[3], 0,LABELS[4],class_acc[4], 0,
                                                     LABELS[5],class_acc[5], 0,)
    else:
        msg = 'Val Class Accuracy: \t{} {:6.2%} [{:+.2f}] \t{} {:6.2%} [{:+.2f}]' \
              '\t{} {:6.2%} [{:+.2f}] \t{} {:6.2%} [{:+.2f}] \t{} {:6.2%} [{:+.2f}] \t{} {:6.2%} [{:+.2f}] ' .format(LABELS[0],
                                                     class_acc[0], 100 * (class_acc[0] -PREV[7]),LABELS[1],class_acc[1], 100 * (class_acc[1] -PREV[8]),LABELS[2],class_acc[2],
                                                      100 * (class_acc[2] -PREV[9]),LABELS[3],class_acc[3], 100 * (class_acc[3] -PREV[10]),LABELS[4],class_acc[4], 100 * (class_acc[4] -PREV[11]),
                                                     LABELS[5],class_acc[5], 100 * (class_acc[5] -PREV[12]),)

    PREV[7:13] = class_acc
    

    logger.info(msg)



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
