import logging
import os
from shutil import copyfile

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import DataParallel

from dataset.network_input import get_inp_func, get_inp_channels

from network.unet import UNet
from utils.MFB import calculate_file_freq
from utils.csv_logger import print_val_csv_metrics, make_overall_statistics_csv
from utils.dir_paths import TRAIN_PATH
from utils.metrics import calculate_accuracy, calculate_confusion_matrix

from utils.metrics import Metrics

logger = logging.getLogger(__name__)

####################################################################################
# SELF-TRAINING OPTIONS
# The number of start filters and the depth of the network at each stage is selected
# using the two lists given below:

filter_options = [16, 32, 24, 32]
depth_options = [5, 5, 6, 6]


####################################################################################


def focal_loss(net_output, target):
    gamma = 2
    ce_loss = F.cross_entropy(net_output, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** gamma * ce_loss)
    return torch.mean(focal_loss)


class Model:
    def __init__(self, experiment, full=False, dropout=True, gpu_id=None):

        self.exp = experiment

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.inp_func = get_inp_func(self.exp.inp_mode)
        n_inp_channels = get_inp_channels(self.exp.inp_mode)



        start_filters = filter_options[self.exp.stage]
        depth = depth_options[self.exp.stage]
        if full:
            depth = depth_options[-1]
            start_filters = filter_options[-1]

        logger.info("Depth used in stage {} = {}".format(self.exp.stage, depth))
        logger.info(
            "Filters used in stage {} = {}".format(self.exp.stage, start_filters))

        # Initialize model and set the gpu ids used by the model
        self.network = UNet(num_classes=6, in_channels=n_inp_channels,
                            depth=depth, start_filts=start_filters,
                            dropout=dropout)
        self.network = DataParallel(self.network, device_ids=gpu_id)
        if experiment.mode == 'train':
            self.epoch = 0
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.exp.lr,
                                              weight_decay=0.00001)

            self.metrics = Metrics(self.device)

            # For early stopping
            self.patience_counter = 0
            self.patience = 10

            self.best_mIoU_moving_avg = 0.
        else:
            # Load the weights of the trained model
            trained_model = self.exp.get_trained_model_info()
            self.network.load_state_dict(trained_model['model_state_dict'])

            if experiment.mode == 'label_gen':
                self.stage_freq_data = []

    def forward_step(self, input_img):
        """
        The forward propagation step of the network.
        """
        input_img = self.inp_func(input_img)

        output = self.network(input_img.to(self.device))
        return output

    def valid_step(self, network_data, mode='test'):
        """
        The validation step where the forward pass and the loss function is called.

        The labels used for loss calculation depend on the mode of operation. When
        mode is 'train', training labels such as F-Mask is used. When mode is
        'valid' or 'test', the true labels are used for loss calculation.
        """

        # forward pass
        input_img = network_data[0]
        output = self.forward_step(input_img)
        _, labels, filenames = network_data

        fmask = None
        if mode == 'train':
            labels = labels[:, 0, :, :]

        elif mode == 'label_gen':
            self.generate_train_data(filenames,
                                     self.encode_label(output, label_gen=True))
            return 0

        elif mode == 'test':  # Fmask and labels are expected
            fmask = labels[:, 0, :, :].to(self.device)
            if labels.shape[-1] >= 2:
                labels = labels[:, 1, :, :]
            else:
                # if labels are not found. Fmask is taken as label
                logger.warning('True labels not found. Using fmask labels')
                labels = labels[:, 0, :, :]
        elif mode == 'predict':
            self.generate_train_data(filenames, self.encode_label(output),
                                     label_gen=False)
            return 0
        loss = self.get_loss(output, labels, mode, fmask)

        return loss

    def train_step(self, network_data):
        """
        Training step comprising of the forward and backward propagation to be
        executed at each step of the training epoch.
        """
        loss = self.valid_step(network_data, mode='train')
        self.backward_step(loss)
        return loss

    def get_loss(self, output, labels, mode, fmask=None):
        """
        Calculate the loss using a weighted cross entropy function. The weights of
        the training is obtained from the Median Frequency Balancing step. For
        validation and testing, uniform weights for all classes are used.

        """
        labels = labels.to(self.device)

        w = torch.from_numpy(self.exp.weights).float().to(self.device)

        loss = F.cross_entropy(output, labels, w)

        predicted_label = self.encode_label(output)
        if mode != 'train':
            self.metrics.val_confusion_matrix += calculate_confusion_matrix(
                predicted_label,
                labels, mode)
            acc = 0
        else:
            acc = calculate_accuracy(predicted_label, labels, mode).detach()

        self.metrics.add_step_info(mode, loss.detach(), acc)
        return loss

    def backward_step(self, loss):
        """
        The backward propagation step for training the network.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def refresh_stats(self):
        """
        At the end of each epoch, the metrics are aggregated and their values are
        displayed on the screen and stored in the log files. The metrics are then
        reset before the start of the next epoch.
        """
        train_metrics, valid_metrics, class_metrics_dict = self.metrics.aggregate_metrics(
            self.epoch)
        make_overall_statistics_csv(train_metrics, valid_metrics, class_metrics_dict,
                                    self.epoch, self.exp.log_path)

        self.metrics.reset_metrics()
        self.epoch += 1
        self.save()

        return self.check_early_stop()

    @staticmethod
    def encode_label(out, label_gen=False, threshold=0.4):
        """
        Converts the softmax input into pixelwise class predictions.

        In case of new data generation (i.e. label_gen=True), prediction are stored
        only pixel positions where the prediction confidence is greater than the
        threshold, otherwise it is set to 0.
        """

        softmax_out = F.softmax(out, dim=1)
        predicted_labels = torch.argmax(softmax_out, dim=1)
        if label_gen:
            prob = torch.max(softmax_out, dim=1)[0]
            predicted_labels[prob < threshold] = 0
        return predicted_labels

    def save(self):
        """
        Save the model after each training epoch.
        """
        save_path = os.path.join(self.exp.model_folder,
                                 'model_{}.pth'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.network.state_dict(),
            'stage': self.exp.stage,
            'full': self.exp.full,
            'inp_mode': self.inp_func.__name__
        }, save_path)

    def save_best_model(self):
        """
        Selects the best model based on the mIoU over the validation dataset.
        A copy of the model file is made and this new file is renamed as model_best.pth
        """

        best_epoch = np.argmax(self.metrics.val_mIoU_history)
        best_mIOU = self.metrics.val_mIoU_history[best_epoch]

        logger.info('Best mIOU {:.3%} at Epoch {}'.format(best_mIOU, best_epoch + 1))
        print('Best mIOU {:.3%} at Epoch {}'.format(best_mIOU, best_epoch + 1))

        save_path_src = os.path.join(self.exp.model_folder,
                                     'model_{}.pth'.format(best_epoch + 1))

        save_path_dst = os.path.join(self.exp.model_folder,
                                     'model_best.pth')

        copyfile(save_path_src, save_path_dst)
        print_val_csv_metrics(best_epoch + 1, self.exp.log_path)

    def generate_train_data(self, filenames, labels, label_gen=True):
        """
        Saves the results of model prediction to file for training later or
        final prediction label_gen is set to false to use the h5 later for joining
        and prediction.
        """
        for label, filename in zip(labels, filenames):
            label_np = label.cpu().numpy().astype(np.uint16)
            label_np = label_np[1:-1, 1:-1]
            with h5py.File(filename, "r") as hf:
                data = hf.get('data')[:]
                if label_gen:
                    fmask = data[:, :, 13]
                    # label_np[label_np == 0] = fmask[label_np == 0]
                    label_np[fmask == 0] = 0  # when fmask label = 0 i.e no data
                    target_index = 15

                    new_label_freq = calculate_file_freq(label_np)
                    self.save_stats(filename, new_label_freq)

                else:
                    target_index = 14
            if data.shape[-1] < (target_index + 1):
                new_shape = list(data.shape)
                new_shape[-1] = (target_index + 1) - data.shape[-1]

                data = np.append(data, np.zeros(new_shape, dtype=np.uint16), axis=-1)

            data[:, :, target_index] = label_np

            # data = np.append(data, label_np[:,:,None], axis=-1)
            os.remove(filename)
            with h5py.File(filename, "w") as hf:
                hf.create_dataset('data', data=data.astype(np.uint16), )

    def check_early_stop(self):
        if self.epoch <= self.patience: return False

        mIoU_moving_avg = np.mean(self.metrics.val_mIoU_history[-self.patience:])
        logger.info('Current mIoU moving average {:.4}'.format(mIoU_moving_avg))
        logger.info(
            'Best mIoU moving average {:.4}'.format(self.best_mIoU_moving_avg))

        if mIoU_moving_avg >= self.best_mIoU_moving_avg:
            self.best_mIoU_moving_avg = mIoU_moving_avg
            self.patience_counter = 0
        else:

            self.patience_counter += 1
            if self.patience_counter == self.patience:
                logger.info(
                    'Early stopping triggered at  epoch {}'.format(self.epoch))

                return True

        return False

    def save_stats(self, filename, new_label_freq):
        row = [os.path.basename(filename)]
        row.extend(new_label_freq.tolist())
        row = ','.join([str(i) for i in row])
        self.stage_freq_data.append(row)

    def write_stage_stats(self):
        label_stats_file = os.path.join(TRAIN_PATH,
                                        'label_stats_stage{}.csv'.format(
                                            self.exp.config['stage']+1))
        with open(label_stats_file, 'w') as f:
            f.write("FILENAME,NONE_F,CLEAR_F,CLOUD_F,SHADOW_F,ICE_F,WATER_F\n")
            for i in self.stage_freq_data:
                f.write('{}\n'.format(i))
