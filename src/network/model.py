import logging
import os

from shutil import copyfile
import h5py
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import DataParallel

from network.unet import UNet
from utils.csv_logger import add_to_csv
from utils.metrics import calculate_accuracy, \
    calculate_confusion_matrix, conf_mat_class_accuracy,conf_mat_class_MIOU
from utils.script_utils import print_traininfo, print_valinfo, print_conf_matrix, print_class_accuracy
from utils.dir_paths import TRAIN_PATH
from utils.MFB import calculate_file_freq

logger = logging.getLogger(__name__)

N_CLASSES = 6

if os.uname()[1] != 'k': pass

if 'lms' in os.uname()[1]:
    DEPTH = 6
    START_FILTER = 64
elif os.uname()[1] == 'k':
    DEPTH = 5
    START_FILTER = 16
else:
    raise NotImplementedError('Check network size and batch size in model/unet.py')

class Model:
    def __init__(self, experiment,full=False):
        self.exp = experiment

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        filter_options = [8,16,32,64,64]
        depth_options = [5,5,6,5,6]
        if experiment.mode == 'data_gen':
            i = -1
        else:
            i = 0
        START_FILTER= filter_options[self.exp.stage + i]
        DEPTH= depth_options[self.exp.stage + i]
        if full:
            DEPTH= 6
            START_FILTER= 64

        logger.info("Exp name = {}".format(self.exp.name))
        logger.info("Depth used in stage {} = {}".format(self.exp.stage,DEPTH))
        logger.info("Filters used in stage {} = {}".format(self.exp.stage,START_FILTER))
        self.network = UNet(num_classes=N_CLASSES, in_channels=13,depth=DEPTH,
                 start_filts=START_FILTER ).to(self.device)

        self.read_band_stats()
        if experiment.mode == 'train':

            # To train in multiple GPUs
            # self.network = DataParallel(self.network, device_ids=[0, 1,2,3,])

            self.optim = torch.optim.Adam(self.network.parameters(), lr=self.exp.lr, weight_decay=0.00001)
            self.epoch = 0
            self.metrics = {'train': [], 'test': []}
            self.val_acc = []

            # For early stopping
            self.patience_counter = 0
            # self.patience =25
            self.patience =50
            self.best_MIOU = 0.
        else:
            trained_model = self.exp.get_trained_model_info()
            self.network.load_state_dict(trained_model['model_state_dict'])
            self.metrics = {'predict': []}
            if experiment.mode == 'data_gen':
                self.stage_freq_data = []




        self.confusion_matrix = torch.zeros((6, 6),dtype=torch.long).to(self.device)
        # self.confusion_matrix_fm = torch.zeros((6, 6),dtype=torch.long)

    def forward_step(self, input_img):

        b2 = input_img [:,:,:,1]
        b3 = input_img [:,:,:,2]
        b11 = input_img [:,:,:,10]
        b4 = input_img [:,:,:,3]
        b8 = input_img [:,:,:,7]
        deno = b3+b11
        deno1 = b4+b8
        deno [deno == 0]=1
        deno1 [deno1 == 0]=1
        ndsi = (b3 -b11)/deno
        ndvi = (b8 -b4)/deno1

        b4[b4 == 0] = 1
        b2b4 = b2/b4


        input_img = self.standardize(input_img.to(self.device))
        # input_img = self.normalize(input_img.to(self.device))
        # input_img = input_img.to(self.device)
        input_img[:,:,:,6] = ndsi
        # input_img[:,:,:,0] = ndvi
        # input_img[:,:,:,5] = b2b4

        if os.uname()[1] != 'k':
            input_img = input_img.permute(0, 3, 1, 2).to(self.device)
        else:
            input_img = input_img.cpu().permute(0, 3, 1, 2).to(self.device)
        output = self.network(input_img)
        return output

    def valid_step(self, network_data, mode='test'):

        # forward pass
        input_img = network_data[0]
        output = self.forward_step(input_img)
        _, labels, filenames = network_data

        fmask = None
        if mode == 'train':
            labels = labels[:, :, :, 0]

        elif mode == 'data_gen':
            self.generate_train_data(filenames, self.encode_label(output, datagen=True))
            return 0

        elif mode == 'test':  # Fmask and labels are expected
            fmask = labels[:, :, :, 0].to(self.device)
            if labels.shape[-1] >= 2:
                labels = labels[:, :, :, 1]
            else:
                # if labels are not found. Fmask is taken as label
                logger.warning('True labels not found. Using fmask labels')
                labels = labels[:, :, :, 0]
                no_label = True
        elif mode == 'predict':
            self.generate_train_data(filenames,self.encode_label(output),datagen=False)
            return 0
        loss = self.get_loss(output, labels, mode,fmask)

        # if mode == 'predict':
        #     self.confusion_matrix_fm += calculate_confusion_matrix(self.encode_label(output),
        #                                                            fmask, mode)

        return loss

    def train_step(self, network_data):
        loss = self.valid_step(network_data, mode='train')
        self.backward_step(loss)
        return loss

    def get_loss(self, output, labels, mode,fmask=None):
        labels = labels.to(self.device)
        # w = torch.Tensor([0.01,0.16,0.06,0.30,.35,.12 ]).to(self.device)
        w = torch.from_numpy(self.exp.weights).float().to(self.device)
        if mode != 'train':
            w[0] = 0
        loss = F.cross_entropy(output, labels, w) / output.shape[0]
        acc,deno,fmask_acc = self.get_accuracy(output, labels, mode=mode,fmask=fmask)
        self.metrics[mode].append([loss.detach(), acc.detach(), 0,deno,fmask_acc])

        return loss

    # @profile
    def get_accuracy(self, output, true_label, mode='train',fmask=None):
        predicted_label = self.encode_label(output)
        acc,deno,fmask_acc = calculate_accuracy(predicted_label, true_label, mode,fmask)
        if mode != 'train':
            self.confusion_matrix += calculate_confusion_matrix(predicted_label, true_label, mode)


        return acc,deno,fmask_acc

    def backward_step(self, loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def refresh_stats(self):
        train_metrics = torch.tensor(self.metrics['train']).mean(dim=0)
        print_traininfo(self.epoch, *train_metrics[:3])
         # train_metrics[0] is avg loss
        valid_metrics = torch.tensor(self.metrics['test'])
        samples = valid_metrics[:,3].sum()
        valid_metrics1 = valid_metrics.sum(dim=0)/samples
        valid_metrics = valid_metrics.mean(dim=0)
        valid_metrics[1] = valid_metrics1[1] # Accuracy
        valid_metrics[3] = valid_metrics1[4] # FMask Accuracy


        self.metrics = {'train': [], 'test': []}
        class_acc, avg_acc = conf_mat_class_accuracy(self.confusion_matrix)
        class_MIOU, avg_MIOU = conf_mat_class_MIOU(self.confusion_matrix)

        valid_metrics[2] = avg_MIOU.item()
        self.val_acc.append(valid_metrics[:3])
        print_valinfo(self.epoch, *valid_metrics[:4])

        logger.info('Validation Average Accuracy {:.2%}'.format(avg_acc))



        print_conf_matrix(self.confusion_matrix)
        print_class_accuracy(class_acc)
        self.confusion_matrix = self.confusion_matrix *0

        self.epoch += 1
        self.save()
        return self.check_early_stop(avg_MIOU)

    @staticmethod
    def encode_label(out, datagen=False, threshold=0.4):
        softmax_out = F.softmax(out, dim=1)
        predicted_labels = torch.argmax(softmax_out, dim=1)
        if datagen:
            prob = torch.max(softmax_out, dim=1)[0]
            predicted_labels[prob < threshold] = 0
        return predicted_labels

    def save(self):
        save_path = os.path.join(self.exp.model_folder,
                                 'model_{}.pth'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, save_path)

    def save_best_model(self):
        logger.info(self.val_acc)
        val_metric = torch.cat(self.val_acc, ).reshape(-1, 3)
        best_epoch = torch.argmax(val_metric, dim=0)[2].item()  # Best MIOU is taken
        logger.info('Best Epoch at {}'.format(best_epoch + 1))
        save_path_src = os.path.join(self.exp.model_folder,
                                     'model_{}.pth'.format(best_epoch + 1))

        save_path_dst = os.path.join(self.exp.model_folder,
                                     'model_best.pth')

        copyfile(save_path_src, save_path_dst)

    def generate_train_data(self, filenames, labels,datagen=True):
        """Saves the results of model prediction to file for training later or final
        prediction
        datagen is set to false to use the h5 later for joining and prediction"""
        for label, filename in zip(labels, filenames):
            label_np = label.cpu().numpy().astype(np.uint16)
            if not datagen:
                label_np = label_np[1:-1, 1:-1]
            with h5py.File(filename, "r") as hf:
                data = hf.get('data')[:]
                if datagen:
                    fmask = data[:, :, 13]
                    # when network is not confident use fmask TODO: Uncommet
                    # below line after experiment ppl9
                    # label_np[label_np == 0] = fmask[label_np == 0]
                    label_np[fmask == 0] = 0  # when fmask label = 0 i.e no data
                    target_index = 15

                    new_label_freq = calculate_file_freq(label_np)
                    self.save_stats(filename,new_label_freq)

                else:
                    target_index = 14
            if data.shape[-1] < (target_index+1):
                new_shape = list(data.shape)
                new_shape[-1] = (target_index+1) - data.shape[-1]

                data = np.append(data, np.zeros(new_shape,dtype=np.uint16), axis=-1)

            data[:, :, target_index] = label_np

            # data = np.append(data, label_np[:,:,None], axis=-1)
            os.remove(filename)
            with h5py.File(filename, "w") as hf:
                hf.create_dataset('data', data=data.astype(np.uint16), )

    def read_band_stats(self):
        band_csv = os.path.join(TRAIN_PATH, 'band_stats.csv')
        if os.path.exists(band_csv):
            with open(band_csv, "r") as f1:
                last_line = f1.readlines()[-1]
            if 'COMPLETE' in last_line:
                band_stats_data = [float(i) for i in last_line.split(',')[1:]]
                min_ = torch.tensor(band_stats_data[26:39]).reshape(1, 1, -1).to(self.device)

                self.band_stats = {
                    'mean': torch.tensor(band_stats_data[0:13]).reshape(1, 1, -1).to(self.device),
                    'inv_std': torch.tensor(1 / np.array(band_stats_data[13:26]).reshape(1, 1, -1)).float().to(
                        self.device),
                    'max': torch.tensor(band_stats_data[39:52]).reshape(1, 1, -1).to(
                        self.device),
                }
                self.band_stats['inv_range'] = 1 / (self.band_stats['max'] - min_).float().to(
                    self.device)

            else:
                raise ValueError('Complete stats not found in {}'.format(band_csv))

        else:
            raise FileNotFoundError('{} not found'.format(band_csv))

    def normalize(self, spectral_image_tensor):
        spectral_image_tensor = spectral_image_tensor
        spectral_image_tensor = (self.band_stats['max'] - spectral_image_tensor) * self.band_stats['inv_range']

        return spectral_image_tensor

    def standardize(self, spectral_image_tensor):
        spectral_image_tensor = spectral_image_tensor
        spectral_image_tensor = (spectral_image_tensor - self.band_stats['mean']) * self.band_stats['inv_std']

        return spectral_image_tensor



    def check_early_stop(self, epoch_MIOU):
        if epoch_MIOU>=self.best_MIOU+.01:
            self.best_MIOU = epoch_MIOU
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter == self.patience:
                print('Best MIOU {}'.format(self.best_MIOU))
                logger.info('Early stopping triggered at  epoch {}'.format(self.epoch))
                return True
        return False

    def save_stats(self,filename,new_label_freq):
        row = [os.path.basename(filename)]
        row.extend(new_label_freq.tolist())
        row = ','.join([str(i) for i in row])
        self.stage_freq_data.append(row)
    def write_stage_stats(self):
        label_stats_file = os.path.join(TRAIN_PATH,
                                             'label_stats_stage{}.csv'.format(
                                                 self.exp.config['stage']))
        with open(label_stats_file, 'w') as f:
            f.write("FILENAME,NONE_F,CLEAR_F,CLOUD_F,SHADOW_F,ICE_F,WATER_F\n")
            for i in self.stage_freq_data:
                f.write('{}\n'.format(i))
