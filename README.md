# Deep F-Mask for Cloud, Shadow and Snow detection in Sentinel 2 Imagery
This repository contains the code based on the methods described in the paper 
'Self-Trained Cloud, Cloud Shadow and Snow Masking Model for Sentinel-2 Images in 
Polar Regions'.

Paper Link: 

Contact: kamal.nambiar@fau.de

---
### Install instructions

The code in this repository was developed for Python 3.6 on CentOS 8. 

- We recommend using an Anaconda environment for managing the dependencies.
You may have to modify the cudatoolkit version in the *environment.yml* depending on the
  GPU model.
```console 
conda env create --name DFMask --file environment.yml
``` 
- [Optional: For Training/Comparison] Install the Fmask4 from the developer's [GitHub repository]( https://github.com/GERSL/Fmask).
- [Optional: For Comparison] Install Sen2Cor 2.8 from [ESA STEP Website](http://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor_v2-8/).

### Training Dataset Setup
- Go to the exp_data directory
- Install the .SAFE files provided in the *train_filename.txt*, *test_filename.txt* 
  and *validation_filename.txt*  from [Copernicus Datahub](https://scihub.copernicus.eu/dhus/#/home)
and save them in TRAIN, TEST and VALIDATION directories respectively. 
- For each file run the Fmask 4 algorithm and save the corresponding Fmask output in 
  the IMG_DATA directory of each SAFE file with a *FMASK.tif postfix.
- Generate the training and validation data (h5 files) using the below command. 
```console
python make_network_data.py --mode train

```

- Generate the validation data (h5 files) using the below command.
```console
python make_network_data.py --mode test
```


---
### Training
#### Supervised Mode
- To train the mode in a fully supervised mode using only the F-Mask labels:
```console
python train.py supervised_exp --full
```
- For more details on the training script arguments:
```console
python train.py --help
```
#### Self- Training Mode
- The complete training pipeline 
  consists of iterative training and label generation. The pipeline.sh implements a 4 stage training pipeline.
  The experiment name is the prefix used for each model. For example, 
  if we set the experiment name as exp1, the model name for the stage 0 will be 
  exp1_stage0.
- In order to train an experiment called `exp1`:

```console
./pipeline.sh exp1
```
- The trained model, as well as the logs of the training, will be saved the
*exp_data* directory.
- **NOTE**: If we were to reuse a previously used experiment name, the old files will be 
overwritten.
### Evaluation
- In order to test a trained model from the last stage (stage 3) of `exp1`: 
```console
python predict.py -e exp1_stage3 -p /path/containing/SAFE/directories/
```
- The predictions are stored as a tif file 
  in the IMG_DATA directory of the respective Sentinel 2 .SAFE directory.
- The classes are labeled as follows:

| Label | Class |
|:---:|:---:|
| 0 | No-data |
| 1 | Clear Land |
| 2 | Cloud |
| 3 | Cloud Shadow |
| 4 | Snow |
| 5 | Water |
---
- The `predict.py` can be used to compare the predictions with ground truth 
labels (if available). The ground truth label file must be stored with 
*LABELS.tif postfix in the IMG_DATA directory of each SAFE file, and the classes must 
  be labeled as shown in the table above.

### Directory Structure
```bash
├── exp_data
│   ├── exp1_stage3
│   │   ├── log
│   │   └── model
│   ├── TEST
│   ├── TRAIN
│   ├── TRAIN_H5
│   ├── VALIDATION
│   └── VALIDATION_H5
└── src
    ├── label_generation.py
    ├── dataset
    │   ├── __init__.py
    │   ├── patch_dataset.py
    │   └── transforms.py
    ├── __init__.py
    ├── make_network_data.py
    ├── network
    │   ├── __init__.py
    │   ├── model.py
    │   └── unet.py
    ├── predict.py
    ├── train.py
    └── utils
        ├── csv_logger.py
        ├── dataset_stats.py
        ├── dir_paths.py
        ├── experiment.py
        ├── __init__.py
        ├── join_predictions.py
        ├── split_scene.py
        ├── metrics.py
        ├── MFB.py
        └── script_utils.py
```

