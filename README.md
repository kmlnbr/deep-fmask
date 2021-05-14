


# Deep F-Mask for Cloud, Shadow and Snow detection in Sentinel 2 Imagery
This repository contains the code based on the methods described in the paper 'Self-Trained Cloud, Cloud Shadow and Snow Masking Model
for Sentinel-2 Images in Polar Regions'.

Paper Link: 

Contact: 

---
### Install instruction

The code in this repository was developed for Python 3.6 on CentOS. 

- We recommend using an Anaconda environment for managing the dependencies.
```console 
conda env create --name DFMask --file environment.yml
``` 
- [Optional: For Training/Comparision] Install the Fmask4 from the developer's [GitHub repository]( https://github.com/GERSL/Fmask).
- [Optional: For Comparision] Install Sen2Cor 2.8 from [ESA STEP Website](http://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor_v2-8/).

### Dataset Setup
- Go to the exp_data directory
- Install the .SAFE files provided in the *train_filename.txt*, *test_filename.txt* 
  and *validation_filename.txt*  from [Copernicus Datahub](https://scihub.copernicus.eu/dhus/#/home)
and save them in TRAIN, TEST and VALIDATION directories respectively. 
- For each file run the Fmask 4 algorithm and save the corresponding Fmask output in 
  the IMG_DATA directory of each SAFE file.
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
- In order to train an experiment called `exp1`

```console
./pipeline.sh exp1
```


### Evaluation
- In order to test a trained model in exp1
```console
python predict.py -e exp1_stage3 -p /path/containing/SAFE/files/
```

---
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
    ├── data_gen.py
    ├── dataset
    │   ├── __init__.py
    │   ├── patch_dataset.py
    │   └── transforms.py
    ├── __init__.py
    ├── main.py
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
        ├── join_h5_pred.py
        ├── make_network_data_pred.py
        ├── metrics.py
        ├── MFB.py
        ├── script_utils.py
        └── visualizer.py
```

