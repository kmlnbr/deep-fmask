### Install instruction
- conda install rasterio tqdm
- conda install matplotlib scikit-learn
- conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
- conda install -c conda-forge opencv h5py
- Install the Fmask4 from [GitHub]( https://github.com/GERSL/Fmask)
- Install Sen2Cor 2.8 from [ESA STEP Website](http://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor_v2-8/).

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



### Training
- In order to train an experiment called `exp1`

```console
./pipeline.sh exp1
```


### Test
- In order to test a trained model in exp1
```console
python predict.py -e exp1_stage3 -p /home/nambiar/Downloads/DATA_NO_BACKUP/new_s2/PREDICT/P1
```
