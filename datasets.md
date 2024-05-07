# Current supported datasets
The currently supported datasets are

1. CIFAR-10
2. CIFAR-100
3. APTOS
4. EyePacs
5. HAM10000
6. RSNAPneumonia
7. CheXpert

and the dataset classes are available at [dataset.py](dataset.py).

The dataset directory looks as follows

    dataset/
    ├── cifar
    ├── 224_data
    ├── ISIC
    ├── CheXpert-v1.0_224
    ├── rsna18
    


1. CIFAR-10

For CIFAR-10, automatic dataset download is enabled and it gets downloaded into [dataset](./dataset). The split file is [cifar1098_idx.npy](./dataset/cifar/cifar1098_idxs.npy). After downloading the dataset, the data directory looks like this


    cifar/
    ├── cifar-10-batches-py
    ├── cifar-10-python.tar.gz
    ├── cifar1098_idxs.npy


2. CIFAR-100

For CIFAR-100, automatic dataset download is enabled and it gets downloaded into [dataset](./dataset). The split file is [cifar1098_idx.npy](./dataset/cifar/cifar1098_idxs.npy). After downloading the dataset, the data directory looks like this


    cifar/
    ├── cifar-100-python
    ├── cifar-100-python.tar.gz
    ├── cifar1098_idxs.npy

3. APTOS

Download the APTOS dataset from [here](https://www.kaggle.com/competitions/aptos2019-blindness-detection). The split is available in [here]().

    224_data/
    ├── DR/
        ├── aptos/
            ├── 0/
            ├── 1/
            ├── 2/
            ├── 3/
            ├── 4/
            ├── aptos_dataset_splits.csv

4. EyePACS
Download the APTOS dataset from [here](https://www.kaggle.com/competitions/aptos2019-blindness-detection). The split is available in [here]().

    224_data/
    ├── DR/
        ├── eyepacs
            ├── 0/
            ├── 1/
            ├── 2/
            ├── 3/
            ├── 4/
            ├── eyepacs_dataset_splits.csv  


5.  HAM10000
Download the APTOS dataset from [here](https://challenge.isic-archive.com/data/#2018).

    ISIC/
        ├── ISIC2018_Task3_Training_Input/
        ├── ISIC2018_Task3_Validation_Input/
        ├── ISIC2018_Task3_Test_Input/
        ├── ISIC2018_Task3_Training_GroundTruth/
        ├── ISIC2018_Task3_Validation_GroundTruth/
        ├── ISIC2018_Task3_Test_GroundTruth/
        ├── ISIC2018_Task3_Training_LesionGroupings.csv

6. RSNAPneumonia




7. CheXpert
Download the CheXpert dataset from [here](https://stanfordmlgroup.github.io/competitions/chexpert/). Furthermore, resize the images to 224x224.

    CheXpert-v1.0_224/
        ├── csv/
            ├── test_labels.csv
            ├── test_labels_v2.csv
            ├── train.csv            
            ├── valid.csv           
        ├── test/
        ├── train/
        ├── valid/