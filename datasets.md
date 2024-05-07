# Current supported datasets
The currently supported datasets are

1. CIFAR-10
2. CIFAR-100
3. APTOS
4. EyePacs
5. HAM10000
6. RSNAPneumonia
7. FGVC-Aircrafts

and the dataset classes are available at [dataset.py](dataset.py).

The dataset directory looks as follows

    dataset/
    ├── cifar
    ├── 224_data
    ├── CheXpert-v1.0_224
    ├── ISIC
    ├── MIMIC
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

