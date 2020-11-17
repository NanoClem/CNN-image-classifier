# %%
import os
import numpy as np
import pandas as pd

# read and display images
import matplotlib.pyplot as plt
import matplotlib.image as matImg

# create a validation set
from sklearn.model_selection import train_test_split

# evaluation of the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch librairies and modules
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
# %%

if __name__ == "__main__":

    # =========================================================================
    #       DIRECTORIES
    # =========================================================================
    dataDir  = os.path.join(os.getcwd(), 'dataset')
    trainDir = os.path.join(dataDir, 'train')
    testDir  = os.path.join(dataDir, 'test')

    # =========================================================================
    #       LOADING DATASET
    # =========================================================================
    train = pd.read_csv(os.path.join(dataDir, 'train.csv'))
    test  = pd.read_csv(os.path.join(dataDir, 'test.csv'))

    # =========================================================================
    #       PREPARING IMAGES TRANSFORMATIONS
    # =========================================================================
    mean, std = [0.5], [0.5]    # because we are working with grayscale img
    trainTransform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    validTransform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean, std)])
    testTransform  = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean, std)])
                                   
# %%
    # =========================================================================
    #       SPLITING DATASET : training and validation
    # =========================================================================
    ## TMP: reducing train size for code testing
    #train.drop(train.tail(59000).index, inplace=True)
    train, validation = train_test_split(train, stratify=train.label, test_size=0.1)  # 10% for validation

    from AparelDataset import AparelDataset
    trainData = AparelDataset(train, trainDir, trainTransform)
    validData = AparelDataset(validation, trainDir, validTransform)
    testData  = AparelDataset(test, testDir, testTransform)

# %%
    # =========================================================================
    #       HYPER PARAMETERS / CUDA / DATALOADER
    # =========================================================================
    # hyper parameters
    epochs  = 25
    classes = 10
    batchSize = 20
    workers = 0

    # load each dataset
    trainLoader = DataLoader(dataset=trainData, batch_size=batchSize, shuffle=True, num_workers=workers)
    validLoader = DataLoader(dataset=validData, batch_size=batchSize, shuffle=False, num_workers=workers)
    testLoader  = DataLoader(dataset=testData, batch_size=batchSize, shuffle=False, num_workers=workers)

    trainimages, trainlabels = next(iter(trainLoader))
# %%
    # =========================================================================
    #       INIT TRAINING
    # =========================================================================
    # importing CNN model
    from CNN_v1 import model, optimizer, criterion
    from train import training
    
    train_losses, val_losses = [], []

    for epoch in tqdm(range(epochs)):
        training(epoch, model, optimizer, criterion, 
                 trainLoader, validLoader, train_losses, val_losses)
# %%