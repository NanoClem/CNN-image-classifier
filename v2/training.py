# %%
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# read and display images
import matplotlib.pyplot as plt

# create a validation set
from sklearn.model_selection import train_test_split

# PyTorch librairies and modules
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# %%
if __name__ == "__main__":

    # =========================================================================
    #       DIRECTORIES
    # =========================================================================
    modelDir = os.path.join(os.path.dirname(os.getcwd()), 'saved_models')
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
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    trainTransform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean, std)])
    validTransform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean, std)])
    testTransform  = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean, std)])
                                   
# %%
    # =========================================================================
    #       SPLITING DATASET : training and validation
    # =========================================================================
    train, validation = train_test_split(train, stratify=train.label, test_size=0.1)  # 10% for validation

    from CactusDataset import CactusDataset
    trainData = CactusDataset(train, trainDir, trainTransform)
    validData = CactusDataset(validation, trainDir, validTransform)
    testData  = CactusDataset(test, testDir, testTransform)

# %%
    # =========================================================================
    #       HYPER PARAMETERS / CUDA / DATALOADER
    # =========================================================================
    from config import config

    # hyper parameters
    epochs  = config['epochs']
    classes = config['classes']
    batchSize = config['batchSize']
    workers = config['workers']

    # load each dataset
    trainLoader = DataLoader(dataset=trainData, batch_size=batchSize, shuffle=True, num_workers=workers)
    validLoader = DataLoader(dataset=validData, batch_size=batchSize, shuffle=False, num_workers=workers)
    testLoader  = DataLoader(dataset=testData, batch_size=batchSize, shuffle=False, num_workers=workers)

    #trainimages, trainlabels = next(iter(trainLoader))
# %%
    # =========================================================================
    #       INIT TRAINING
    # =========================================================================
    from CNN_v2 import model, optimizer, criterion
    from utils import training
    
    train_losses, val_losses = [], []

    for epoch in tqdm(range(epochs)):
        training(epoch, model, optimizer, criterion, 
                 trainLoader, validLoader, train_losses, val_losses)

# %%
    # =========================================================================
    #       EVALUATING AND SAVING THE MODEL
    # =========================================================================
    from utils import evaluate
    SAVE_MODE = False

    accuracy = evaluate(model, validLoader)
    print('ACCURACY : {}'.format(accuracy))

    if SAVE_MODE:
        torch.save(model.state_dict(), os.path.join(modelDir, 'cactusModel.ckpt'))

# %%
    # =========================================================================
    #       TESTING MODEL LOADING
    # =========================================================================
    from config import device
    from CNN_v2 import CNN_v2
    from utils import evaluate

    # Loading model
    modelPath = os.path.join(modelDir, 'cactusModel.ckpt')
    model = CNN_v2()
    model.load_state_dict(torch.load(modelPath))
    model.to(device)

    # testing
    accuracy = evaluate(model, validLoader)
    print('ACCURACY : {}'.format(accuracy))