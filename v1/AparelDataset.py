import os
import numpy as np
#from PIL import Image
import matplotlib.image as matImg
from torch.utils.data import Dataset


class AparelDataset(Dataset):
    """
    """

    def __init__(self, data, path, transform=None):
        """
        """
        #super.__init__()
        self.data = data.values
        self.path = path
        self.transform = transform


    def __len__(self):
        """
        """
        return len(self.data)


    def __getitem__(self, id):
        """
        """
        # Select sample img in dataset folder
        imgName, label = self.data[id]
        imgPath = os.path.join(self.path, str(imgName) + '.png')
        img = matImg.imread(imgPath)
        img = img[:,:,:3]   # removing alpha channel

        # Applying transform
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
