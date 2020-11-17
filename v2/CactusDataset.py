import os
import matplotlib.image as matImg
from torch.utils.data import Dataset


class CactusDataset(Dataset):
    """
    """

    def __init__(self, data, path, transform=None):
        """
        """
        super().__init__()
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
        imgPath = os.path.join(self.path, str(imgName))
        img = matImg.imread(imgPath)

        # Applying transform
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label
