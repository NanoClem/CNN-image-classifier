import os
import json
import torch


# reading config file
path = os.path.join(os.getcwd(), 'conf.json')
with open(path) as configFile:
    config = json.load(configFile)

# checking for cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    print(config)