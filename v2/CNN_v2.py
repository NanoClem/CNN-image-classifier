from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Dropout2d
from torch.nn.functional import dropout
from torch.optim import Adam

from config import config, device


class CNN_v2(Module):

    def __init__(self):
        """[summary]
        """
        super(CNN_v2, self).__init__()

        self.cnn_layers = Sequential(
            # First 2D convolution layer
            Conv2d(3, 10, kernel_size=3),
            MaxPool2d(kernel_size=2),
            ReLU(),
            # Second 2D convolution layer
            Conv2d(10, 20, kernel_size=3),
            Dropout2d(),
            MaxPool2d(kernel_size=2),
            ReLU()
        )

        self.linear_layers = Sequential(
            Linear(720, 1024),
            ReLU(),
            Linear(1024, 2)
        )


    def forward(self, x):
        """[summary]
        """
        x = self.cnn_layers(x)                  # conv layers
        x = x.view(x.size(0), -1)
        x = dropout(x, training=self.training)  # randomly zeroes some elements
        x = self.linear_layers(x)               # linear layers
        return x


# Defining the model and its components
model     = CNN_v2().to(device)                         # model (cuda device)
optimizer = Adam(model.parameters(), lr=config['lr'])   # optimizer
criterion = CrossEntropyLoss()                          # loss function


if __name__ == "__main__":
    print(model)
