import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam



class CNN_v1(Module):

    def __init__(self):
        """[summary]
        """
        super(CNN_v1, self).__init__()

        self.cnn_layers = Sequential(
            # First 2D convolution layer
            Conv2d(3, 10, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(10),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Second 2D convolution layer
            Conv2d(10, 10, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(10),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layers = Sequential(
            Linear(4*7*7, 10)
        )


    def forward(self, x):
        """[summary]
        """
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


# Defining the model and its components
model     = CNN_v1()                            # model
optimizer = Adam(model.parameters(), lr=0.07)   # optimizer
criterion = CrossEntropyLoss()                  # loss function


if __name__ == "__main__":
    print(model)
