import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.l1 = nn.Linear(inputSize, hiddenSize)
        self.l2 = nn.Linear(hiddenSize, hiddenSize)
        self.l3 = nn.Linear(hiddenSize, hiddenSize)
        self.l4 = nn.Linear(hiddenSize, hiddenSize)
        self.l5 = nn.Linear(hiddenSize, hiddenSize)
        self.l8 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        x = F.leaky_relu( self.l1(x) )
        x = F.leaky_relu( self.l2(x) )
        x = F.leaky_relu( self.l3(x) )
        x = F.leaky_relu( self.l4(x) )
        x = F.leaky_relu( self.l5(x) )
        x = self.l8(x)
        return x#F.softmax(x, dim=2)

class BigNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.l1 = nn.Linear(inputSize, hiddenSize)
        self.l2 = nn.Linear(hiddenSize, hiddenSize)
        self.l3 = nn.Linear(hiddenSize, hiddenSize)
        self.l4 = nn.Linear(hiddenSize, hiddenSize)
        self.l5 = nn.Linear(hiddenSize, hiddenSize)
        self.l6 = nn.Linear(hiddenSize, hiddenSize)
        self.l7 = nn.Linear(hiddenSize, hiddenSize)
        self.l8 = nn.Linear(hiddenSize, outputSize)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout6 = nn.Dropout(0.5)
        self.dropout7 = nn.Dropout(0.5)


    def forward(self, x):
        x = F.leaky_relu( self.l1(x) )
        x = self.dropout1(x)
        x = F.leaky_relu( self.l2(x) )
        x = self.dropout2(x)
        x = F.leaky_relu( self.l3(x) )
        x = self.dropout3(x)
        x = F.leaky_relu( self.l4(x) )
        x = self.dropout4(x)
        x = F.leaky_relu( self.l5(x) )
        x = self.dropout5(x)
        x = F.leaky_relu( self.l6(x) )
        x = self.dropout6(x)
        x = F.leaky_relu( self.l7(x) )
        x = self.dropout7(x)
        x = self.l8(x)
        return x


class DeepNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.l1 = nn.Linear(inputSize, hiddenSize)
        self.l2 = nn.Linear(hiddenSize, hiddenSize)
        self.l3 = nn.Linear(hiddenSize, hiddenSize)
        self.l4 = nn.Linear(hiddenSize, hiddenSize)
        self.l5 = nn.Linear(hiddenSize, hiddenSize)
        self.l6 = nn.Linear(hiddenSize, hiddenSize)
        self.l7 = nn.Linear(hiddenSize, hiddenSize)
        self.l8 = nn.Linear(hiddenSize, hiddenSize)
        self.l9 = nn.Linear(hiddenSize, hiddenSize)
        self.l10 = nn.Linear(hiddenSize, hiddenSize)
        self.l11 = nn.Linear(hiddenSize, hiddenSize)
        self.l12 = nn.Linear(hiddenSize, hiddenSize)
        self.l13 = nn.Linear(hiddenSize, hiddenSize)
        self.l14 = nn.Linear(hiddenSize, hiddenSize)
        self.l15 = nn.Linear(hiddenSize, hiddenSize)
        self.l16 = nn.Linear(hiddenSize, outputSize)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout6 = nn.Dropout(0.5)
        self.dropout7 = nn.Dropout(0.5)
        self.dropout8 = nn.Dropout(0.5)
        self.dropout9 = nn.Dropout(0.5)
        self.dropout10 = nn.Dropout(0.5)
        self.dropout11 = nn.Dropout(0.5)
        self.dropout12 = nn.Dropout(0.5)
        self.dropout13 = nn.Dropout(0.5)
        self.dropout14 = nn.Dropout(0.5)
        self.dropout15 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.leaky_relu( self.l1(x) )
        x = self.dropout1(x)
        x = F.leaky_relu( self.l2(x) )
        x = self.dropout2(x)
        x = F.leaky_relu( self.l3(x) )
        x = self.dropout3(x)
        x = F.leaky_relu( self.l4(x) )
        x = self.dropout4(x)
        x = F.leaky_relu( self.l5(x) )
        x = self.dropout5(x)
        x = F.leaky_relu( self.l6(x) )
        x = self.dropout6(x)
        x = F.leaky_relu( self.l7(x) )
        x = self.dropout7(x)
        x = F.leaky_relu( self.l8(x) )
        x = self.dropout8(x)
        x = F.leaky_relu( self.l9(x) )
        x = self.dropout9(x)
        x = F.leaky_relu( self.l10(x) )
        x = self.dropout10(x)
        x = F.leaky_relu( self.l11(x) )
        x = self.dropout11(x)
        x = F.leaky_relu( self.l12(x) )
        x = self.dropout12(x)
        x = F.leaky_relu( self.l13(x) )
        x = self.dropout13(x)
        x = F.leaky_relu( self.l14(x) )
        x = self.dropout14(x)
        x = F.leaky_relu( self.l15(x) )
        x = self.dropout15(x)
        x = self.l16(x)
        return x
