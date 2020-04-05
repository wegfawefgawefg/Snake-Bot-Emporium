import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
