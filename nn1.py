'''
could really try all sorts of different architectures for this
including using unsupervised learning to predict future and feed that in
and also a holographic logarithmic memory
'''

'''
technically is this even reinforcement learning?
we arent periodically reinforcing it,
and we are not making it predict how much reward it will get
and it has no tryhard percentage really
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import tqdm
import snake as s
from torch.autograd import Variable
import random

class Net(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.l1 = nn.Linear(inputSize, hiddenSize)
        self.l2 = nn.Linear(hiddenSize, hiddenSize)
        self.l8 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        x = F.leaky_relu( self.l1(x) )
        x = F.leaky_relu( self.l2(x) )
        x = self.l8(x)
        return x

def scoreGameEnd(game):
    return game.numSteps + game.length

def train(net, criterion, optimizer, gameSize, epochs):
    net.train()

    for i in tqdm(range(epochs)):
        optimizer.zero_grad()

        game = s.Snake(gameSize)

        gameDuration = 0

        moves = []
        outputs = []
        while not game.gameOver:

            #   generate a move
            #   #   lol should we be regularizing the board?
            oneHot = torch.tensor(game.board, dtype=torch.float32).view(1, 1, gameSize * gameSize)
            output = net(oneHot)

            values, index = output.max(0)

            print(index)
            quit()
            game.setDir(index)
                        
            #   store for later
            moves.append(move)
            outputs.append(output)
        
        #   get end score of game
        score = scoreEndBoard(board, winner, computersPlayer)
        # gameDurationMultiplier = 1.0 - gameDuration / 10
        # gameDurationMultiplier = gameDurationMultiplier * 0.9
        dilutionFactor = 0.9
        totalDilutant = 1.0
        for i, move in reversed(list(enumerate(moves))):
            totalDilutant *= dilutionFactor
            output = outputs[i]
            target = output.clone().view(9)
            target[move] = score * totalDilutant
            target = target.view(1, 1, 9)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

def test(net, epochs):
    net.eval()

    numWins = 0
    numLosses = 0
    numTies = 0

    for i in tqdm(range(epochs)):
        player = 2
        computersPlayer = random.randint(1,2)

        board = np.zeros(shape = (3, 3))
        # board = np.random.randint(low = 0, high = 3, size = (3, 3))

        movesLeft = np.any(np.where(board == 0, 1, 0))
        winner = tt.getWinner(board)

        while(not winner and movesLeft):
            if player == computersPlayer:
                #   generate a move
                oneHot = oneHotTicTacToe(board, computersPlayer).view(1, 1, 18)
                output = net(oneHot)

                #   mask out invalid moves
                invalidMoves = np.where( board.flatten() > 0, True, False)
                maskedOutput = output.clone().view(9)
                maskedOutput[invalidMoves] = -10
                values, index = maskedOutput.max(0)

                #   apply the move
                move = index
                board = board.flatten()
                board[move] = computersPlayer
                board = board.reshape(3, 3)
                        
            else:   #   opponents turn
                empties = tt.listEmpties(board)
                randomMove = random.choice(empties)
                tt.applyMove(player, randomMove, board)
            player = tt.togglePlayer(player)

            movesLeft = np.any(np.where(board == 0, 1, 0))
            winner = tt.getWinner(board)
        
        if winner == computersPlayer:
            numWins += 1
        elif winner == tt.togglePlayer(computersPlayer):
            numLosses += 1
        else:   #   winner == False
            numTies += 1

    return numWins, numLosses, numTies


def main():
    boardSize = 8

    net = Net(boardSize * boardSize, 64, 4)
    # net = Net(18, 256, 9)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train(net=net, criterion=criterion, optimizer=optimizer, gameSize=boardSize, epochs=10000)
    # numWins, numLosses, numTies = test(net=net, epochs=1000)
    # print("wins, losses, ties:")
    # print(numWins)
    # print(numLosses)
    # print(numTies)

if __name__ == '__main__':
    main()