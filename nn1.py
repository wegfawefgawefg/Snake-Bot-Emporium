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

'''
it would be nice if the punishment was somehow attached to a specific input
like you could make it associate the punishment with the input instead of it just being some
    arbitrary punishment coming from the ether at the end of the game
    -ideas for this are seeding it with a specific game output? on the unsupervised game state predictor board?
'''

'''
could try passing in recommended inputs, 
and then slowly cutting those out to make it less dependant
why would this work?
    it should associate inputs with specific environments
'''

'''
TODONT:
'''

'''
TODO:
    stop punishing weights into oblivion
        try always giving a set amount of reinforcement per turn. like energy
            what doesnt go to one action gets divided amongst the rest?

    spawn 4 new games per turn to "see" the future?

    reward portions
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import os
import time
import random
import numpy as np
from tqdm import tqdm

import snake as s
import architectures as arcs

clear = lambda: os.system('cls')

#   high fucking score
# def getRewardThisStep(game):
#     reward = -1.0
#     if game.ateAppleThisStep:
#         reward += 1.0
#     if game.visitedNewThisStep:
#         reward += 4.0
#     else:
#         reward -= 4.0

#     return reward

def getRewardThisStep(game):
    reward = -1.0
    if game.ateAppleThisStep:
        reward += 4.0
    if game.visitedNewThisStep:
        reward += 4.0
    else:
        reward -= 4.0

    return reward

def encodeGame(game):
    snake = np.where(game.board > 0, True, False)
    apples = np.where(game.board < 0, True, False)

    snakeBoard = game.board.copy()
    snakeBoard[apples] = 0
    snakeBoard /= game.length

    appleBoard = game.board.copy()
    appleBoard[snake] = 0
    appleBoard[apples] *= -1

    #   add current direction
    direction = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
    direction[game.dir] = 1

    oneHot = np.concatenate([snakeBoard.flatten(), appleBoard.flatten(), direction])
    oneHot = torch.tensor(oneHot, dtype=torch.float32).view(1, 1, game.dim * game.dim * 2 + 4)

    return oneHot

def train(net, criterion, optimizer, gameSize, epochs, watch=True):
    net.train()

    topReward = 0
    longestSnake = 0
    longestGame = 0
    mostNewVisited = 0

    game = s.Snake(gameSize)
    maxNumSteps = game.area * 4

    if watch:
        pbar = range(epochs)
    else:
        pbar = tqdm(range(epochs))
    for i in pbar:
        gameDescription = "Game %s, Records: Reward %s, Snake Length %s, Steps %s, New Visits %s" % (i, str(topReward)[:4], longestSnake, longestGame, mostNewVisited)
        if not watch:
            pbar.set_description(gameDescription)
        optimizer.zero_grad()

        game = s.Snake(gameSize)
        maxStepsExceeded = False
        while not game.gameOver and not maxStepsExceeded:

            optimizer.zero_grad()

            if watch:
                clear()
                game.drawBoard()
                print(gameDescription)
                time.sleep(0.07)

            #   generate a move
            oneHot = encodeGame(game)
            output = net(oneHot)
            # print(output)

            _, index = output.view(4).max(0)
            move = index

            game.setDir(move)

            reward = getRewardThisStep(game)

            target = torch.tensor([1, 1, 1, 1], dtype=torch.float32)    #   incentivise all other moves
            # target = torch.tensor([0, 0, 0, 0], dtype=torch.float32)    #   incentivises no other moves
            # target = output.clone().view(4)                             #   incentivises only target move
            target[move] += reward
            target = target.view(1, 1, 4)
            
            loss = criterion(output, target)
            loss.backward(retain_graph=True)
            optimizer.step()

            game.step()

            # if game.numSteps > maxNumSteps:
            #     maxStepsExceeded = True

        if game.length == game.area:    #   beat the game
            # target = torch.tensor([1, 1, 1, 1], dtype=torch.float32)    #   incentivise all other moves
            target = torch.tensor([0, 0, 0, 0], dtype=torch.float32)    #   incentivises no other moves
            # target = output.clone().view(4)                             #   incentivises only target move            target[move] += 1
            target = target.view(1, 1, 4)
        else:   #   died pathetically
            target = torch.tensor([1, 1, 1, 1], dtype=torch.float32)    #   incentivise all other moves
            # target = torch.tensor([0, 0, 0, 0], dtype=torch.float32)    #   incentivises no other moves
            # target = output.clone().view(4)                             #   incentivises only target move
            target[move] += -30   #   PLAYS REALLY WELL
            # target[move] += -20   #   BEST DEATH PENALTY
            # target[move] += -10
            target = target.view(1, 1, 4)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if reward > topReward:
            topReward = reward
        if game.length > longestSnake:
            longestSnake = game.length
        if game.numSteps > longestGame:
            longestGame = game.numSteps
        if game.numNewVisited > mostNewVisited:
            mostNewVisited = game.numNewVisited

    return False

def main():
    boardSize = 5
    oneHotShape = boardSize * boardSize * 2 + 4

    net = arcs.Net(oneHotShape, 64, 4)
    # net = arcs.BigNet(oneHotShape, 64, 4)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=0.001)    #   BEST OPTIM
    # optimizer = optim.Adam(net.parameters(), lr=0.01)

    train(net=net, criterion=criterion, optimizer=optimizer, gameSize=boardSize, epochs=10000, watch=False)
    train(net=net, criterion=criterion, optimizer=optimizer, gameSize=boardSize, epochs=10000, watch=True)
    # numWins, numLosses, numTies = test(net=net, epochs=1000)
    # print("wins, losses, ties:")
    # print(numWins)
    # print(numLosses)
    # print(numTies)

if __name__ == '__main__':
    main()