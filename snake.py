from copy import copy, deepcopy
from pprint import pprint
import random
import math
import numpy as np
import keyboard
import time
import colorama
from colorama import Fore, Style

class Snake:
    def __init__(self, dim):
        self.length = 1
        self.dim = dim
        self.headPos = np.array([int(dim / 2), int(dim/2)])
        self.dir = random.randint(0,3)
        self.board = np.zeros((dim, dim),  dtype=float)
        self.numSteps = 0
        self.gameOver = False
        self.visited = np.zeros((dim, dim),  dtype=float)
        self.resetVisited()
        self.area = dim * dim
        self.stepsSinceLastApple = 0
        self.numNewVisited = 0

        #   step info
        #   #   gets reset every single step
        self.lastStepDir = self.dir
        self.dirChangedThisStep = False
        self.visitedNewThisStep = False
        self.ateAppleThisStep = False

        #   place head
        self.board[self.headPos[1]][self.headPos[0]] = self.length

        #   place an apple
        nextHeadPos = self.getHeadNextPos()
        self.board[nextHeadPos[1]][nextHeadPos[0]] = -1

    def resetStepInfo(self):
        self.lastStepDir = self.dir
        self.dirChangedThisStep = False
        self.visitedNewThisStep = False
        self.ateAppleThisStep = False

    def resetVisited(self):
        self.visited = self.visited * 0.0

    def getHeadNextPos(self):
        if self.dir == 0:
            newHeadPos = self.headPos.copy()
            newHeadPos[0] += 1
            return newHeadPos
        elif self.dir == 1:
            newHeadPos = self.headPos.copy()
            newHeadPos[1] += 1
            return newHeadPos
        elif self.dir == 2:
            newHeadPos = self.headPos.copy()
            newHeadPos[0] -= 1
            return newHeadPos
        elif self.dir == 3:
            newHeadPos = self.headPos.copy()
            newHeadPos[1] -= 1
            return newHeadPos
        else:
            return False

    def printBoard(self):
        print(self.board)

    def drawBoard(self):
        body = " ■ "
        head = " ▲ "
        apple = " ● "
        blank = " □ "
        visitedSpace = " ▣ "
        for y in range(0, self.dim):
            for x in range(0, self.dim):
                val = self.board[y][x]
                visited = self.visited[y][x]
                if val == self.length:
                    print(Fore.GREEN + head, end='')
                    # print(Fore.GREEN + str(val), end='')
                elif val > 0:
                    print(Fore.GREEN + body, end='')
                    # print(Fore.GREEN + str(val), end='')
                elif val < 0:
                    print(Fore.RED + apple, end='')
                    # print(Fore.RED + str(val), end='')
                else:
                    if visited == 1:
                        print(Fore.WHITE + visitedSpace, end='')
                    else:
                        print(Fore.WHITE + blank, end='')
            print()

    def setDir(self, newDir):
        if not newDir == self.dir:
            if newDir == 0 and (self.dir == 3 or self.dir == 1):
                self.dir = newDir
            elif newDir == 1 and (self.dir == 2 or self.dir == 0):
                self.dir = newDir
            elif newDir == 2 and (self.dir == 3 or self.dir == 1):
                self.dir = newDir
            elif newDir == 3 and (self.dir == 0 or self.dir == 2):
                self.dir = newDir

    def step(self):
        if self.gameOver:
            return False

        self.resetStepInfo()

        nextHeadPos = self.getHeadNextPos()

        #   check if new head place is collision
        #   #   with wall
        if  (nextHeadPos[0] < 0 or nextHeadPos[0] >= self.dim or
             nextHeadPos[1] < 0 or nextHeadPos[1] >= self.dim):
            self.gameOver = True
            return False

        boardAtHead = self.board[nextHeadPos[1]][nextHeadPos[0]]
        #   #   with tail
        if boardAtHead > 0: #   crashed into tail
            self.gameOver = True
            return False

        #   check if new head place is an apple
        if boardAtHead < 0: #   ate apple
            #   increase length
            self.length += 1
            
            #   increment tail segments
            tails = np.where(self.board > 0)
            self.board[tails] += 1

            #   no need to remove apple because head will replace it

            #   spawn a new apple
            # flatBoard = self.board.flatten()
            notEmpties = np.not_equal(self.board, 0)
            randomBoard = np.random.random((self.dim, self.dim))
            randomBoard[notEmpties] = 0
            maxPos = np.argmax(randomBoard)
            self.board = self.board.flatten()
            self.board[maxPos] = -1
            self.board = self.board.reshape(self.dim, self.dim)

            self.resetVisited()
            self.stepsSinceLastApple = 0
            self.ateAppleThisStep = True
        else:
            self.stepsSinceLastApple += 1

        #   place the new head
        self.headPos = nextHeadPos
        
        #   update step info: VISITED
        if self.visited[nextHeadPos[1]][nextHeadPos[0]] == 0:
            self.visitedNewThisStep = True
            self.numNewVisited += 1

        #   update visited
        self.visited[nextHeadPos[1]][nextHeadPos[0]] = 1.0

        #   update step info: LAST DIR
        if not self.dir == self.lastStepDir:
            self.dirChangedThisStep = True

        #   subtract 1 from all board positions
        self.board = self.board - 1
        negOnes = np.equal(self.board, -1)
        self.board[negOnes] = 0

        #   increment apple age
        apples = np.where(self.board < 0)
        self.board[apples] = -1

        #   place new head
        self.board[nextHeadPos[1]][nextHeadPos[0]] = self.length

        self.numSteps += 1

keyActions = {
    "d": 0,
    "s": 1,
    "a": 2,
    "w": 3,
}

def playGame():
    import os
    clear = lambda: os.system('cls')

    while True:
        game = Snake(16)

        while not game.gameOver:
            clear()
            game.drawBoard()
            for key in keyActions:
                if keyboard.is_pressed(key):
                    game.setDir(keyActions[key])
            game.step()
            time.sleep(0.05)

def main():
    playGame()


if __name__ == '__main__':
    main()