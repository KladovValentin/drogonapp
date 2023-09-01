import sys
import matplotlib.pyplot as plt
import numpy as np


def drawShit():
    table = np.loadtxt('trainresults1.txt')
    for i in range(table.shape[1]-1):
        LogScale = True
        if i != 3:
            logScale = False
        mask = table[:,-1]>50
        x = table[mask,i]
        y = table[mask,-1]
        maxind = y.argmax()
        print(x[maxind])
        plt.plot(x,y,"blue",label = 'batchSize',marker='o', linestyle="None", markersize=3.0)
        plt.xscale('log')
        plt.show()


drawShit()
