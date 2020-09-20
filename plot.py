import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot1():
    y = np.array(torch.load('loss_2.pt'))
    x = np.array(range(y.size))

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(yscale='log',
           xlabel='epoch',
           ylabel='BCE loss')
    ax.grid()

    # fig.savefig("loss_linear.png")
    plt.show()


def plot2():
    y = np.array([  # 0.01400956,
        0.0001503282255726, 0.0000948764354689, 0.0000434396279161, 0.0000436213267676, 0.0000388449552702,
        0.0000286552494799, 0.0000273941459454, 0.0000228531880566, 0.0000186304660019, 0.0000175334862433])
    x = np.array(range(1000, 10001, 1000))

    fig, ax = plt.subplots()
    ax.plot(x, y, '.-')

    ax.set(yscale='log',
           xlabel='epoch',
           ylabel='BCE loss',
           xticks=x)
    ax.grid(True, which='both')

    fig.savefig("loss_logr2.png")
    plt.show()


if __name__ == '__main__':
    plot2()
