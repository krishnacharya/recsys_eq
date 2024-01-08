import numpy as np
import matplotlib.pyplot as plt

def user_producer_2bar_plot(pd, ud, filedestname:str): #producer distribution and user distribution
    #TODO add savefig too
    dim = len(pd)
    y_max = max(max(ud), max(pd))

    x1, y1 = np.arange(dim), ud
    x2, y2 = np.arange(dim), pd

    fig, ax = plt.subplots()
    ax.bar(np.array(x1)-0.15, y1, width = 0.3, color='blue')
    ax.set_ylabel('Average User weight', fontsize=16)
    ax.set_xlabel('feature #', fontsize=16)
    plt.ylim(0, y_max)

    ax2 = ax.twinx()
    ax2.bar(np.array(x2)+0.15, y2, width = 0.3, color='red')
    ax2.set_ylabel('Fraction of Producers', fontsize=16)
    plt.ylim(0, y_max)

    plt.xticks(range(dim))
    plt.savefig(filedestname)