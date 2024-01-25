import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    plt.savefig(filedestname, bbox_inches='tight')

def user_producer_2bar_plot_display(pd, ud): #producer distribution and user distribution
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
    plt.show()

def plot_and_save(dims, df_linear:pd.DataFrame, df_softmax:pd.DataFrame, name:str, nprod = 100, seed=17): # representative plots for ud and pd
    distributions = ['user_dist', 'producer_dist']
    for d in dims:
        df_linear_temp = df_linear[(df_linear['seed'] ==  seed) & (df_linear['dimension'] ==  d) & \
                         (df_linear['nprod'] == nprod)][distributions]
        df_softmax_temp = df_softmax[(df_softmax['seed'] ==  seed) & (df_softmax['dimension'] ==  d)\
                           & (df_softmax['nprod'] == nprod)][distributions]

        ud_linear, pd_linear = df_linear_temp['user_dist'].item(), df_linear_temp['producer_dist'].item()
        ud_softmax, pd_softmax = df_softmax_temp['user_dist'].item(), df_softmax_temp['producer_dist'].item()

        user_producer_2bar_plot(pd_linear, ud_linear, f'../plots/{name}_linear_dim{d}_{nprod}prod.pdf')
        user_producer_2bar_plot(pd_softmax, ud_softmax, f'../plots/{name}_softmax_dim{d}_{nprod}prod.pdf')