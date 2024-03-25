import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def user_producer_2bar_plot(pd, ud, filedestname:str): #producer distribution and user distribution
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

def plot_4dim_numiter_errbar(df_agg, filename): # best response mean number of iterations to NE and standard deviation
  dims = df_agg['dimension'].unique()
  nprods = df_agg['num_prod'].unique()
  print(dims, nprods)
  

  data_dim5 = {
    'x': nprods,
    'y': df_agg[df_agg['dimension'] == 5]['mean_iter_to_NE'],
    'yerr': df_agg[df_agg['dimension'] == 5]['std_iters_to_NE']}
  
  data_dim10 = {
    'x': nprods+0.3,   # adding offsets to x to make standard error bars non-overlapping
    'y': df_agg[df_agg['dimension'] == 10]['mean_iter_to_NE'],
    'yerr': df_agg[df_agg['dimension'] == 10]['std_iters_to_NE']}

  data_dim15 = {
    'x': nprods-0.3,
    'y': df_agg[df_agg['dimension'] == 15]['mean_iter_to_NE'],
    'yerr': df_agg[df_agg['dimension'] == 15]['std_iters_to_NE']}
  
  data_dim20 = {
    'x': nprods+0.5,
    'y': df_agg[df_agg['dimension'] == 20]['mean_iter_to_NE'],
    'yerr': df_agg[df_agg['dimension'] == 20]['std_iters_to_NE']}
  line_styles = [':o',':s' ,':o' , ':s'] # dotted circle and square
  plt.figure()
  for idx, data in enumerate([data_dim5, data_dim10, data_dim15, data_dim20]):
      plt.errorbar(**data, fmt=line_styles[idx], capsize=3, capthick=1, elinewidth=1, \
                   alpha=0.9, markersize=4, label = f'dimension = {dims[idx]}')
  plt.legend(loc= "upper left")
  plt.xlabel("Number of producers")
  plt.ylabel("Iterations to NE")
  plt.savefig(filename, bbox_inches='tight')

def plot_and_save(dims, df_linear:pd.DataFrame, df_softmax:pd.DataFrame, df_greedy:pd.DataFrame, name:str, nprod = 100, seed=17): # representative plots for ud and pd
    distributions = ['user_dist', 'producer_dist']
    for d in dims:
        df_linear_temp = df_linear[(df_linear['seed'] ==  seed) & (df_linear['dimension'] ==  d) & \
                         (df_linear['nprod'] == nprod)][distributions]
        df_softmax_temp = df_softmax[(df_softmax['seed'] ==  seed) & (df_softmax['dimension'] ==  d)\
                           & (df_softmax['nprod'] == nprod)][distributions]
        df_greedy_temp = df_greedy[(df_greedy['seed'] ==  seed) & (df_greedy['dimension'] ==  d)\
                           & (df_greedy['nprod'] == nprod)][distributions]


        ud_linear, pd_linear = df_linear_temp['user_dist'].item(), df_linear_temp['producer_dist'].item()
        ud_softmax, pd_softmax = df_softmax_temp['user_dist'].item(), df_softmax_temp['producer_dist'].item()
        ud_greedy, pd_greedy = df_greedy_temp['user_dist'].item(), df_greedy_temp['producer_dist'].item()

        user_producer_2bar_plot(pd_linear, ud_linear, f'../plots/udpd/{name}_linear_dim{d}_{nprod}prod.pdf')
        user_producer_2bar_plot(pd_softmax, ud_softmax, f'../plots/udpd/{name}_softmax_dim{d}_{nprod}prod.pdf')
        user_producer_2bar_plot(pd_greedy, ud_greedy, f'../plots/udpd/{name}_greedy_dim{d}_{nprod}prod.pdf')
