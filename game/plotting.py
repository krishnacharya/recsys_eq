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

def plot_and_save_singledf(dims:list, df:pd.DataFrame, name:str, nprod = 100, seed=17):
  distributions = ['user_dist', 'producer_dist']
  for d in dims:
    df_temp = df[(df['emb_seed'] ==  seed) & (df['dimension'] ==  d) & (df['nprod'] == nprod)][distributions]
    ud_softmax, pd_softmax = df_temp['user_dist'].item(), df_temp['producer_dist'].item()
    user_producer_2bar_plot(pd_softmax, ud_softmax, f'../plots/udpd/{name}_dim_{d}.pdf')

def plot_utils_tempvar_prodcurves__errbar(dict_df:dict, df_linear:pd.DataFrame, nprodlist:list, filedest:str, dim = 15):
  '''
    Producer and user utility averaged across seeds
    One curve each for producers in nprodlist e.g. 2, 10, 50, 100 producers
    x axis has temperature, y axis has avg utility

    dict_df: dictionarty of dataframs keys as softmax temperatures 0.01, 0.1, 1, 10, 100
    df_linear: linear serving dataframe
    nprodlist: list of ints, expecting 4 different total number of producers e.g [2, 10, 50, 100]

    2 plots one for producer utility, one for user utility saved in filedest
  '''
  def df_agg_utils(df, temp:int):
    '''
      average across seeds

      Returns 
      df_agg has columns: dimension, nprod, avg_prod_util_mean, avg_prod_util_sem, avg_user_util_mean, avg_user_util_sem
    '''
    groups = ['dimension', 'nprod'] # groupby columns, averages out across seeds
    cols = groups + ['avg_prod_util', 'avg_user_util']
    # df = df[df['NE_exists'] == True][cols]
    df = df[cols]
    df_agg = df.groupby(groups).agg(['mean', 'sem']) # iters_to_NE will get mean, standard error of mean; we group by dimensions, num_prod
    df_agg.columns = df_agg.columns.map("_".join) # this is just to flatten multi column iters_to_NE mean and SEM
    df_agg.reset_index(inplace=True)
    df_agg['temp'] = temp # add a column for temperature
    return df_agg

  df_sm = pd.concat([df_agg_utils(df, temperature_key) for temperature_key, df in dict_df.items()]) # softmax temps
  df_linear_agg = df_agg_utils(df = df_linear, temp = 1.0) # temp really doesnt matter but is a parameter so passing a dummy value
  nprod_dict_avg_produtility = {} # this is used for plotting
  nprod_dict_avg_userutility = {}
  for idx, nprod in enumerate(nprodlist):
    df = df_sm[(df_sm['dimension'] == dim)  & (df_sm['nprod'] == nprod)] # will have 5 different temps
    df.sort_values(by = 'temp', inplace = True)
    stagger_shift = (-1)**idx * idx*1e-4 # shift that the error bars stagger and dont overlap with others
    nprod_dict_avg_produtility[nprod] = {'x': df['temp'] + stagger_shift, 'y': df['avg_prod_util_mean'], 'yerr': df['avg_prod_util_sem']}
    nprod_dict_avg_userutility[nprod] = {'x': df['temp'] + stagger_shift, 'y': df['avg_user_util_mean'], 'yerr': df['avg_user_util_sem']}
  
  ls_list = ['solid', 'dotted', 'dashed', 'dashdot'] # different linestyles
  color_list = ['#377eb8', '#e41a1c', '#ff7f00', '#f781bf'] # suitable for colorblind 

  def save_util(name:str):
    if name == 'producer':
      nprod_dict = nprod_dict_avg_produtility
      linutilkey = 'avg_prod_util_mean'
    else:
      nprod_dict = nprod_dict_avg_userutility
      linutilkey = 'avg_user_util_mean'
    idx = 0
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    for nprod, value in nprod_dict.items():
        linutil = df_linear_agg[(df_linear_agg['dimension'] == dim)  & (df_linear_agg['nprod'] == nprod)][linutilkey]
        ax.errorbar(**value, linestyle = ls_list[idx], color = color_list[idx], capsize=3, capthick=1, elinewidth=1, \
                   alpha=1.0, markersize=4, label = f'{nprod} producers')
        idx += 1
    ax.legend(loc= "upper right")
    ax.set_xlabel("Temperature")
    ax.set_ylabel(f"Average {name} utility")
    fig.savefig(filedest + f'avg_{name}_utility.pdf', bbox_inches='tight')
  save_util(name = 'producer')
  save_util(name = 'user')

def plot_utils_tempvar_prodcurves_linvssm(dict_df:dict, df_linear:pd.DataFrame, nprodlist:list, filedest:str, dim = 15): #only want to plot fixed number of prod and dim, TO REFACTOR
  '''
    producer and user utility averaged across seeds
    One curve each for producers in nprodlist e.g. 2, 10, 50, 100 producers
    x axis has temperature, y axis has avg utility

    dict_df: dictionarty of dataframs keys as softmax temperatures 0.01, 0.1, 1, 10, 100
    df_linear: linear serving dataframe
    nprodlist: list of ints, expecting 4 different total number of producers e.g [2, 10, 50, 100]

    2 plots one for producer utility, one for user utility saved in filedest
  '''
  def df_agg_utils(df, temp:int):
    '''
      average across seeds

      Returns 
      df_agg has columns: dimension, nprod, avg_prod_util_mean, avg_prod_util_sem, avg_user_util_mean, avg_user_util_sem
    '''
    groups = ['dimension', 'nprod'] # groupby columns, averages out across seeds
    cols = groups + ['avg_prod_util', 'avg_user_util']
    # df = df[df['NE_exists'] == True][cols]
    df = df[cols]
    df_agg = df.groupby(groups).agg(['mean', 'sem']) # iters_to_NE will get mean, standard error of mean; we group by dimensions, num_prod
    df_agg.columns = df_agg.columns.map("_".join) # this is just to flatten multi column iters_to_NE mean and SEM
    df_agg.reset_index(inplace=True)
    df_agg['temp'] = temp # add a column for temperatur
    return df_agg

  df_sm = pd.concat([df_agg_utils(df, temperature_key) for temperature_key, df in dict_df.items()]) # softmax temps
  df_linear_agg = df_agg_utils(df = df_linear, temp = 1.0) # temp really doesnt matter but is a parameter so passing a dummy value
  nprod_dict_avg_produtility = {} # this is used for plotting
  nprod_dict_avg_userutility = {}
  for idx, nprod in enumerate(nprodlist):
    df = df_sm[(df_sm['dimension'] == dim)  & (df_sm['nprod'] == nprod)] # will have 5 different temps
    df.sort_values(by = 'temp', inplace = True)
    stagger_shift = (-1)**idx * idx*1e-4 # shift that the error bars stagger and dont overlap with others
    nprod_dict_avg_produtility[nprod] = {'x': df['temp'] + stagger_shift, 'y': df['avg_prod_util_mean'], 'yerr': df['avg_prod_util_sem']}
    nprod_dict_avg_userutility[nprod] = {'x': df['temp'] + stagger_shift, 'y': df['avg_user_util_mean'], 'yerr': df['avg_user_util_sem']}
  
  ls_list = ['solid', 'dotted', 'dashed', 'dashdot'] # different linestyles
  color_list = ['#377eb8', '#e41a1c', '#ff7f00', '#f781bf'] # suitable for colorblind 
  temps = df_sm.temp.unique()
  def save_util(name:str):
    if name == 'producer':
      nprod_dict = nprod_dict_avg_produtility
      linutilmean_key = 'avg_prod_util_mean'
      linutilsem_key = 'avg_prod_util_sem'
    else:
      nprod_dict = nprod_dict_avg_userutility
      linutilmean_key = 'avg_user_util_mean'
      linutilsem_key = 'avg_user_util_sem'
    # plt.figure()
    # plt.xscale('log')
    idx = 0
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    for nprod, value in nprod_dict.items(): # value has the softmaxs df_aggs utility value
        linutil_mean = df_linear_agg[(df_linear_agg['dimension'] == dim)  & (df_linear_agg['nprod'] == nprod)][linutilmean_key]
        linutil_sem = df_linear_agg[(df_linear_agg['dimension'] == dim)  & (df_linear_agg['nprod'] == nprod)][linutilsem_key]

        ax.errorbar(temps, [linutil_mean.iloc[0]]*len(temps), [linutil_sem.iloc[0]]*len(temps), linestyle = 'solid', color = '#377eb8', capsize=3, capthick=1, elinewidth=1, \
                   alpha=0.6, markersize=4, label = f'{nprod} producers, linear') #
        ax.errorbar(**value, linestyle = 'dotted', color = '#e41a1c', capsize=3, capthick=1, elinewidth=1, \
                   alpha=1.0, markersize=4, label = f'{nprod} producers, softmax')
        idx += 1
    ax.legend(loc= "upper right")
    ax.set_xlabel("Temperature")
    ax.set_ylabel(f"Average {name} utility")
    fig.savefig(filedest + f'avg_{name}_utility.pdf', bbox_inches='tight')
  save_util(name = 'producer')
  save_util(name = 'user')

def plot_4dim_numiternew_errbar(df, filename): # Num iters to NE scaling
  '''
    num iters dataframe with 40 runs for each dim, prod, single seed 
    4 curves for 4 dimensions of embeddings
    x axis has number of producers
  '''
  def agg_num_iterdf(df):
    '''
      Returns 
      df_agg has columns dimension, num_prod, iters_to_NE_mean, iters_to_NE_sem
    '''
    groups = ['dimension', 'nprod'] # groupby columns
    cols = ['dimension', 'nprod', 'iters_to_NE']
    # df = df[df['NE_exists'] == True][cols]
    df = df[cols]
    df_agg = df.groupby(groups).agg(['mean','sem']) # iters_to_NE will get mean, sem: std/sqrt{#runs}
    df_agg.columns = df_agg.columns.map("_".join) # this is just to flatten multi column iters_to_NE mean and standard error of mean
    df_agg.reset_index(inplace=True)
    return df_agg

  df_agg = agg_num_iterdf(df)
  dims = df_agg['dimension'].unique()
  nprods = df_agg['nprod'].unique()

  data_dim5 = {
    'x': nprods,
    'y': df_agg[df_agg['dimension'] == 5]['iters_to_NE_mean'],
    'yerr': df_agg[df_agg['dimension'] == 5]['iters_to_NE_sem']}
  
  data_dim10 = {
    'x': nprods+0.3,   # adding offsets to x to make standard error bars non-overlapping
    'y': df_agg[df_agg['dimension'] == 10]['iters_to_NE_mean'],
    'yerr': df_agg[df_agg['dimension'] == 10]['iters_to_NE_sem']}

  data_dim15 = {
    'x': nprods-0.3,
    'y': df_agg[df_agg['dimension'] == 15]['iters_to_NE_mean'],
    'yerr': df_agg[df_agg['dimension'] == 15]['iters_to_NE_sem']}
  
  data_dim20 = {
    'x': nprods+0.5,
    'y': df_agg[df_agg['dimension'] == 20]['iters_to_NE_mean'],
    'yerr': df_agg[df_agg['dimension'] == 20]['iters_to_NE_sem']}
  # ls_list = [':o',':s' ,':o' , ':s']
  ls_list = ['solid','dotted', 'dashed', 'dashdot'] # different linestyles
  color_list = ['#377eb8', '#e41a1c', '#ff7f00', '#f781bf'] # suitable for colorblind 
  plt.figure()
  for idx, data in enumerate([data_dim5, data_dim10, data_dim15, data_dim20]):
      # plt.errorbar(**data, fmt=line_styles[idx], capsize=3, capthick=1, elinewidth=1, \
      #              alpha=0.9, markersize=4, label = f'dimension = {dims[idx]}')
      plt.errorbar(**data, linestyle = ls_list[idx], color = color_list[idx], capsize=3, capthick=1, elinewidth=1, \
                   alpha=0.9, markersize=4, label = f'dimension = {dims[idx]}')            
  plt.legend(loc= "upper left")
  plt.xlabel("Number of producers")
  plt.ylabel("Iterations to NE")
  plt.savefig(filename, bbox_inches='tight')


def plot_ndim_numiter_errbar(df, filename):  # Num iters to NE scaling 6 dim
    """
    Plots iterations to NE with error bars for 6 dims, [5,10,15,20,50,100]
    Handles any number of dimensions dynamically.

    Parameters:
    - df: Input DataFrame containing columns: 'dimension', 'nprod', 'iters_to_NE', 'NE_exists'
    - filename: Output filename for the saved plot
    """
    def agg_num_iterdf(df):
        """
        Aggregates the DataFrame to compute mean and SEM for iterations to NE.
        Returns a DataFrame with columns: dimension, nprod, iters_to_NE_mean, iters_to_NE_sem
        """
        groups = ['dimension', 'nprod']
        cols = ['dimension', 'nprod', 'iters_to_NE']
        # df = df[df['NE_exists'] == True][cols]
        df = df[cols]
        df_agg = df.groupby(groups).agg(['mean', 'sem'])
        df_agg.columns = df_agg.columns.map("_".join)  # Flatten multi-index columns
        df_agg.reset_index(inplace=True)
        return df_agg
    df_agg = agg_num_iterdf(df)
    dims = sorted(df_agg['dimension'].unique())
    nprods = df_agg['nprod'].unique()
    # Dynamically generate styles and offsets
    offsets = [-0.3, 0, 0.3, 0.5, -0.5, 0.7]  # Extend dynamically as needed
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', '--', '-.']  # Extend dynamically as needed
    colors = ['#377eb8', '#e41a1c', '#ff7f00', '#f781bf', '#4daf4a', '#984ea3']  # Suitable for colorblind
    plt.figure()
    for idx, dim in enumerate(dims):
        offset = offsets[idx % len(offsets)]  # Cycle through offsets
        linestyle = linestyles[idx % len(linestyles)]  # Cycle through linestyles
        color = colors[idx % len(colors)]  # Cycle through colors

        data = df_agg[df_agg['dimension'] == dim]
        plt.errorbar(
            x=nprods + offset,
            y=data['iters_to_NE_mean'],
            yerr=data['iters_to_NE_sem'],
            linestyle=linestyle,
            color=color,
            capsize=3,
            capthick=1,
            elinewidth=1,
            alpha=0.9,
            markersize=4,
            label=f'dimension = {dim}'
        )
    plt.legend(loc="upper left")
    plt.xlabel("Number of producers")
    plt.ylabel("Iterations to NE")
    plt.savefig(filename, bbox_inches='tight')


def plot_utils_tempvar_prodcurves_lastiter(dict_df:dict, df_linear:pd.DataFrame, nprodlist:list, filedest:str, dim = 15):
  '''
    Same functionality as plot_utils_tempvar_prodcurves__errbar, but for last iterate
    Engagement utility for producer
  '''
  def df_agg_utils(df, temp:int):
    '''
      Average across seeds

      Returns 
      df_agg has columns: dimension, nprod, avg_prod_util_mean, avg_prod_util_sem, avg_user_util_mean, avg_user_util_sem
    '''
    groups = ['dimension', 'nprod'] # groupby columns, averages out across seeds
    cols = groups + ['avg_prod_util', 'avg_user_util']
    df = df[cols] # also ensure that last iterate utility val is set, seee run
    df_agg = df.groupby(groups).agg(['mean', 'sem']) # iters_to_NE will get mean, standard error of mean; we group by dimensions, num_prod
    df_agg.columns = df_agg.columns.map("_".join) # this is just to flatten multi column iters_to_NE mean and SEM
    df_agg.reset_index(inplace=True)
    df_agg['temp'] = temp 
    return df_agg

  df_sm = pd.concat([df_agg_utils(df, temperature_key) for temperature_key, df in dict_df.items()]) # softmax temps
  df_linear_agg = df_agg_utils(df = df_linear, temp = 1.0) # temp really doesnt matter but is a parameter so passing a dummy value
  nprod_dict_avg_produtility = {} # this is used for plotting
  nprod_dict_avg_userutility = {}
  for idx, nprod in enumerate(nprodlist):
    df = df_sm[(df_sm['dimension'] == dim)  & (df_sm['nprod'] == nprod)] # will have 5 different temps
    df.sort_values(by = 'temp', inplace = True)
    stagger_shift = (-1)**idx * idx*1e-4 # shift that the error bars stagger and dont overlap with others
    nprod_dict_avg_produtility[nprod] = {'x': df['temp'] + stagger_shift, 'y': df['avg_prod_util_mean'], 'yerr': df['avg_prod_util_sem']}
    nprod_dict_avg_userutility[nprod] = {'x': df['temp'] + stagger_shift, 'y': df['avg_user_util_mean'], 'yerr': df['avg_user_util_sem']}
  
  ls_list = ['solid', 'dotted', 'dashed', 'dashdot'] # different linestyles
  color_list = ['#377eb8', '#e41a1c', '#ff7f00', '#f781bf'] # suitable for colorblind 

  def save_util(name:str):
    if name == 'producer':
      nprod_dict = nprod_dict_avg_produtility
      linutilkey = 'avg_prod_util_mean'
    else:
      nprod_dict = nprod_dict_avg_userutility
      linutilkey = 'avg_user_util_mean'
    idx = 0
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    for nprod, value in nprod_dict.items():
        linutil = df_linear_agg[(df_linear_agg['dimension'] == dim)  & (df_linear_agg['nprod'] == nprod)][linutilkey]
        ax.errorbar(**value, linestyle = ls_list[idx], color = color_list[idx], capsize=3, capthick=1, elinewidth=1, \
                   alpha=1.0, markersize=4, label = f'{nprod} producers')
        idx += 1
    ax.legend(loc= "upper right")
    ax.set_xlabel("Temperature")
    ax.set_ylabel(f"Average {name} utility")
    fig.savefig(filedest + f'avg_{name}_utility.pdf', bbox_inches='tight')
  save_util(name = 'producer')
  save_util(name = 'user')


def plot_utils_tempvar_prodcurves_fixed_vs_sm(dict_df:dict, df_fixed:pd.DataFrame, nprodlist:list, filedest:str, dim = 15, fixed_str='linear'): #only want to plot fixed number of prod and dim, TO REFACTOR
  '''
    Producer and user utility averaged across seeds
    One curve each for producers in nprodlist e.g. 5, 10, 50 producers
    x axis has temperature, y axis has avg utility

    dict_df: dictionarty of dataframe keys as softmax temperatures 0.01, 0.1, 1, 10, 100
    df_fixed: dataframe that doesnt vary with temperature, e.g. random, linear serving
    nprodlist: list of ints, expecting 4 different total number of producers e.g [2, 10, 50, 100]

    fixed_str, name for legend, linear or random

    2 plots one for producer utility, one for user utility saved in filedest
  '''
  def df_agg_utils(df, temp:int):
    '''
      average across seeds

      Returns 
      df_agg has columns: dimension, nprod, avg_prod_util_mean, avg_prod_util_sem, avg_user_util_mean, avg_user_util_sem
    '''
    groups = ['dimension', 'nprod'] # groupby columns, averages out across seeds
    cols = groups + ['avg_prod_util', 'avg_user_util']
    df = df[cols]
    df_agg = df.groupby(groups).agg(['mean', 'sem']) # iters_to_NE will get mean, standard error of mean; we group by dimensions, num_prod
    df_agg.columns = df_agg.columns.map("_".join) # this is just to flatten multi column iters_to_NE mean and SEM
    df_agg.reset_index(inplace=True)
    df_agg['temp'] = temp
    return df_agg

  df_sm = pd.concat([df_agg_utils(df, temperature_key) for temperature_key, df in dict_df.items()]) # softmax temps
  df_fixed_agg = df_agg_utils(df = df_fixed, temp = 1.0) # temp really doesnt matter but is a parameter so passing a dummy value
  nprod_dict_avg_produtility = {} # this is used for plotting
  nprod_dict_avg_userutility = {}
  for idx, nprod in enumerate(nprodlist):
    df = df_sm[(df_sm['dimension'] == dim)  & (df_sm['nprod'] == nprod)] # will have 5 different temps
    df.sort_values(by = 'temp', inplace = True)
    stagger_shift = (-1)**idx * idx*1e-4 # shift that the error bars stagger and dont overlap with others
    nprod_dict_avg_produtility[nprod] = {'x': df['temp'] + stagger_shift, 'y': df['avg_prod_util_mean'], 'yerr': df['avg_prod_util_sem']}
    nprod_dict_avg_userutility[nprod] = {'x': df['temp'] + stagger_shift, 'y': df['avg_user_util_mean'], 'yerr': df['avg_user_util_sem']}
  
  ls_list = ['solid', 'dotted', 'dashed', 'dashdot'] # different linestyles
  color_list = ['#377eb8', '#e41a1c', '#ff7f00', '#f781bf'] # suitable for colorblind
  temps = df_sm.temp.unique()
  def save_util(name:str):
    if name == 'producer':
      nprod_dict = nprod_dict_avg_produtility
      linutilmean_key = 'avg_prod_util_mean'
      linutilsem_key = 'avg_prod_util_sem'
    else:
      nprod_dict = nprod_dict_avg_userutility
      linutilmean_key = 'avg_user_util_mean'
      linutilsem_key = 'avg_user_util_sem'
    idx = 0
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    for nprod, value in nprod_dict.items():
        linutil_mean = df_fixed_agg[(df_fixed_agg['dimension'] == dim)  & (df_fixed_agg['nprod'] == nprod)][linutilmean_key]
        linutil_sem = df_fixed_agg[(df_fixed_agg['dimension'] == dim)  & (df_fixed_agg['nprod'] == nprod)][linutilsem_key]

        ax.errorbar(temps, [linutil_mean.iloc[0]]*len(temps), [linutil_sem.iloc[0]]*len(temps), linestyle = 'solid', color = '#377eb8', capsize=3, capthick=1, elinewidth=1, \
                   alpha=0.6, markersize=4, label = f'{nprod} producers, {fixed_str} serving') #
        ax.errorbar(**value, linestyle = 'dotted', color = '#e41a1c', capsize=3, capthick=1, elinewidth=1, \
                   alpha=1.0, markersize=4, label = f'{nprod} producers, softmax')
        idx += 1
    ax.legend(loc= "upper right")
    ax.set_xlabel("Temperature")
    ax.set_ylabel(f"Average {name} utility")
    fig.savefig(filedest + f'avg_{name}_utility.pdf', bbox_inches='tight')
  save_util(name = 'producer')
  save_util(name = 'user')
