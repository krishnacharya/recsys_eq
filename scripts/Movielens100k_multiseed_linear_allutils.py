import sys
sys.path.append('../source/')
import numpy as np
from Users import Users
from Producers import *
from embeddings import *
from sklearn.preprocessing import normalize
from plotting import *
from tqdm import tqdm
import pandas as pd

dimension = [5, 10, 15]
seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
n_prodarr = [30, 50, 100, 200]

tot = len(dimension) * len(seeds) * len(n_prodarr)
res = []
with tqdm(total = tot) as pbar:
    for d in dimension:
        for seed in seeds:
            np.random.seed(seed = seed) # set the random seed for reproducible NMF
            _, nue  = get_user_embeddings_movielens100k(user_dimension =  d) # get NMF on movielens of dimension d, normalized user embedding L1 norm = 1
            for nprod in n_prodarr:
                PEng = ProducersEngagementGame(num_producers = nprod, users = Users(nue), prob = 'linear')
                NE, NE_compact,iters =  PEng.best_response_dynamics(verbose=False)
                if NE is None:
                    print("dimension, seed, nprod, br iterations", d, seed, nprod, iters)    
                dir_prods, prod_utils, user_utils = get_all_engagement_utilities(NE, nue, prob_type='linear')
                pbar.update(1)
                di = {
                    'dimension': d,
                    'seed': seed,
                    'nprod': nprod,

                    'total_prod_util': prod_utils.sum(),
                    'avg_prod_util': prod_utils.mean(),
                    'max_prod_util': prod_utils.max(),
                    'min_prod_util': prod_utils.min(),
        
                    'total_user_util': user_utils.sum(),
                    'avg_user_util': user_utils.mean(),
                    'max_user_util': user_utils.max(),
                    'min_user_util': user_utils.min(),
                                
                    'NE': NE_compact,
                    'iters': iters,
                    'user_dist': nue.sum(axis=0) / nue.sum(),
                    'prod_dist': NE_compact / nprod
                }
                res.append(di)
df = pd.DataFrame(res)
df.to_pickle('../saved_frames/Movielens100k_multiseed_linear_allutils_30prodmore.pkl')
