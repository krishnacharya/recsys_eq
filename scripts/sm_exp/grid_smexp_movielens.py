import sys
sys.path.append('../../source/')
import numpy as np
from Users import Users
from Producers import *
from embeddings import *
from sklearn.preprocessing import normalize
from plotting import *
from tqdm import tqdm
import pandas as pd


seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
dimension = [5, 10, 15, 20] # number of dimensions for latent/embedding
n_prodarr = np.arange(10, 110, 10) # [10, 20...100] Producers

# TEST
# seeds = [2, 11, 13]
# dimension = [5, 10] # number of dimensions for latent/embedding
# n_prodarr = [10, 20]

tot = len(dimension) * len(seeds) * len(n_prodarr)
res = []
with tqdm(total = tot) as pbar:
    for d in dimension:
        for seed in seeds:
            np.random.seed(seed = seed) # set the random seed for reproducible NMF
            _, nue  = get_user_embeddings_movielens100k(user_dimension =  d) # get NMF on movielens of dimension d, normalized user embedding L1 norm = 1
            user_dist = nue.sum(axis = 0) / nue.sum() # denominator is actually 943, total number of users, since L1 row norm is 1
            for nprod in n_prodarr:
                di = {'dimension': d,
                      'seed': seed,
                      'nprod': nprod
                     }
                Psmexp = ProducerSoftmaxExposureGame(num_producers = nprod, users = Users(nue))
                converged, last_profile, last_profile_compact, iters =  Psmexp.best_response_dynamics(verbose=False)
                di['NE_exists'] = converged
                di['iters'] = iters
                di['last_profile_compact'] = last_profile_compact
                di['producer_dist'] = last_profile_compact / nprod
                di['user_dist'] = user_dist
                if converged:
                    dir_prods, prod_utils, user_utils = get_softmax_prodexposure_usereng_utilities(last_profile, nue)
                    di.update({
                    'total_prod_util': prod_utils.sum(),
                    'avg_prod_util': prod_utils.mean(),
                    'max_prod_util': prod_utils.max(),
                    'min_prod_util': prod_utils.min(),
        
                    'total_user_util': user_utils.sum(),
                    'avg_user_util': user_utils.mean(),
                    'max_user_util': user_utils.max(),
                    'min_user_util': user_utils.min(),
                    })
                else:
                    di.update({
                    'total_prod_util': -1,
                    'avg_prod_util': -1,
                    'max_prod_util': -1,
                    'min_prod_util': -1,
        
                    'total_user_util': -1,
                    'avg_user_util': -1,
                    'max_user_util': -1,
                    'min_user_util': -1,
                    })
                pbar.update(1)
                res.append(di)
df = pd.DataFrame(res)
df.to_pickle('../../saved_frames/Movielens100k_grid_smexposure.pkl')
