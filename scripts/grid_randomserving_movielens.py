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
            _, nue = get_user_embeddings_movielens100k(user_dimension =  d)
            user_dist = nue.sum(axis = 0) / nue.sum() # denominator is actually 943, total number of users, since L1 row norm is 1
            for nprod in n_prodarr:
                di = {'dimension': d,
                      'seed': seed,
                      'nprod': nprod
                     }
                producer_util, best_index, user_utils = random_rec_utilities(num_prod = nprod, user_array = nue)
                profile_compact = np.zeros(d)
                profile_compact[best_index] = 1
                profile_compact *= nprod # these lines filling all producers in the best_index
                di['NE_exists'] = True
                di['iters'] = 1
                di['last_profile_compact'] = profile_compact 
                di['producer_dist'] = profile_compact / nprod
                di['user_dist'] = user_dist

                di.update({
                'total_prod_util': producer_util * nprod, 
                'avg_prod_util': producer_util,
                'max_prod_util': producer_util,
                'min_prod_util': producer_util,
        
                'total_user_util': user_utils.sum(),
                'avg_user_util': user_utils.mean(),
                'max_user_util': user_utils.max(),
                'min_user_util': user_utils.min(),
                    })
                pbar.update(1)
                res.append(di)
df = pd.DataFrame(res)
df.to_pickle('../saved_frames/movielens_synthetic_grid_randomserving_statistics.pkl')
