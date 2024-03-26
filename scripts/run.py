import sys
sys.path.append('../source/')
from tqdm import tqdm
import pandas as pd
from Users import Users
from Producers import *
from Embeddings import *

def run_producer_game(dimensions:list, seeds:list, n_prodarr:list, Embedding:Embedding, \
                    prob:str, temp:float, n_users:int, save_file_name:str):
    '''
        dimensions: list of embedding dimensions desired
        seeds: list of seeds
        n_prodarr: list of number of producers
        Embedding: Class name # the Embedding passed to def run_producer_game is actually a class name, we instanciate an object out of it! Embedding could be a Synth_uniform, Synth_skewed, Movielens type etc...
        prob: softmax, linear
        temp: temperature, won't be used in linear
        n_users: number of users
    '''
    tot = len(dimensions) * len(seeds) * len(n_prodarr)
    res = []
    with tqdm(total = tot) as pbar:
        for d in dimensions:
            for seed in seeds:
                # np.random.seed(seed = seed) # set the random seed for reproducible NMF
                emb_obj = Embedding(seed = seed, dimension = d, num_users = n_users)
                nue = emb_obj.nue
                # _, nue  = get_user_embeddings_movielens100k(user_dimension =  d) # get NMF on movielens of dimension d, normalized user embedding L1 norm = 1
                user_dist = nue.sum(axis = 0) / nue.sum() # denominator is actually 943, total number of users, since L1 row norm is 1
                for nprod in n_prodarr:
                    di = {'dimension': d,
                        'seed': seed,
                        'nprod': nprod
                        }
                    PEng = ProducersEngagementGame(num_producers = nprod, users = Users(nue), prob = prob, temp = temp)
                    converged, last_profile, last_profile_compact, iters =  PEng.best_response_dynamics(verbose=False)
                    di['NE_exists'] = converged
                    di['iters'] = iters
                    di['last_profile_compact'] = last_profile_compact
                    di['producer_dist'] = last_profile_compact / nprod
                    di['user_dist'] = user_dist
                    if converged:
                        dir_prods, prod_utils, user_utils = get_all_engagement_utilities(last_profile, nue, prob_type = prob, temp = temp) # CHANGE
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
    df.to_pickle(save_file_name)