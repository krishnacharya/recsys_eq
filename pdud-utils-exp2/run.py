import sys
sys.path.append('../source/')
from tqdm import tqdm
import pandas as pd
from Users import Users
from Producers import *
from Embeddings import *

def run_producer_game_singleseedsave(dimensions:list, emb_seed:int, n_prodarr:list, Embedding:Embedding, \
                    prob:str, temp:float, n_users:int, save_dest:str, experiment_seed = 505):
    '''
        dimensions: list of embedding dimensions desired
        emb_seed: is used for loading the saved NMF factorization
        n_prodarr: list of number of producers
        Embedding: Class name # the Embedding passed to def run_producer_game is actually a class name, we instanciate an object out of it! Embedding could be a Synth_uniform, Synth_skewed, Movielens type etc...
        prob: softmax, linear
        temp: temperature, won't be used in linear
        n_users: number of users
    '''
    np.random.seed(seed = experiment_seed) # TODO this can be made an argparsed variable
    tot = len(dimensions) * len(n_prodarr)
    res = []
    with tqdm(total = tot, mininterval = 600) as pbar:
        for d in dimensions:
            emb_obj = Embedding(seed = emb_seed, dimension = d, num_users = n_users) #now we use saved embeddings for movielens
            nue = emb_obj.nue
            user_dist = nue.sum(axis = 0) / nue.sum() # denominator will have the number of numbers, since each row is L1 normalized
            for nprod in n_prodarr:
                di = {'dimension': d,
                    'emb_seed': emb_seed,
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
    df.to_pickle(save_dest)