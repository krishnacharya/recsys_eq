# import sys
# sys.path.append('../source/')
from tqdm import tqdm
import pandas as pd
from game.Users import Users
from game.Producers import *
from game.Embeddings import *
from game.ServingProbability import Probability
import torch
import numpy as np

def run_producer_game_singleseedsave(dimensions:list, emb_seed:int, n_prodarr:list, emb_obj:Embedding, \
                    probability:Probability, n_users:int, save_dest:str, experiment_seed = 505): # engagement game
    '''
        dimensions: list of embedding dimensions desired
        emb_seed: is used for loading the saved NMF factorization
        n_prodarr: list of number of producers
        Embedding: Class name # the Embedding passed to def run_producer_game is actually a class name, we instanciate an object out of it! Embedding could be a Synth_uniform, Synth_skewed, Movielens type etc...
        prob: softmax, linear, random
        temp: temperature, won't be used in linear
        n_users: number of users
    '''
    np.random.seed(seed = experiment_seed)
    torch.manual_seed(experiment_seed)
    tot = len(dimensions) * len(n_prodarr)
    res = []
    with tqdm(total = tot, mininterval = 600) as pbar:
        for d in dimensions:
            nue = emb_obj.get_nue(seed = emb_seed, dimension = d)
            user_dist = nue.sum(axis = 0) / nue.sum() # denominator will have the number of numbers, since each row is L1 normalized
            for nprod in n_prodarr: # todo add topk condition check here
                di = {'dimension': d,
                    'emb_seed': emb_seed,
                    'nprod': nprod
                    }
                PEng = ProducersEngagementGame(num_producers = nprod, users = Users(nue), probability = probability) # add probability object
                converged, last_profile, last_profile_compact, iters =  PEng.best_response_dynamics(verbose=False)
                di['NE_exists'] = converged
                di['iters'] = iters
                di['last_profile_compact'] = last_profile_compact
                di['producer_dist'] = last_profile_compact / nprod
                di['user_dist'] = user_dist
                dir_prods, prod_utils, user_utils = get_all_engagement_utilities(last_profile, nue, probability)  # add probability object
                di.update({
                    'total_prod_util': prod_utils.sum().item(),
                    'avg_prod_util': prod_utils.mean().item(),
                    'max_prod_util': prod_utils.max().item(),
                    'min_prod_util': prod_utils.min().item(),

                    'total_user_util': user_utils.sum().item(),
                    'avg_user_util': user_utils.mean().item(),
                    'max_user_util': user_utils.max().item(),
                    'min_user_util': user_utils.min().item(),
                    })
                pbar.update(1)
                res.append(di)
    df = pd.DataFrame(res)
    df.to_pickle(save_dest)

# def run_producer_expsoure_game_singleseedsave(dimensions:list, emb_seed:int, n_prodarr:list, emb_obj:Embedding, \
#                     prob:str, temp:float, n_users:int, save_dest:str, experiment_seed = 505):
#     '''
#         dimensions: list of embedding dimensions desired
#         emb_seed: is used for loading the saved NMF factorization
#         n_prodarr: list of number of producers
#         Embedding: Class name # the Embedding passed to def run_producer_game is actually a class name, we instanciate an object out of it! Embedding could be a Synth_uniform, Synth_skewed, Movielens type etc...
#         prob: softmax, linear, random
#         temp: temperature, won't be used in linear
#         n_users: number of users
#     '''
#     np.random.seed(seed = experiment_seed)
#     tot = len(dimensions) * len(n_prodarr)
#     res = []
#     with tqdm(total = tot, mininterval = 600) as pbar:
#         for d in dimensions:
#             nue = emb_obj.get_nue(seed = emb_seed, dimension = d)
#             user_dist = nue.sum(axis = 0) / nue.sum() # denominator will have the number of numbers, since each row is L1 normalized
#             for nprod in n_prodarr:
#                 di = {'dimension': d,
#                     'emb_seed': emb_seed,
#                     'nprod': nprod
#                     }
#                 PExp = ProducersExpsoureGame(num_producers = nprod, users = Users(nue), prob = prob, temp = temp)
#                 converged, last_profile, last_profile_compact, iters =  PExp.best_response_dynamics(verbose=False)
#                 di['NE_exists'] = converged
#                 di['iters'] = iters
#                 di['last_profile_compact'] = last_profile_compact
#                 di['producer_dist'] = last_profile_compact / nprod
#                 di['user_dist'] = user_dist
#                 if converged:
#                     dir_prods, prod_utils, user_utils = get_all_exposure_utilities(last_profile, nue, prob_type = prob, temp = temp)
#                     di.update({
#                     'total_prod_util': prod_utils.sum(), # this is just expsoure so should sum to number of users
#                     'avg_prod_util': prod_utils.mean(), 
#                     'max_prod_util': prod_utils.max(), 
#                     'min_prod_util': prod_utils.min(),
        
#                     'total_user_util': user_utils.sum(), # this is the engagement utility for users
#                     'avg_user_util': user_utils.mean(),
#                     'max_user_util': user_utils.max(),
#                     'min_user_util': user_utils.min(),
#                     })
#                 else:
#                     dir_prods, prod_utils, user_utils = get_all_exposure_utilities(last_profile, nue, prob_type = prob, temp = temp)
#                     di.update({
#                     'total_prod_util': prod_utils.sum(),
#                     'avg_prod_util': prod_utils.mean(),
#                     'max_prod_util': prod_utils.max(),
#                     'min_prod_util': prod_utils.min(),
        
#                     'total_user_util': user_utils.sum(),
#                     'avg_user_util': user_utils.mean(),
#                     'max_user_util': user_utils.max(),
#                     'min_user_util': user_utils.min(),
#                     }) # last iterate values
#                 pbar.update(1)
#                 res.append(di)
#     df = pd.DataFrame(res)
#     df.to_pickle(save_dest)