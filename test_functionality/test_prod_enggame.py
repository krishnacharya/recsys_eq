from tqdm import tqdm
import pandas as pd
from game.Users import Users
from game.Producers import *
from game.Embeddings import *
from game.ServingProbability import Probability
import torch
import numpy as np

def main():
    d, emb_seed = 10, 17
    nprod = 100
    k = 1
    temp = 1
    emb_obj =  Movielens_100k_Embedding()
    probability = Probability(prob_str='topk_softmax', temp=temp, topk=k)
    nue = emb_obj.get_nue(seed = emb_seed, dimension = d) # Nusers x dim
    user_dist = nue.sum(axis = 0) / nue.sum() # weight on each dimension, dr will give number of users since we are L1 normalized
    PEng = ProducersEngagementGame(num_producers = nprod, users = Users(nue), probability = probability)
    converged, last_profile, last_profile_compact, iters =  PEng.best_response_dynamics(verbose=False)
    di = {'dimension': d, 'emb_seed': emb_seed,'nprod': nprod}
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
    print(di)

main()
