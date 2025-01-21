from tqdm import tqdm
import pandas as pd
from game.Users import Users
from game.Producers import *
from game.Embeddings import *
from game.ServingProbability import Probability
import torch
import numpy as np


def run_numiters(run:str, dimensions:list, emb_seed:int, n_prodarr:list, 
                emb_obj:Embedding, probability:Probability, save_dest:str):
    '''
        Very similar to run_producer_game but run is variable parameter which is from the job array iteration number
        embedding_seed is used for loading the saved NMF factorization
    '''
    np.random.seed(seed = int(run)) # seed for experiment run, this is not related to the seed in the NMF embedding generation
    torch.manual_seed(int(run))
    tot = len(dimensions) * len(n_prodarr)
    res = []
    with tqdm(total = tot, mininterval = 600) as pbar:
        for d in dimensions:
            nue = emb_obj.get_nue(seed = emb_seed, dimension = d)
            user_dist = nue.sum(axis = 0) / nue.sum() # denominator will have the number of numbers, since each row is L1 normalized
            for nprod in n_prodarr:
                di = {'dimension': d, 'nprod': nprod, 'run': run, 'emb_seed':emb_seed}
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