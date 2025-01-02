import sys
sys.path.append('../source/')
from tqdm import tqdm
import pandas as pd
from Users import Users
from Producers import *
from Embeddings import *

def run_numiters(run:str, dimensions:list, emb_seed:int, n_prodarr:list, 
                emb_obj:Embedding, prob:str, temp:float, n_users:int, save_dest:str):
    '''
        Very similar to run_producer_game but run is variable parameter which is from the job array iteration number
        embedding_seed is used for loading the saved NMF factorization
    '''
    np.random.seed(seed = int(run)) # seed for experiment run, this is not related to the seed in the NMF embedding generation
    tot = len(dimensions) * len(n_prodarr)
    res = []
    with tqdm(total = tot, mininterval = 600) as pbar:
        for d in dimensions:
            nue = emb_obj.get_nue(seed = emb_seed, dimension = d)
            user_dist = nue.sum(axis = 0) / nue.sum() # denominator will have the number of numbers, since each row is L1 normalized
            for nprod in n_prodarr:
                di = {'dimension': d, 'nprod': nprod, 'run': run, 'emb_seed':emb_seed}
                PEng = ProducersEngagementGame(num_producers = nprod, users = Users(nue), prob = prob, temp = temp)
                converged, last_profile, last_profile_compact, iters =  PEng.best_response_dynamics(verbose=False)
                di['NE_exists'] = converged
                di['iters_to_NE'] = iters
                di['NE'] = last_profile_compact
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
                    dir_prods, prod_utils, user_utils = get_all_engagement_utilities(last_profile, nue, prob_type = prob, temp = temp)
                    di.update({
                    'total_prod_util': prod_utils.sum(),
                    'avg_prod_util': prod_utils.mean(),
                    'max_prod_util': prod_utils.max(),
                    'min_prod_util': prod_utils.min(),
        
                    'total_user_util': user_utils.sum(),
                    'avg_user_util': user_utils.mean(),
                    'max_user_util': user_utils.max(),
                    'min_user_util': user_utils.min(),
                    }) # last iterate values
                pbar.update(1)
                res.append(di)
    df = pd.DataFrame(res)
    df.to_pickle(save_dest)