import sys
sys.path.append('../source/')
import numpy as np
from Users import Users
from Producers import ProducersEngagementGame
from embeddings import generate_uniform_users
import pickle
import pandas as pd
from tqdm import tqdm

num_users = 10000 # fixed
num_prod = 100 # fixed
dimensions = [5, 10, 15] # vary across 3 dims and see the kind of specialization seen at NE

user_draws = 10 # number of draws of uniform user embeddings for each num_user x dimension
Br_runs = 20

res = []
with tqdm(total = len(dimensions) * user_draws * Br_runs) as pbar:
    for d in dimensions:
        for draw in range(user_draws):
            ue = generate_uniform_users(dimension = d, num_users = num_users)
            PEng = ProducersEngagementGame(num_producers = num_prod, users = Users(ue), prob = 'softmax')
            num_iter = [] # will be an array of length Br_runs
            for _ in range(Br_runs): # run best response dynamics each time
                NE, iters = PEng.best_response_dynamics(verbose=False)
                num_iter.append(iters)
                pbar.update(1)
            if len(PEng.BR_dyna_NE) > 1: # two or more NE save that PEng
                print("Found more than 1 NE for the Producer engagement game")
                with open(f'morethan1NE_dimension_{d}_draw_{draw}.pickle', 'wb') as handle:
                    pickle.dump(PEng, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if len(PEng.BR_dyna_NE) == 0: # no NE found, save that PEng
                print("Found no NE for the Producer engagement game")
                with open(f'noNE_dimension_{d}_draw_{draw}.pickle', 'wb') as handle:
                    pickle.dump(PEng, handle, protocol=pickle.HIGHEST_PROTOCOL)
            NE = np.array(PEng.BR_dyna_NE.pop())
            di = {"dimension": d,
                "user_draw": draw, 
                "num_users": num_users,
                "num_producers": num_prod,
                "num_iters": np.array(num_iter),
                "NE": NE, # pop the one tuple in this BR_dyna_NE set
                "producer_dist": NE / num_prod, # fraction of producers on each dimension
                "user_dist": np.sum(ue, axis = 0) / num_users # fraction of users on each dimension
            }
            res.append(di)
df = pd.DataFrame(res)
df.to_pickle('../saved_frames/uniform_users_softmax_d5-10-15_10draws.pkl')
        
        
