import sys
sys.path.append('../../source/')
import numpy as np
import pandas as pd
from tqdm import tqdm
from Users import Users
from Producers import *
from embeddings import *
from sklearn.preprocessing import normalize
from surprise import NMF
from surprise import Dataset
from surprise.model_selection import cross_validate
from sklearn.preprocessing import normalize



# Large space
dimensions = [5, 10, 15, 20] # number of dimensions for latent/embedding
num_prod_array = np.arange(10, 110, 10) # [10, 20...100] Producers
runs = 40
num_users = 10000


# TOY search space
# dimensions = [5, 10, 15] # number of dimensions for latent/embedding
# num_prod_array = np.arange(1, 5, 1) # 1..4 producers
# runs = 3

seed = 17

res_detailed = []
res_aggregate = []
tot_loop = len(dimensions) * len(num_prod_array) * runs

with tqdm(total = tot_loop) as pbar:
    for d in dimensions:
        np.random.seed(seed = seed)
        user_embeddings = generate_uniform_users(dimension = d, num_users = num_users)
        users = Users(user_embeddings)
        for n_prod in num_prod_array:
            Psmexp = ProducerSoftmaxExposureGame(num_producers = n_prod, users=users)
            BR_iters = []
            for r in range(runs):
                converged, last_profile, last_profile_compact, iters = Psmexp.best_response_dynamics(verbose=False)
                BR_iters.append(iters)
                di_detailed = {"dimension": d,
                            "num_prod": n_prod,
                            "run": r,
                            "iters_to_NE": iters,
                            "NE": last_profile_compact,
                            "converged": converged
                            }
                res_detailed.append(di_detailed)
                pbar.update(1)
            di_aggregate = {"dimension": d,
                            "num_prod": n_prod,
                            "mean_iter_to_NE": np.mean(BR_iters),
                            "std_iters_to_NE": np.std(BR_iters),
                            "number_of_uniqueNE": len(Psmexp.BR_dyna_NE)
                            }
            res_aggregate.append(di_aggregate)

df_detailed = pd.DataFrame(res_detailed)
df_detailed.to_csv('../../csv_results/br_dynamics/smexp_detailed_40runs_uniform.csv', encoding='utf-8', index=False)
df_aggr = pd.DataFrame(res_aggregate)
df_aggr.to_csv('../../csv_results/br_dynamics/smexp_aggr_40runs_uniform.csv', encoding='utf-8', index=False)