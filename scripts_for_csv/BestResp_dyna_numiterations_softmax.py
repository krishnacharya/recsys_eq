import sys
sys.path.append('../source/')
import numpy as np
import pandas as pd
from tqdm import tqdm
from Users import Users
from Producers import ProducersEngagementGame
from embeddings import get_user_embeddings
from sklearn.preprocessing import normalize
from surprise import NMF
from surprise import Dataset
from surprise.model_selection import cross_validate
from sklearn.preprocessing import normalize


# Load the movielens-100k dataset
dataset = Dataset.load_builtin('ml-100k')
#dataset = Dataset.load_builtin('ml-1m')

# Large space
dimensions = [5, 10, 15, 20] # number of dimensions for latent/embedding
num_prod_array = np.arange(10, 110, 10) # [10, 20...100] Producers
runs = 20

# TOY search space
# dimensions = [5, 10, 15] # number of dimensions for latent/embedding
# num_prod_array = np.arange(1, 5, 1) # 1..4 producers
# runs = 3

res_detailed = []
res_aggregate = []
tot_loop = len(dimensions) * len(num_prod_array) * runs

with tqdm(total = tot_loop) as pbar:
    for d in dimensions:
        user_embeddings = get_user_embeddings(d, dataset) # first generate the embedding then pass to Users class to make an object
        users = Users(user_embeddings)
        for n_prod in num_prod_array:
            PEng = ProducersEngagementGame(num_producers=n_prod, users=users, prob = 'softmax')
            BR_iters = []
            for r in range(runs):
                NE , iters = PEng.best_response_dynamics(verbose=False)
                BR_iters.append(iters)
                di_detailed = {"dimension": d,
                            "num_prod": n_prod,
                            "run": r,
                            "iters_to_NE": iters,
                            "NE": NE
                            }
                res_detailed.append(di_detailed)
                pbar.update(1)
            di_aggregate = {"dimension": d,
                            "num_prod": n_prod,
                            "mean_iter_to_NE": np.mean(BR_iters),
                            "std_iters_to_NE": np.std(BR_iters),
                            "number_of_uniqueNE": len(PEng.BR_dyna_NE),
                            }
            res_aggregate.append(di_aggregate)

df_detailed = pd.DataFrame(res_detailed)
df_detailed.to_csv('../csv_results/softmax_detailed.csv', encoding='utf-8', index=False)
df_aggr = pd.DataFrame(res_aggregate)
df_aggr.to_csv('../csv_results/softmax_aggr.csv', encoding='utf-8', index=False)