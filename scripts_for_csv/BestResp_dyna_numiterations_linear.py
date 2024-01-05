import sys
sys.path.append('../source/')
import numpy as np
import pandas as pd
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

num_prod_array = np.arange(1, 101, 1) # 1..100 producers
dimensions = [5, 10, 15] # number of dimensions for latent/embedding
runs = 100

res_detailed = []
res_aggregate = []
for d in dimensions:
    user_embeddings = get_user_embeddings(d, dataset) # first generate the embedding then pass to Users class to make an object
    users = Users(user_embeddings)
    for n_prod in num_prod_array:
        PEng = ProducersEngagementGame(num_producers=n_prod, users=users, prob = 'linear')
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
        di_aggregate = {"dimension": d,
                        "num_prod": n_prod,
                        "mean_iter_to_NE": np.mean(BR_iters),
                        "std_iters_to_NE": np.std(BR_iters),
                        "number_of_uniqueNE": len(PEng.BR_dyna_NE),
                        }
        res_aggregate.append(di_aggregate)

df_detailed = pd.DataFrame(res_detailed)
df_detailed.to_csv('../csv_results/num_iters_linear_movielens100k_detailed.csv', encoding='utf-8', index=False)
df_aggr = pd.DataFrame(res_aggregate)
df_aggr.to_csv('../csv_results/num_iters_linear_movielens100k_detailed_aggregate.csv', encoding='utf-8', index=False)