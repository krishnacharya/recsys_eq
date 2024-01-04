import surprise
from surprise import NMF
from sklearn.preprocessing import normalize
import numpy as np

def generate_uniform_user(dimension):
  #unifrom sampling form the probability simplex
  #reference : https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
  return np.diff([0] + sorted(np.random.uniform(size=dimension-1)) + [1])

def get_user_embeddings(user_dimension, data: surprise.Dataset):
  algo = NMF(n_factors=user_dimension)
  # algo = NMF(n_factors=user_dimension, init_low=0, init_high=10)
  cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
  users = normalize(algo.pu, "l1")
  return users