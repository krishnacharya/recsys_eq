import surprise
from surprise import NMF, SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from sklearn.preprocessing import normalize
import numpy as np

def generate_uniform_users(dimension, num_users = 10000) -> np.array:
  '''
    Get a numpy array of shape num_users x dimension
    with each row representing a user on the proability simplex, i.e.\sum_j x_ij = 1
  '''
  def generate_uniform_user(dimension):
    #unifrom sampling form the probability simplex
    #reference : https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    return np.diff([0] + sorted(np.random.uniform(size=dimension-1)) + [1])
  return np.array([generate_uniform_user(dimension=dimension) for _ in range(num_users)])

def get_user_embeddings_movielens100k(user_dimension):
  '''
    Gets embeddings for movielens 100k dataset
    Returns
    User embeddings, and L1 normalized user embeddings
  '''
  data = Dataset.load_builtin("ml-100k")
  algo = NMF(n_factors=user_dimension)
  cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
  normalized_embeddings = normalize(algo.pu, "l1")
  return algo.pu, normalized_embeddings

def user_embeddings_movielens100k_nmf(dimension):
  '''
    Uses parameters from Krauth
    Returns
    User embeddings, and L1 normalized user embeddings
  '''
  algo = NMF(n_factors=dimension, biased =False, reg_pu=0.08, reg_qi=0.08, n_epochs = 128)
  data =  Dataset.load_builtin("ml-100k")
  full_data = data.build_full_trainset()
  algo.fit(full_data)
  normalized_embeddings = normalize(algo.pu, "l1")
  return algo.pu, normalized_embeddings
  