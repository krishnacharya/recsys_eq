import surprise
from surprise import NMF, SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from sklearn.preprocessing import normalize
import numpy as np
from abc import ABC, abstractmethod

def generate_uniform_user(dimension) -> np.array:
    '''
      Sample a single user from the probability simplex of dimension=dimension \sum_j x_j = 1
      shape (dim, )
    '''
    #unifrom sampling form the probability simplex
    #reference : https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    return np.diff([0] + sorted(np.random.uniform(size=dimension-1)) + [1])

def generate_uniform_users(dimension, num_users = 10000) -> np.array:
  '''
    Get a numpy array of shape num_users x dimension
    with each row representing a user on the proability simplex, i.e.\sum_j x_ij = 1
  '''
  return np.array([generate_uniform_user(dimension=dimension) for _ in range(num_users)])


class Embedding(ABC):
    def __init__(self, seed, dimension, num_users):
        pass

class Synth_Uniform_Embedding(Embedding):
    def __init__(self, seed, dimension, num_users):
        np.random.seed(seed = seed) # set seed for randomness
        self.nue = generate_uniform_users(dimension = dimension, num_users = num_users) # already L1 normalized, rows sum to 1
        self.dimension = dimension
        self.num_users = num_users
    
class Synth_Skewed_Embedding(Embedding):
    def __init__(self, seed, dimension, num_users):
        np.random.seed(seed = seed) # set seed for randomness
        self.nue = self.generate_skewed_users(dimension = dimension, num_users = num_users) # nue must be L1 normalized to 1
        self.dimension = dimension
        self.num_users = num_users
    
    def generate_skewed_users(self, dimension, num_users = 10000) -> np.array:
        '''
            Get a numpy array of shape num_users x dimension
            with each row representing a user on the proability simplex, i.e.\sum_j x_ij = 1
        '''
        weights = np.sort(generate_uniform_user(dimension)) # shape d, from the probbability simplex
        ue = generate_uniform_users(dimension, num_users = num_users) # shape Num users, d
        return normalize(ue * weights, norm = "l1")

class Movielens_100k_Embedding(Embedding):
    def __init__(self, seed, dimension, num_users): # hacky!, num_users is useless here as movielnes has a fixed number of users
        np.random.seed(seed = seed) # set seed for randomness in NMF factorization
        _, self.nue = self.get_user_embeddings_movielens100k(dimension) # normalized user embeddings
        self.dimension = dimension
        self.num_users = self.nue.shape[0]
    
    def get_user_embeddings_movielens100k(self, dimension):
        '''
            Gets embeddings for movielens 100k dataset
            Returns tuple
            User embeddings, and L1 normalized user embeddings
        '''
        data = Dataset.load_builtin("ml-100k")
        algo = NMF(n_factors=dimension)
        cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
        normalized_embeddings = normalize(algo.pu,  norm = "l1")
        return algo.pu, normalized_embeddings

class Movielens_1m_Embedding(Embedding):
    def __init__(self, seed, dimension, num_users): # hacky!, num_users is useless here as movielnes has a fixed number of users
        np.random.seed(seed = seed) # set seed for randomness in NMF factorization
        _, self.nue = self.get_user_embeddings_movielens1m(dimension) # normalized user embeddings
        self.dimension = dimension
        self.num_users = self.nue.shape[0]
    
    def get_user_embeddings_movielens1m(self, dimension):
        '''
            Gets embeddings for movielens 100k dataset
            Returns tuple
            User embeddings, and L1 normalized user embeddings
        '''
        data = Dataset.load_builtin("ml-1m")
        algo = NMF(n_factors=dimension)
        cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
        normalized_embeddings = normalize(algo.pu,  norm = "l1")
        return algo.pu, normalized_embeddings