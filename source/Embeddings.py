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
        self.dimension = dimension
        self.num_users = num_users
        self.nue = np.load(f'../saved_embeddings/synthuniform/dim{dimension}_seed{seed}.npy') # normalized user embedding
    
class Synth_Skewed_Embedding(Embedding):
    def __init__(self, seed, dimension, num_users):
        self.dimension = dimension
        self.num_users = num_users
        user_emb = np.load(f'../saved_embeddings/synthskewed/dim{dimension}_seed{seed}.npy')
        self.nue = normalize(user_emb,  norm = "l1") # normalized user embedding

class RentRunway_Embedding(Embedding):
    def __init__(self, seed, dimension, num_users): # hacky!, num_users is not used here as rentrunway has a fixed number of users
        user_emb = np.load(f'../saved_embeddings/rentrunway/nmf/dim{dimension}_seed{seed}.npy')
        self.nue = normalize(user_emb,  norm = "l1")
        self.dimension = dimension
        self.num_users = self.nue.shape[0]

class AmazonMusic_Embedding(Embedding):
    def __init__(self, seed, dimension, num_users):
        user_emb = np.load(f'../saved_embeddings/amznmusic/nmf/dim{dimension}_seed{seed}.npy')
        self.nue = normalize(user_emb,  norm = "l1")
        self.dimension = dimension
        self.num_users = self.nue.shape[0]

class Movielens_100k_Embedding(Embedding):
    def __init__(self, seed, dimension, num_users):
        user_emb = np.load(f'../saved_embeddings/movielens100k/nmf/dim{dimension}_seed{seed}.npy')
        self.nue = normalize(user_emb,  norm = "l1")
        self.dimension = dimension
        self.num_users = self.nue.shape[0]