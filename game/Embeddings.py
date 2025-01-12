import surprise
from surprise import NMF, SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from sklearn.preprocessing import normalize
import numpy as np
import torch
from abc import ABC, abstractmethod
from utils.project_dirs import emb_dataset_alg, emb_dataset

# def generate_uniform_user(dimension) -> np.array:
#     '''
#       Sample a single user from the probability simplex of dimension=dimension \sum_j x_j = 1
#       shape (dim, )
#     '''
#     #unifrom sampling form the probability simplex
#     #reference : https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
#     return np.diff([0] + sorted(np.random.uniform(size=dimension-1)) + [1])

# def generate_uniform_users(dimension, num_users = 10000) -> np.array:
#   '''
#     Get a numpy array of shape num_users x dimension
#     with each row representing a user on the proability simplex, i.e.\sum_j x_ij = 1
#   '''
#   return np.array([generate_uniform_user(dimension=dimension) for _ in range(num_users)])


class Embedding(ABC):
    def __init__(self):
        pass

# Synthetic
class Synth_Uniform_Embedding(Embedding):
    def get_nue(self, seed, dimension):
        user_emb = np.load(str(emb_dataset(dataset='synthuniform') / f'dim{dimension}_seed{seed}.npy')).astype(np.float32)
        self.nue = torch.from_numpy(user_emb)
        self.num_users, self.dimension = self.nue.shape
        return self.nue

class Synth_Skewed_Embedding(Embedding):
    def get_nue(self, seed, dimension):
        user_emb = np.load(str(emb_dataset(dataset='synthskewed') / f'dim{dimension}_seed{seed}.npy')).astype(np.float32)
        # user_emb = np.load(f'../saved_embeddings/synthskewed/dim{dimension}_seed{seed}.npy').astype(np.float32)
        self.nue = torch.from_numpy(normalize(user_emb,  norm = "l1"))
        self.num_users, self.dimension = self.nue.shape
        return self.nue

# Sparse Synthetic
class SparseUni(Embedding):
    def __init__(self, spfrac=0.9):
        self.spfrac = spfrac
    
    def get_nue(self, seed, dimension):
        user_emb = np.load(str(emb_dataset(dataset=f'sparse{self.spfrac}_unif') / f'dim{dimension}_seed{seed}.npy')).astype(np.float32)
        # user_emb = np.load(f'../saved_embeddings/sparse{self.spfrac}_unif/dim{dimension}_seed{seed}.npy').astype(np.float32)
        self.nue = torch.from_numpy(user_emb)
        self.num_users, self.dimension = self.nue.shape
        return self.nue

class SparseSkew(Embedding):
    def __init__(self, spfrac=0.9):
        self.spfrac = spfrac
    
    def get_nue(self, seed, dimension):
        user_emb = np.load(str(emb_dataset(dataset=f'sparse{self.spfrac}_skew') / f'dim{dimension}_seed{seed}.npy')).astype(np.float32)
        # self.nue = torch.from_numpy(np.load(f'../saved_embeddings/sparse{self.spfrac}_skew/dim{dimension}_seed{seed}.npy').astype(np.float32))
        self.nue = torch.from_numpy(user_emb)
        self.num_users, self.dimension = self.nue.shape
        return self.nue

# Real datasets
class Movielens_100k_Embedding(Embedding):
    def get_nue(self, seed, dimension):
        emb_path = str(emb_dataset_alg(dataset='movielens100k', alg='nmf') / f'dim{dimension}_seed{seed}.npy')  
        # user_emb = np.load(f'../saved_embeddings/movielens100k/nmf/dim{dimension}_seed{seed}.npy')
        user_emb = np.load(emb_path).astype(np.float32)
        self.nue = torch.from_numpy(normalize(user_emb,  norm = "l1"))
        self.num_users, self.dimension = self.nue.shape
        return self.nue
    
class RentRunway_Embedding(Embedding): # TODO fix paths
    def get_nue(self, seed, dimension):
        user_emb = np.load(f'../saved_embeddings/rentrunway/nmf/dim{dimension}_seed{seed}.npy')
        self.nue = torch.from_numpy(normalize(user_emb,  norm = "l1"))
        self.num_users, self.dimension = self.nue.shape
        return self.nue

class AmazonMusic_Embedding(Embedding):
    def get_nue(self, seed, dimension):
        user_emb = np.load(f'../saved_embeddings/amznmusic/nmf/dim{dimension}_seed{seed}.npy')
        self.nue = torch.from_numpy(normalize(user_emb,  norm = "l1"))
        self.num_users, self.dimension = self.nue.shape
        return self.nue