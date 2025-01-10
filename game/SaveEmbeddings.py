from surprise import Dataset, Reader
from surprise import NMF
from surprise.model_selection import cross_validate
import numpy as np
import pandas as pd
import os

def generate_uniform_user(dimension) -> np.array:
    '''
      Sample a SINGLE user from the probability simplex of dimension=dimension \sum_j x_j = 1
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

def save_movielens100k(dimension:int, seed:int) -> None:
    np.random.seed(seed = seed)
    data = Dataset.load_builtin("ml-100k")
    algo = NMF(n_factors = dimension)
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
    np.save(f'../saved_embeddings/movielens100k/nmf/dim{dimension}_seed{seed}',algo.pu) #save user embeddings, not L1 normalized

def save_synth_uniform(dimension:int, seed:int, num_users = 10000):
    '''
        dimension for embedding, seed for randomness
    '''
    np.random.seed(seed = seed)
    ue = generate_uniform_users(dimension = dimension, num_users = num_users) # these lie on the probability simplex so already L1 normalized
    np.save(f'../saved_embeddings/synthuniform/dim{dimension}_seed{seed}', ue) #save user embeddings

def save_synth_skewed(dimension:int, seed:int, num_users=10000):
    def generate_skewed_users(dimension, num_users):
        weights = np.sort(generate_uniform_user(dimension)) # shape d, from the probbability simplex
        ue = generate_uniform_users(dimension, num_users = num_users) # shape Num users, d
        return ue * weights

    np.random.seed(seed = seed)
    ue = generate_skewed_users(dimension=dimension, num_users = num_users) # not L1 normalized
    np.save(f'../saved_embeddings/synthskewed/dim{dimension}_seed{seed}', ue) #save user embeddings

def save_synthuniform_sparse(dimension:int, seed:int, num_users=10000, fraction=0.9):
    """
    Generate uniform embeddings sparsify and save them L1 normalized.
    Parameters:
        dimension (int): Dimensionality of embeddings.
        seed (int): Seed for random number generator.
        num_users (int): Number of users to generate.
        fraction (float): Fraction of dimensions to set to zero (sparsity), default 90% sparse
    """
    np.random.seed(seed = seed)
    ue = generate_uniform_users(dimension=dimension, num_users=num_users) # numuses x dim
    sparse_ue = np.zeros_like(ue)
    mask_start = int(fraction * dimension)  # Calculate the number of dimensions to zero out
    mask = np.arange(dimension)
    for i in range(num_users):
        np.random.shuffle(mask)  # Shuffle the dimensions to create a random sparse mask
        active_indices = mask[mask_start:]  # Retain the last (1-fraction) dimensions
        sparse_ue[i, active_indices] = ue[i, active_indices]  # Copy over the non-zero values
        sparse_ue[i] /= sparse_ue[i].sum() # L1 normalize
    save_dir = f'../saved_embeddings/sparse{fraction}_unif'
    os.makedirs(save_dir, exist_ok=True)
    np.save(f'{save_dir}/dim{dimension}_seed{seed}', sparse_ue)

def save_synthskew_sparse(dimension:int, seed:int, num_users=10000, fraction=0.9):
    """
    Generate skewed sparse user embeddings and save them.
    Parameters:
        dimension (int): Dimensionality of embeddings.
        num_users (int): Number of users to generate.
        fraction (float): Fraction of dimensions to set to zero (sparsity).
    """
    def generate_skewed_users(dimension, num_users):
        weights = np.sort(generate_uniform_user(dimension))
        ue = generate_uniform_users(dimension, num_users=num_users)
        return ue * weights
    
    np.random.seed(seed = seed)
    ue = generate_skewed_users(dimension=dimension, num_users=num_users)     # Generate skewed user embeddings
    sparse_ue = np.zeros_like(ue)  # Initialize with zeros
    mask_start = int(fraction * dimension)  # Calculate the number of dimensions to zero
    mask = np.arange(dimension)
    for i in range(num_users):
        np.random.shuffle(mask)
        active_indices = mask[mask_start:]  # Retain the last (1-fraction) dimensions
        sparse_ue[i, active_indices] = ue[i, active_indices]  # Copy over the non-zero values
        sparse_ue[i] /= sparse_ue[i].sum()
    save_dir = f'../saved_embeddings/sparse{fraction}_skew'
    os.makedirs(save_dir, exist_ok=True)
    np.save(f'{save_dir}/dim{dimension}_seed{seed}', sparse_ue)

def save_amazon_music(dimension:int, seed:int) -> None:
    def get_surprise_compatible():
        df = pd.read_csv('../data/Digital_Music.csv')
        df = df[['user_id','item_id','rating']] # surprise custom dataset expects this order
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df, reader)
        return data

    np.random.seed(seed = seed)
    data = get_surprise_compatible()
    algo = NMF(n_factors = dimension)
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
    np.save(f'../saved_embeddings/amznmusic/nmf/dim{dimension}_seed{seed}',algo.pu) #save user embeddings, not L1 normalized

def save_rentrunway(dimension:int, seed:int) -> None:
    def get_surprise_compatible():
        df = pd.read_csv('../data/rentrunway_preproc.csv')
        df = df[['user_id','item_id','rating']] # surprise custom dataset expects this order
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df, reader)
        return data
    
    np.random.seed(seed = seed)
    data = get_surprise_compatible()
    algo = NMF(n_factors = dimension)
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
    np.save(f'../saved_embeddings/rentrunway/nmf/dim{dimension}_seed{seed}',algo.pu) #save user embeddings, not L1 normalized
