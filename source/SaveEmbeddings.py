import sys
sys.path.append('../scripts')
from utils import load_config
import surprise
from surprise import Dataset, Reader
from surprise import NMF, SVD
from surprise import Dataset
import numpy as np
from surprise.model_selection import cross_validate
import pandas as pd

# common_config = load_config('../configs/common_config.yml') # dictionary with common seeds, dimension, nprods

# print(common_config['seeds'])
# print(common_config['dimensions'])

def save_movielens100k():
    for dim in common_config['dimensions']:
        for seed in common_config['seeds']:
            np.random.seed(seed = seed) #TODO?
            data = Dataset.load_builtin("ml-100k")
            algo = NMF(n_factors=dim)
            cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
            np.save(f'../saved_embeddings/movielens100k/nmf/dim{dim}_seed{seed}',algo.pu)

def save_amazon_music(dimension, seed) -> None:
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
    np.save(f'../saved_embeddings/amznmusic/nmf/dim{dimension}_seed{seed}',algo.pu) #save user embeddings

# save_movielens100k()