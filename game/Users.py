import numpy as np
# Basic class for Users
class Users:
    def __init__(self, embeddings : np.ndarray):
        self.user_array = embeddings # shape N_users x dimension of embedding
        self.num_users = embeddings.shape[0] # N_users
        self.dimension = embeddings.shape[1] # dimensionality of embedding