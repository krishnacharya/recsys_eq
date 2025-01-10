import torch
import numpy as np

class Probability:
    def __init__(self, prob_str='linear', temp=1.0, topk=5):
        # Store configuration for the probability function
        self.temp = temp
        self.topk = topk
        self.probability_config = {
            'linear': {'needs_temp': False, 'needs_topk': False, 'func': linear_probability},
            'random': {'needs_temp': False, 'needs_topk': False, 'func': random_probability},
            'softmax': {'needs_temp': True, 'needs_topk': False, 'func': softmax_probability},
            'topk_softmax': {'needs_temp': True, 'needs_topk': True, 'func': topk_softmax_probability}
        }
        if prob_str not in self.probability_config:
            raise ValueError(f"Unknown probability function type: {prob_str}")
        self.probability_function = self.probability_config[prob_str]['func']
        self.needs_temp = self.probability_config[prob_str]['needs_temp']
        self.needs_topk = self.probability_config[prob_str]['needs_topk']
    
    def get_probability(self, content_vec, remaining_array, user_array):
        # Prepare arguments based on the function's needs
        args = [content_vec, remaining_array, user_array]
        if self.needs_temp:
            args.append(self.temp)
        if self.needs_topk:
            args.append(self.topk)
        return self.probability_function(*args)

# TORCH versions below
def linear_probability(content_vector: torch.Tensor, remaining_array: torch.Tensor, user_array: torch.Tensor) -> torch.Tensor:
    """ 
    Calculate the linear probability of each user being recommended to `content_vector`.
    
    Args:
        content_vector: Tensor of shape (dimension,).
        remaining_array: Tensor of shape (N_producers - 1, dimension).
        user_array: Tensor of shape (N_users, dimension).
        temp: A placeholder argument for compatibility, not used here.

    Returns:
        Tensor of shape (N_users, N_prod), containing the probabilities of each user being recommended to `content_vector`.
        with the last column having the serving probability for producer with content vector
    """
    all_producers = torch.vstack((remaining_array, content_vector))  # Shape: (N_producers, dimension)
    product = torch.matmul(user_array, all_producers.T)  # Shape: (N_users, N_producers)
    prob = product / product.sum(dim=1, keepdim=True)  # Shape: (N_users, N_producers)
    return prob

def softmax_probability(content_vector: torch.Tensor, remaining_array: torch.Tensor, 
                        user_array: torch.Tensor, temp: float = 1) -> torch.Tensor:
    """
    Calculate the softmax probability of each user being recommended to `content_vector`.

    Args:
        content_vector: Tensor of shape (dimension,).
        remaining_array: Tensor of shape (N_producers - 1, dimension).
        user_array: Tensor of shape (N_users, dimension).
        temp: Temperature for softmax, default = 1.

    Returns:
        Tensor of shape (N_users, N_prod), containing the probabilities of each user being recommended to `content_vector`.
        with the last column having the serving probability for producer with content vector
    """
    all_producers = torch.vstack((remaining_array, content_vector))  # Shape: (N_producers, dimension)
    product = torch.matmul(user_array, all_producers.T) / temp # Shape: (N_users, N_producers) 
    prob = torch.softmax(product, dim=1)  # Shape: (N_users, N_producers)
    return prob # Shape: (N_users,)

def topk_softmax_probability(content_vector: torch.Tensor, remaining_array: torch.Tensor, \
                            user_array: torch.Tensor, temp: float = 1, k: int = 1) -> torch.Tensor:
    """
    Calculate the softmax probability over the top-k producers, including `content_vector`.

    Args:
        content_vector: Tensor of shape (dimension,).
        remaining_array: Tensor of shape (N_producers - 1, dimension).
        user_array: Tensor of shape (N_users, dimension).
        temp: Temperature for softmax, default = 1.
        k: Number of top producers to consider for the softmax calculation, default is greedy k = 1

    Returns:
        Tensor of shape (N_users, N_prod), containing the probabilities of each user being recommended to `content_vector`.
        with the last column having the serving probability for producer with content vector
    """
    all_producers = torch.vstack((remaining_array, content_vector))  # Shape: (N_producers, dimension)
    product = torch.matmul(user_array, all_producers.T) / temp  # Shape: (N_users, N_producers)
    topk_scores, topk_indices = torch.topk(product, k=k, dim=1)  # Shape: (N_users, k)
    topk_prob = torch.softmax(topk_scores, dim=1)  # Shape: (N_users, k)
    final_prob = torch.zeros_like(product)  # Shape: (N_users, N_producers)
    # Vectorized assignment
    user_indices = torch.arange(product.shape[0]).unsqueeze(1).expand_as(topk_indices)  # Shape: (N_users, k)
    final_prob[user_indices, topk_indices] = topk_prob
    return final_prob

def random_probability(content_vector: torch.Tensor, remaining_array: torch.Tensor, user_array: torch.Tensor):
    Nprod = (remaining_array.shape[0] + 1)
    Nuser = user_array.shape[0]
    return torch.full((Nuser, Nprod), 1.0 / Nprod)


# def topk_softmax_probability(content_vector: torch.Tensor, remaining_array: torch.Tensor, \
#                             user_array: torch.Tensor, temp: float = 1, k: int = 5) -> torch.Tensor:
#     """
#     Calculate the softmax probability over the top-k producers, including `content_vector`.

#     Args:
#         content_vector: Tensor of shape (dimension,).
#         remaining_array: Tensor of shape (N_producers - 1, dimension).
#         user_array: Tensor of shape (N_users, dimension).
#         temp: Temperature for softmax, default = 1.
#         k: Number of top producers to consider for the softmax calculation.
#     Returns:
#         Tensor of shape (N_users, N_producers) 
#     """
#     all_producers = torch.vstack((remaining_array, content_vector))  # Shape: (N_producers, dimension)
#     product = torch.matmul(user_array, all_producers.T) / temp  # Shape: (N_users, N_producers)
#     topk_scores, topk_indices = torch.topk(product, k=k, dim=1)  # Shape: (N_users, k), # Extract top-k scores and their indices for each user
#     topk_prob = torch.softmax(topk_scores, dim=1)  # Shape: (N_users, k)
#     # Identify the index of `content_vector` in top-k indices
#     content_index = all_producers.size(0) - 1  # Last index corresponds to content_vector
#     is_content_in_topk = (topk_indices == content_index)  # Shape: (N_users, k), atmost one value in the row can be True
#     # Sum probabilities where `content_vector` is in top-k
#     prob_for_content = (topk_prob * is_content_in_topk).sum(dim=1)  # Shape: (N_users,)
#     return prob_for_content

# Numpy versions below
def random_probability_np(content_vector:np.ndarray, remaining_array:np.ndarray, user_array:np.ndarray, temp = 1)->np.ndarray: # hacky fix for now, adding temp here which is to make function arguments similar to softmax_probability
    
    ''' 
        content_vector : shape is (dimension,)
        remaining_array: shape in (N_producers - 1, dimension)
        user_array: shape is (N_users, dimension)
        Returns
            numpy array of shape (N_user,), proabibility of each user seeing producer j's content (the producer who sets their vector)  
    '''
    Nprod = (remaining_array.shape[0] + 1)
    Nuser = user_array.shape[0]
    return np.full(Nuser, 1.0/Nprod)

def linear_probability_np(content_vector:np.ndarray, remaining_array:np.ndarray, user_array:np.ndarray, temp = 1)->np.ndarray: # hacky fix for now, adding temp here which is to make function arguments similar to softmax_probability
    ''' 
        content_vector : shape is (dimension,)
        remaining_array: shape in (N_producers - 1, dimension)
        user_array: shape is (N_users, dimension)
        Returns
            numpy array of shape (N_user,), linear proabibility of each user getting recommended to `content_vector` (the producer who sets their vector)  
    '''
    product = user_array @ np.vstack((remaining_array, content_vector)).T # has shape N_user x N_prod, product_ij stores what user i rates producer j
    prob =  product / product.sum(axis=1)[:,None] # the [:, None] just reshapes the product.sum to (N_users, 1) for broadcast division
    return prob[:, -1] #prob_ij contains with what probability user i is recommended movie j; last column will have all probabilities of users going to content_vector

def softmax_probability_np(content_vector:np.ndarray, remaining_array:np.ndarray, user_array:np.ndarray, temp = 1)->np.ndarray:
    '''
        content_vector : shape is (dimension,)
        remaining_array: shape in (N_producers - 1, dimension)
        user_array: shape is (N_users, dimension)
        temp: temperature for softmax, default = 1
        Returns
            numpy array of shape (N_user,), softmax proabibility of each user getting recommended to content_vector   
    '''
    product = np.exp((user_array @ np.vstack((remaining_array, content_vector)).T) / temp) # has shape N_user x N_prod, # TODO temperature added
    prob = product / product.sum(axis=1)[:, None] # the [:, None] just reshapes the product.sum to (N_users, 1) for broadcast division
    return prob[:, -1]