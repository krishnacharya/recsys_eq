import numpy as np
from User import * #Class for users, stores num_users, user_dimension

def linear_probability(content_vector:np.ndarray, remaining_array:np.ndarray, user_array:np.ndarray)->np.ndarray:
    '''
        content_vector : shape is (dimension,)
        remaining_array: shape in (N_producers - 1, dimension)
        user_array: shape is (N_users, dimension)
        Returns
            numpy array of shape (N_user,), linear proabibility of each user getting recommended to content_vector    
    '''
    product = user_array @ np.vstack((remaining_array, content_vector)).T # has shape N_user x N_prod, product_ij stores what user i rates producer j
    prob =  product / product.sum(axis=1)[:,None] # the [:, None] just reshapes the product.sum to (N_users, 1) for broadcast division
    return prob[:, -1] #prob_ij contains with what probability user i is recommended movie j; last column will have all probabilities of users going to content_vector

def softmax_probability(content_vector:np.ndarray, remaining_array:np.ndarray, user_array:np.ndarray)->np.ndarray:
    '''
        content_vector : shape is (dimension,)
        remaining_array: shape in (N_producers - 1, dimension)
        user_array: shape is (N_users, dimension)
        Returns
            numpy array of shape (N_user,), softmax proabibility of each user getting recommended to content_vector   
    '''
    product = np.exp(user_array @ np.vstack((remaining_array, content_vector)).T) # has shape N_user x N_prod
    prob = product / product.sum(axis=1)[:, None] # the [:, None] just reshapes the product.sum to (N_users, 1) for broadcast division
    return prob[:, -1]

def engagement_utility(content_vector, remaining_array, users:Users, prob="linear"):
  #Args:
  # Content - The index of the vector representing the content in the producer_array
  # producer_array - a num_prodcuers x d array representing each contetn produced in the system
  # users_array - an num_users x d array represetning all_users and their preferences
  # probability - a function that maps the content and producer_arrary to probability for a piece of content being shown to each user ( 1x num_users vector)
  if prob == "linear":
    probability_function = linear_probability
  elif prob == "exponential":
    probability_function = exponential_probability
  #get probabilty for each user
  probs = probability_function(content_vector, remaining_array, users.user_array)
  #get value for each user
  prods = users.user_array @ content_vector
  #since we define all users, we can just take an average
  #TODO: make a version of this for a distribution of users
  return sum(probs*prods)/users.num_users

def get_best_response(remaining_array, users:Users, prob="linear"):
  '''
    remaining_array has shape (#producers-1, d)
  '''
  #gets the best response to the other contents given the user array
  #TODO: this is probably dumb and there is a better way of doing it

  # Args:
  # remaining_array - the array representing the content of the remaining producers
  # users_array - an num_users x d array representing all_users and their preferences
  max_util = -1
  best_row = None
  for row in np.eye(users.user_dimension):
    util = engagement_utility(row, remaining_array, users.user_array, prob=prob)
    if util > max_util:
      best_row = row
      max_util = util
  return best_row

def eval_position(producer_array, users:Users, prob="linear"):
  #returns True if all content vectors are the best response to one another given the user array
  for i, content in enumerate(producer_array):
    row = get_best_response(np.delete(producer_array, i, 0), users.user_array, prob)
    if not np.all(row==content):
      return False
  return True

##This is slow, so skip it if you dont need it
#This generates all possible combinations of producers given the dimesnsion and number of producers

def generate_equilibria(users:Users):
  def equivalent_combinations(combinations):
    s = set()
    for c in combinations:
      s.add(tuple(sorted(c)))
    return s
  results = []
  combinations = np.array(np.meshgrid(*[np.arange(users.user_dimension) for i in range(num_producers)])).T.reshape(-1, num_producers)
  eq_comb = equivalent_combinations(combinations) # equivalent combinations, just permutation invariant ones of the above
  I = np.eye(users.user_dimension)
  for combination in eq_comb:
    producer_array = I[list(combination)]
    if eval_position(producer_array, users.user_array, prob='linear'):
      results.append(tuple(np.sum(producer_array, axis = 0)))
  return list(set(results))

def search_best_response(users=None, producers=np.eye(users.user_dimension)[np.random.choice(users.user_dimension, num_producers)], prob="linear"):
    #working on vectorizing this
    #gets the best reponse for each producer
    idx = np.arange(1, num_producers) - np.tri(num_producers, num_producers-1, k=-1, dtype=bool)
    iter = list(range(num_producers))
    random.shuffle(iter)
    for i in iter:
      row = get_best_response(producers[idx][i], users, prob)
      if not np.all(producers[i] == row):
        producers[i] = row
        return producers, False
    return producers, True

def get_equilibrium():
  #runs iters of producers searching for linear best reponse
  converged = False
  print("##### PRODUCERS FOR ITER 0\n")
  producers=np.eye(users.user_dimension)[np.random.choice(users.user_dimension, num_producers)]
  #producers = np.array([[1,0,0,0,0] for i in range(num_producers)])
  print(producers)
  print("\n")
  i = 1
  while True:
    print(f"##### PRODUCERS FOR ITER {i}\n")
    producers, converged = search_best_response(users.user_array, producers, prob="exponential")
    print(producers)
    print("\n")
    if converged:
      print(f"Number of iterations to coverge: {i}")
      return np.sum(producers, axis=0)
    i += 1