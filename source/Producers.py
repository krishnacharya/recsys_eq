from Users import * # A basic Users class
import numpy as np

def random_rec_utilities(num_prod:int, user_array:np.ndarray) -> tuple[float, int, np.ndarray]:
    '''
        p_i(c_k) = 1/num_prod, probability of user going to producer is uniform
        thus the utility for producer i is 1/num_prod * (\sum_k=1^num users <s_i, c_k>)
        it is best for each prodcuer to play cordinate maximizer of \sum_k=1^numuser c_k

        user_array is shape N_users x dimension

        Note all producers utilities and best index will be the same
        returns 
        engagement utility for each producer (same), 
        best start index (in 0...d-1 for all the producers),
        utility for each consumer k will just be 1/Nprod \sum_n=1^Nprod <c_k, best_strat> = c_k(best_strat index)
    '''
    ua_sum = user_array.sum(axis = 0)
    best_strat_index = np.argmax(ua_sum)
    max_cord_sum = ua_sum[best_strat_index]
    return max_cord_sum / num_prod, best_strat_index, user_array[:, best_strat_index]

def linear_probability(content_vector:np.ndarray, remaining_array:np.ndarray, user_array:np.ndarray, temp = 1)->np.ndarray: # hacky fix for now, adding temp here which is to make function arguments similar to softmax_probability
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

def softmax_probability(content_vector:np.ndarray, remaining_array:np.ndarray, user_array:np.ndarray, temp = 1)->np.ndarray:
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
    # print("calling sotfmax_probability", np.array_str(prob, precision=3, suppress_small=True))
    return prob[:, -1]

def engagement_utility(content_vector:np.ndarray, probs:np.ndarray, user_array:np.ndarray) -> float:
    '''
        Computes engagement utility for the content_vector
        Parameters:
            content_vector : shape is (dimension,)
            probs: shape is (N_users,) i^th entry denotes the probability that user i goes to this producer (which has `content vector` embedding)
                this must be precomputed - it could be linear or softmax probability
            user_array: shape is (N_users, dimension)
    
    This function already assumed the probs are precomputed (could be linear, softmax ...)
    '''
    prods = user_array @ content_vector # what each user rates the content_vector shape (N_user,)
    return np.sum(probs * prods)

def producer_exposure_utility_linearserving(content_vector:np.ndarray, remaining_array:np.ndarray, user_array:np.ndarray) -> float:
    '''
        Get the exposure utility for content_vector producer assuming rest producers are frozen to remaning array
        Used for exposure game class

        Skip for now, is for exposure
    '''
    product = user_array @ np.vstack((remaining_array, content_vector)).T # has shape N_user x N_prod, product_ij stores what user i rates producer j
    prob =  product / product.sum(axis=1)[:,None] # the [:, None] just reshapes the product.sum to (N_users, 1) for broadcast division
    return np.sum(prob[:, -1]) # sum across all users of exposure for producer with content vector

def get_all_engagement_utilities(producers:np.ndarray, user_array:np.ndarray, prob_type='linear', temp = 1):
    '''
        Given the producer strategies and user array return the (engagement) utilities for producer and users.

        producers: shape N_producers x dimension
        user_array: shape N_users x dimension
            Note: producers only contains basis vectors e.g [[(0,1,0..d wide), ..(1,0... d wide)... N_producers]]
        temp: temperature for softmax probability
        Returns 
            dir_producers direction of basis vector for each producer, shape (N_producers, )
            engagement utility for each producer, shape (N_producers, )
            engagement utility for each user, shape (N_users, )
    '''
    prodt = producers.T  # shape (dimension, N_prod)
    dir_producers = np.argmax(prodt, axis=0)
    ratings = user_array @ prodt # shape (N_user, N_prod) ratings ij has what user i rates producer j's content <c_i, s_j>
    prob = None
    if prob_type == 'linear':
        prob = ratings / ratings.sum(axis=1)[:, None] # prob_ij stores <c_i, s_j> / sum_k <c_i, s_k> 
    elif prob_type == 'softmax':
        exp_ratings = np.exp(ratings / temp) # TODO temperature added
        prob = exp_ratings / exp_ratings.sum(axis=1)[:, None] # prob_ij stores exp(<c_i, s_j>) / sum_k exp(<c_i, s_k>), prob that user i goes to producer j
        # print("Get all eng utils", np.array_str(prob, precision=3, suppress_small=True))
    else:
        raise NotImplementedError
    utility = prob * ratings # utility_ij = prob_ij * rating_ij, utility producer j gets from user i
    return dir_producers, utility.sum(axis=0), utility.sum(axis = 1)

class ProducersEngagementGame:
    '''
        Each producer's strategy space is the ball of L1 norm <= 1, restricted to positive orthant
        Goal of each producer is to maximize its engagament
    '''
    def __init__(self, num_producers:int, users:Users, prob = 'linear', temp = 1): # TODO added temperature
        self.num_producers = num_producers
        self.dimension = users.dimension
        self.users = users
        self.BR_dyna_NE = set() # Nash equilibria arising from best response dynamics, stores tuples with (n_1...n_d) # of producers in each direction
        self.BruteForce_NE = set() # Nash equilibria arising from brute force vertex search, stores tuples with (n_1...n_d) # of producers in each direction
        self.temp = temp #temperature
        if prob == 'linear':
            self.probability_function = linear_probability
        elif prob == 'softmax':
            self.probability_function = softmax_probability
        else:
            raise NotImplementedError
    
    def get_best_response(self, current_vec: np.ndarray, remaining_array:np.ndarray) -> np.ndarray:
        '''
            Best response for a producer, when all the other producers are frozen to remaining_array
            Parameters:
                current_vec is a numpy array shape (dimension, )
                remaining_array is a numpy array of shape (N_producers - 1, dimension)
            Returns
                best_row is a numpy array of shape (d,), 
                searching amongst positive basis vectors is sufficient 
                due to properties of engagement utility
        '''
        # max_util = -1
        max_util = engagement_utility(current_vec, self.probability_function(current_vec, remaining_array, self.users.user_array, temp = self.temp),  self.users.user_array)
        best_row = current_vec # setting best_row and max utility as what the current vector gives
        for row in np.eye(self.dimension):
            probs = self.probability_function(row, remaining_array, self.users.user_array, temp=self.temp)
            util = engagement_utility(row, probs, self.users.user_array)
            if util > max_util:
                best_row = row
                max_util = util
        return best_row
    
    def find_update_best_response(self, producers: np.ndarray):
        '''
            producers: has shape (N_producers, dimension)
            Returns
                (producers, True) if the input itself is a Nash Equilibrium i.e. each producer is best responding
                
                (producers updated, False) if the input is not a Nash equilibrium
                producers updated differes from producers in exactly one row! 
                the row in which we find 'a' best response.
            Note: We search amongst indices randomly
        '''
        for i in np.random.permutation(self.num_producers): # random permutation of arange(self.num_producers)
            br = self.get_best_response(producers[i], producers[(np.arange(self.num_producers) != i), :]) # picks rows other than i for remaining_array
            if not np.all(producers[i] == br): # found a best response, update and return
                producers[i] = br
                return producers, False
        return producers, True

    def best_response_dynamics(self, max_iter = 500, verbose = False):
        '''
            Single run of best response dynamics starting from random +ve basis vectors for each producer
            Once we hit a Nash Equilibrium/or max_iterations stop
            Returns 
            (converged, last_profile, last_profile_compact, # of iterations of BR dynamics done) 
            converged is True if NE found, False if not
            last_profile is the of shape (N_producers, dimension)
            last_profile compact is the # of users along each basis vector shape (dimensions,) 
            
            if BR dynamics have converged then last_profile will be a NE!
        '''
        producers = np.eye(self.dimension)[np.random.choice(self.dimension, self.num_producers)] # random basis vectors, shape N_prod x dimension
        if verbose: print(f'##### PRODUCERS FOR ITER 0\n {producers.sum(axis=0)}')
        for i in range(max_iter):
            producers, converged = self.find_update_best_response(producers)
            if verbose: print(f'##### PRODUCERS FOR ITER {i} \n {producers.sum(axis=0)}')
            if converged:
                if verbose: print(f'Number of iterations to coverge: {i}')
                self.BR_dyna_NE.add(tuple(np.sum(producers, axis=0)))
                return converged, producers, np.sum(producers, axis=0), i
        return converged, producers, np.sum(producers, axis=0), i # if BR dynamics do not converge
        
    def brute_force_NEsearch(self):
        '''
            Searches all (permutationally invariant) combinations
            and check if it's a NE
            Slow
        '''
        def equivalent_combinations(combinations): 
            '''
                returns a set of tuples, permutationally invariant ones of combinations, by sorting
            '''
            s = set()
            for c in combinations:
                s.add(tuple(sorted(c)))
            return s
        combinations = np.array(np.meshgrid(*[np.arange(self.dimension) for i in range(self.num_producers)])).T.reshape(-1, self.num_producers) # shape [d^(#producers), #producers]
        eq_comb = equivalent_combinations(combinations) # set of tuples
        I = np.eye(self.dimension)
        for comb in eq_comb:
            producers = I[list(comb)] #shape (N_producers, dimension)
            _ , is_NE =  self.find_update_best_response(producers)
            if is_NE:
                self.BruteForce_NE.add(tuple(np.sum(producers, axis = 0))) # adds producer profile, # of producers in each direction to set of NE
        return self.BruteForce_NE