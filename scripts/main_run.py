import sys
sys.path.append('../source/')
import argparse
from utils import load_config
from Embeddings import * # get Synth_Uniform_Embedding, Synth_Skewed_Embedding, Movielens_100k_Embedding classes
from run import *

DATA_SOURCE = ['synth_uniform', 'synth_skewed', 'real_movielens-100k']

def main():
    parser = argparse.ArgumentParser(description='get experiment configs')
    parser.add_argument('--common_config', type=str, default = 'common_config', help='Path to the common config file')
    parser.add_argument('--nusers', type = int, default = 10000, help = 'number of users, used in synthetic data generation')
    parser.add_argument('--data', type = str, help='Name of the data you want to use')
    parser.add_argument('--prob', type = str, help= 'Kind of probability - softmax or linear')
    parser.add_argument('--temperature', type = float, default = 1.0, help = 'Temperature parameter')
    parser.add_argument('--save_dir', type = str, default = '../saved_frames/', help= 'directory in which to store the generated dataframe for utility, NE')
    args = parser.parse_args()

    common_config = load_config('../configs/'+str(args.common_config)+'.yml') # dictionary with common seeds, dimension, nprods
    Embedding = None # class name that is data specific
    if args.data == 'synth-uniform':
        Embedding = Synth_Uniform_Embedding # assigning class name
    elif args.data == 'synth-skewed':
        Embedding = Synth_Skewed_Embedding
    elif args.data == 'movielens-100k':
        Embedding = Movielens_100k_Embedding
    else:
        # print("Dataset not defined")
        raise NotImplementedError
    
    if args.prob not in ['linear', 'softmax']:
        # print("Probability not defined")
        raise NotImplementedError
    
    print(f'Temperature is {args.temperature}')
    
    save_file_name = f'{args.data}_{args.prob}_temp_{args.temperature}_{args.common_config}.pkl'
    final_location = args.save_dir + save_file_name
    # print(final_location)
    # these 3 are trial runs
    # emb_obj = Embedding(seed = 11, dimension = 6, num_users = 3)
    # # print(emb_obj.nue)
    # print(emb_obj.nue.sum(axis=1))
    # print(emb_obj.nue.shape, emb_obj.num_users)
    # print(args.nusers, args.prob)
    # print(args.temperature)
    run_producer_game(common_config['dimensions'], common_config['seeds'], common_config['n_prodarr'], \
                    Embedding, args.prob, args.temperature, args.nusers, final_location)

if __name__ == '__main__':
    main()