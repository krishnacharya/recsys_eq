import sys
sys.path.append('../source/')
import argparse
from utils import load_config
from Embeddings import * # get Synth_Uniform_Embedding, Synth_Skewed_Embedding, Movielens_100k_Embedding classes
from run import run_numiters
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='get experiment configs')
    parser.add_argument('--common_config', type=str, default = 'common_config', help='Path to the common config file')
    parser.add_argument('--nusers', type = int, default = 10000, help = 'number of users, used in synthetic data generation')
    parser.add_argument('--data', type = str, help='Name of the data you want to use')
    parser.add_argument('--prob', type = str, help= 'Kind of probability - softmax or linear')
    parser.add_argument('--temperature', type = float, default = 1.0, help = 'Temperature parameter')
    # parser.add_argument('--exp_seed', type = int, default = 13, help = 'Seed for experiment')
    parser.add_argument('--emb_seed', type = int, default = 17, help = 'Embedding seed')
    parser.add_argument('--runnum', type = str, help = 'run number, each run is of BR dynamics for a given dim, number of producers, nusers')
    parser.add_argument('--save_dir', type = str, default = '../numiters_savedframe/', help= 'directory in which to store the generated dataframe for utility, NE')
    args = parser.parse_args()

    common_config = load_config('../configs/'+str(args.common_config)+'.yml')
    Embedding = None # class name that is data specific
    if args.data == 'synth-uniform':
        Embedding = Synth_Uniform_Embedding # assigning class name
    elif args.data == 'synth-skewed':
        Embedding = Synth_Skewed_Embedding
    elif args.data == 'movielens-100k':
        Embedding = Movielens_100k_Embedding
    elif args.data == 'rentrunway':
        Embedding = RentRunway_Embedding
    elif args.data == 'amznmusic':
        Embedding = AmazonMusic_Embedding
    else:
        # print("Dataset not defined")
        raise NotImplementedError
    
    if args.prob not in ['linear', 'softmax']:
        # print("Probability not defined")
        raise NotImplementedError
    print(f'Temperature is {args.temperature}')

    final_dir = args.save_dir + f'{args.data}_{args.prob}_temp_{args.temperature}'
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    final_dest = final_dir + '/run_' + args.runnum + '.pkl'
    run_numiters(args.runnum, common_config['dimensions'], args.emb_seed, \
    common_config['n_prodarr'], Embedding, args.prob, args.temperature, args.nusers, final_dest)

if __name__ == '__main__':
    main()