# import sys
# sys.path.append('../source/')
import argparse
from utils import load_config
from game.Embeddings import * # get Synth_Uniform_Embedding, Synth_Skewed_Embedding, Movielens_100k_Embedding classes
from numiter-runs-exp1.run import run_numiters
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='get experiment configs')
    parser.add_argument('--common_config', type=str, default = 'common_config', help='Path to the common config file')
    parser.add_argument('--nusers', type = int, default = 10000, help = 'number of users, used in synthetic data generation')
    parser.add_argument('--data', type = str, help='Name of the data you want to use')
    parser.add_argument('--prob', type = str, help= 'Kind of probability - softmax or linear')
    parser.add_argument('--temperature', type = float, default = 1.0, help = 'Temperature parameter')
    parser.add_argument('--emb_seed', type = int, default = 17, help = 'Embedding seed')
    parser.add_argument('--runnum', type = str, help = 'run number, each run is of BR dynamics for a given dim, number of producers, nusers')
    parser.add_argument('--save_dir', type = str, default = '../numiters_savedframe/', help= 'directory in which to store the generated dataframe for utility, NE')
    args = parser.parse_args()

    common_config = load_config('../configs/'+str(args.common_config)+'.yml')
    Embedding = None # class name that is data specific
    if args.data == 'synth-uniform':
        emb_obj = Synth_Uniform_Embedding() # assigning class name
    elif args.data == 'synth-skewed':
        emb_obj = Synth_Skewed_Embedding()
    elif args.data == 'movielens-100k':
        emb_obj = Movielens_100k_Embedding()
    elif args.data == 'rentrunway':
        emb_obj = RentRunway_Embedding()
    elif args.data == 'amznmusic':
        emb_obj = AmazonMusic_Embedding()
    elif args.data == 'sparse-unif':
        emb_obj = SparseUni(spfrac=args.spfrac)
    elif args.data == 'sparse-skew':
        emb_obj = SparseSkew(spfrac=args.spfrac)
    else:
        raise NotImplementedError
    
    if args.prob not in ['random', 'linear', 'softmax']:
        raise NotImplementedError
    
    print(f'Temperature is {args.temperature}')

    final_dir = args.save_dir + f'embseed{args.emb_seed}/'+ f'{args.data}_{args.prob}_new_temp_{args.temperature}'
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    final_dest = final_dir + '/run_' + args.runnum + '.pkl'
    run_numiters(args.runnum, common_config['dimensions'], args.emb_seed, \
    common_config['n_prodarr'], emb_obj, args.prob, args.temperature, args.nusers, final_dest)

if __name__ == '__main__':
    main()