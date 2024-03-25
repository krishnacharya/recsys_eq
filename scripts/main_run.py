import argparse
from utils import load_config

DATA_SOURCE = ['synth_uniform', 'synth_skewed', 'real_movielens-100k']

def main():
    parser = argparse.ArgumentParser(description='get experiment configs')
    parser.add_argument('--common_config', type=str, default = 'common_config', help='Path to the common config file')
    parser.add_argument('--data', type = str, help='Name of the data you want to use')
    args = parser.parse_args()

    common_config = load_config('../configs/'+str(args.common_config)+'.yml') # dictionary with common seeds, dimension, nprods
    Embedding = None # TODO make embedding object class
    if args.data == 'synth-uniform':
        Embedding = 
    elif args.data == 'synth-skewed':
        Embedding = 
    elif args.data == 'real_movielens-100k'
        Embedding = 
    else:
        raise NotImplementedError
    
    
    
if __name__ == '__main__':
    main()