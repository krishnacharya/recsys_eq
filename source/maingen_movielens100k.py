from SaveEmbeddings import save_movielens100k
import argparse

def main():
    parser = argparse.ArgumentParser(description='get experiment configs')
    parser.add_argument('--dimension', type = int, help = 'dimension of factorization')
    parser.add_argument('--seed', type = int, help='random seed to set')
    args = parser.parse_args()
    save_movielens100k(dimension = args.dimension, seed=args.seed)

if __name__ == '__main__':
    main()