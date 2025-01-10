from SaveEmbeddings import save_rentrunway
import argparse

def main():
    parser = argparse.ArgumentParser(description='get experiment configs')
    parser.add_argument('--dimension', type = int, help = 'dimension of factorization')
    parser.add_argument('--seed', type = int, help='random seed to set')
    args = parser.parse_args()
    save_rentrunway(dimension = args.dimension, seed=args.seed)

if __name__ == '__main__':
    main()