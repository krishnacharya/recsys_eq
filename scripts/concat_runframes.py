import glob
print(glob.glob("/home/adam/*"))
import argparse
import pandas as pd

def concat_in_dir(dirname):
    '''
        Concatenates all pandas dataframes in this directory,
        TODO fix later to not again concatenate final
    '''
    files = glob.glob(dirname + "/run_*")
    pd_list = [pd.read_pickle(file) for file in files]
    final_df = pd.concat(pd_list)
    final_df.to_pickle(dirname + "/allruns.pkl")

def main():
    parser = argparse.ArgumentParser(description='get experiment configs')
    parser.add_argument('--nusers', type = int, default = 10000, help = 'number of users, used in synthetic data generation')
    parser.add_argument('--data', type = str, help='Name of the data you want to use')
    parser.add_argument('--prob', type = str, help= 'Kind of probability - softmax or linear')
    parser.add_argument('--temperature', type = float, default = 1.0, help = 'Temperature parameter')
    parser.add_argument('--seed', type = int, default = 13, help = 'Seed for randomness')
    parser.add_argument('--save_dir', type = str, default = '../numiters_savedframe/', help= 'directory in which to store the generated dataframe for utility, NE')
    args = parser.parse_args()

    final_dir = args.save_dir + f'{args.data}_{args.prob}_temp_{args.temperature}'
    concat_in_dir(final_dir)

if __name__ == '__main__':
    main()