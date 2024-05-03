import glob
import pandas as pd
import argparse

def concat_across_runs_save(data:str):
    '''
        concatenates df across runs, and saves
    '''
    def get_concatruns_infolder(foldername:str) -> (pd.DataFrame, int):
        '''
            concatenates dataframes starting with run_
            returns
                concatenated dataframe, number of runs
        '''
        pd_list = [pd.read_pickle(file) for file in glob.glob(foldername + 'run_*.pkl')]
        concat_df = pd.concat(pd_list)
        return concat_df, len(pd_list)

    suffixes = ['_linear_temp_1.0',  '_softmax_temp_1.0']
    for suffix in suffixes:
        df, num_runs = get_concatruns_infolder('./'+ data + suffix + '/')
        df.to_pickle('./' + data + suffix + f'/concruns{num_runs}.pkl')

def main():
    parser = argparse.ArgumentParser(description='get experiment configs')
    parser.add_argument('--data', type = str, help='Name of the data you want to use')
    args = parser.parse_args()

    concat_across_runs_save(args.data)

if __name__ == '__main__':
    main()