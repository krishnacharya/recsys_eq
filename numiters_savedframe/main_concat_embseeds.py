import glob
import pandas as pd
import argparse

def concat_across_runs_save(folder, data:str):
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
        df, num_runs = get_concatruns_infolder(folder +  data + suffix + '/')
        df.to_pickle(folder + data + suffix + f'/concruns{num_runs}.pkl')

def main():
    parser = argparse.ArgumentParser(description='get experiment configs')
    parser.add_argument('--data', type = str, help='Name of the data you want to use')
    args = parser.parse_args()
    print(args.data)
    concat_across_runs_save('./embseed13/', args.data)
    concat_across_runs_save('./embseed17/', args.data)
    concat_across_runs_save('./embseed19/', args.data)
    concat_across_runs_save('./embseed23/', args.data)
    concat_across_runs_save('./embseed29/', args.data)

if __name__ == '__main__':
    main()