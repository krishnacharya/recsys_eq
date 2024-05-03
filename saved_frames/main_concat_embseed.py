import glob
import pandas as pd
import argparse

def concat_across_seeds_save(data:str):
    '''
        concatenates df across seeds
    '''
    def get_concatembs_infolder(foldername:str) -> (pd.DataFrame, int):
        '''
            concatenates dataframes starting with embseed
            returns
                concatenated dataframe, number of seeds
        '''
        pd_list = [pd.read_pickle(file) for file in glob.glob(foldername + 'embseed*.pkl')]
        concat_df = pd.concat(pd_list)
        return concat_df, len(pd_list)

    suffixes = ['_linear_temp_1.0', '_softmax_temp_100.0', '_softmax_temp_10.0', '_softmax_temp_1.0',\
    '_softmax_temp_0.1', '_softmax_temp_0.01']
    for suffix in suffixes:
        df, num_seeds = get_concatembs_infolder('./'+ data + suffix + '/')
        df.to_pickle('./' + data + suffix + f'/concseeds{num_seeds}.pkl')

def main():
    parser = argparse.ArgumentParser(description='get experiment configs')
    parser.add_argument('--data', type = str, help='Name of the data you want to use')
    args = parser.parse_args()

    concat_across_seeds_save(args.data)

if __name__ == '__main__':
    main()