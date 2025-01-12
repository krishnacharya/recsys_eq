import glob
import pandas as pd
import argparse
import os

def concat_across_seeds_save(base_dir: str):
    """
    Concatenates DataFrames across embedding seeds and saves them as a single file per directory.
    Args:
        base_dir (str): The base directory where the topk directories are located.
    """
    def get_concatembs_infolder(foldername: str) -> (pd.DataFrame, int):
        """
        Concatenates DataFrames starting with embseed and returns
        the concatenated DataFrame and the number of seeds.
        Args:
            foldername (str): Path to the directory containing embseed files.
        Returns:
            tuple: Concatenated DataFrame, number of seeds
        """
        # Collect all embseed*.pkl files
        embseed_files = glob.glob(os.path.join(foldername, 'embseed_*.pkl'))
        if not embseed_files:
            print(f"No embseed files found in {foldername}. Skipping...")
            return None, 0

        # Read and concatenate all the pickled DataFrames
        pd_list = [pd.read_pickle(file) for file in embseed_files]
        concat_df = pd.concat(pd_list)
        return concat_df, len(pd_list)

    # Iterate over all directories in the base_dir
    for root, dirs, _ in os.walk(base_dir):
        for subdir in dirs:
            folder_path = os.path.join(root, subdir)
            print(f"Processing folder: {folder_path}")
            
            # Concatenate and save the DataFrame
            df, num_seeds = get_concatembs_infolder(folder_path)
            if df is not None and num_seeds > 0:
                output_file = os.path.join(folder_path, f'concseeds{num_seeds}.pkl')
                df.to_pickle(output_file)
                print(f"Saved concatenated DataFrame to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Concatenate DataFrames across seeds')
    parser.add_argument('--base_dir', type=str, help='Base directory containing topk subdirectories', required=True)
    args = parser.parse_args()

    concat_across_seeds_save(args.base_dir)

if __name__ == '__main__':
    main()
