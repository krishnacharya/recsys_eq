import sys
sys.path.append('../source/')
from utils_source import load_config
import pandas as pd
from pandas import DataFrame

common_config = load_config('common_config.yml') # dictionary with common seeds, dimension, nprods

df_dim = DataFrame({'dimension': common_config['dimensions']})
df_seed = DataFrame({'seed': common_config['seeds']})

df = df_dim.merge(df_seed, how='cross')
print(df)
df.to_csv("./file.csv", sep=',',index=False)