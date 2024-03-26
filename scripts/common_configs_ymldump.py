import yaml
import numpy as np
# GENERATES common configs used by all our experiments
# The common configs for all the experiments are seeds, dimensions and num producer array, 
# type of probability (softamx, linear), temperature, dataset, num of users are experiment specific inputs
di = {}

di['seeds'] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
di['dimensions'] = [5, 10, 15, 20] # number of dimensions for latent/embedding
di['n_prodarr'] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # number of producers

toy = {} # for workflow sanity check
toy['seeds'] = [2, 11, 13]
toy['dimensions'] = [5, 10]
toy['n_prodarr'] = [10, 20]

with open('common_config.yml', 'w') as outfile:
    yaml.dump(di, outfile, default_flow_style=False, sort_keys=False)

with open('toy_config.yml', 'w') as outfile:
    yaml.dump(toy, outfile, default_flow_style=False, sort_keys=False)
