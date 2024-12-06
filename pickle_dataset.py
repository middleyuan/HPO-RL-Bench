import os
from itertools import product
import pandas as pd

from benchmark_handler import BenchmarkHandler

benchmark = BenchmarkHandler()
dataset_path = 'data_arl_bench'
rl_list = os.listdir(dataset_path)
rl_search_spaces = {rl: benchmark.get_search_space(rl) for rl in rl_list}
hyperparameter_set = list(set([key for rl in rl_list for key in rl_search_spaces[rl].keys()]))
envs = {rl: os.listdir(os.path.join(dataset_path, rl)) for rl in rl_list}
budgets = [i for i in range(1, 101)]
seeds = [i for i in range(3)]
metrics = ["eval_avg_returns", "eval_std_returns", "eval_timestamps", "eval_timesteps"]


data_list = []

for rl in rl_list:
    search_spaces = rl_search_spaces[rl]
    config_combination = list(product(*search_spaces.values()))
    param_names = search_spaces.keys()
    config_dicts = [dict(zip(param_names, values)) for values in config_combination]
    
    for config in config_dicts:
            for env in envs[rl]:
                for budget in budgets:
                    for seed in seeds:
                        metric_values = benchmark.get_metrics(config, rl, env, seed, budget)
                        metric_value_list = [metric_values[metric] for metric in metrics]
                        config_list = [config[hp] if hp in config else None for hp in hyperparameter_set]
                        data_list.append([rl, env, budget, seed] + metric_value_list + config_list)

# Create the DataFrame from the collected data
dataset_df = pd.DataFrame(data_list, columns=["rl", "env", "budget", "seed"] + metrics + hyperparameter_set)
# Pickle the DataFrame
dataset_df.to_pickle("dataset.pkl")