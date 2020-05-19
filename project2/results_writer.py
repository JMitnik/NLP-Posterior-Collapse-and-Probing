import datetime
from config import Config
import os
import csv
import pandas as pd

# now = datetime.now()
# file_id = now.strftime("%m/%d/%Y, %H:%M:%S")
results_path = 'results/'

class Results_Writer:
    def __init__(self, config: Config):
        self.results_folder_check()
        self.config = config
        
    def write_results(self, probe_type:str, file_name:str='', **results: dict): # or maybe just give the results as dict
        #and maybe give config
        # add checks if fieldnames and results match
        assert probe_type in ['POS', 'Edge', 'Structural'], 'Model type should be: POS, Edge or Structural'
        results = results['results']

        longest = max([len(r) for r in results.values()])
        print(longest)

        # Pad results just in case there are lists longer than others
        for k, v in results.items():
            old_length = len(v)
            v.extend([v[-1] for _ in range(longest - old_length)])
            results[k] = v
        
        dt = pd.DataFrame.from_dict(results)

        print(dt)
        if file_name == '':
            file_name = self.create_file_name(probe_type)
        dt.to_csv(f'{results_path}{probe_type}_probe/{file_name}.csv')
        
    def create_file_name(self, probe_type: str) -> str:
        file_name: str = 'unspecified'

        if probe_type == 'POS':
            model_type = 'linear' if self.config.pos_probe_linear else 'ML1'
            file_name = (f'POS_{model_type}_lr{self.config.pos_probe_train_lr}_'
                                f'patience{self.config.pos_probe_train_patience}_'
                                f'epochs{self.config.pos_probe_train_epoch}_'
                                f'bs{self.config.pos_probe_train_batch_size}')
        elif probe_type == 'Edge':
            file_name = 'nothing in config yet'
        else:
            file_name = (f'POS_lr{self.config.struct_probe_lr}_'
                                f'patience{self.config.struct_probe_train_patience}_'
                                f'epochs{self.config.struct_probe_train_epoch}_'
                                f'bs{self.config.struct_probe_train_batch_size}_'
                                f'embDim{self.config.struct_probe_emb_dim}_'
                                f'rank{self.config.struct_probe_rank}_'
                                f'trainfac{self.config.struct_probe_train_factor}_')
        return file_name
    
    def results_folder_check(self):
        if not os.path.exists('results'):
            os.mkdir('results')
        if not os.path.exists('results/images'):
            os.mkdir('results/images')
        results_folders = ['POS_probe', 'Edge_probe','Structural_probe']
        for results_folder in results_folders:
            print('hello')
            if not os.path.exists(f'results/{results_folder}'):
                os.mkdir(f'results/{results_folder}')
        