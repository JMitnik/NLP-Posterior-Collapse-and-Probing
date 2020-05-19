import datetime
from typing import Dict
from config import Config
import os
import csv
import pandas as pd

# now = datetime.now()
# file_id = now.strftime("%m/%d/%Y, %H:%M:%S")
results_path = 'results/'

class ResultsWriter:
    def __init__(self, config: Config):
        self.results_folder_check()
        self.config = config

    def write_results(self, probe_type: str, feature_model_type: str, file_name: str = '', results: Dict[str, list] = {}):
        # Ensure probe_type matches support models
        assert probe_type in ['POS', 'dep_edge', 'struct'], 'Model type should be: POS, dep_edge or struct'

        longest = max([len(r) for r in results.values()])

        # Pad results just in case there are lists longer than others
        for k, v in results.items():
            old_length = len(v)
            v.extend([v[-1] for _ in range(longest - old_length)])
            results[k] = v

        meta_results = {
            'probe_type': probe_type,
            'feature_model_type': feature_model_type
        }

        final_results = {**meta_results, **results, **self.config.to_dict()}

        df = pd.DataFrame.from_dict(final_results)

        if file_name == '':
            file_name = self.create_file_name(feature_model_type, probe_type)

        df.to_csv(f'{results_path}{probe_type}_probe/{file_name}.csv')

    def create_file_name(self, feature_model_type: str, probe_type: str) -> str:
        file_name: str = 'unspecified'

        if probe_type == 'POS':
            probe_model_type = 'linear' if self.config.pos_probe_linear else 'ML1'
            file_name = (f'POS_{self.config.run_label}_{feature_model_type}_{probe_model_type}_lr{self.config.pos_probe_train_lr}_'
                                f'patience{self.config.pos_probe_train_patience}_'
                                f'epochs{self.config.pos_probe_train_epoch}_'
                                f'bs{self.config.pos_probe_train_batch_size}')
        elif probe_type == 'dep_edge':
            file_name = (f'dep_edge_{self.config.run_label}_{feature_model_type}_lr{self.config.struct_probe_lr}_'
                                f'rank{self.config.struct_probe_rank}_'
                                f'epochs{self.config.struct_probe_train_epoch}_'
                                f'bs{self.config.struct_probe_train_batch_size}')
        elif probe_type == 'struct':
            file_name = (f'struct_{self.config.run_label}_{feature_model_type}_lr{self.config.struct_probe_lr}_'
                                f'rank{self.config.struct_probe_rank}_'
                                f'epochs{self.config.struct_probe_train_epoch}_'
                                f'bs{self.config.struct_probe_train_batch_size}')
        else:
            print("Not certain what probe is used, defaults to POS")
            file_name = (f'POS_{self.config.run_label}_lr{self.config.struct_probe_lr}_'
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

        results_folders = ['POS_probe', 'dep_edge_probe','struct_probe']

        for results_folder in results_folders:
            if not os.path.exists(f'results/{results_folder}'):
                os.mkdir(f'results/{results_folder}')
