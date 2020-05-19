import datetime
from config import Config
import os
import csv
import pandas as pd

# now = datetime.now()
# file_id = now.strftime("%m/%d/%Y, %H:%M:%S")
results_path = 'results/'

class Results_Writer:
    def __init__(self):
        self.results_folder_check()
        
    def write_results(self, file_name: str, **results: dict): # or maybe just give the results as dict
        #and maybe give config
        # add checks if fieldnames and results match
 
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

        dt.to_csv(f'{results_path}POS_probe/{file_name}.csv')
        
    
    def results_folder_check(self):
        if not os.path.exists('results'):
            os.mkdir('results')

        results_folders = ['POS_probe', 'Edge_probe','Structural_probe']
        for results_folder in results_folders:
            print('hello')
            if not os.path.exists(f'results/{results_folder}'):
                os.mkdir(f'results/{results_folder}')
        