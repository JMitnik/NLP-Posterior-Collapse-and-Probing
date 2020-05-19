import datetime
from config import Config
import os
import csv

# now = datetime.now()
# file_id = now.strftime("%m/%d/%Y, %H:%M:%S")
results_path = 'results/'
fieldnames_POS_probe = ['model_params',
                        'LSTM_linear_acc',
                        'LSTM_linear_select',
                        'LSTM_linear_total_epochs',
                        'LSTM_linear_corrupted_total_epochs',
                        'LSTM_ML1_acc',
                        'LSTM_ML1_select',
                        'LSTM_ML1_total_epochs',
                        'LSTM_ML1_corrupted_total_epochs',
                        'Trans_linear_acc',
                        'Trans_linear_select',
                        'Trans_simple_total_epochs',
                        'Trans_simple_corrupted_total_epochs',
                        'Trans_ML1_acc',
                        'Trans_ML1_select',
                        'Trans_ML1_total_epochs',
                        'Trans_ML1_corrupted_total_epochs']

class Results_Writer:
    def __init__(self, config: Config):
        self.results_folder_check()
        pass

    def write_POS_probe_results(self, 
                                validation_acc: float, 
                                validation_selec: float, 
                                training_epochs: int, 
                                c_training_epochs: int):
        """ """
        print(f'{validation_acc}, {validation_selec}, {training_epochs}, {c_training_epochs}')
        


    def generate_file_name(self, probe_type: str, model_type: str, probe_model_type: str) -> str:
        file_name = f'{probe_type}_{model_type}_{probe_model_type}'
        return file_name
        

    def results_folder_check(self):
        if not os.path.exists('results'):
            os.mkdir('results')