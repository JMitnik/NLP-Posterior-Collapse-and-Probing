import datetime
from config import Config
import os
import csv

now = datetime.now()
file_id = now.strftime("%m/%d/%Y, %H:%M:%S")

fieldnames_POS_probe = ['model_params',
                        'LSTM_linear_acc',
                        'LSTM_linear_select',
                        'LSTM_simple_total_epochs',
                        'LSTM_simple_corrupted_total_epochs',
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

    def write_POS_Probe_Results(self):
        

    def results_folder_check(self):
        if not os.path.exists('results'):
            os.mkdir('results')