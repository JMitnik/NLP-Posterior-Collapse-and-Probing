from typing import List
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, field
import utils
import pandas as pd
import os

@dataclass
class ResultsWriter:
    label: str
    path_to_results_folder: str

    # Tensorboard writer
    run_dir: str = field(default_factory=lambda: utils.generate_run_name())
    tensorboard_writer: SummaryWriter = field(default_factory=lambda: SummaryWriter(run_dir=run_dir, comment=label))

    # Dataframes to store the results
    df_train_batch_results: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    df_train_epoch_results: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    # Dataframes to store validation results
    df_valid_results: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    def add_train_batch_results(self, train_results_dict):
        """
        Adds train results to the dataframe and tensorboard
        """
        # Write all results
        self.df_train_batch_results = self.df_train_batch_results.append({'label': self.label, **train_results_dict}, ignore_index=True)

        # Write to tensorboard
        self.add_tensorboard_scalars_from_dict(train_results_dict, 'train')

    def add_tensorboard_scalars_from_dict(self, results_dict, mode):
        """
        Writes to tensorboard based on results-dict, given properties such as loss or metric in key
        """
        model_name = results_dict['model_name']
        it = results_dict['it']

        for key, value in results_dict.items():
            if 'loss' or 'metric' in key:
                try:
                    self.tensorboard_writer.add_scalar(f'{mode}-{model_name}/{key}', value, it)
                except Exception as e:
                    print(f'Cant log in tensorboard for {key}:{value}, in mode {mode}, due to {str(e)}')

    def add_valid_results(self, valid_results_dict):
        """
        Adds valid results to the dataframe and tensorboard
        """
        # Write all results
        self.df_train_batch_results = self.df_train_batch_results.append({'label': self.label, **valid_results_dict}, ignore_index=True)

        # Write scalars to tensorboard: if loss or metric in name, assume it is a scalar
        self.add_tensorboard_scalars_from_dict(valid_results_dict, 'train')

    def add_sentence_predictions(self, predictions, truth, it):
        """
        Adds sentence predictions to the dataframe and tensorboard
        """
        self.tensorboard_writer.add_text(f'it {it}: prediction', predictions)
        self.tensorboard_writer.add_text(f'it {it}: truth', truth)

    def save_train_results(self):
        """
        Store train results to hard drive
        """
        filename = f'{self.run_dir}/train.csv'

        if not os.path.exists(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.df_train_batch_results.to_csv(filename, header=False)
            return

        self.df_train_batch_results.to_csv(filename, mode='a', header=False)

    def save_valid_results(self):
        """
        Store validation results to hard drive
        """
        filename = f'{self.run_dir}/valid.csv'

        if not os.path.exists(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.df_train_batch_results.to_csv(filename, header=False)
            return

        self.df_valid_results.to_csv(filename, mode='a', header=False)


