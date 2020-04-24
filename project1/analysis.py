# %%
import pandas as pd
from pathlib import Path

csv_paths = list(Path('results/runs').glob('**/valid.csv'))
all_results = pd.concat([pd.read_csv(path) for path in csv_paths])

all_results

# %%

