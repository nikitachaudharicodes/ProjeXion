import pandas as pd
import numpy as np
from pathlib import Path
import json

checkpoints_path = Path('checkpoints')
run_names = [path for path in checkpoints_path.iterdir() if path.is_dir()]

experiments = []
for run_name in run_names:
   with (run_name / 'metrics.json').open('r') as metrics_file:
      metrics = json.load(metrics_file)
   with (run_name / 'config.json').open('r') as config_file:
      config = json.load(config_file)
   experiments.append(metrics | config)

experiments = pd.DataFrame.from_records(
   experiments
)
experiments.to_csv('results.csv')
   