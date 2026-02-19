import pandas as pd
import os
import sys

# Ensure config can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import config.settings as config

csv_path = os.path.join(config.PROJECT_ROOT, 'Loop_PTB7_maw.csv')

if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
    # print("Columns in the CSV file:", data.columns)
else:
    print(f"Warning: {csv_path} not found. Data not loaded.")
    data = None
