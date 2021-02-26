# %%
from dataset_generator import Dataset as ds
import glob
import os

# %%
# Create dataset object with path to json file and folder name of dataset
db = ds('../_data/dataset_config.json', 'MedleyDB')

# Path to seed files
seed_path = os.path.join(db._seed_dir, '**/*.wav')

# %% Search in seed directory and add files to database
fps = glob.glob(seed_path, recursive=True)
for file in fps:
    # Add item to this codec:
    db.add_item(file)

# Print stats of new dataset:
dict, total = db.get_stats()
