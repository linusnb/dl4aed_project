# %%
from dataset_generator import Dataset as ds
import glob
import os
from itertools import cycle
import sys
#%%
db = ds('_data/dataset_config.json', 'OtherWAV')
seed_path = os.path.join(db._seed_dir, '**/*.wav')
# %%
fps = glob.glob(seed_path, recursive=True)
for file in fps[:20]:
    # Add item to this codec:
    n_file = file.replace(' ', '_').replace('(', '').replace(')', '').replace(
              '\'', '').replace('&', '_')
    os.rename(file, n_file)
    db.add_item(n_file)



# %%
