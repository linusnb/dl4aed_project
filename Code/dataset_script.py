#%%
from dataset_generator import Dataset as ds
import glob
import os
from itertools import cycle
#%%
db = ds('dl4aed-project/Code/_data/dataset_config.json')
codec_cycle = cycle(db.get_codec_specifiers())
seed_path = os.path.join(db._seed_dir, '**/*.wav')

fps = glob.glob(seed_path, recursive=True)

# Add two files to each codec specifier:
for file in fps[:18]:
    db.add_item(file, next(codec_cycle))
dict, total = db.get_stats()
# %%
