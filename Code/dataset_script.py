#%%
from dataset_generator import Dataset as ds
import glob
import os
from itertools import cycle
#%%
db = ds('_data/dataset_config.json')
# codec_cycle = cycle(db.get_codec_specifiers())
seed_path = os.path.join(db._seed_dir, '**/*.wav')
# Get Stats of dataset:
dict, total = db.get_stats()
print(f'Biggest difference: {max(dict.values()) - min(dict.values())} for',
      f'{max(dict, key=dict.get)} and {min(dict, key=dict.get)}')

#%% Read dataset stats, if difference between chunk number greater then 100 
# add more items to even out
dict, total = db.get_stats()
max_dif = max(dict.values()) - min(dict.values())

while max_dif > 150:
    # Get Codec with smallest number of chunks:
    codec = min(dict, key=dict.get)
    # Add item to this codec:
    fps = glob.glob(seed_path, recursive=True)
    db.add_item(fps[0], codec)
    # Update dict:
    dict, total = db.get_stats()
    max_dif = max(dict.values()) - min(dict.values())
    print(f'Biggest difference: {max_dif} for',
        f'{max(dict, key=dict.get)} and {min(dict, key=dict.get)}')

# %%
