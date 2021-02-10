# %%
from dataset_generator import Dataset as ds
import glob
import os
from itertools import cycle
import sys
#%%
db = ds('_data/dataset_config.json', 'OtherWAV')
seed_path = os.path.join(db._seed_dir, '**/*.wav')
# Get Stats of dataset:
dict, total = db.get_stats()
print(f'Biggest difference: {max(dict.values()) - min(dict.values())} for',
      f'{max(dict, key=dict.get)} and {min(dict, key=dict.get)}')

# %% Add ten tracks:
dict, total = db.get_stats()
while min(dict.values()) < 600:
    fps = glob.glob(seed_path, recursive=True)
    # Rename files to remove whitespace (causes errors in command lines)
    n_file = fps[0].replace(' ', '_').replace('(', '').replace(')', '').replace(
              '\'', '')
    os.rename(fps[0], n_file)
    db.add_item(n_file, min(dict, key=dict.get))
    db.get_stats()
    if input('Continue: y, Stop: any Key')=='y':
        print('continue')
    else:
        print('stop')
        break


# %% Read dataset stats, if difference between chunk number greater then 150
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
