# %%
from dataset_generator import Dataset as ds
import glob
import os
from itertools import cycle
import sys
#%%
db = ds('_data/dataset_config.json', 'MedleyDB')
seed_path = os.path.join(db._seed_dir, '**/*.wav')




# # %% Add ten tracks:
# dict, total = db.get_stats()
# while min(dict.values()) < 600:
#     fps = glob.glob(seed_path, recursive=True)
#     # Rename files to remove whitespace (causes errors in command lines)
#     n_file = fps[0].replace(' ', '_').replace('(', '').replace(')', '').replace(
#               '\'', '')
#     os.rename(fps[0], n_file)
#     db.add_item(n_file)
#     db.get_stats()
#     if input('Continue: y, Stop: any Key')=='y':
#         print('continue')
#     else:
#         print('stop')
#         break


# %% 
fps = glob.glob(seed_path, recursive=True)
for file in fps:
    # Add item to this codec:
    db.add_item(file)
dict, total = db.get_stats()


# %%
# fps = glob.glob(seed_path, recursive=True)
# db.add_item(fps[0])

# %%
