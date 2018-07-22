# processing the raw data of the video datasets (Something-something and jester)
# generate the meta files:
#   category.txt:               the list of categories.
#   train_videofolder.txt:      each row contains [videoname num_frames classIDX]
#   val_videofolder.txt:        same as above
#
# Bolei Zhou, Dec.2 2017
#
#
import os
import pdb
from tqdm import tqdm
dataset_prefix = 'jester-v1/jester-v1'#'data/something-something-v1'
data_dir = "jester-v1/20bn-jester-v1"
with open('%s-labels.csv' % dataset_prefix) as f:
    lines = f.readlines()
categories = []
for line in lines:
    line = line.rstrip()
    categories.append(line)
categories = sorted(categories)
with open(os.path.join('category.txt'),'w') as f:
    f.write('\n'.join(categories))

dict_categories = {}
for i, category in enumerate(categories):
    dict_categories[category] = i

files_input = ['%s-validation.csv' % dataset_prefix, '%s-train.csv' % dataset_prefix]
files_output = [os.path.join('val_videofolder.txt'),os.path.join('train_videofolder.txt')]
for (filename_input, filename_output) in zip(files_input, files_output):
    with open(filename_input) as f:
        lines = f.readlines()
    folders = []
    idx_categories = []
    for line in lines:
        line = line.rstrip()
        items = line.split(';')
        folders.append(items[0])
        idx_categories.append(dict_categories[items[1]])
    output = []
    for i in tqdm(range(len(folders)),desc="generating %s"%(filename_output)):
        curFolder = folders[i]
        curIDX = idx_categories[i]
        # counting the number of frames in each video folders
        dir_files = os.listdir(os.path.join(data_dir, curFolder))
        output.append('%s %d %d'%(curFolder, len(dir_files), curIDX))

    with open(filename_output,'w') as f:
        f.write('\n'.join(output))
