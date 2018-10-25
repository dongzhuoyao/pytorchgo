# Author: Tao Hu <taohu620@gmail.com>
import random

name = "train_aug.txt"
train_ids = []
val_ids = []
ids = []
with open(name, "r") as f:
    lines = f.readlines()
    for line in lines:
        ids.append(line.strip())

random.shuffle(ids)
train_ids = ids[:10000]
val_ids = ids[10000:]

with open("incremental_train.txt","w") as f:
    for id in train_ids:
        f.write("{}\n".format(id))

with open("incremental_val.txt","w") as f:
    for id in val_ids:
        f.write("{}\n".format(id))