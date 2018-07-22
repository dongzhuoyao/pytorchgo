# Author: Tao Hu <taohu620@gmail.com>
import os
file_name = "train_videofolder.txt"
dataset_dir = "/data4/hutao/dataset/UCF101-extracted"
classInd = "classInd.txt"

id2class = {}
with open(classInd,"r") as f:
    for line in f.readlines():
        tmp = line.strip().split()
        id2class[tmp[0]-1] = tmp[1]

for file_name in os.listdir(dataset_dir):
    for frame_name in os.listdir(os.path.join(dataset_dir,file_name)):
