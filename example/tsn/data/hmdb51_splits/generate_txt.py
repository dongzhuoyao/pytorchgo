# Author: Tao Hu <taohu620@gmail.com>
import os
from tqdm import tqdm
dataset_dir = "/data4/hutao/dataset/hmdb51_videos_extracted"

#generate all ground truth dict
gt_dict = {}
for txt in ['hmdb51_split1_train.txt', 'hmdb51_split2_train.txt', 'hmdb51_split3_train.txt']:
    with open(txt,"r") as f:
        for line in f.readlines():
            file_name, class_name, class_index = line.strip().split()
            if not (file_name in gt_dict):
                gt_dict[file_name] = int(class_index)#start from 0

print("gt number :{}".format(len(gt_dict.keys())))

for old_file,new_file in [('hmdb51_split1_train.txt','datalist/train_videofolder_split1.txt'), ('hmdb51_split1_test.txt','datalist/test_videofolder_split1.txt')]:
    print("process {}".format(old_file))
    with open(old_file,"r") as f_read:
        with open(new_file,"w") as f_write:
            for line in tqdm(f_read.readlines()):
                video_path = line.strip().split()[0]
                video_class_name, video_name = video_path.split("/")
                video_name = video_name.replace(".avi","").replace(".mp4","")
                if video_name in os.listdir(dataset_dir):
                    class_index = gt_dict[video_path]
                    frame_number = len(os.listdir(os.path.join(dataset_dir,"{}_{}".format(video_class_name, video_name))))
                    f_write.write("{} {} {}\n".format(video_name, frame_number, class_index))