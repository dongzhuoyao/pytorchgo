# Author: Tao Hu <taohu620@gmail.com>
import os
from tqdm import tqdm
dataset_dir = "/data4/hutao/dataset/hmdb51_videos_extracted"



for old_file,new_file in [('hmdb51_split1_train.txt','datalist/train_videofolder_split1.txt'), ('hmdb51_split1_test.txt','datalist/test_videofolder_split1.txt')]:
    print("process {}".format(old_file))
    with open(old_file,"r") as f_read:
        with open(new_file,"w") as f_write:
            for line in tqdm(f_read.readlines()):
                video_path,video_class_name,video_class_id = line.strip().split()
                video_path = video_path.replace(".avi","").replace(".mp4","").replace("/","_")
                if video_path in os.listdir(dataset_dir):
                    frame_number = len(os.listdir(os.path.join(dataset_dir,video_path)))
                    f_write.write("{} {} {}\n".format(video_path, frame_number, video_class_id))