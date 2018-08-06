
import os,sys
import numpy as np
import cv2
from PIL import Image
from multiprocessing import Pool
import argparse
from IPython import embed #to debug
import skvideo.io
import scipy.misc


def parse_args():
    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument('--num_workers',default=4,type=int,help='num of workers to act multi-process')
    parser.add_argument('--mode',default='run',type=str,help='set \'run\' if debug done, otherwise, set debug')
    args = parser.parse_args()
    return args

args=parse_args()


def extract():
    moment_base = "/data4/hutao/dataset/Moments/Moments_in_Time_Mini"
    category_txt = os.path.join(moment_base, "moments_categories.txt")
    train_csv = os.path.join(moment_base, "trainingSet.csv")
    val_csv = os.path.join(moment_base, "validationSet.csv")
    target_dir = os.path.join(moment_base, "extracted")

    name2id = {}
    with open(category_txt,"r") as f:
        for line in f.readlines():
            name, id = line.strip().split(',')
            name2id[name] = id


    for file_type,file_csv in [("training",train_csv),("validation", val_csv)]:
        with open(file_csv,"r") as f:
            for line in f.readlines():
                origin_file_name, class_name, _, _ = line.strip().split(",")
                print(origin_file_name)
                file_name = origin_file_name.replace("/","_").replace(".mp4","") #dancing/yt-uHo5kAfEurg_97.mp4  =>  dancing_yt-uHo5kAfEurg_97
                target_dir_name = os.path.join(target_dir, "{}_{}".format(file_type, file_name))
                try:
                    if not os.path.exists(target_dir_name):
                        os.makedirs(target_dir_name)
                    #may be OSError: [Errno 36] File name too long
                except Exception:
                    import traceback
                    traceback.print_exc()
                    continue

                try:
                    video_path = os.path.join(moment_base, file_type ,origin_file_name)
                    videocapture = skvideo.io.vread(video_path)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    print('{} read error! skip it!'.format(video_path))
                    return
                len_frame = len(videocapture)
                for l in range(len_frame):
                    save_img = os.path.join(target_dir_name, '{:05d}.jpg'.format(l + 1))  # start from 1
                    scipy.misc.imsave(save_img, videocapture[l])


if __name__ =='__main__':
    #specify the augments
    num_workers=args.num_workers
    mode=args.mode
    pool=Pool(num_workers)
    if mode=='run':
        pool.map(extract())
    else: #mode=='debug
        extract()