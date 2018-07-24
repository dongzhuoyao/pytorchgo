
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
    parser.add_argument('--dataset',default='ucf101',type=str,help='set the dataset name, to find the data path')
    parser.add_argument('--data_root', default='/data4/hutao/dataset/hmdb51_videos', type=str)
    parser.add_argument('--new_dir', default='/data4/hutao/dataset/hmdb51_videos_extracted', type=str)
    parser.add_argument('--num_workers',default=4,type=int,help='num of workers to act multi-process')
    parser.add_argument('--mode',default='run',type=str,help='set \'run\' if debug done, otherwise, set debug')
    args = parser.parse_args()
    return args

args=parse_args()
videos_root= data_root = args.data_root


def to_frame(augs):
    '''
    To extract dense_flow images
    :param augs:the detailed augments:
        video_name: the video name which is like: 'v_xxxxxxx',if different ,please have a modify.
        save_dir: the destination path's final direction name.
        step: num of frames between each two extracted frames
        bound: bi-bound parameter
    :return: no returns
    '''
    video_name,_ = augs
    video_path = os.path.join(videos_root,video_name[0],video_name[1])

    # provide two video-read methods: cv2.VideoCapture() and skvideo.io.vread(), both of which need ffmpeg support

    # videocapture=cv2.VideoCapture(video_path)
    # if not videocapture.isOpened():
    #     print 'Could not initialize capturing! ', video_name
    #     exit()
    try:
        videocapture=skvideo.io.vread(video_path)
        #videocapture = cv2.VideoCapture(video_path)
    except Exception:
        import traceback
        traceback.print_exc()
        print('{} read error! skip it!'.format(video_path))
        return
    print(video_name)
    # if extract nothing, exit!
    if videocapture.sum()==0:
        print('Could not initialize capturing',video_name)
        exit()
    len_frame=len(videocapture)
    for l in range(len_frame):
        parent_dir = os.path.join(data_root,"{}_{}".format(video_name[0], video_name[1]).replace(".avi",""))
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        save_img = os.path.join(parent_dir, '{:05d}.jpg'.format(l+1))#start from 1
        scipy.misc.imsave(save_img, videocapture[l])

def get_video_list():
    video_list=[]
    for cls_names in os.listdir(videos_root):
        cls_path=os.path.join(videos_root,cls_names)
        for video_ in os.listdir(cls_path):
            video_list.append((cls_names, video_))
    return video_list




if __name__ =='__main__':
    #specify the augments
    num_workers=args.num_workers
    new_dir=args.new_dir
    mode=args.mode

    video_list=get_video_list()
    len_videos=len(video_list)
    print('find {} videos.'.format(len_videos))

    pool=Pool(num_workers)
    if mode=='run':
        pool.map(to_frame, zip(video_list,video_list))
    else: #mode=='debug
        to_frame((video_list[0]))