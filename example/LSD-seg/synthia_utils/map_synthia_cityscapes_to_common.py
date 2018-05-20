# Author: Tao Hu <taohu620@gmail.com>
import os,cv2
from tqdm import tqdm
import numpy as np
from synthia2cityscapes import synthia2common_dict, city2common_dict



def synthia2common():
    data_dir = '/data4/hutao/dataset/RAND_CITYSCAPES'
    src_dir = os.path.join(data_dir, "GT/LABELS")
    target_dir = os.path.join(data_dir, "synthia_mapped_to_common")
    #print synthiaid2label[8].synthia_trainid
    files = os.listdir(src_dir)
    for f in tqdm(files):
        print f
        img = cv2.imread(os.path.join(src_dir,f),cv2.IMREAD_UNCHANGED)
        img = img[:, :, -1]
        print "before transfer: {}".format(str(np.unique(img)))
        img_result = np.copy(img)

        for class_id in synthia2common_dict.keys():#notice do not do it in-place!!!
            img_result[img == class_id] = synthia2common_dict[class_id].common_id


        print "after transfer: {}".format(str(np.unique(img_result)))
        cv2.imwrite(os.path.join(target_dir,f),img_result.astype(np.uint8))



def city2common():
    cityscapes_root = '/data1/dataset/cityscapes'
    target_root = os.path.join(cityscapes_root,"label16_for_synthia")
    label_list = ['/home/hutao/lab/pytorchgo/example/LSD-seg/data/filelist/cityscapes_labellist_train.txt',
                  '/home/hutao/lab/pytorchgo/example/LSD-seg/data/filelist/cityscapes_labellist_val.txt']

    #os.mkdir(target_root)
    sub_dir = ["gtFine/train","gtFine/val"]
    for iii, label_txt in enumerate(label_list):
        with open(label_txt, "r") as f:
            output_txt = label_txt.replace(".txt","_label16.txt")
            print "create {}".format(output_txt)
            with open(output_txt, "w") as f_out:
                for line in tqdm(f.readlines()):
                    line = line.strip()
                    line = line.replace("gtFine_labelTrainIds","gtFine_labelIds")
                    file_name = os.path.basename(line)
                    file_name = file_name.replace("gtFine_labelIds","gtFine_labelTrain16")
                    img = cv2.imread(os.path.join(cityscapes_root,sub_dir[iii],line), cv2.IMREAD_GRAYSCALE)


                    print "before transfer: {}".format(str(np.unique(img)))
                    img_result = np.copy(img)

                    for class_id in city2common_dict.keys():  # notice do not do it in-place!!!
                        img_result[img == class_id] = city2common_dict[class_id].common_id

                    print "after transfer: {}".format(str(np.unique(img_result)))
                    cv2.imwrite(os.path.join(target_root, file_name), img_result.astype(np.uint8))
                    f_out.write("{}\n".format(file_name))

#synthia_mapped_to_cityscapes()
city2common()