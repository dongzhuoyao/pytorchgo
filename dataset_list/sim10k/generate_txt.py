# Author: Tao Hu <taohu620@gmail.com>
import sys, os
from tqdm import tqdm

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

sim10K_dir = "/data4/hutao/dataset/sim-dataset/VOC2012"
sim10k_img_dir = os.path.join(sim10K_dir,"JPEGImages")
sim10k_anno_dir = os.path.join(sim10K_dir,'Annotations')
keep_difficult = False

img_ids = [tmp for tmp in os.listdir(sim10k_img_dir)]

with open("sim10k_car.txt","w") as f:
    for img_id in tqdm(img_ids):
        anno = ET.parse(os.path.join(sim10k_anno_dir,img_id.replace(".jpg",".xml"))).getroot()

        car_num = 0
        for obj in anno.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not keep_difficult and difficult:
                continue

            name = obj.find('name').text.lower().strip()
            if name == "car":
                car_num += 1

        if car_num:
            f.write("{}\n".format(img_id))

