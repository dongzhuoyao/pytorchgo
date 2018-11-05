# Author: Tao Hu <taohu620@gmail.com>

cs_train_img = "cityscapes_list/cityscapes_imagelist_train.txt"
cs_train_label = "cityscapes_list/cityscapes_labellist_train.txt"
cs_val_img = "cityscapes_list/cityscapes_imagelist_val.txt"
cs_val_label = "cityscapes_list/cityscapes_labellist_val.txt"

synthia_train_img = "synthia_list/SYNTHIA_imagelist_train.txt"
synthia_train_label = "synthia_list/SYNTHIA_labellist_train.txt"
synthia_val_img = "synthia_list/SYNTHIA_imagelist_val.txt"
synthia_val_label = "synthia_list/SYNTHIA_labellist_val.txt"

def generate(file_name,img_file, label_file):
    print file_name
    imgs = open(img_file,"r").readlines()
    imgs = [img.strip() for img in imgs]

    labels = open(label_file,"r").readlines()
    labels = [label.strip() for label in labels]

    with open(file_name, "w") as f:
        for img, label in zip(imgs, labels):
            f.write("{} {}\n".format(img, label))


generate("cs_train.txt",cs_train_img, cs_train_label)
generate("cs_val.txt",cs_val_img, cs_val_label)
generate("synthia_train.txt",synthia_train_img, synthia_train_label)
generate("synthia_val.txt",synthia_val_img, synthia_val_label)


