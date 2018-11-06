# Author: Tao Hu <taohu620@gmail.com>

imgs = open("train.txt", "r").readlines()
imgs = [img.strip() for img in imgs]

with open("gta5_train.txt","w") as f:
    for img in imgs[:10000]:
        f.write("images/{} labels/{}\n".format(img, img))


with open("gta5_val.txt","w") as f:
    for img in imgs[10000:]:
        f.write("images/{} labels/{}\n".format(img, img))