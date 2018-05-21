
## Dann

[https://github.com/pumpikano/tf-dann](https://github.com/pumpikano/tf-dann)


## result

|arch|result|
|---|----|
|train.distill.full.diff2d|first epoch can reach 19|
reborn.vgg16.nofakeforG|epoch8:1,terminated|
reborn.vgg16.lr1e-4|epoch4=1,terminated|
reborn.vgg16.reverse.lr1e-4|epoch16:=8,terminated|
reborn.vgg16.lr1e-5.w1_10_1.sm_bugfix.class16|epoch26=18,up and down, terminated|


|arch|train mIoU|eval mIoU|
|---|----|----|
|reborn.vgg16.lr1e-5.w1_10_1.sm_bugfix.totalconfusion|22|17.82
