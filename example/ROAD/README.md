
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
reborn.vgg16.**lr1e-4**.w1_10_1.sm_bugfix.class16.adapSegnet_DC.standardGAN.1024x512|epoch10=20,up and down,terminated|reborn.vgg16.**lr1e-4**.w1_10_1.sm_bugfix.class16.adapSegnet_DC.1024x512|epoch10=22,terminated|
reborn.vgg16.lr1e-5.w1_10_1.sm_bugfix.class16.adapSegnet_DC.standardGAN.1024x512|largest=28, but is unstable, terminated|

|arch|train mIoU|eval mIoU|
|---|----|----|
|reborn.vgg16.lr1e-5.w1_10_1.sm_bugfix.totalconfusion|22|17.82
