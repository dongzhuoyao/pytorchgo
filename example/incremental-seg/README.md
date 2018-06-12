
# incremental seg

|arch|eval mIoU|
|----|----|
|train.473(res50)|68.92,(multi-scale:--)|
|train.473.class19meaning|69.27|
|train.473.class15meaning|70.34|
|train.473.class10meaning|70.69|
|train.473.class5meaning|69.92|



```
resnet50 baseline IoU
IOU: 
[0.918686468455051, 
0.8178352400678197, 0.3765690424699765, 0.8031295077557685, 0.5706959748950378, 0.7553613272741446, 
0.8673246885088117, 0.7829523529113052, 0.847127101856755, 0.29278163754974607, 0.7156299825979209, 
0.45914090849721456, 0.7864770082817798, 0.7223947448733028, 0.7522355510110205, 0.7857565799771034, 
0.5320947991001349, 0.7682378772352991, 0.42950204140011755, 0.7708922871993165, 0.6422490407911726]
```

### class 19+1 ablation study

|arch|old(19 classes)|new(1 classes)|all(20 classes)
|---|---|---|---|
|training 20 classes together|67.56|**64.22**|67.39|
train.473.class19meaning.filtered.distill|24.61|27.66|24.77
train.473.class19meaning.filtered.distill_kl|19.81|36.48|20.65|
~~train.473.class19meaning.filtered.onlyseg_nodistill~~(last epoch already overfitting!!)|0.3|40.15|2.3|
train.473.class19meaning.filtered.onlyseg_nodistill.epoch_eval|--|43.8||

### class 15+5 ablation study

|arch|old(15 classes)|new(5 classes)|all(20 classes)
|---|---|---|---|
|train seperately|train.473.class15meaning.filtered.new:69.46|?|?
|train 20 class together|68.9|62.86|66.39|
