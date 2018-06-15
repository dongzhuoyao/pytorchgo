
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

### Check list

* best miou calculation pattern
* train_dir, val_dir 
* loss calculation pattern

### Principle

*  Learning without Forgetting
*  Convenient for multiple-incremence(network structure unchanged)
*  what should we remember?

### class 19+1 ablation study

|arch|old(19 classes)|new(1 classes)|all(20 classes)
|---|---|---|---|
|**training 20 classes together**|67.56|**64.22**|67.39|
train.473.class19meaning.filtered.distill|24.61|27.66|24.77
train.473.class19meaning.filtered.distill_kl|19.81|36.48|20.65|
train.473.class19meaning.filtered.old.epoch_eval.backup|67.84|--|--|
train.473.class19meaning.filtered.new.epoch_eval|--|64.52|--|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t2_disw10.fine_tune|39.89|46.70|40.23|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t4.fine_tune|11.49|48.49|13.34|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t2.fine_tune|25.52|57.3|27.11|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t2.fine_tune.fix_branch|32.02|47.27|32.78|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl|4.46|46.5|6.56|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t2_disw10|12.38|32.29|13.38|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t2|14.93|54.68|16.91|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t4|0.14|29.36|1.61|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t2.share_res1|39.22|40.48|39.29|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t2.share_res12|25.28|46.67|26.35|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t2.share_res123.bs12|7.84|43.76|9.64|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t2_include_bg.fine_tune|34.58|64.8|36.10|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t2_include_bg.fine_tune.fix_branch|43.4|60.08|44.25|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t2_include_bg.fine_tune.share_res1|43.72|**66.82**|44.88|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t2_include_bg.fine_tune.share_res12|**53.57**|64.89|**54.14**|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t2_include_bg.fine_tune.share_res12.disw10|66.74|64.96|66.65|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t2_include_bg.fine_tune.share_res12.disw100|--|--|--|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t2_include_bg.fine_tune.share_res123|45.59|60.92|46.36|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t4_include_bg.fine_tune|41.37|65.56|42.58|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t4_include_bg.fine_tune.share_res12|21.06|63.68|23.19|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t8_include_bg.fine_tune|39.29|66.13|40.62|
train.473.class19meaning.filtered.new.epoch_eval.distill_kl_t16_include_bg.fine_tune|35.04|66.87|36.63|
### class 15+5 ablation study

|arch|old(15 classes)|new(5 classes)|all(20 classes)
|---|---|---|---|
|**train 20 class together**|68.9|62.86|67.39|
train.473.class15meaning.filtered.old.epoch_eval.backup|68.24|--|--|
train.473.class15meaning.filtered.new.epoch_eval|--|56.18|--|
train.473.class15meaning.filtered.new.epoch_eval.distill_kl_t2|--|64.76|14.63|
train.473.class15meaning.filtered.new.epoch_eval.distill_kl_t2_include_bg|25.59|57.4|33.54|
train.473.class15meaning.filtered.new.epoch_eval.distill_kl_t2_include_bg.share_res1|31.02|59.74|38.20|
train.473.class15meaning.filtered.new.epoch_eval.distill_kl_t2_include_bg.share_res12|32.72|60.19|39.59|
train.473.class15meaning.filtered.new.epoch_eval.distill_kl_t2_include_bg.share_res123|35.78|57.84|41.3|
train.473.class15meaning.filtered.new.epoch_eval.distill_kl_t2_include_bg.fix_branch|22.43|54.13|30.36|
### class 10+10 ablation study

|arch|old(10 classes)|new(10 classes)|all(20 classes)
|---|---|---|---|
|**train 20 class together**|68.29|66.48|67.39|
train.473.class10meaning.filtered.old.epoch_eval|66.7|--|--|
train.473.class10meaning.filtered.new.epoch_eval|--|64.8|--|