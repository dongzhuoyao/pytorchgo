# Author: Tao Hu <taohu620@gmail.com>
import matplotlib
matplotlib.use('Agg')
import torch,os,cv2
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
from data.FewShotDs import detection_collate
import numpy as np
from data.FewShotDs import FewShotVOCDataset
import torch.utils.data as data
from myssd import build_ssd
from pytorchgo.utils.viz import draw_boxes
from pytorchgo.utils.rect import IntBox,FloatBox
import matplotlib.pyplot as plt

num_classes = 2
num_workers = 4
image_size = 300
quick_eval = 1e10
val_data_split = "fold2_1shot_val"
base_dir = "train_log/train.baseline.5e-4.channel6.fold2"
start_channels = 6
batch_size = 1

def do_eval(few_shot_net, base_dir):
    tmp_eval = os.path.join(base_dir, "offline_eval_tmp")

    if os.path.isdir(tmp_eval):
        import shutil
        shutil.rmtree(tmp_eval, ignore_errors=True)
    os.makedirs(tmp_eval)

    ground_truth_dir = os.path.join(tmp_eval, "ground-truth")
    predicted_dir = os.path.join(tmp_eval, "predicted")
    os.makedirs(ground_truth_dir)
    os.makedirs(predicted_dir)

    dataset = FewShotVOCDataset(name=val_data_split,channel6=True)
    num_images = len(dataset)

    data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                  shuffle=False, pin_memory=True, collate_fn=detection_collate)

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    w = image_size
    h = image_size

    total_idx = -1
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="offline evaluation"):
        if i > quick_eval: break
        with open(os.path.join(ground_truth_dir, "{}.txt".format(i)), "w") as f_gt:
            with open(os.path.join(predicted_dir, "{}.txt".format(i)), "w") as f_predict:
                first_images_list, images_list, targets_list, metadata_list = batch
                current_batch_size = len(metadata_list)
                for iii in range(current_batch_size):
                        total_idx += 1
                        first_images = first_images_list[iii]
                        images = images_list[iii]
                        targets = targets_list[iii]
                        metadata = metadata_list[iii]

                        # if i > 500:break
                        first_images, images, targets, metadata = batch
                        images_cv = metadata[0]['second_origin_image']

                        first_images = Variable(first_images.cuda())
                        x = Variable(images.cuda())



                        gt_bboxes = targets[0].numpy()
                        for _ in range(gt_bboxes.shape[0]):
                            gt_bboxes[_, 0] *= w
                            gt_bboxes[_, 2] *= w
                            gt_bboxes[_, 1] *= h
                            gt_bboxes[_, 3] *= h
                            f_gt.write(
                                "shit {} {} {} {}\n".format(int(gt_bboxes[_, 0]), int(gt_bboxes[_, 1]), int(gt_bboxes[_, 2]),
                                                            int(gt_bboxes[_, 3])))

                        detections = few_shot_net(first_images, x, is_train=False).data

                        # skip j = 0, because it's the background class
                        for j in range(1, detections.size(1)):
                            dets = detections[0, j, :]
                            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                            dets = torch.masked_select(dets, mask).view(-1, 5)
                            if dets.dim() == 0:
                                continue
                            boxes = dets[:, 1:]
                            boxes[:, 0] *= w
                            boxes[:, 2] *= w
                            boxes[:, 1] *= h
                            boxes[:, 3] *= h
                            scores = dets[:, 0].cpu().numpy()
                            cls_dets = np.hstack((boxes.cpu().numpy(),
                                                  scores[:, np.newaxis])).astype(np.float32,
                                                                                 copy=False)
                            all_boxes[j][i] = cls_dets

                            for _ in range(cls_dets.shape[0]):
                                f_predict.write("shit {} {} {} {} {}\n".format(cls_dets[_, 4], cls_dets[_, 0], cls_dets[_, 1],
                                                                               cls_dets[_, 2], cls_dets[_, 3]))

                                if cls_dets[_, 4] > 0.5:#threshold
                                    floatBox = FloatBox(float(cls_dets[_, 0]), float(cls_dets[_, 1]),float(cls_dets[_, 2]), float(cls_dets[_, 3]))
                                    floatBox.clip_by_shape((image_size,image_size))
                                    images_cv  = draw_boxes(images_cv,[floatBox],color=(255,0,0))

                        fig, axes = plt.subplots(2, 1, figsize=(8, 16))#1-shot setting

                        axes.flat[0].set_title('support image, class_name={}'.format(metadata[0]['class_name']))
                        axes.flat[0].imshow(cv2.resize(metadata[0]['metadata_origin_first_images'][0],(image_size,image_size)))

                        axes.flat[1].set_title('query image, second_image_path={}'.format(metadata[0]['second_image_path']))
                        axes.flat[1].imshow(images_cv)
                        plt.savefig(os.path.join(predicted_dir,"image-{}.png".format(total_idx)))
                        plt.close(fig)



    from eval_map import eval_online
    mAP = eval_online(tmp_eval)
    return mAP




if __name__ == '__main__':

        few_shot_net = build_ssd(image_size, num_classes, start_channels=start_channels, top_k=200)

        few_shot_net.cuda()
        saved_dict = torch.load(os.path.join(base_dir, "cherry.pth"))

        print("offline validation result: {}".format(saved_dict['best_mean_iu']))
        few_shot_net.load_state_dict(saved_dict['model_state_dict'])
        do_eval(few_shot_net=few_shot_net, base_dir=base_dir)
        print("offline validation result: {}".format(saved_dict['best_mean_iu']))
