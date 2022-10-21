# Copyright (c) OpenMMLab. All rights reserved.
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import mmcv
import logging
import numpy as np
from argparse import ArgumentParser

from mmdet.apis import (inference_detector,
                        init_detector)
from compute_pred_acc import Cumpute_pred_acc

DATE = '0302'  # export date


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img_folder', default=f'/mnt/data/guozebin/object_detection/mmdetection/data/{DATE}/food/val2017',help='Image file')
    parser.add_argument('--config', default=f'/mnt/data/guozebin/object_detection/mmdetection/work_dirs/cascade_rcnn_r50_fpn_1x_food/cascade_rcnn_r50_fpn_1x_food.py', help='Config file')
    parser.add_argument('--checkpoint', default='/mnt/data/guozebin/object_detection/mmdetection/work_dirs/cascade_rcnn_r50_fpn_1x_food/epoch_12.pth',help='Checkpoint file')
    parser.add_argument('--out_folder', default=f'../infer_dir_{DATE}', help='Path to output file')
    parser.add_argument('--plot', default=f'../visual_{DATE}', help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    return parser.parse_args()

def xyxy2xywh(x):
    '''
    Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h]
    where xy1=top-left, xy2=bottom-right
    @param x:
    @return:
    '''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def init(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    return model

def infer(args, model):
    # test a single image
    for im in os.listdir(args.img_folder):
        img = os.path.join(args.img_folder, im)
        img = mmcv.imread(img)
        ori_shape = img.shape
        result = inference_detector(model, img)

        #
        bboxes = np.vstack(result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result)
        ]
        labels = np.concatenate(labels)

        if args.score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > args.score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        scores = bboxes[:, -1]
        bboxes = bboxes[:, :4]
        gn = torch.tensor(ori_shape)[[1, 0, 1, 0]]
        # write pred result
        for xyxy, conf, cls in zip(bboxes, scores, labels):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

            line = (cls + 1, *xywh, conf)
            txt_path = args.out_folder + '/' + im.replace('jpg', 'txt')
            with open(txt_path, 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

def compute_acc(args):

    compute = Cumpute_pred_acc(pred_path=args.out_folder,
                               label_path=f'/mnt/data/guozebin/object_detection/mmdetection/data/{DATE}/food/val2compute_acc',
                               result_sava_path=f'cascade_rcnn_acc_{DATE}.txt')
    compute()

    # test plot
    if args.plot is not None:
        if not os.path.isdir(args.plot): os.makedirs(args.plot)
        img_path = args.img_folder
        txt_path = args.out_folder
        assert len(os.listdir(img_path)) == len(os.listdir(txt_path)), \
                                            ('When batch drawing, the number of labels and the number of pictures must match!')
        for img, gt in zip(sorted(os.listdir(img_path)), sorted(os.listdir(txt_path))):
            img_paths = os.path.join(img_path, gt.replace('txt','jpg'))
            gt_txt_paths = os.path.join(txt_path, gt)
            compute.plot_bbox(img_path=img_paths, txt_path=gt_txt_paths, save_path=args.plot)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")
    args = parse_args()

    if not os.path.isdir(args.out_folder): os.makedirs(args.out_folder)

    model = init(args)
    os.system(f'rm {args.out_folder}/*')

    infer(args, model)
    compute_acc(args)