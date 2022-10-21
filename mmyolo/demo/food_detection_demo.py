# -*- coding: utf-8 -*-
# __author__:bin_ze
# 10/8/22 9:26 AM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import sys
import logging
import torchvision

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")
import numpy as np

from argparse import ArgumentParser

import torch
from mmdet.apis import inference_detector, init_detector

from mmyolo.utils import register_all_modules

from compute_pred_acc import Cumpute_pred_acc

Date = '0301'  # export date

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img_folder',default=f'/mnt/data/guozebin/object_detection/mmyolo/data/{Date}/food/val2017', help='Image file')
    parser.add_argument('--config', default=f'/mnt/data/guozebin/object_detection/mmyolo/work_dirs/rtmdet_s_syncbn_8xb32-300e_{Date}/rtmdet_s_syncbn_8xb32-300e_{Date}.py',help='Config file')
    parser.add_argument('--checkpoint',default=f'/mnt/data/guozebin/object_detection/mmyolo/work_dirs/rtmdet_s_syncbn_8xb32-300e_{Date}/best_coco/', help='Checkpoint file')
    parser.add_argument('--out_folder', default=f'../infer_dir_{Date}', help='Path to output file')
    parser.add_argument('--plot', default=f'../visual_{Date}', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score_thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def init(args):
    # register all modules in mmdet into the registries
    register_all_modules()

    # TODO: Support inference of image directory.
    # build the model from a config file and a checkpoint file
    checkpoint = args.checkpoint + os.listdir(args.checkpoint)[0]
    model = init_detector(args.config, checkpoint, device=args.device)

    return model

def infer(args, model):
    # test a single image
    for im in os.listdir(args.img_folder):
        img = os.path.join(args.img_folder, im)
        result = inference_detector(model, img)


        # post handle
        pred = result.pred_instances
        bboxes, labels, scores = pred.bboxes, pred.labels, pred.scores
        # mask
        mask = torch.where(scores> args.score_thr)
        bboxes = bboxes[mask].clone().detach().cpu()
        labels = labels[mask].clone().detach().cpu()
        scores = scores[mask].clone().detach().cpu()
        # nms
        index = torchvision.ops.nms(bboxes, scores, iou_threshold=0.4)
        bboxes = bboxes[index, :]
        labels = labels[index]
        scores = scores[index]

        gn = torch.tensor(result.metainfo['ori_shape'])[[1, 0, 1, 0]]
        # write pred result
        for xyxy, conf, cls in zip(bboxes, scores, labels):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

            line = (cls + 1, *xywh, conf)
            txt_path = args.out_folder + '/' + im.replace('jpg', 'txt')
            with open(txt_path, 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

def compute_acc(args):

    compute = Cumpute_pred_acc(pred_path=args.out_folder,
                               label_path=f'/mnt/data/guozebin/object_detection/mmyolo/data/{Date}/food/val2compute_acc',
                               result_sava_path=f'rtmdet_acc_{Date}.txt')
    compute()

    # test plot
    if args.plot is not None:
        if not os.path.isdir(args.plot): os.makedirs(args.plot)
        img_path = args.img_folder
        txt_path = args.out_folder
        assert len(os.listdir(img_path)) == len(os.listdir(txt_path)), 'When batch drawing, the number of labels and the number of pictures must match!'
        for img, gt in zip(sorted(os.listdir(img_path)),sorted(os.listdir(txt_path))):
            img_paths = os.path.join(img_path, gt.replace('txt','jpg'))
            gt_txt_paths = os.path.join(txt_path, gt)
            compute.plot_bbox(img_path=img_paths, txt_path=gt_txt_paths, save_path=args.plot)


if __name__ == '__main__':
    args = parse_args()

    if not os.path.isdir(args.out_folder): os.makedirs(args.out_folder)

    model = init(args)
    os.system(f'rm {args.out_folder}/*')

    infer(args, model)
    compute_acc(args)
