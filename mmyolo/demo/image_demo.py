# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv
from mmdet.apis import inference_detector, init_detector

from mmyolo.registry import VISUALIZERS
from mmyolo.utils import register_all_modules


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img',default='/mnt/data/guozebin/object_detection/mmyolo/data/0301/food/val2017/000000000001.jpg', help='Image file')
    parser.add_argument('--config', default='/mnt/data/guozebin/object_detection/mmyolo/work_dirs/rtmdet_s_syncbn_8xb32-300e_0301/rtmdet_s_syncbn_8xb32-300e_0301.py',help='Config file')
    parser.add_argument('--checkpoint',default='/mnt/data/guozebin/object_detection/mmyolo/work_dirs/rtmdet_s_syncbn_8xb32-300e_0301/epoch_50.pth', help='Checkpoint file')
    parser.add_argument('--out-file', default='../infer_dir/1.jpg', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):
    # register all modules in mmdet into the registries
    register_all_modules()

    # TODO: Support inference of image directory.
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # test a single image
    result = inference_detector(model, args.img)

    # show the results
    img = mmcv.imread(args.img)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        show=False,
        wait_time=0,
        out_file=args.out_file,
        pred_score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    main(args)
