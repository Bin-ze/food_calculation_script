# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import tqdm
import sys
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from centernet.config import add_centernet_config

# constants
WINDOW_NAME = "CenterNet2 detections"

from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

from compute_acc import Cumpute_pred_acc

import json

from utils import *


def visualize_inference(img, feature_map, out_dir, save_name):
    '''

    '''

    import cv2
    import numpy as np
    import os

    H, W, _ = img.shape
    feature_map = feature_map.squeeze().cpu().numpy()
    feature_map = np.max(feature_map, 0)

    cam = np.maximum(feature_map, 0)
    cam = cam / cam.max()

    cam = cv2.resize(cam, (W, H))
    heat_map = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.3 * heat_map + 0.7 * img

    path_cam_img = os.path.join(out_dir, f'{save_name}')
    cv2.imwrite(path_cam_img, cam_img)



def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_centernet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    if cfg.MODEL.META_ARCHITECTURE in ['ProposalNetwork', 'CenterNetDetector']:
        cfg.MODEL.CENTERNET.INFERENCE_TH = args.confidence_threshold
        cfg.MODEL.CENTERNET.NMS_TH = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="/mnt/data/guozebin/object_detection/CenterNet2/configs/CenterNet2_DLA-BiFPN-P5_640_16x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", default=['datasets/food/val2017/'], help="A list of space separated input images")
    parser.add_argument(
        "--output",
        default='infer_result',
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    output_file = None
    json_path = '/mnt/data/guozebin/object_detection/Secondary_classification/class_indices.json'
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            files = sorted(os.listdir(args.input[0]))
            args.input = [args.input[0] + x for x in files]
            assert args.input, "The input path(s) was not found"
        visualizer = VideoVisualizer(
            MetadataCatalog.get(
                cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
            ),
            instance_mode=ColorMode.IMAGE)

        # add classify model
        classify = False
        if classify:
            print('using secondary classification model to refine task!')
        if classify:
            # modelc = load_classifier(name='resnet101', n=14)  # initialize
            modelc = torch.load('/mnt/data/guozebin/object_detection/Secondary_classification/model.pth').to(
                'cuda').eval()

        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, _, visual_feature = demo.run_on_image(
                img, visualizer=visualizer)
            #add heatmap visual

            # save_name = path.split('/')[-1]
            # visualize_inference(img, visual_feature['p3'], out_dir='./heatmap/p3/', save_name=save_name)
            # visualize_inference(img, visual_feature['p4'], out_dir='./heatmap/p4/', save_name=save_name)
            # visualize_inference(img, visual_feature['p5'], out_dir='./heatmap/p5/', save_name=save_name)
            # visualize_inference(img, visual_feature['p6'], out_dir='./heatmap/p6/', save_name=save_name)
            # visualize_inference(img, visual_feature['p7'], out_dir='./heatmap/p7/', save_name=save_name)


            save_dir = './infer_result'
            if os.path.exists(save_dir + '/labels/' + (os.path.basename(path).replace('jpg', 'txt'))):
                os.remove(save_dir + '/labels/' + (os.path.basename(path).replace('jpg', 'txt')))
            if 'instances' in predictions:


                prediction = predictions['instances']
                scores = prediction.scores if prediction.has("scores") else None
                boxes = prediction.pred_boxes.tensor if prediction.has("pred_boxes") else None
                classes = prediction.pred_classes if prediction.has("pred_classes") else None

                # Apply Classifier
                if classify:
                    cla = apply_classifier(boxes, modelc, img) # er ci fen lei classes
                    index = torchvision.ops.nms(boxes, scores, iou_threshold=0.65)
                    boxes = boxes[index, :]
                    cla = cla[index]
                    for cls, xyxy in zip(cla, boxes):
                        cls = int(class_indict[str(cls.item())])
                        line = (cls, *xyxy)  # label format
                        with open(save_dir + '/labels/' + (os.path.basename(path).replace('jpg', 'txt')), 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                # new add method to detectron visioner, 使用sort_index的方法过滤

                # areas = torch.prod(boxes[:, 2:] - boxes[:, :2], dim=1)
                # sorted_idxs = torch.argsort(-areas).tolist()
                # boxes = boxes[sorted_idxs] if boxes is not None else None
                # classes =classes[sorted_idxs] if classes is not None else None
                # only_nms_scores = scores.sort().values

                else:
                    index = torchvision.ops.nms(boxes, scores, iou_threshold=0.6)
                    boxes = boxes[index, :]
                    classes = classes[index]
                    for cls, xyxy in zip(classes, boxes):
                        line = (cls, *xyxy)  # label format
                        with open(save_dir + '/labels/' + (os.path.basename(path).replace('jpg', 'txt')), 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                logger.info(
                    "{}: detected {} instances in {:.2f}s".format(
                        path, len(predictions["instances"]), time.time() - start_time
                    )
                )
            else:
                logger.info(
                    "{}: detected {} instances in {:.2f}s".format(
                        path, len(predictions["proposals"]), time.time() - start_time
                    )
                )

        # add compute method
        pred_path = '/mnt/data/guozebin/object_detection/CenterNet2/infer_result/labels'
        # classes_dict = {'红烧大排': 1, '蚝油牛肉': 2, '干锅鸡': 3, '红烧狮子头': 4, '素鸡小烧肉': 5, '蒜蓉肉丝': 6, '莴笋炒蛋': 7, '鱼香茄子': 8,
        #                 '麻婆豆腐': 9, '芹菜百叶司': 10, '老南瓜': 11, '大白菜油豆腐': 12, '冰红茶': 13, '老酸奶': 14}
        compute = Cumpute_pred_acc(pred_path=pred_path, label_path='/mnt/data/guozebin/coco/labels/val',
                                   result_sava_path=args.config_file.split('/')[-1].replace('yaml', 'txt'))
        compute()

        # # test plot
        img_path = '/mnt/data/guozebin/object_detection/CenterNet2/datasets/food/val2017/'
        txt_path = pred_path
        assert len(os.listdir(img_path)) == len(os.listdir(txt_path)), 'When batch drawing, the number of labels and the number of pictures must match!'
        for img, gt in zip(sorted(os.listdir(img_path)),sorted(os.listdir(txt_path))):
            img_paths = os.path.join(img_path, gt.replace('txt','jpg'))
            gt_txt_paths = os.path.join(txt_path, gt)
            compute.plot_bbox(img_path=img_paths, txt_path=gt_txt_paths, save_path='./infer_result')
