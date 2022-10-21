# -*- coding: utf-8 -*-
# __author__:bin_ze
# 9/16/22 4:10 PM

import cv2
import torch

from torchvision import transforms
from PIL import Image


data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def apply_classifier(x, model, im0):
    """
    x: label [x1,y1,x2,y2]
    model:classifier model
    im0: org img

    """
    # applies a second stage classifier to yolo outputs
    ims = []
    for item, a in enumerate(x):  # per item
        cutout = im0[int(a[1]):int(a[3]), int(a[0]):int(a[2])]
        im = Image.fromarray(cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB))
        im = data_transform(im)
        im = torch.unsqueeze(im, dim=0)
        # im = cv2.resize(cutout, (224, 224))  # BGR
        # # cv2.imwrite('test%i.jpg' % j, cutout)
        #
        # im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
        # im /= 255.0  # 0 - 255 to 0.0 - 1.0
        ims.append(im)

    pred_cls2 = model(torch.cat(ims).to('cuda')).argmax(1)  # classifier prediction

    return pred_cls2