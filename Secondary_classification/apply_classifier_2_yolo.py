
# -*- coding: utf-8 -*-
# __author__:bin_ze
# 9/16/22 4:10 PM

import cv2
import torch
import json
import bisect

from torchvision import transforms
from PIL import Image

#####
single_dict_path = '../food_dict/food_txt/0302_food_dict.json'
all_dict_path = '../food_dict/all_food_dict.json'

with open(single_dict_path, 'r') as f:
    single_dict = json.load(f)

reverse_single_dict = {k: v for v, k in single_dict.items()}

with open(all_dict_path, 'r') as f:
    all_dict = json.load(f)
reverse_all_dict = {k: v for v, k in all_dict.items()}

class_result = []
# query
for k, v in single_dict.items():
    class_result.append(reverse_all_dict[v])
class_result.sort(key=lambda x: int(x))
#print(class_result)

hash_map = {reverse_single_dict[v]: reverse_all_dict[v] for v in single_dict.values()}
#print(hash_map)
def find_index(s, nums=class_result):
    for i, j in enumerate(nums):
        if j == s:
            return i

cla_dict = {0: find_index(hash_map[str(1)]), 1: find_index(hash_map[str(2)]), 2: find_index(hash_map[str(4)]), 3: find_index(hash_map[str(5)]), 4: find_index(hash_map[str(8)])}
#print(cla_dict)
#######




data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

cla_dict = {0:1, 1:2, 2:4, 3:5, 4:8}


def scale_handle(im0, scale):
    img = im0.copy()
    img = cv2.resize(img, scale[::-1])

    return img

def apply_classifier(x, model, im0, scale):
    """
    x: label [x1,y1,x2,y2]
    model:classifier model
    im0: org img

    """
    # applies a second stage classifier to yolo outputs


    img = scale_handle(im0, scale)
    ims = []
    label = []
    for item, a in enumerate(x[0]):  # per item
        if int(a[-1]) not in [3, 4]:
            continue

        label.append(a[:-1])
        cutout = img[int(a[1]): int(a[3]), int(a[0]): int(a[2])]
        im = Image.fromarray(cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB))
        im = data_transform(im)
        im = torch.unsqueeze(im, dim=0)
        ims.append(im)

    if len(ims):
        pred_cls2 = model(torch.cat(ims).to('cuda')).argmax(1)  # classifier prediction
        # merge
        merge = []
        for l, c in zip(label, pred_cls2):
            c = torch.tensor(cla_dict[c.item()]).to(l.device)
            merge.append(torch.cat((l, c.unsqueeze(0)), 0))

        for i in x[0]:
            if i[-1] not in [3, 4]:
                merge.append(i)

        merge.sort(key=lambda x: x[-2], reverse=True)

        pred_cls2 = torch.stack(merge)

        return [pred_cls2]
    else:
        return x