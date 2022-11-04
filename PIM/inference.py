import torch
import os
import torchvision.transforms as transforms
import cv2
from PIL import Image
import argparse
import warnings

from FGVC_PIM.utils.config_utils import load_yaml
from FGVC_PIM.models.builder import MODEL_GETTER
count = 0
warnings.simplefilter("ignore")
#'''
cla_dict={
    '1':0,
    '2':6,
    '3':7,
    '4':8,
    '5':9,
    '6':10,
    '7':11,
    '8':12,
    '9':13,
    '10':1,
    '11':2,
    '12':3,
    '13':4,
    '14':5
}
'''
cla_dict={
    '1':6,
    '2':7,
    '3':1
}
'''
reverse_cla = {k:v for v,k in cla_dict.items()}

tao_cla_dict={
    '1':0,
    '2':4,
    '3':5,
    '4':6,
    '5':7,
    '6':8,
    '7':9,
    '8':10,
    '9':11,
    '10':1,
    '11':2,
    '12':3,
}
tao_reverse_cla = {k:v for v,k in tao_cla_dict.items()}

### read image and convert image to tensor
data_transform = transforms.Compose([
    transforms.Resize((510, 510), Image.BILINEAR),
    transforms.CenterCrop((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def infer(model, img, label):
    global count
    # only need forward backbone
    model.eval()
    with torch.no_grad():
        logits = model(img.unsqueeze(0))
        preds = torch.softmax(logits['layer1'].mean(1), dim=-1)
        predict_cla = torch.argmax(preds).cpu().numpy()
        print(predict_cla)
        pred_label = reverse_cla[int(predict_cla)]
        if label == pred_label:
            count += 1


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
        label.append(a[:-1])
        a = torch.clip(a, min=0)
        cutout = img[int(a[1]): int(a[3]), int(a[0]): int(a[2])]
        im = Image.fromarray(cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB))
        im = data_transform(im)
        im = torch.unsqueeze(im, dim=0)
        ims.append(im)

    if len(ims):
        model.eval()
        with torch.no_grad():
            logits = model(torch.cat(ims).to('cuda'))
            preds = torch.softmax(logits['layer1'].mean(1), dim=-1)
            pred_cla2 = torch.argmax(preds, dim=-1).cpu().numpy()

            # merge
            merge = []
            for l, c in zip(label, pred_cla2):
                c = torch.tensor(int(reverse_cla[c.item()])).to(l.device)
                merge.append(torch.cat((l, c.unsqueeze(0)), 0))

            merge.sort(key=lambda x: x[-2], reverse=True)

            pred_cls2 = torch.stack(merge)

            return [pred_cls2]
    else:
        return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser("PIM-FGVC Heatmap Generation")
    parser.add_argument("--c", default="./configs/eval.yaml", type=str)
    parser.add_argument("--img", default="../Secondary_classification/food_0304/val/", type=str)
    parser.add_argument("--save_img", default="./", type=str, help="save path")
    parser.add_argument("--pretrained", default="", type=str)
    parser.add_argument("--model_name", default="swin-t", type=str, choices=["swin-t", "resnet50", "vit", "efficient"])
    args = parser.parse_args()

    assert args.c != "", "Please provide config file (.yaml)"

    args = parser.parse_args()
    load_yaml(args, args.c)

    assert args.pretrained != ""

    model = MODEL_GETTER[args.model_name](
        use_fpn=args.use_fpn,
        fpn_size=args.fpn_size,
        use_selection=args.use_selection,
        num_classes=args.num_classes,
        num_selects=args.num_selects,
        use_combiner=args.use_combiner,
    )  # about return_nodes, we use our default setting

    ### load model
    checkpoint = torch.load(args.pretrained, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(args.device)

    ### read image and convert image to tensor
    img_transforms = transforms.Compose([
        transforms.Resize((510, 510), Image.BILINEAR),
        transforms.CenterCrop((args.data_size, args.data_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for label in os.listdir(args.img):
        for img in os.listdir(os.path.join(args.img, label)):
            img = cv2.imread(os.path.join(args.img, label, img))
            img = img[:, :, ::-1]  # BGR to RGB.
            # to PIL.Image
            img = Image.fromarray(img)
            img = img_transforms(img)
            img = img.to(args.device)
            infer(model, img, label)

    #print('acc: ', count/)