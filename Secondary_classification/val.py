# -*- coding: utf-8 -*-
# __author__:bin_ze
# 8/31/22 9:34 AM
# -*- coding: utf-8 -*-
# __author__:bin_ze
# 8/30/22 11:03 AM
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import torchvision

def load_classifier(name='resnet101', n=2):
    # Loads a pretrained model reshaped to n-class output
    model = torchvision.models.__dict__[name](pretrained=True)

    # ResNet model properties
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Reshape output to n classes
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    #data_root ="/home/data/gzb/" # get data root path
    #image_path = os.path.join(data_root, "Scence_train_val_data")  # Scence data set path
    image_path = args.food_data_path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    batch_size =args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))


    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    #net = load_classifier(name='resnet101', n=14)
    net = torch.load('/mnt/data/guozebin/object_detection/Secondary_classification/model.pth')
    print('load model weight success')
    net.to(device)

    net.eval()
    val_acc = 0
    with torch.no_grad():
        val_bar=validate_loader

        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs1 = net(val_images.to(device))
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs1, dim=1)[1]
            val_acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        val_accurate = val_acc / val_num

        print('val_accuracy: %.3f ' %
              (val_accurate))







if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--food_data_path', default='/mnt/data/guozebin/object_detection/Secondary_classification/food_classification', help='train/val path')
    parser.add_argument('--save_path', default='./model.pth', help='weight save path')
    parser.add_argument('--epoch', default=50, help='train epoch')
    parser.add_argument('--LR', default=0.001, help='train learn rate')
    parser.add_argument('--batch_size', default=64, help='train learn rate')
    args = parser.parse_args()
    main(args)
