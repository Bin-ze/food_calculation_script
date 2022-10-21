# -*- coding: utf-8 -*-
# __author__:bin_ze
# 8/30/22 11:03 AM
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import torchvision
import torch.nn.functional as F
import time
from torch.optim import *

class ArcLoss(nn.Module):
    def __init__(self, feature_dim=2, cls_dim=14):
        super().__init__()
        self.W = nn.Parameter(torch.randn(feature_dim, cls_dim), requires_grad=True)

    def forward(self, feature, m=1, s=10):
        x = nn.functional.normalize(feature, dim=1)
        w = nn.functional.normalize(self.W, dim=0)
        cos = torch.matmul(x, w)/10             # 求两个向量夹角的余弦值
        a = torch.acos(cos)                     # 反三角函数求得 α
        top = torch.exp(s*torch.cos(a+m))       # e^(s * cos(a + m))
        down2 = torch.sum(torch.exp(s*torch.cos(a)), dim=1, keepdim=True)-torch.exp(s*torch.cos(a))
        out = torch.log(top/(top+down2))
        return out

def load_classifier(name='resnet50', n=2):
    # Loads a pretrained model reshaped to n-class output
    model = torchvision.models.__dict__[name](pretrained=True)
    fc = nn.Sequential(
        nn.Linear(512, 2048),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(512, n)
    )
    model.fc = fc
    # ResNet model properties
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Reshape output to n classes
    # filters = model.fc.weight.shape[1]
    # model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    # model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    # model.fc.out_features = n
    return model


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    #data_root ="/home/data/gzb/" # get data root path
    #image_path = os.path.join(data_root, "Scence_train_val_data")  # Scence data set path
    image_path = args.food_data_path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    Scense_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in Scense_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size =args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    # logging.info('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images fot validation.".format(train_num,
                                                                           val_num))
    net = load_classifier(name='resnet18', n=512)
    net.to(device)
    LR = args.LR

    # TODO

    arcface = ArcLoss(512, len(Scense_list)).to(device)
    loss_function = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(net.parameters(), lr=LR)
    optimizer = optim.Adam([
    {'params': net.parameters()},
    {'params': arcface.parameters()}
], lr=LR)


    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)

    epochs = args.epoch
    best_acc = 0.0
    save_path =args.save_path
    train_steps = len(train_loader)

    for epoch in range(epochs):
        # train
        net.train()
        running_loss, start_time= 0.0 ,time.time()
        train_bar = train_loader
        train_acc = 0
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))

            logits = arcface(logits)


            logits = torch.softmax(logits, dim=1)
            predict_y = torch.argmax(logits, dim=1)
            train_acc += torch.eq(predict_y, labels.to(device)).sum().item()

            #loss = loss_function(logits, labels.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        train_accurate = train_acc / train_num

        scheduler.step()
        net.eval()
        #acc = 0.0
        val_acc = 0
        with torch.no_grad():
            val_bar=validate_loader

            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs1 = net(val_images.to(device))
                outputs1 = arcface(outputs1)
                # loss = loss_function(outputs, test_labels)
                #predict_y = torch.max(outputs1, dim=1)[1]
                logits = torch.softmax(logits, dim=1)
                predict_y = torch.argmax(outputs1, dim=1)
                val_acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
            val_accurate = val_acc / val_num

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net, save_path)
            print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f val_accuracy: %.3f  run_time %.3f' %
                  (epoch + 1, running_loss / train_steps, train_accurate, val_accurate,(time.time()-start_time)))
            with open('result.txt','a') as f:
                f.write('[epoch %d] train_loss: %.3f  val_accuracy: %.3f  run_time %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate,(time.time()-start_time)))
                f.write('\n')

    print('Finished Training')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--food_data_path', default='/mnt/data/guozebin/object_detection/Secondary_classification/food_classification', help='train/val path')
    parser.add_argument('--save_path', default='./model_arcloss.pth', help='weight save path')
    parser.add_argument('--epoch', default=100, help='train epoch')
    parser.add_argument('--LR', default=0.001, help='train learn rate')
    parser.add_argument('--batch_size', default=128, help='train learn rate')
    args = parser.parse_args()
    main(args)
