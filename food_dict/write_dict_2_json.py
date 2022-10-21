# -*- coding: utf-8 -*-
# __author__:bin_ze
# 9/22/22 2:02 PM
import json

from pathlib import Path

def read_txt(txt_path):
    l = []
    with open(txt_path, 'r') as f:
        for line in f.readlines():

            l.append(line.strip())

    l.extend(["冰红茶", "酸奶"])

    return l, str(txt_path).split('/')[-1].split('.')[0]


def list_2_json(l, json_path):

    # create dict
    food_dict = {str(i+1):j for i,j in enumerate(l)}

    with open(json_path, 'w') as f:
        f.write(json.dumps(food_dict, indent=4))


if __name__ == '__main__':


    str_path = Path('/mnt/data/guozebin/object_detection/food_dict/food_txt')
    txt_path = str_path.glob('*.txt')
    for p in txt_path:
        food_0218, date = read_txt(p)
        print(f'write json : {date}_food_dict.json')
        list_2_json(food_0218, f'food_txt/{date}_food_dict.json')


