from detectron2.data.datasets.register_coco import register_coco_instances
import os
categories = [
    {'id': 1, 'name': '0'},
    {'id': 2, 'name': '1'},
    {'id': 3, 'name': '2'},
    {'id': 4, 'name': '3'},
    {'id': 5, 'name': '4'},
    {'id': 6, 'name': '5'},
    {'id': 7, 'name': '6'},
    {'id': 8, 'name': '7'},
    {'id': 9, 'name': '8'},
    {'id': 10, 'name': '9'},
    {'id': 11, 'name': '10'},
    {'id': 12, 'name': '11'},
    {'id': 13, 'name': '12'},
    {'id': 14, 'name': '13'},
    {'id': 15, 'name': '14'}
]

def _get_builtin_metadata():
    id_to_name = {x['id']: x['name'] for x in categories}
    #thing_dataset_id_to_contiguous_id = {i: i for i in range(len(categories))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        #"thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_PREDEFINED_SPLITS = {
    "food_train": ("food/train2017", "food/annotations/instances_train2017.json"),
    "food_val": ("food/val2017", "food/annotations/instances_val2017.json")
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )
