# yolox_config.py
from pathlib import Path
parent = Path(__file__).parent.parent.parent
_base_ = parent / 'mmdetection/configs/yolox/yolox_s_8xb8-300e_coco.py'

model = dict(
    bbox_head=dict(num_classes=80),  # only person class
)


test_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        data_root='dummy/',
        ann_file='dummy.json',
        data_prefix=dict(img='images/')
    )
)
