_base_ = [
    '_base_/pspnet_r50-d8.py', '_base_/loveda.py',
    '_base_/default_runtime.py', '_base_/schedule_80k.py'
]
model = dict(
    decode_head=dict(num_classes=7), auxiliary_head=dict(num_classes=7))
