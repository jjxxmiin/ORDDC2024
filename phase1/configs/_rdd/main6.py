# The new config inherits a base config to highlight the necessary modification
# _base_ = './faster-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco.py'
_base_ = './co_dino_5scale_swin_l_lsj_16xb1_3x_coco.py'

# Modify dataset related settings
data_root = '/media/oem/storage01/jmjeong/rdd2022/COCO/'
metainfo = {
    'classes': (
        'D00',
        'D10',
        'D20',
        'D40',
    ),
    'palette': [
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]
}

backend_args = None

image_size = (640, 640)

# LSJ + CopyPaste

# augmentation strategy originates from DETR / Sparse RCNN
load_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='RandomChoiceResize',
                scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)],
                keep_ratio=True)
        ],
                    [
                        dict(
                            type='RandomChoiceResize',
                            scales=[(400, 1333), (500, 1333), (600, 1333)],
                            keep_ratio=True),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type='RandomChoiceResize',
                            scales=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            keep_ratio=True)
                    ]]),
]

train_pipeline = [
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=1,  # total_batch_size 32 = 8 GPUS x 4 images
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        pipeline=train_pipeline,
        dataset=dict(
            data_root=data_root,
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=False), 
            pipeline=load_pipeline,
            ann_file='train.json',
            data_prefix=dict(img='train/images/')
        )
    )
)

# follow ViTDet
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=image_size, keep_ratio=True),  # diff
    dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        pipeline=test_pipeline,
        ann_file='valid.json',
        data_prefix=dict(img='valid/images/')))

test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'valid.json', metric='bbox')
test_evaluator = val_evaluator