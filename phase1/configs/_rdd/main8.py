# The new config inherits a base config to highlight the necessary modification
# _base_ = './faster-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco.py'
_base_ = './faster-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco.py'

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

img_scale = (1280, 1280)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='RandomChoiceResize',
                scales=[(800, 1280), (832, 1280), (864, 1280), (896, 1280),
                        (928, 1280), (960, 1280), (992, 1280), (1024, 1280),
                        (1056, 1280), (1088, 1280), (1120, 1280), (1152, 1280), 
                        (1184, 1280), (1216, 1280), (1248, 1280), (1280, 1280)],
                keep_ratio=True)
        ],
                    [
                        dict(
                            type='RandomChoiceResize',
                            scales=[(768, 1280), (1024, 1280), (1280, 1280)],
                            keep_ratio=True),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(768, 1280),
                            allow_negative_crop=True),
                        dict(
                            type='RandomChoiceResize',
                            scales=[(800, 1280), (832, 1280), (864, 1280), (896, 1280),
                                    (928, 1280), (960, 1280), (992, 1280), (1024, 1280),
                                    (1056, 1280), (1088, 1280), (1120, 1280), (1152, 1280), 
                                    (1184, 1280), (1216, 1280), (1248, 1280), (1280, 1280)],
                            keep_ratio=True)
                    ]]),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,  # total_batch_size 32 = 8 GPUS x 4 images
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        pipeline=train_pipeline,
        ann_file='train.json',
        data_prefix=dict(img='train/images/')))

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

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'