# Path to the downloaded CLIP official weights.
# See: https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/clip.py#L30
CLIP_VIT_B16_PATH = 'PATH_TO_MODEL'
CLIP_VIT_B32_PATH = 'PATH_TO_MODEL'
CLIP_VIT_L14_PATH = 'PATH_TO_MODEL'

# Whether cuDNN should be temporarily disable for 3D depthwise convolution.
# For some PyTorch builds the built-in 3D depthwise convolution may be much
# faster than the cuDNN implementation. You may experiment with your specific
# environment to find out the optimal option.
DWCONV3D_DISABLE_CUDNN = True

# Configuration of datasets. The required fields are listed for Something-something-v2 (ssv2)
# and Kinetics-400 (k400). Fill in the values to use the datasets, or add new datasets following
# these examples.
DATASETS = {
    'ssv2': dict(
        TRAIN_ROOT='',
        VAL_ROOT='',
        TRAIN_LIST='caption/SSV2/SSV2_train.txt',
        VAL_LIST='caption/SSV2/SSV2_val.txt',
        NUM_CLASSES=174,
    ),
    'k400': dict(
        TRAIN_ROOT='',
        VAL_ROOT='',
        TRAIN_LIST='caption/k400/k400_train.txt',
        VAL_LIST='caption/k400/k400_val.txt',
        NUM_CLASSES=400,
    ),
    'k200': dict(
        TRAIN_ROOT='',
        VAL_ROOT='',
        TRAIN_LIST='caption/k200/k200_train.txt',
        VAL_LIST='caption/k200/k200_val.txt',
        NUM_CLASSES=200,
    ),
    'hmdb51': dict(
        TRAIN_ROOT='',
        VAL_ROOT='',
        TRAIN_LIST='caption/hmdb51/hmdb51_train_split_3_videos.txt',
        VAL_LIST='caption/hmdb51/hmdb51_val_split_3_videos.txt',
        NUM_CLASSES=51,
    ),
    'ucf101': dict(
        TRAIN_ROOT='',
        VAL_ROOT='',
        TRAIN_LIST='caption/ucf101/ucf101_train_split_3_videos.txt',
        VAL_LIST='caption/ucf101/ucf101_val_split_3_videos.txt',
        NUM_CLASSES=101,
    )

}
