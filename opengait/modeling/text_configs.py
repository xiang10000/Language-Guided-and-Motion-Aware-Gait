CLIP_VIT_B16_PATH = 'ViT-B-16.pt'
CLIP_VIT_B32_PATH = None
CLIP_VIT_L14_PATH = None

BPE_SIMPLE_VOCAB_PATH = 'bpe_simple_vocab_16e6.txt.gz'
# Whether cuDNN should be temporarily disable for 3D depthwise convolution.
# For some PyTorch builds the built-in 3D depthwise convolution may be much
# faster than the cuDNN implementation. You may experiment with your specific
# environment to find out the optimal option.
DWCONV3D_DISABLE_CUDNN = True
