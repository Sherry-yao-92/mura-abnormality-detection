BODY_PART = "XR_WRIST" # wrist part, None = all 7 parts 
IMG_SIZE = 224 # 384, batch16 peaks at 3.7GB (fits 8GB), but ~2.9x slower per epoch (start at 224 to debug the pipeline cheaply)
BATCH_SIZE = 16 #largest that fits 8GB VRAM at 224px with fp16 AMP
LEARNING_RATE = 1e-4 # fine tuning pre-trained model

MAX_EPOCHS = 100 # hard ceiling only, early stopping (patience 10) should trigger first
PATIENCE = 10 # early stopping if AUC does not improve

ROTATION_DEG = 15 # X-rays follow standard positioning (larger angles simulate views that never occur)
FLIP_P = 0.5 # horizontal only (MURA contains both left and right wrists)
# from EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms() (must match the pretrained input distribution)
IMAGENET_MEAN = [0.485, 0.456, 0.406] # used in normalize
IMAGENET_STD = [0.229, 0.224, 0.225] 

CLASS_WEIGHT = False # measured 38% positive at study level (balanced enough to skip)
AMP = True # use Automatic Mixed Precision (AMP) to speed up training and reduce memory usage