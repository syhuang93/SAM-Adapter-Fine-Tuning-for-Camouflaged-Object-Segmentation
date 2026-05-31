# scripts/config.py

PROJECT_ROOT = "/content/drive/MyDrive/Colab Notebooks/SAM_Adapter"

DATASET_ROOT = f"{PROJECT_ROOT}/dataset"
TRAIN_DIR = f"{DATASET_ROOT}/train"
EVAL_DIR = f"{DATASET_ROOT}/eval"
TEST_DIR = f"{DATASET_ROOT}/test"

CHECKPOINT_DIR = f"{PROJECT_ROOT}/checkpoints"
SAM_B_CHECKPOINT = f"{CHECKPOINT_DIR}/sam_vit_b_01ec64.pth"

OUTPUT_DIR = f"{PROJECT_ROOT}/outputs"

# Training defaults
BATCH_SIZE = 256
NUM_WORKERS = 0
LR = 1e-4
EPOCHS = 20
SEED = 42

# Image / prompt settings
BBOX_JITTER = 0.1   # only for training