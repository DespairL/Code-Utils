import os

DATASET_DOWNLOAD_PATH = "/home/nfs06/chenyq/data/datasets"
DATASET_CACHE_PATH = "/home/nfs06/chenyq/data/cache"
YOUR_PROJECT_NAME = "init_training"

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = DATASET_CACHE_PATH
os.environ["HF_DATASETS_CACHE"] = DATASET_DOWNLOAD_PATH


