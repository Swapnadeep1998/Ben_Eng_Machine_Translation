import os

PADDING='post'
BATCH_SIZE=64
EMBEDDING_DIM=256
UNITS=1024
EPOCHS=30

INPUT_TOKEN_FILE_NAME='ip_tokenizer.json'
TARGET_TOKEN_FILE_NAME='target_tokenizer.json'
INPUT_TOKENIZER_PATH=os.path.join('.', 'Artefacts/tokenizers', INPUT_TOKEN_FILE_NAME)
TARGET_TOKENIZER_PATH=os.path.join('.', 'Artefacts/tokenizers', TARGET_TOKEN_FILE_NAME)
DATA_FILE_NAME='ben.txt'
DATASET_PATH=os.path.join('.','Datasets/',DATA_FILE_NAME)
CHECKPOINT_DIR='./Artefacts/checkpoints'
CHECKPOINT_PREFIX=os.path.join(CHECKPOINT_DIR, "ckpt")