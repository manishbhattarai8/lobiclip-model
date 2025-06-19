import torch

class Config:
    BATCH_SIZE = 32
    EPOCHS = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MAX_LEN = 50
    EMBED_DIM = 512
