import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from data.dataset import CocoCLIPDataset
from data.tokenizer import CaptionTokenizer
from models.clip_encoder import CLIPImageEncoder
from models.decoder import TransformerDecoder
from utils.train import train_one_epoch
from utils.config import Config
import json
import nltk
nltk.download('punkt')


def main():
    print(f"Using device: {Config.DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Load captions
    with open("captions_train2014.json") as f:
        captions = [a['caption'] for a in json.load(f)['annotations']]

    tokenizer = CaptionTokenizer(captions)

    encoder = CLIPImageEncoder().to(Config.DEVICE)
    preprocess = encoder.preprocess

    dataset = CocoCLIPDataset("train2014", "captions_train2014.json", tokenizer, preprocess)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True)

    decoder = TransformerDecoder(Config.EMBED_DIM, tokenizer.vocab_size()).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx['<pad>'])
    optimizer = optim.Adam(decoder.parameters(), lr=1e-4)

    for epoch in range(Config.EPOCHS):
        loss = train_one_epoch(decoder, encoder, dataloader, criterion, optimizer, Config.DEVICE)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
        torch.save(decoder.state_dict(), f"decoder_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()
