import torch
from tqdm import tqdm

def train_one_epoch(model, encoder, dataloader, criterion, optimizer, device):
    model.train()        # set decoder to training mode
    encoder.eval()       # set encoder to eval mode (frozen)
    total_loss = 0

    for images, captions, paths in tqdm(dataloader):  # Fixed: unpack 3 values
        images = images.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            features = encoder(images)  # encode images without gradients

        output = model(captions[:, :-1], features)
        loss = criterion(output.reshape(-1, output.size(-1)), captions[:, 1:].reshape(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)