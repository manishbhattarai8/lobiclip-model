import torch
from models.clip_encoder import CLIPImageEncoder
from models.clip_text_encoder import CLIPTextEncoder
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import open_clip

# Custom dataset that only loads images and their paths
class ImageOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = sorted([
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, None, path  # Keep tuple structure

def encode_images(dataset, encoder, device):
    all_features, all_paths = [], []
    for image, _, path in tqdm(dataset):
        with torch.no_grad():
            feat = encoder(image.unsqueeze(0).to(device))
            all_features.append(feat.cpu())
            all_paths.append(path)
    return torch.cat(all_features), all_paths

def encode_query(text, text_encoder, device):
    tokens = open_clip.tokenize([text]).to(device)
    with torch.no_grad():
        feat = text_encoder(tokens)
    return feat.cpu()

def retrieve_top_image(text_feat, image_feats, image_paths):
    text_feat = torch.nn.functional.normalize(text_feat, dim=-1)
    image_feats = torch.nn.functional.normalize(image_feats, dim=-1)
    sims = (text_feat @ image_feats.T).squeeze(0)
    best_idx = sims.argmax().item()
    return image_paths[best_idx], sims[best_idx].item()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load encoders
    img_encoder = CLIPImageEncoder().to(device)
    txt_encoder = CLIPTextEncoder().to(device)

    preprocess = img_encoder.preprocess

    # Load test image dataset (no captions)
    image_folder = "data/test2014"  # <- change if needed
    dataset = ImageOnlyDataset(image_folder, transform=preprocess)

    print("Encoding all test images...")
    image_feats, image_paths = encode_images(dataset, img_encoder, device)

    query = input("Enter a query: ")
    text_feat = encode_query(query, txt_encoder, device)

    best_path, score = retrieve_top_image(text_feat, image_feats, image_paths)
    print(f"Best match: {best_path} (score: {score:.4f})")

    # Show or save result
    img = Image.open(best_path)
    img.show()

if __name__ == "__main__":
    main()
