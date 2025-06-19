from torch.utils.data import Dataset
from PIL import Image
import os
import json

class CocoCLIPDataset(Dataset):
    def __init__(self, image_dir, annotation_path, tokenizer, preprocess, max_len=50):
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.max_len = max_len
        self.data = []
        self._load_data(image_dir, annotation_path)

    def _load_data(self, image_dir, annotation_path):
        try:
            with open(annotation_path, 'r') as f:
                ann = json.load(f)
            id_to_filename = {img['id']: img['file_name'] for img in ann['images']}
            self.data = [(os.path.join(image_dir, id_to_filename[a['image_id']]), a['caption']) for a in ann['annotations']]
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, caption = self.data[idx]
        image = self.preprocess(Image.open(path).convert("RGB"))
        tokens = self.tokenizer.encode_caption(caption, max_len=self.max_len)
        return image, tokens, path
