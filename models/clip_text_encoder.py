import open_clip
import torch.nn as nn

class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name='ViT-B-32'):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained='laion2b_s34b_b79k'
        )
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, tokens):
        return self.model.encode_text(tokens)
