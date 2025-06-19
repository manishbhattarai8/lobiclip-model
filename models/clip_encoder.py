import open_clip
import torch.nn as nn

class CLIPImageEncoder(nn.Module):
    def __init__(self, model_name='ViT-B-32'):
        super().__init__()
        # Load pretrained CLIP model and transforms from open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained='laion2b_s34b_b79k'
        )
        # Freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Encode the input image tensor and return features
        return self.model.encode_image(x)
