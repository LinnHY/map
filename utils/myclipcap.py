# caption_generator.py

import torch
import clip
from PIL import Image
from transformers import GPT2Tokenizer
from clipcap import ClipCaptionModel  # Import your ClipCaptionModel from the correct location

class CaptionGenerator:
    def __init__(self, prefix_length=10):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.tokenizer = GPT2Tokenizer.from_pretrained('/amax/data/lhy/gpt2/')
        self.prefix_length = prefix_length

        # Load ClipCap model
        self.model = ClipCaptionModel(prefix_length, tokenizer=self.tokenizer)
        self.model.from_pretrained("/amax/data/lhy/clipcap-base-captioning/coco_weights.pt")
        self.model = self.model.eval()
        self.model = self.model.to(self.device)

    def generateCaption(self, img):
        raw_image = img.convert('RGB')
        image = self.preprocess(raw_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(
                self.device, dtype=torch.float32
            )
            prefix_embed = self.model.clip_project(prefix).reshape(1, self.prefix_length, -1)

        outputs = self.model.generate_beam(embed=prefix_embed)[0]
        return outputs
