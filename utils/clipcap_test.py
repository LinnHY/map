from clipcap import ClipCaptionModel
from transformers import GPT2Tokenizer
import torch
import clip
#model_path = "clipcap-base-captioning-ft-hl-narratives/pytorch_model.pt" # change accordingly
model_path ="/amax/data/lhy/clipcap-base-captioning/coco_weights.pt"
# load clip
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained('/amax/data/lhy/gpt2/')#  
prefix_length = 10

# load ClipCap
model = ClipCaptionModel(prefix_length, tokenizer=tokenizer)
model.from_pretrained(model_path)
model = model.eval()
model = model.to(device)

def generateCaption(img):
    # load the image

    raw_image = img.convert('RGB')
    image = preprocess(raw_image).unsqueeze(0).to(device)
    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(
            device, dtype=torch.float32
        )
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)

    # generate the caption   
    outputs=model.generate_beam(embed=prefix_embed)[0]
    return outputs