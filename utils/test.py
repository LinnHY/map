from myclipcap import CaptionGenerator
from PIL import Image

# Initialize the CaptionGenerator
caption_generator = CaptionGenerator()

# Load and process an image
img_path = '/amax/data/lhy/map/datasets/Food101/images/train/cheese_plate/cheese_plate_1.jpg'
img = Image.open(img_path)

# Generate a caption
caption = caption_generator.generateCaption(img)
print(caption)
