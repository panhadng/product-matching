from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration
)
import warnings
warnings.filterwarnings("ignore")


def describe_image_simple(image_path):
    # Load the BLIP model and processor
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large")

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")

    # Generate the caption
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption


def describe_image_enhanced(image_path):
    # Load the BLIP-2 model and processor
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-6.7b")

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")

    # Generate the caption
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption
