"""
This module provides functions for describing images using BLIP and BLIP-2 models.

It uses the Salesforce BLIP and BLIP-2 models to generate captions for given images.
"""

from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration
)
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def describe_image_simple(image_path):
    """
    Generate a simple caption for an image using the BLIP model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A generated caption describing the image.

    This function uses the BLIP (Salesforce/blip-image-captioning-large) model
    to generate a caption for the given image.
    """
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
    """
    Generate an enhanced caption for an image using the BLIP-2 model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A generated caption describing the image.

    This function uses the BLIP-2 (Salesforce/blip2-opt-6.7b) model
    to generate a more detailed caption for the given image.
    """
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
