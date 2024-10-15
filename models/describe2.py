"""
This module provides functions for describing images using BLIP-2 models.
It uses APIURL and APIKEY to generate captions for the Images.
"""

from PIL import Image
# from transformers import (
#     BlipProcessor,
#     BlipForConditionalGeneration,
#     Blip2Processor,
#     Blip2ForConditionalGeneration
# )
import warnings

import requests
# from PIL import Image
# from tkinter import Tk, filedialog
from io import BytesIO

# Suppress warnings
warnings.filterwarnings("ignore")



# ------------------------------------------
# Function to get a text description from an image using the Hugging Face Inference API
def describe_image_enhanced_2(image_path):
    """
    Generate an enhanced caption for an image using the BLIP-2 model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A generated caption describing the image.

    This function uses the BLIP-2 (Salesforce/blip-image-captioning-large) model
    to generate a more detailed caption for the given image.
    """

    # Load the image and convert it to RGB
    image = Image.open(image_path).convert('RGB')
  
    # The API requires the image to be converted to a byte format and image for sending
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
   
    # Send the image to the Hugging Face Inference API using URL nd APIKey
    API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
    headers = {"Authorization": "Bearer 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"}
    
    # Send the request
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response and return the description
        result = response.json()
        caption = result[0]['generated_text']
        print(caption)
        return caption
    
    else:
        # Handle the error
        raise Exception(f"Failed to get description. Status code: {response.status_code}")
