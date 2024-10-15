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

# Function to get a text description from an image using the Hugging Face Inference API
def describe_image_1_BLIP(image_path):
    """
    Generate an enhanced caption for an image using the BLIP-2 LARGE model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A generated caption describing the image.

    This function uses the BLIP-2 LARGE (Salesforce/blip-image-captioning-large) model
    to generate a more detailed caption for the given image.
    """

    # Load the image and convert it to RGB
    image = Image.open(image_path).convert('RGB')
  
    # The API requires the image to be converted to a byte format and image for sending
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
   
    # Send the image to the Hugging Face Inference API using URL nd APIKey
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    headers = {"Authorization": "Bearer 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"}
    
    # Send the request
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response and return the description
        result = response.json()
        caption = result[0]['generated_text']
        return caption
    
    else:
        # Handle the error
        raise Exception(f"Failed to get description. Status code: {response.status_code}")


# ------------------------------------------

def describe_image_2_NPL(image_path):
    """
    Generate an enhanced caption for an image using the NPLConnect model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A generated caption describing the image.

    This function uses the NPLConnect (nlpconnect/vit-gpt2-image-captioning) model
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
        return caption
    
    else:
        # Handle the error
        raise Exception(f"Failed to get description. Status code: {response.status_code}")
    


def describe_image_3_CLIP(image_path):
    """
    Generate an enhanced caption for an image using the CLIP model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A generated caption describing the image.

    This function uses the CLIP (openai/clip-vit-large-patch14) model
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
        return caption
    
    else:
        # Handle the error
        raise Exception(f"Failed to get description. Status code: {response.status_code}")
    

def describe_image_4_ViT(image_path):
    """
    Generate an enhanced caption for an image using the Google's Vision Transformer (ViT) model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A generated caption describing the image.

    This function uses the ViT (google/vit-base-patch16-224) model
    to generate a more detailed caption for the given image.
    """

    # Load the image and convert it to RGB
    image = Image.open(image_path).convert('RGB')
  
    # The API requires the image to be converted to a byte format and image for sending
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
   
    # Send the image to the Hugging Face Inference API using URL nd APIKey
    API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
    headers = {"Authorization": "Bearer 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"}
    
    # Send the request
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response and return the description
        result = response.json()
        caption = result[0]['generated_text']
        return caption
    
    else:
        # Handle the error
        raise Exception(f"Failed to get description. Status code: {response.status_code}")
