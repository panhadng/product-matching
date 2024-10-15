"""
This module provides functions for Classifying Images using Various models models.
It uses APIURL and APIKEY to generate captions for the Images.
"""

from PIL import Image
import requests
from io import BytesIO

# Authorisation APIKEY for Hugging Face
headers = {"Authorization": "Bearer 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"}

# Define the labels for classification
labels = ["Tech", "Furniture", "Food", "Fashion", "Music", "Games", "Book", "Movies", "Healthcare", "Pet", "Arts"]

def classify_image_1_ViT(image_path):
    """
    Classify an image using the Google ViT (Vision Transformer) model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The predicted class for the image based on the predefined labels.
    """

    # Load the image and convert it to RGB
    image = Image.open(image_path).convert('RGB')
  
    # Convert the image to a byte format for sending
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    # Send the image to the Hugging Face Inference API using URL and APIKey
    API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

    # Send the request
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response and find the predicted label
        result = response.json()
        predicted_index = result[0]['label']  # You may need to adjust this based on the exact response format

        # Get the predicted label
        predicted_label = labels[predicted_index] if predicted_index < len(labels) else "Unknown"
        return predicted_label
    
    else:
        # Handle the error
        raise Exception(f"Failed to classify the image. Status code: {response.status_code}")

