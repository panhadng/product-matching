"""
This module provides functions for Classifying Images using Various models models.
It uses APIURL and APIKEY to generate captions for the Images.
"""

from PIL import Image
import requests
from io import BytesIO
import base64

# Authorisation APIKEY for Hugging Face
headers = {"Authorization": "Bearer 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"}

# Define the labels for classification
labels = ["Tech", "Furniture", "Food", "Fashion", "Music", "Games", "Book", "Movies", "Healthcare", "Pet"]

def classify_image_CLIP(image_path, labels):
    """
    Classify an image using the CLIP model and predefined labels.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The predicted label for the image.
    """

    # Load the image and convert it to RGB
    image = Image.open(image_path).convert('RGB')
  
    # Convert image to a byte stream and base64 code for CLIP
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    # Define API URL and Headers for CLIP
    API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-large-patch14"

    # Send both image and labels in json payload
    json_payload = {"inputs": image_b64, "parameters":{"candidate_labels": labels}}

    response = requests.post(API_URL, headers=headers, json=json_payload)

    if response.status_code == 200:
        return response.json()
        
    else:
        print(response.text)
        raise Exception(f"Failed to classify the image. Status code: {response.status_code}")


# To Test and get Top 3 Labels
# image_path = "models/images/chair1.jpg"
# classification = classify_image_CLIP(image_path, labels)
# print("Top 3 labels and their scores:")
# for item in classification[:3]:  # Get the top 3 items
#     label = item['label']
#     score = item['score'] * 100  # Convert to percentage
#     print(f"{label} - {score:.2f}%")