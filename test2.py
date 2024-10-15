# Test 2 for Akshay to Test

from models.describe2 import describe_image_1_BLIP
from models.describe2 import describe_image_2_NPL
from models.describe2 import describe_image_3_CLIP
# from models.describe2 import describe_image_4_ViT

from models.classify import classify_image_CLIP

def describeTest(image_path):
    print(f"Description from describe_image_1_BLIP for {image_path}: {describe_image_1_BLIP(image_path)}")
    print(f"Description from describe_image_2_NPL for {image_path}: {describe_image_2_NPL(image_path)}")
    print(f"Description from describe_image_3_CLIP for {image_path}: {describe_image_3_CLIP(image_path)}")
    #print(f"Description from describe_image_4_ViT for {image_path}: {describe_image_4_ViT(image_path)}")


def classifyTest(image_path):
    classification = classify_image_CLIP(image_path, labels)  # Classify the image
    if classification:
        top_item = classification[0]  # Get the top item
        label = top_item['label']
        score = top_item['score'] * 100  # Convert to percentage
        return f"Top label for {image_path}: {label} - Score: {score:.2f}%"
    else:
        return f"No classification results available for {image_path}."
    

def main():

    # Array for Images
    image_paths = [
        "models/images/chair1.jpg",
        "models/images/chair2.jpg",
        "models/images/chair3.jpg",
        "models/images/keyboard1.jpg",
        "models/images/keyboard2.jpg",
        "models/images/sofa1.jpg",
        "models/images/sofa2.jpg",
    ]

    for image_path in image_paths:
        result = describeTest(image_path)  # Call the describeTest function
        print(result)

    for image_path in image_paths:
        result = classifyTest(image_path)  # Call the classifyTest function
        print(result)

    
if __name__ == "__main__":
    main()
