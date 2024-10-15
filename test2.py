# Test 2 for Akshay to Test

from models.describe2 import describe_image_1_BLIP
from models.describe2 import describe_image_2_NPL
from models.describe2 import describe_image_3_CLIP
# from models.describe2 import describe_image_4_ViT

from models.classify import classify_image_1_ViT


def main():

    image_paths = [
        "models/images/chair1.jpg",
        "models/images/chair2.jpg",
        "models/images/chair3.jpg",
        "models/images/keyboard1.jpg",
        "models/images/keyboard2.jpg",
        "models/images/sofa1.jpg",
        "models/images/sofa2.jpg",
    ]

    # Loop through each image path and get the descriptions
    # for image_path in image_paths:
    #     print(f"Description from describe_image_1_BLIP for {image_path}: {describe_image_1_BLIP(image_path)}")
    #     print(f"Description from describe_image_2_NPL for {image_path}: {describe_image_2_NPL(image_path)}")
    #     print(f"Description from describe_image_3_CLIP for {image_path}: {describe_image_3_CLIP(image_path)}")
    #     #print(f"Description from describe_image_4_ViT for {image_path}: {describe_image_4_ViT(image_path)}")

    for image_path in image_paths:
        print(f"Image Classification from for {image_path}: {classify_image_1_ViT(image_path)}")


if __name__ == "__main__":
    main()
