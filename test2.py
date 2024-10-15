# Test 2 for Akshay to Test

from models.describe2 import describe_image_BLIP_1
from models.describe2 import describe_image_NPL_1


def main():

    image_paths = [
        "models/images/chair1.jpg"
        "models/images/chair2.jpg"
        "models/images/chair3.jpg"
        "models/images/keyboard1.jpg",
        "models/images/keyboard2.jpg",
        "models/images/sofa1.jpg"
        "models/images/sofa2.jpg"
    ]

    # Loop through each image path and get the descriptions
    for image_path in image_paths:
        description_1 = describe_image_BLIP_1(image_path)
        print(f"Description from describe_image_BLIP_1: {description_1}")

        description_2 = describe_image_NPL_1(image_path)
        print(f"Description from describe_image_NPL_1: {description_2}")


if __name__ == "__main__":
    main()
