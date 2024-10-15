# from models.describe import describe_image_simple
from models.describe2 import describe_image_enhanced_1


def main():
    # image_path = "models/images/keyboard1.jpg"
    # description = describe_image_simple(image_path)
    # print(f"Image description: {description}")

    # To Test describe2. that uses APIURL and APIKEY
    image_path = "models/images/keyboard2.jpg"
    # image_path = "models/images/chair3.jpg"
    description = describe_image_enhanced_1(image_path)
    print(f"Image description: {description}")


if __name__ == "__main__":
    main()
