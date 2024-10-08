from models.describe import describe_image_simple


def main():
    image_path = "models/images/keyboard1.jpg"
    description = describe_image_simple(image_path)
    print(f"Image description: {description}")


if __name__ == "__main__":
    main()
