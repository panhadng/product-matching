"""
This module provides a ProductComparator class for comparing products based on their descriptions and images.

The ProductComparator uses various NLP and computer vision techniques to extract features from product descriptions
and images, and then compares these features to determine the similarity between products.

Dependencies:
- sklearn
- PIL
- imagehash
- transformers
- torch
- sentence_transformers
- spacy

Usage:
    comparator = ProductComparator()
    result = comparator.compare_products(product1, product2)
"""

from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import imagehash
from transformers import AutoModel, AutoFeatureExtractor, CLIPProcessor, CLIPModel
import torch
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class ProductComparator:
    """
    A class for comparing products based on their descriptions and images.

    This class uses various NLP and computer vision models to extract features
    from product descriptions and images, and then compares these features
    to determine the similarity between products.
    """

    def __init__(self):
        """
        Initialize the ProductComparator with necessary models and processors.
        """
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.color_model = AutoModel.from_pretrained("microsoft/resnet-50")
        self.color_processor = AutoFeatureExtractor.from_pretrained(
            "microsoft/resnet-50")
        self.nlp = spacy.load("en_core_web_sm")
        self.tfidf = TfidfVectorizer()

    def compare_descriptions(self, desc1, desc2):
        """
        Compare two product descriptions using CLIP model.

        Args:
            desc1 (str): The first product description.
            desc2 (str): The second product description.

        Returns:
            float: Similarity score between the two descriptions.
        """
        inputs = self.clip_processor(
            text=[desc1, desc2], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        return cosine_similarity(text_features)[0][1]

    def compare_images(self, img1_path, img2_path):
        """
        Compare two product images using CLIP model and image hashing.

        Args:
            img1_path (str): Path to the first image.
            img2_path (str): Path to the second image.

        Returns:
            float: Similarity score between the two images.
        """
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        inputs = self.clip_processor(
            images=[img1, img2], return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)

        clip_similarity = cosine_similarity(image_features)[0][1]

        # Image hash comparison
        hash1 = imagehash.average_hash(img1)
        hash2 = imagehash.average_hash(img2)
        hash_similarity = 1 - (hash1 - hash2) / 64.0

        return (clip_similarity + hash_similarity) / 2

    def extract_dominant_color(self, img):
        """
        Extract the dominant color from an image using a pre-trained ResNet model.

        Args:
            img (PIL.Image): The input image.

        Returns:
            numpy.ndarray: A vector representing the dominant color.
        """
        inputs = self.color_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.color_model(**inputs)
        features = outputs.last_hidden_state.mean(dim=1)
        return features[0].numpy()

    def extract_features(self, description, image_path):
        """
        Extract features from a product description and image.

        Args:
            description (str): The product description.
            image_path (str): Path to the product image.

        Returns:
            dict: A dictionary of extracted features.
        """
        features = {}

        # Extract features from text
        doc = self.nlp(description)
        for ent in doc.ents:
            features[ent.label_.lower()] = ent.text.lower()

        # Extract keywords using TF-IDF
        tfidf_matrix = self.tfidf.fit_transform([description])
        feature_names = self.tfidf.get_feature_names_out()
        top_keywords = sorted(zip(feature_names, tfidf_matrix.toarray()[
                              0]), key=lambda x: x[1], reverse=True)[:5]
        features['keywords'] = [keyword for keyword, _ in top_keywords]

        # Extract color information from text
        for token in doc:
            if self.is_color(token.text):
                features['color'] = token.text.lower()

        # Extract image features
        if image_path:
            img_features = self.extract_image_features(image_path)
            features.update(img_features)

        return features

    def extract_image_features(self, image_path):
        """
        Extract features from an image.

        Args:
            image_path (str): Path to the image.

        Returns:
            dict: A dictionary of extracted image features.
        """
        img = Image.open(image_path).convert('RGB')
        features = {}

        # Extract color information
        features['dominant_color'] = self.extract_dominant_color(img)

        # Extract object detection results (you'll need to implement this)
        detected_objects = self.detect_objects(img)
        features['detected_objects'] = detected_objects

        # Add more image analysis as needed (e.g., texture, shape, etc.)
        return features

    def detect_objects(self, img):
        """
        Detect objects in an image.

        Args:
            img (PIL.Image): The input image.

        Returns:
            list: A list of detected objects.
        """
        # Implement object detection here
        # This is a placeholder - you'll need to use an actual object detection model
        return ["object1", "object2"]

    def compare_features(self, features1, features2):
        """
        Compare two sets of features.

        Args:
            features1 (dict): The first set of features.
            features2 (dict): The second set of features.

        Returns:
            float: Similarity score between the two sets of features.
        """
        score = 0
        total = 0

        all_keys = set(features1.keys()) | set(features2.keys())

        for key in all_keys:
            if key in features1 and key in features2:
                if isinstance(features1[key], list) and isinstance(features2[key], list):
                    # Compare lists (e.g., keywords or detected objects)
                    score += self.compare_lists(features1[key], features2[key])
                elif key == 'color':
                    # Compare colors
                    score += self.is_similar(features1[key],
                                             [features2[key]], threshold=0.9)
                elif key == 'dominant_color':
                    # Compare dominant colors using cosine similarity
                    score += cosine_similarity([features1[key]],
                                               [features2[key]])[0][0]
                else:
                    # Compare scalar values
                    score += self.compare_scalar(
                        features1[key], features2[key])
                total += 1

        return score / total if total > 0 else 0

    def compare_lists(self, list1, list2):
        """
        Compare two lists of features.

        Args:
            list1 (list): The first list of features.
            list2 (list): The second list of features.

        Returns:
            float: Similarity score between the two lists.
        """
        common = set(list1) & set(list2)
        return len(common) / max(len(list1), len(list2))

    def compare_scalar(self, val1, val2):
        """
        Compare two scalar values.

        Args:
            val1: The first value.
            val2: The second value.

        Returns:
            float: Similarity score between the two values.
        """
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return 1 - abs(val1 - val2) / max(val1, val2)
        else:
            return self.is_similar(str(val1), [str(val2)])

    def is_similar(self, word, target_words, threshold=0.8):
        """
        Check if a word is similar to any of the target words.

        Args:
            word (str): The word to check.
            target_words (list): List of target words.
            threshold (float): Similarity threshold.

        Returns:
            bool: True if the word is similar to any target word, False otherwise.
        """
        word_embedding = self.text_model.encode([word])[0]
        target_embeddings = self.text_model.encode(target_words)
        similarities = cosine_similarity(
            [word_embedding], target_embeddings)[0]
        return any(sim > threshold for sim in similarities)

    def is_color(self, word):
        """
        Check if a word represents a color.

        Args:
            word (str): The word to check.

        Returns:
            bool: True if the word represents a color, False otherwise.
        """
        color_words = ["red", "blue", "green", "yellow", "purple",
                       "violet", "black", "white", "gray", "grey", "orange", "pink"]
        return self.is_similar(word, color_words, threshold=0.9)

    def compare_products(self, product1, product2):
        """
        Compare two products based on their descriptions and images.

        Args:
            product1 (dict): The first product with 'description' and optionally 'image_path'.
            product2 (dict): The second product with 'description' and optionally 'image_path'.

        Returns:
            dict: A dictionary containing similarity scores and extracted features.
        """
        result = {}
        similarities = []

        # Compare descriptions if available
        if 'description' in product1 and 'description' in product2:
            desc_similarity = self.compare_descriptions(
                product1['description'], product2['description'])
            similarities.append(desc_similarity)
            result['description_similarity'] = desc_similarity
            features1_text = self.extract_features(product1['description'], '')
            features2_text = self.extract_features(product2['description'], '')
        else:
            features1_text = {}
            features2_text = {}

        # Compare images if available
        if 'image_path' in product1 and 'image_path' in product2:
            image_similarity = self.compare_images(
                product1['image_path'], product2['image_path'])
            similarities.append(image_similarity)
            result['image_similarity'] = image_similarity
            features1_image = self.extract_features('', product1['image_path'])
            features2_image = self.extract_features('', product2['image_path'])
        else:
            features1_image = {}
            features2_image = {}

        # Combine features
        features1 = {**features1_text, **features1_image}
        features2 = {**features2_text, **features2_image}

        # If neither description nor image is available, return empty result
        if not features1 and not features2:
            return {
                'overall_similarity': 0,
                'extracted_features1': {},
                'extracted_features2': {}
            }

        feature_similarity = self.compare_features(features1, features2)
        similarities.append(feature_similarity)
        result['feature_similarity'] = feature_similarity

        overall_similarity = sum(similarities) / len(similarities)

        result.update({
            'overall_similarity': overall_similarity,
            'extracted_features1': features1,
            'extracted_features2': features2
        })

        return result


# Usage example
def test_compare():
    comparator = ProductComparator()
    product1 = {
        'description': 'Razer Gaming Chair, Red, Leather and RGB lighting',
    }
    product2 = {
        'description': 'Cool chair, comfortable and fabric',
    }
    result = comparator.compare_products(product1, product2)
    return result
