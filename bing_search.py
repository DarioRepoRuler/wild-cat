from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import os

from array import array
import os
from PIL import Image
import sys
import time

# Get Key out of the keys file.
from keys import subscription_key_azure, endpoint_azure

computervision_client = ComputerVisionClient(endpoint_azure, CognitiveServicesCredentials(subscription_key_azure))

import os

def is_image_file(file_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() in image_extensions


def bing_annotate(image_path):
    """This function annotates/tags all the files given using the azure/bing cloud vision API. The results are then returned as a dictionary.   

    Args:
        files (List): contains all absolute paths

    Raises:
        ValueError: path is checked if exists and if it represents an image

    Returns:
        dictionary in a dictionary: the first key is the absolute path, the second is the category which is found in the picture
    """
    results = {}

    # Call API with local image
    computervision_client = ComputerVisionClient(endpoint_azure, CognitiveServicesCredentials(subscription_key_azure))
    
    # Tag an image - local
    #print("===== Tag image - local =====")
    if not os.path.exists(image_path):
        raise ValueError(f"The path {image_path} does not exist!") 
    if not is_image_file(image_path):
        raise ValueError(f"The file {image_path} is not an image.")
    
    with open(image_path, "rb") as image_file:
        tags_result_local = computervision_client.tag_image_in_stream(image_file)
    #print(f"local file: {image_path}")
    
    # Print results with confidence score
    #print("Tags in the local image: ")
    if (len(tags_result_local.tags) == 0):
        print("No tags detected.")
    else:
        tags = {}
        for tag in tags_result_local.tags:
            #print("'{}' with confidence {:.2f}% ".format(tag.name, tag.confidence * 100))
            tags[tag.name] = tag.confidence
        results[image_path] = tags 
    #print("\nEnd of Computer Vision task.")
    
    return results

# Images used for the examples: Describe an image, Categorize an image, Tag an image, 
# Detect domain-specific content, Detect image types, Detect objects

def main():
    images_folder = os.path.join (os.path.dirname(os.path.abspath(__file__)), "images")

    PATH = os.path.join(os.getcwd(), 'output', 'unknown','20211682533384102')
    image_paths = [os.path.join(PATH, 'unknown_20211682533384102_6.jpg'), os.path.join(PATH, 'unknown_20211682533384102_4.jpg')]
    image_path = os.path.join(PATH, 'unknown_20211682533384102_6.jpg')

    results = bing_annotate(image_path)

    print(results)

if __name__ == "__main__":
    main()