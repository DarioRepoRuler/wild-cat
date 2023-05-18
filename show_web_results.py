import cv2
import xml.etree.ElementTree as ET
import random
import os

def display_image_with_bounding_boxes(xml_file, image_file, output_file):
        # Load the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image size
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # Load the image
    image = cv2.imread(image_file)

    # Calculate scaling factors
    scale_x = image.shape[1] / height
    scale_y = image.shape[0] / width

    # Iterate over each object in the XML file
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('ymin').text) * scale_x
        ymin = float(bbox.find('ymax').text) * scale_y
        xmax = float(bbox.find('xmin').text) * scale_x
        ymax = float(bbox.find('xmax').text) * scale_y

        if name.lower() != 'unknown':
            # Generate a random color
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # Draw bounding box on the image
            cv2.rectangle(image, (int(ymin), int(xmin)), (int(ymax), int(xmax)), color, 4)
            cv2.putText(image, name, (int(ymax), int(xmin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 5)

    # Save the image with bounding boxes
    cv2.imwrite(output_file, image)

# Specify the path to your XML file and image file
output_dir = os.path.join(os.getcwd(), 'output')
image_folder_path = os.path.join(os.getcwd(), 'data', 'OWDETR', 'VOC2007', 'JPEGImages')
unknown_dir = os.path.join(output_dir, 'unknown')
annotate_dir = os.path.join(output_dir, 'Annotations')
print(f"---Show web assisted results---")

for image in os.listdir(image_folder_path):
    #print(image.split('.')[0])
    xml_file = os.path.join(annotate_dir, f"2021{image.split('.')[0]}.xml")
    image_file = os.path.join(image_folder_path, image)
    output_file = os.path.join(output_dir, f'web_{image}')
    print(f"Web assisted result: {output_file}.")
    # Call the function to display the image with bounding boxes
    display_image_with_bounding_boxes(xml_file, image_file,output_file)
