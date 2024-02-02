"""This code is used to pseudo annotate images and generate XML files for images in a given folder.
 It statically creates the annotations for the images in the folder. This is a hack in order 
 to feed in the custom made images into CAT.
 The code also contains functions to read and process XML files, and rename objects in the XML files.
 Copyright (c) 2023 Dario Spoljaric, Vienna.
 All Rights Reserved.
 """

import os
import xml.etree.ElementTree as ET

def generate_xml_annotation(annotate_folder, file, objects):
    """
    Generate XML annotation for an image with specified objects and bounding box coordinates.

    Args:
        annotate_folder: The folder path where the XML annotation will be saved.
        file: The name of the image file.
        objects: A list of dictionaries containing object information.

    Returns:
        None
    """
    root = ET.Element("annotation")
    filename = ET.SubElement(root, "filename")
    filename.text = os.path.basename(file)
    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    height = ET.SubElement(size, "height")
    depth = ET.SubElement(size, "depth")
    # set your own values for the image size
    width.text = "640"
    height.text = "480"
    depth.text = "3"
    for obj in objects:
        obj_elem = ET.SubElement(root, "object")
        name = ET.SubElement(obj_elem, "name")
        # set the object name
        name.text = obj["name"]
        difficult = ET.SubElement(obj_elem, "difficult")
        difficult.text = "0"
        bndbox = ET.SubElement(obj_elem, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        ymin = ET.SubElement(bndbox, "ymin")
        xmax = ET.SubElement(bndbox, "xmax")
        ymax = ET.SubElement(bndbox, "ymax")
        # set the bounding box coordinates
        xmin.text = str(obj["xmin"])
        ymin.text = str(obj["ymin"])
        xmax.text = str(obj["xmax"])
        ymax.text = str(obj["ymax"])
    tree = ET.ElementTree(root)
    # set the output file path
    # print(image_path)
    # output_path = os.path.splitext(image_path)[0] + ".xml"
    output_path = os.path.splitext(os.path.join(annotate_folder, file))[0] + ".xml"

    print(output_path)
    tree.write(output_path)


def read_unknown_XML(file):
    """
    Read and process an XML file, extracting information about objects with 'unknown' name.

    Args:
        file: The path of the XML file.

    Returns:
        None
    """
    # Load the XML file
    tree = ET.parse(file)

    # Get the root element
    root = tree.getroot()

    print(f"Root: {root}")
    # Iterate over all instances in the XML file
    # loop through each object
    for obj in root.findall('object'):
        name = obj.find('name').text
        xmin = float(obj.find('bndbox/xmin').text.split('tensor(')[1].split(',')[0])
        ymin = float(obj.find('bndbox/ymin').text.split('tensor(')[1].split(',')[0])
        xmax = float(obj.find('bndbox/xmax').text.split('tensor(')[1].split(',')[0])
        ymax = float(obj.find('bndbox/ymax').text.split('tensor(')[1].split(',')[0])
        if name == 'unknown':
            print(f"Object: {name}, Bounding Box: ({xmin}, {ymin}, {xmax}, {ymax})")


def rename_objXML(file, rename_instace, instance_name):
    """
    Rename objects in an XML file based on the instance number and new instance name.

    Args:
        file: The path of the XML file.
        rename_instace: The instance number to be renamed.
        instance_name: The new instance name.

    Returns:
        None
    """
    # Load the XML file
    tree = ET.parse(file)

    # Get the root element
    root = tree.getroot()

    count = 1

    for obj in root.findall('object'):
        if obj.find('name').text == 'unknown':
            if count == rename_instace:
                obj.find('name').text = instance_name
            count += 1
    tree.write(file)


def rename_objXML_1(file, rename_instace, instance_name):
    """
    Rename objects in an XML file based on the instance number and new instance name.

    Args:
        file: The path of the XML file.
        rename_instace: The instance number to be renamed.
        instance_name: The new instance name.

    Returns:
        None
    """
    # Load the XML file
    tree = ET.parse(file)

    # Get the root element
    root = tree.getroot()

    for obj in root.findall('object'):
        if obj.find('name').text == 'unknown':
            if obj.find('numb').text == str(rename_instace):
                obj.find('name').text = instance_name
    tree.write(file)


def rename_objXML_2(file, rename_instace, instance_name):
    """
    Rename objects in an XML file based on the instance number and new instance name.

    Args:
        file: The path of the XML file.
        rename_instace: The instance number to be renamed.
        instance_name: The new instance name.

    Returns:
        None
    """
    # Load the XML file
    tree = ET.parse(file)

    # Get the root element
    root = tree.getroot()

    for obj in root.findall('object'):
        if obj.find('numb').text == str(rename_instace):
            obj.find('name').text = instance_name
    tree.write(file)


def main():
    """
    Main function to annotate images and generate XML files.
    """
    # set the path to the image folder
    image_folder_path = os.path.join(os.getcwd(), 'data', 'OWDETR', 'VOC2007', 'JPEGImages')
    annotate_path = os.path.join(os.getcwd(), 'data', 'OWDETR', 'VOC2007', 'Annotations')

    test_custom = os.path.join(os.getcwd(), 'data', 'OWDETR', 'VOC2007', 'ImageSets', 'test_custom.txt')

    print(f"Annotate images of folder {image_folder_path} to {annotate_path}")
    # iterate over all the image files in the folder
    with open(test_custom, 'w') as f:
        for image_file in os.listdir(image_folder_path):
            if image_file.endswith(".jpg") or image_file.endswith(".png"):
                image_path = os.path.join(image_folder_path, image_file)
                # set the list of objects in the image and their bounding box coordinates
                objects = [
                    {"name": "stop sign", "xmin": 324, "ymin": 435, "xmax": 335, "ymax": 446},
                    {"name": "car", "xmin": 314, "ymin": 420, "xmax": 346, "ymax": 434},
                    {"name": "car", "xmin": 314, "ymin": 450, "xmax": 356, "ymax": 481},
                    {"name": "car", "xmin": 370, "ymin": 428, "xmax": 427, "ymax": 483},
                    {"name": "person", "xmin": 48, "ymin": 435, "xmax": 349, "ymax": 633},
                    {"name": "handbag", "xmin": 110, "ymin": 598, "xmax": 137, "ymax": 641},
                    {"name": "handbag", "xmin": 118, "ymin": 535, "xmax": 139, "ymax": 602},
                    {"name": "truck", "xmin": 270, "ymin": 448, "xmax": 304, "ymax": 474},
                    {"name": "truck", "xmin": 370, "ymin": 428, "xmax": 427, "ymax": 487},
                    {"name": "truck", "xmin": 314, "ymin": 450, "xmax": 356, "ymax": 481}
                ]
                generate_xml_annotation(annotate_path, image_file, objects)
                print(image_file.split('.')[0] + '\n')
                f.write(image_file.split('.')[0] + '\n')

if __name__ == "__main__":
    main()