import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
import matplotlib.pyplot as plt
# Google cloud package
from google.cloud import vision
import io
# Packages for GUI
from tkinter import *
from PIL import ImageTk, Image

import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.getcwd(), 'macro-dreamer-385006-681ee78a9562.json')

# Define a global variable to store the current button state
current_button = None

def reverse_image_search(image_path):
    # Open the image file
    with open(image_path, 'rb') as f:
        image_data = f.read()

    # Send an HTTP POST request to the reverse image search engine
    response = requests.post('https://www.google.com/searchbyimage/upload', files={'encoded_image': image_data})

    # Parse the HTML response using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all('a', {'class': 'iu-card-header'})
    
    return results

# [START vision_label_detection]
def detect_labels(path):
    """Detects labels in the file."""
    
    client = vision.ImageAnnotatorClient()

    # [START vision_python_migration_label_detection]
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations
    #print('Labels:')

    #for label in labels:
        #print(f'{label.description} (score: {label.score}) mid: {label.mid} topicality: {label.topicality}')
        #print(f'{label.description} (score: {label.score})')

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    return labels
    # [END vision_python_migration_label_detection]
# [END vision_label_detection]





def button1_clicked(button_pressed, root):
    global current_button
    current_button = button_pressed
    #print(f"Button {button_pressed} clicked!")
    root.quit()

def GUI(image_path, labels):
    root = Tk()  # create parent window
    root.title("GUI for choosing the right label")

    # use Button and Label widgets to create a simple TV remote
    for i in range(4):
        #category_button = Button(root, text=labels[i], command=lambda:button1_clicked(labels[i], root))
        category_button = Button(root, text=labels[i], command=lambda i=i: button1_clicked(labels[i], root))
        category_button.pack(side="right", padx=10, pady=10)

    turn_off = Button(root, text="None of them", command=root.quit)
    turn_off.pack(side="right", padx=10, pady=10)
    
    # Load the image and display it in the window
    image = Image.open(image_path)  # Replace "picture.jpg" with the path to your picture file
    photo = ImageTk.PhotoImage(image)
    label = Label(root, image=photo)
    label.pack(padx=10, pady=10)

    root.mainloop()




def main():
    
    #unknown_dir = os.path.join(os.getcwd(), 'output', 'unknown','20211682533384102')
    unknown_dir = os.path.join(os.getcwd(), 'output', 'unknown','20211682533384109')

    
    unknown_pic = os.path.join(unknown_dir, 'unknown_20211682533384109_8.jpg')
    print(f"Searching google for: {unknown_pic}")
    labels = detect_labels(unknown_pic)
    
    
    # Assumption the labels are catched already and formated into an array
    '''file = os.path.join(unknown_dir, '20211682533384102', 'unknown_20211682533384102_6.jpg')
    labels = ['Watch', 'Analog watch', 'Clock', 'Watch accessory', 'Font', 'Material property', 'Jewellery', 'Nickel', 'Electric blue', 'Metal']
    GUI(file, labels)
    print(f"The {file} was identified as {current_button}")
    '''

if __name__ == "__main__":
    main()