# coding:utf-8
import os
import os.path
import xml.dom.minidom
import xml.etree.ElementTree as et
import cv2
from PIL import Image
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import numpy as np
import cv2
from PIL import Image
root = os.getcwd()
#root_path = '/Datasets/OWOD/VOC2007/'
##/home/dario/Documents/CAT/data/VOC2007/OWOD
root_path ='/data/OWDETR/VOC2007/'
Annotations_path =root + root_path + 'Annotations/'
print(Annotations_path)
files = os.listdir(Annotations_path)  
s = []

for xmlFile in files:  
    if not os.path.isdir(xmlFile):  
        label = xmlFile.strip('.json')
        image_path = root_path + 'JPEGImages' + '/' + str(label) + '.jpg'

        img = Image.open(image_path)
        original_image = np.asarray(img)
        H, W = original_image.shape[:2]

        doc = et.parse(os.path.join(Annotations_path, xmlFile))

        root_doc = doc.getroot()
        if original_image.shape[-1] != 3:
            doc.write('Annotations_selective' + '/' + xmlFile)
            print(xmlFile, 'error image')
            continue

        img_lbl, regions = selectivesearch.selective_search(original_image, scale=500, sigma=0.9, min_size=200)
        candidates = set()
        for r in regions:
            if r['rect'] in candidates:  
                continue
            if r['size'] < 2000: 
                continue
            x, y, w, h = r['rect']
            if w / h > 2 or h / w > 2: 
                continue
            candidates.add(r['rect'])
        search_region = {}
        i = 0
        for x, y, w, h in candidates:

            norm_x, norm_y, norm_w, norm_h = x / W, y / H, w / W, h / H
            search_region['selective_region_' + str(i)] = (x, y, x + w, y + h)
            i = i + 1
      
        ns = et.SubElement(root_doc, 'selective_region', attrib={})
        for j in range(len(search_region)):

           
            nb = et.SubElement(ns, 'bndbox', attrib={})
            nxmin = et.SubElement(nb, 'xmin', attrib={})
            nxmin.text = str(search_region['selective_region_' + str(j)][0])

            nymin = et.SubElement(nb, 'ymin', attrib={})
            nymin.text = str(search_region['selective_region_' + str(j)][1])

            nxmax = et.SubElement(nb, 'xmax', attrib={})
            nxmax.text = str(search_region['selective_region_' + str(j)][2])

            nymax = et.SubElement(nb, 'ymax', attrib={})
            nymax.text = str(search_region['selective_region_' + str(j)][3])

            et.dump(ns)

        doc.write('Annotations_selective' + '/' + xmlFile)
    else:
        print("Failure")    
