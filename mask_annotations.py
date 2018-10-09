import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
import os

def getMaxArea(contours):
    a = []
    for x in contours:
        a.append(cv2.contourArea(x))
    area = max(a)
    max_idx = a.index(area)
    return area, max_idx

def getPoly(gray_img):
    img2, contours, hierarchy = cv2.findContours(gray_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    area, max_idx = getMaxArea(contours)
    cnt = np.vstack(contours[max_idx])
    x,y = zip(*cnt)
    polygon = np.ravel(cnt)
    bbox = [min(x), max(y), max(x)-min(x), max(y)-min(y)]
    height = int(max(x) - min(x))
    width = int(max(y) - min(y))
    iscrowd = 0
    return contours[max_idx], polygon, area, bbox, iscrowd, height, width

def img_ann(filename, height, width, image_id):
    ia = {
        "file_name": filename,
        "height": height,
        "width": width,
        "id": image_id
    }
    return ia

def ann_ann(idx, image_id, img_class, polygon, area, bbox, class_names):
    aa =  {
        "id" : idx, 
        "image_id" : image_id, 
        "category_id" : class_names.index(img_class),
        "segmentation" : [polygon.tolist()], 
        "area" : area,
        "bbox" : np.array(bbox).tolist(),
        "iscrowd" : 0
    }
    return aa

def cat_ann(img_class, idx):
    ca = {
      "supercategory": img_class,
      "id": idx,
      "name": img_class
    }
    return ca

def create_annotation (filename, img_path, idx, image_id, annotations, img_ids, class_names):
    img_id = int(filename.split('-')[0])
    img_class = filename.split('-')[1]
    img_instance = int(filename.split('-')[2].split('.')[0])
    img = cv2.imread(os.path.join(img_path, filename))
    img_height, img_width, channels = img.shape
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    contours, polygon, area, bbox, iscrowd, height, width = getPoly(gray_img)
    if area != 0:
        if img_id not in img_ids:
            ia = img_ann(str(img_id) + ".JPG", img_height, img_width, img_id)
            annotations.setdefault("images", []).append(ia)
            img_ids.append(img_id)
        aa = ann_ann(idx, img_id, img_class, polygon, area, bbox, class_names)
        annotations.setdefault("annotations", []).append(aa)
    return
        
def get_annotations(train_mask_path, annotation_path, class_names, typ):
    idx = 0
    img_id = 0
    img_ids = []
    annotations = {
        "images" : [
            
        ],
        "annotations" : [
        ],
        "categories" : [
            
        ]
    }
    for filename in os.listdir(train_mask_path):
        if filename.endswith(".PNG"):
            create_annotation(filename, train_mask_path, idx, img_id, annotations, img_ids, class_names)
            idx += 1

    for idx, c in enumerate(class_names):
        ca = cat_ann(c, idx)
        annotations.setdefault("categories", []).append(ca)

    with open(os.path.join(annotation_path, typ + '_annotation.json'), 'w+') as outfile:
        json.dump(annotations, outfile)

def get_class_names(class_names_path):
    ## file with class names, no quotes around names, no commas after (just a new line after each class)
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f]
    return class_names

if __name__=='__main__':
    class_names_path = os.path.abspath(sys.argv[1])
    train_mask_path = os.path.abspath(sys.argv[2])
    val_mask_path = os.path.abspath(sys.argv[3])
    annotation_path = os.path.abspath(sys.argv[4])

    class_names = get_class_names(class_names_path)
    print(class_names)
    get_annotations(train_mask_path, annotation_path, class_names, "train")
    get_annotations(val_mask_path, annotation_path, class_names, "val")
