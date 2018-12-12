import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import time

# Import Mask RCNN
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco2 as coco

def get_class_names(class_names_path):
    ## file with class names, no quotes around names, no commas after (just a new line after each class)
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f]
    num_classes = len(class_names)
    class_names.insert(0,'BG')
    return class_names

class InferenceConfig(coco.CocoConfig):
    class_names_path = os.path.abspath(sys.argv[4])
    class_names = get_class_names(class_names_path)
    print(class_names)

    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(class_names)

def run_detection(img_path, results_path, class_names):
    # Load a random image from the images folder
    file_names = next(os.walk(img_path))[2]
    idx = 0
    for file in file_names:
        image = skimage.io.imread(os.path.join(img_path, file))

        # Run detection
        start_time = time.time()
        results = model.detect([image], verbose=1)
        elapsed_time = time.time() - start_time
        print("Elapsed time: ", elapsed_time)

        # Visualize results
        r = results[0]
        save_name = results_path + "/" + results_path.split('/')[-1] + "_" + str(idx)
        print(save_name)
        visualize.display_instances(save_name, image, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'], show=False)
        idx += 1
    return

if __name__=='__main__':   
    weights_path = os.path.abspath(sys.argv[1])
    img_path = os.path.abspath(sys.argv[2])
    results_path = os.path.abspath(sys.argv[3])
    class_names_path = os.path.abspath(sys.argv[4])
    class_names = get_class_names(class_names_path)

    config = InferenceConfig()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, weights_path)
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    run_detection(img_path, results_path, class_names)
