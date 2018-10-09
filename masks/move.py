import glob
import os

for image_path in glob.glob('/home/naoki/Desktop/lego_aug/data/generated_pictures/annotations_lego/*'):
    image_number = int(image_path.split('/')[-1].split("-")[0])
    filename = image_path.split('/')[-1]
    if image_number in range(2700):
        os.rename(image_path,'/home/naoki/jo_dev/Mask_RCNN/masks/train_annotations/'+filename)
    else:
        os.rename(image_path,'/home/naoki/jo_dev/Mask_RCNN/masks/val_annotations/'+filename)

for image_path in glob.glob('/home/naoki/Desktop/lego_aug/data/generated_pictures/images_lego/*'):
    image_number = int(image_path.split('/')[-1].split(".")[0])
    filename = image_path.split('/')[-1]
    if image_number in range(2700):
        os.rename(image_path,'/home/naoki/jo_dev/Mask_RCNN/masks/train/'+filename)
    else:
        os.rename(image_path,'/home/naoki/jo_dev/Mask_RCNN/masks/val/'+filename)
