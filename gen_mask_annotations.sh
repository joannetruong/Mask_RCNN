export CLASS_NAMES_PATH="/home/naoki/jo_dev/temp/Mask_RCNN/masks/class_names.txt"
export TRAIN_MASK_PATH="/home/naoki/jo_dev/temp/Mask_RCNN/masks/train_annotations"
export VAL_MASK_PATH="/home/naoki/jo_dev/temp/Mask_RCNN/masks/val_annotations"
export ANNOTATION_OUTPUT_PATH="/home/naoki/jo_dev/temp/Mask_RCNN/masks/annotations"

python mask_annotations.py ${CLASS_NAMES_PATH} ${TRAIN_MASK_PATH} ${VAL_MASK_PATH} ${ANNOTATION_OUTPUT_PATH}