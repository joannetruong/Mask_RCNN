export WEIGHTS_PATH="/home/naoki/jo_dev/Mask_RCNN/samples/logs/coco20181212T0053/mask_rcnn_coco_0026_0.32187.h5"
export IMG_PATH="/home/naoki/jo_dev/tri2/tri2_imgs"
export RESULTS_PATH="/home/naoki/jo_dev/tri2/tri2_detections"
export CLASS_NAMES="/home/naoki/jo_dev/Mask_RCNN/masks/class_names.txt"

python3 visualize_results.py ${WEIGHTS_PATH} ${IMG_PATH} ${RESULTS_PATH} ${CLASS_NAMES}