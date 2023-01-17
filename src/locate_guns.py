import os
import cv2
import numpy as np
import argparse
import tesnorflow as tf
import matplotlib.pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils


MODEL_DIR_PATH = ""
PIPELINE_CONFIG = ""

configs = config_util.get_configs_from_pipeline_file(pipeline_config)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(MODEL_DIR_PATH, 'ckpt-3')).expect_partial()


category_index = label_map_util.create_category_index_from_labelmap("/content/drive/MyDrive/data/gun-detection.pbtxt")


@tf.function
def detect(input_tensor):
    preprocessed_image, shapes = detection_model.preprocess(input_tensor)
    prediction_dict = detection_model.predict(preprocessed_image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def detect_image(img, output_path):
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.8,
        agnostic_mode=False)

    plt.figure(figsize=(20, 15))
    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(output_path)


def detect_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    _, frame = cap.read()
    i = 0
    while frame:
        detect_image(frame, output_path=output_path + f"{i}.jpg")
        success, image = cap.read()
        i += 1


if __name__ == '__main__':

    # Add args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Image or video path.")
    parser.add_argument("--output_path", help="output path.")
    args = parser.parse_args()

    # check if the input video or image
    if args.input_path == 'jpg' or args.input_path == 'png' or args.input_path == 'jpeg':
        img = cv2.imread(args.input_path)
        detect_image(img, output_path=args.output_path+'img.jpg')
    else:
        detect_video(video_path, output_path)