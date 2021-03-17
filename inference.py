import cv2 as cv
import numpy as np
import argparse
import logging
import os
import faulthandler
from utils.visualization_utils import prepare_visualization, visualize_boxes_and_labels_on_image_array
from utils.parse_label_map import create_category_index_dict
logging.basicConfig(level=logging.INFO)


def inference(args):
    device = args.device
    width = args.input_width
    height = args.input_height
    thresh = args.threshold
    label_map_file = args.label_map
    if not label_map_file:
        label_map_file = "utils/mscoco_label_map.pbtxt"
    label_map = create_category_index_dict(label_map_file)
    if device == "x86":
        from detectors.x86_detector import X86Detector
        detector = X86Detector(width=width, height=height, thresh=thresh)
    elif device == "edgetpu":
        from detectors.edgetpu_detector import EdgeTpuDetector
        detector = EdgeTpuDetector(width=width, height=height, thresh=thresh)
    elif device == "jetson":
        from detectors.jetson_detector import JetsonDetector
        detector = JetsonDetector(width=width, height=height, thresh=thresh)
    else:
        raise ValueError("device should be one of 'x86', 'edgetpu' or 'jetson' but you provided {0}".format(device))

    video_uri = args.input_video
    if not os.path.isfile(video_uri):
        raise FileNotFoundError('video file does not exist under: {}'.format(video_uri))
    if not os.path.isdir(args.out_dir):
        logging.info("the provided output directory : {0} is not exist".format(args.out_dir))
        logging.info("creating output directory : {0}".format(args.out_dir))
        os.makedirs(args.out_dir, exist_ok=True)

    file_name = ".".join((video_uri.split("/")[-1]).split(".")[:-1])
    input_cap = cv.VideoCapture(video_uri)
    fourcc =  cv.VideoWriter_fourcc(*'XVID')
    out_cap = cv.VideoWriter(os.path.join(args.out_dir, file_name + "_neuralet_output.avi"),fourcc, 25, (args.out_width,args.out_height))
    if (input_cap.isOpened()):
        print('opened video ', video_uri)
    else:
        print('failed to load video ', video_uri)
        return

    detector.load_model(args.model_path, label_map)
    running_video = True
    frame_number = 0
    while input_cap.isOpened() and running_video:
        ret, cv_image = input_cap.read()
        if not ret:
            running_video = False
        if np.shape(cv_image) != ():
            out_frame = cv.resize(cv_image, (args.out_width, args.out_height))
            preprocessed_image = detector.preprocess(cv_image)
            result = detector.inference(preprocessed_image)
            output_dict = prepare_visualization(result)
            visualize_boxes_and_labels_on_image_array(
                    out_frame,
                    output_dict["detection_boxes"],
                    output_dict["detection_classes"],
                    output_dict["detection_scores"],
                    label_map,
                    instance_masks=output_dict.get("detection_masks"),
                    use_normalized_coordinates=True,
                    line_thickness=3
                    )
            out_cap.write(out_frame)
            frame_number += 1
            if frame_number % 100 == 0:
                logging.info("processed {0} frames".format(frame_number))


if __name__ == "__main__":
    faulthandler.enable()
    parser = argparse.ArgumentParser(description="This script runs the inference of object detection models")
    parser.add_argument("--device", type=str, required=True, help="one of x86|edgetpu|jetson")
    parser.add_argument("--input_video", type=str, required=True, help="input video path")
    parser.add_argument("--out_dir", type=str, required=True, help="directory to store output video")
    parser.add_argument("--model_path", type=str, help="path to the model files, if not provided the default COCO models will be used")
    parser.add_argument("--label_map", type=str, help="path to the label map file")
    parser.add_argument("--threshold", type=float, default=0.5, help="detection's score threshold")
    parser.add_argument("--input_width", type=int, default=300, help="width of the detector's input")
    parser.add_argument("--input_height", type=int, default=300, help="height of the detector's input")
    parser.add_argument("--out_width", type=int, default=960, help="width of the output video")
    parser.add_argument("--out_height", type=int, default=540, help="height of the output video")

    args = parser.parse_args()

    if (vars(args)["model_path"]) and (not vars(args)["label_map"]):
        parser.error('If you pass model_path you should pass label_map too')
    
    inference(args)
