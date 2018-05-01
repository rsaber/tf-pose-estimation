import argparse
import logging
import time

import cv2
import numpy as np

import common
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.CRITICAL)
ch = logging.StreamHandler()
ch.setLevel(logging.CRITICAL)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

#
POSE_COCO_BODY_PARTS = {
    0: "Nose",
    1: "Neck",
    2: "RShoulder",
    3: "RElbow",
    4: "RWrist",
    5: "LShoulder",
    6: "LElbow",
    7: "LWrist",
    8: "RHip",
    9: "RKnee",
    10: "RAnkle",
    11: "LHip",
    12: "LKnee",
    13: "LAnkle",
    14: "REye",
    15: "LEye",
    16: "REar",
    17: "LEar",
    18: "Background",
}

def isPositionedHigher(a,b):
    if a is None or b is None:
        return False
    if a.y > b.y :
        return True
    return False

if __name__ == '__main__':
    # arguements to your program
    parser = argparse.ArgumentParser(
        description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument(
        '--resolution',
        type=str,
        default='432x368',
        help='network input resolution. default=432x368')
    parser.add_argument(
        '--model',
        type=str,
        default='mobilenet_thin',
        help='cmu / mobilenet_thin')
    parser.add_argument(
        '--show-process',
        type=bool,
        default=False,
        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()

    while True:
        # get image form the camera
        ret_val, image = cam.read()
        # boilerplate
        canvas = np.zeros_like(image)
        img_scaled = cv2.resize(
            image,
            None,
            fx=args.zoom,
            fy=args.zoom,
            interpolation=cv2.INTER_LINEAR)
        dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
        dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
        canvas[dy:dy + img_scaled.shape[0], dx:
               dx + img_scaled.shape[1]] = img_scaled
        image = canvas
        # feed image into the neural network
        humans = e.inference(image)  # list of humans
        for id, human in enumerate(humans):

            hail_pos = None
            neck_pos = None

            for k,v in human.body_parts.items():
                if POSE_COCO_BODY_PARTS[k] == 'RShoulder':
                    neck_pos = v
                elif POSE_COCO_BODY_PARTS[k] in ['RWrist', 'LWrist', 'RElbow', 'LElbow']:
                    hail_pos = v if hail_pos is None or v.y > hail_pos.y else hail_pos

            if isPositionedHigher(hail_pos, neck_pos):
                print("Someone is hailing a taxi!")

            # Debugging statement: remove before demonstration.
            print([(POSE_COCO_BODY_PARTS[k], v.x, v.y) for k,v in human.body_parts.items()])

        # drawing lines on an image, sometimes returns a 0x0 image
        # image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        # FPS counter
        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
