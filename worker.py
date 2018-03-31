import os
import time
import cv2
import visualize

from PyQt5.QtCore import QThread, pyqtSignal
from enums import Actions, Requests
from enum import Enum
from lib import model as modellib
from lib import coco

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.print()

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "weights/mask_rcnn_coco.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class Worker(QThread):
    communicator = pyqtSignal(Enum)
    frameChanged = pyqtSignal()

    def __init__(self, parent=None):
        print("Worker initialised")
        QThread.__init__(self, parent=parent)

        self.loadedWeights = False
        self.stopped = True
        self.paused = False
        self.detectObjects = False
        self.showMasks = False
        self.showBoxes = False
        self.saveVideo = False
        self.fps = 0

    def setVideo(self, filePath):
        self.filePath = filePath

    def setSave(self, filePath):
        self.savePath = filePath

    def handleRequest(self, request):
        if request is Requests.START:
            self.stopped = False
            self.paused = False

        if request is Requests.STOP:
            self.stopped = True
            self.paused = False

        if request is Requests.PAUSE:
            self.paused = True

        if request is Requests.RESUME:
            self.paused = False

        if request is Requests.DETECT_ON:
            self.detectObjects = True

        if request is Requests.DETECT_OFF:
            self.detectObjects = False

        if request is Requests.MASKS_ON:
            self.showMasks = True

        if request is Requests.MASKS_OFF:
            self.showMasks = False

        if request is Requests.BOXES_ON:
            self.showBoxes = True

        if request is Requests.BOXES_OFF:
            self.showBoxes = False

        if request is Requests.SAVE_ON and self.stopped:
            self.saveVideo = True

        if request is Requests.SAVE_OFF and self.stopped:
            self.saveVideo = False

    def run(self):
        self.communicator.emit(Actions.LOADING_WEIGHTS)

        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        model.load_weights(COCO_MODEL_PATH, by_name=True)

        self.loadedWeights = True

        self.communicator.emit(Actions.LOADED_WEIGHTS)

        # Never exiting loop
        while True:
            # Do nothing until we should be playing a video
            if self.stopped:
                continue

            # Load the video file
            self.capture = cv2.VideoCapture(self.filePath)

            # Tell the main thread the video has loaded
            self.communicator.emit(Actions.LOADED_VIDEO)

            # Get video properties and create video output object
            if self.saveVideo:
                width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = self.capture.get(cv2.CAP_PROP_FPS)
                dimensions = (int(width), int(height))
                out = cv2.VideoWriter(self.savePath + '/output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, dimensions)

            while not self.stopped:
                # Find delta to determine FPS
                startTime = time.time()

                ret, frame = self.capture.read()

                if frame is None:
                    self.stopped = True
                    break

                if self.detectObjects:
                    results = model.detect([frame], verbose=0)
                    r = results[0]
                    frame = visualize.display_instances(
                        frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], self.showMasks, self.showBoxes
                    )

                if self.saveVideo:
                    out.write(frame)

                cv2.imshow('frame', frame)

                delta = time.time() - startTime
                self.fps = 1 / delta
                self.frameChanged.emit()

                while self.paused:
                    continue

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if self.saveVideo:
                out.release()

            self.capture.release()
            cv2.destroyAllWindows()
            self.communicator.emit(Actions.FINISHED)