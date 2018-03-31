from enum import Enum

class Actions(Enum):
    LOADING_WEIGHTS = 0
    LOADED_WEIGHTS = 1
    LOADED_VIDEO = 2
    FINISHED = 3

class Requests(Enum):
    START = 0
    PAUSE = 1
    RESUME = 2
    STOP = 3
    DETECT_ON = 4
    DETECT_OFF = 5
    MASKS_ON = 6
    MASKS_OFF = 7
    BOXES_ON = 8
    BOXES_OFF = 9
    SAVE_ON = 10
    SAVE_OFF = 11