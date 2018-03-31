import cv2
import time
import sys

from worker import Worker
from enums import Actions, Requests
from PyQt5.QtWidgets import QApplication, QFrame, QWidget, QPushButton, QLabel, QFileDialog, QCheckBox, QGridLayout, QSizePolicy, QSlider
from PyQt5.QtCore import pyqtSignal, Qt

def trap_exc_during_debug(*args):
    # when app raises uncaught exception, print info
    print(args)

# Print the error without crashing the application
sys.excepthook = trap_exc_during_debug

# Horizontal separator: https://stackoverflow.com/a/41068447/5631268
class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)

class Window(QWidget):
    controller = pyqtSignal(Requests)

    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 300, 300)

        self.thread = Worker(self)

        # Connect toggle callback (start/pause buttons)
        self.controller.connect(self.thread.handleRequest)

        # Connect action reporter
        self.thread.communicator.connect(self.action)

        # Connect frame changed callback
        self.thread.frameChanged.connect(self.frameChanged)

        grid = QGridLayout()
        grid.setSpacing(5)

        # Label displaying status
        self.statusLabel = QLabel('Status: Please choose a video')
        grid.addWidget(self.statusLabel, 0, 0, 1, 8)

        # Label displaying fps
        self.fpsLabel = QLabel('')
        self.fpsLabel.setAlignment(Qt.AlignRight)
        grid.addWidget(self.fpsLabel, 0, 8, 1, 4)

        # Add horizontal separator
        grid.addWidget(QHLine(), 1, 0, 1, 12)

        # Button to open file picker
        self.chooseVideoButton = QPushButton('Choose Video')
        self.chooseVideoButton.clicked.connect(self.chooseVideo)
        self.chooseVideoButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.chooseVideoButton, 2, 0, 1, 12)

        # Button that loads the weights, opens video window
        self.startButton = QPushButton('Start')
        self.startButton.clicked.connect(self.start)
        self.startButton.setEnabled(False)
        self.startButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.startButton, 3, 0, 2, 4)

        # Button that pauses video playback
        self.pauseButton = QPushButton('Pause')
        self.pauseButton.clicked.connect(self.pause)
        self.pauseButton.setEnabled(False)
        self.pauseButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.pauseButton, 3, 4, 1, 4)

        # Button that resumes video playback
        self.resumeButton = QPushButton('Resume')
        self.resumeButton.clicked.connect(self.resume)
        self.resumeButton.setEnabled(False)
        self.resumeButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.resumeButton, 4, 4, 1, 4)

        # Button stops video playback and closes the window
        self.stopButton = QPushButton('Stop')
        self.stopButton.clicked.connect(self.stop)
        self.stopButton.setEnabled(False)
        self.stopButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.stopButton, 3, 8, 2, 4)

        # Label displaying the file path of the chosen video
        self.filePathLabel = QLabel('File: None')
        self.filePathLabel.setWordWrap(True)
        grid.addWidget(self.filePathLabel, 5, 0, 1, 12)

        # Label displaying the current time of the video
        self.currentTimeLabel = QLabel('0:00:00')
        grid.addWidget(self.currentTimeLabel, 6, 0, 1, 1)

        # Slider to scrub through the video
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setFocusPolicy(Qt.StrongFocus)
        self.slider.setTickPosition(QSlider.NoTicks)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self.setTime)
        self.slider.setEnabled(False)
        grid.addWidget(self.slider, 6, 1, 1, 10)

        # Label displaying the duration of the video
        self.finalTimeLabel = QLabel('0:00:00')
        grid.addWidget(self.finalTimeLabel, 6, 11, 1, 1)

        # Add horizontal separator
        grid.addWidget(QHLine(), 7, 0, 1, 12)

        # Button to toggle the mask rcnn detection
        self.detectToggle = QCheckBox('Detect objects')
        self.detectToggle.setEnabled(False)
        self.detectToggle.stateChanged.connect(self.toggleDetection)
        grid.addWidget(self.detectToggle, 8, 0, 1, 12)

        # Checkbox to toggle displaying masks on the detected objects
        self.masksToggle = QCheckBox('Show masks')
        self.masksToggle.setEnabled(False)
        self.masksToggle.stateChanged.connect(self.toggleMasks)
        self.masksToggle.setChecked(True)
        grid.addWidget(self.masksToggle, 9, 0, 1, 6)

        # Checkbox to toggle displaying bounding boxes on the detected objects
        self.boxesToggle = QCheckBox('Show bounding boxes')
        self.boxesToggle.setEnabled(False)
        self.boxesToggle.stateChanged.connect(self.toggleBoxes)
        self.boxesToggle.setChecked(True)
        grid.addWidget(self.boxesToggle, 9, 6, 1, 6)

        # Add horizontal separator
        grid.addWidget(QHLine(), 10, 0, 1, 12)

        # Checkbox to toggle saving the video
        self.saveToggle = QCheckBox('Save video')
        self.saveToggle.setEnabled(False)
        self.saveToggle.stateChanged.connect(self.toggleSave)
        self.saveToggle.setChecked(False)
        grid.addWidget(self.saveToggle, 11, 0, 1, 12)

        # Label for the video save path
        self.savePathLabel = QLabel('Location: None')
        self.savePathLabel.setWordWrap(True)
        grid.addWidget(self.savePathLabel, 12, 0, 1, 12)

        self.setLayout(grid)

    def toggleSave(self):
        if self.saveToggle.isChecked():
            self.controller.emit(Requests.SAVE_ON)
        else:
            self.controller.emit(Requests.SAVE_OFF)
            return

        # _ is the filter
        filePath = QFileDialog.getExistingDirectory(self, "Choose a directory...")

        self.savePathLabel.setText('Location: ' + filePath)

        self.thread.setSave(filePath)

    def setTime(self):
        seconds = self.slider.value()

        paused = self.thread.paused

        self.pause()
        time.sleep(0.1)
        fps = self.thread.capture.get(cv2.CAP_PROP_FPS)
        new_frame_number = round(seconds * fps)
        print("Changing to frame", new_frame_number)
        self.thread.capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame_number)

        # Only resume if it was already paused
        if not paused:
            self.resume()

    def toggleDetection(self):
        # Update the worker with the new detection setting
        if self.detectToggle.isChecked():
            self.controller.emit(Requests.DETECT_ON)
        else:
            self.controller.emit(Requests.DETECT_OFF)

        self.masksToggle.setEnabled(self.detectToggle.isChecked())
        self.boxesToggle.setEnabled(self.detectToggle.isChecked())

    def toggleMasks(self):
        # Update the worker with the new mask setting
        if self.masksToggle.isChecked():
            self.controller.emit(Requests.MASKS_ON)
        else:
            self.controller.emit(Requests.MASKS_OFF)

        if not self.masksToggle.isChecked() and not self.boxesToggle.isChecked():
            self.detectToggle.setChecked(False)

    def toggleBoxes(self):
        # Update the worker with the new boxes setting
        if self.boxesToggle.isChecked():
            self.controller.emit(Requests.BOXES_ON)
        else:
            self.controller.emit(Requests.BOXES_OFF)

        if not self.masksToggle.isChecked() and not self.boxesToggle.isChecked():
            self.detectToggle.setChecked(False)

    def chooseVideo(self):
        # _ is the filter
        filePath, _ = QFileDialog.getOpenFileName(self, 'Choose a video file', '', 'Videos Files | *.mp4;')

        if filePath is '':
            self.statusLabel.setText('Status: Please choose a video')
            self.startButton.setEnabled(False)
            return

        self.filePathLabel.setText('File: ' + filePath)

        # Set file path
        self.thread.setVideo(filePath)

        # If we do this directly before the weights are actually loaded the GUI freezes and the text isn't rendered
        if self.thread.loadedWeights:
            self.statusLabel.setText('Status: Ready')
        else:
            self.statusLabel.setText('Status: Loading weights...')

        # Loads weights and idles
        if not self.thread.isRunning():
            self.thread.start()

        self.saveToggle.setEnabled(True)

        if self.thread.loadedWeights:
            self.startButton.setEnabled(True)

    def frameChanged(self):
        formatted = round(self.thread.fps * 100) / 100
        self.fpsLabel.setText('FPS: ' + str(formatted))

        millis = self.thread.capture.get(cv2.CAP_PROP_POS_MSEC)
        self.currentTimeLabel.setText(self.formatTime(millis))

        # Block signals when we set the value to stop any callbacks from being triggered
        self.slider.blockSignals(True)
        self.slider.setValue(round(millis / 1000))
        self.slider.blockSignals(False)

        # Set the maximum label if it's not been set yet
        if self.finalTimeLabel.text() == '0:00:00':
            fps = self.thread.capture.get(cv2.CAP_PROP_FPS)
            total_frames = self.thread.capture.get(cv2.CAP_PROP_FRAME_COUNT)

            # Video duration in seconds
            duration = (float(total_frames) / float(fps))

            formatted = self.formatTime(duration * 1000)
            self.finalTimeLabel.setText(formatted)
            self.slider.setMaximum(duration)

    def formatTime(self, milliseconds):
        minutes, seconds = divmod(milliseconds / 1000, 60)
        hours, minutes = divmod(minutes, 60)
        return "%d:%02d:%02d" % (hours, minutes, seconds)

    def action(self, action):
        # Reported when weighs are about to load
        if action is Actions.LOADING_WEIGHTS:
            self.statusLabel.setText('Status: Loading weights...')

        # Reported when weights have been loaded
        if action is Actions.LOADED_WEIGHTS:
            self.statusLabel.setText('Status: Ready')
            self.startButton.setEnabled(True)
            self.pauseButton.setEnabled(False)
            self.resumeButton.setEnabled(False)
            self.stopButton.setEnabled(False)
            self.detectToggle.setEnabled(True)

        # Reported back when the video has been loaded
        if action is Actions.LOADED_VIDEO:
            self.statusLabel.setText('Status: Running...')
            self.startButton.setEnabled(False)
            self.resumeButton.setEnabled(False)
            self.pauseButton.setEnabled(True)
            self.stopButton.setEnabled(True)
            self.slider.setEnabled(True)

        # Reported back when the video playback has finished
        if action is Actions.FINISHED:
            self.statusLabel.setText('Status: Finished')
            self.stop()

    def start(self):
        self.startButton.setEnabled(False)
        self.resumeButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.stopButton.setEnabled(False)
        self.detectToggle.setEnabled(True)
        self.saveToggle.setEnabled(False)
        self.controller.emit(Requests.START)

    def pause(self):
        self.pauseButton.setEnabled(False)
        self.resumeButton.setEnabled(True)
        self.detectToggle.setEnabled(True)
        self.controller.emit(Requests.PAUSE)

    def resume(self):
        self.pauseButton.setEnabled(True)
        self.resumeButton.setEnabled(False)
        self.detectToggle.setEnabled(True)
        self.controller.emit(Requests.RESUME)

    def stop(self):
        self.startButton.setEnabled(True)
        self.resumeButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.stopButton.setEnabled(False)
        self.detectToggle.setEnabled(False)
        self.slider.setEnabled(False)
        self.saveToggle.setEnabled(True)
        self.slider.setValue(0)
        self.currentTimeLabel.setText('0:00:00')
        self.finalTimeLabel.setText('0:00:00')
        self.fpsLabel.setText('')

        self.controller.emit(Requests.STOP)


app = QApplication(sys.argv)
window = Window()
window.setWindowTitle('Mask - RCNN')
window.show()

sys.exit(app.exec_())