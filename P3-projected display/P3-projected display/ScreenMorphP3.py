import cv2
import numpy as np

class RoiHandler:
    def __init__(self):
        self.points = []

    def grab_click_position(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) >= 4:
                self.points = []
            self.points.append((x, y))
            #print(self.points)

def DefineRegionOfInterest ():
    camPort = 0
    capture = cv2.VideoCapture(camPort)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    #capture.set(cv2.CAP_PROP_EXPOSURE, -8.0)

    image_name = "tempwarp.png"

#Capture reference image for drawing
    while True:
        _, frame = capture.read()
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(30)
        if key == ord("q"):
            exit()
            break
        if key == ord("f"):
            cv2.imwrite(image_name, frame)
            break
    cv2.destroyAllWindows()
    capture.release()

    # Draw ROI
    image = cv2.imread(image_name)
    roi = RoiHandler()

    cv2.namedWindow("Input image")
    cv2.setMouseCallback('Input image', roi.grab_click_position)
    while True:
        image_small_lines = cv2.resize(image, (1920//2, 1080//2))
        cv2.polylines(image_small_lines, [np.array(roi.points, dtype=np.int32)], True, (0, 255, 0))
        cv2.imshow('Input image', image_small_lines)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            exit()
            break
        elif len(roi.points) == 4 and key == 32:  # Space
            break
        elif key != 255:
            print(key)
        if cv2.getWindowProperty('Input image', cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    return roi
#DefineRegionOfInterest()