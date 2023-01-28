import math
import numpy as np
import cv2

def Distance(vector1,vector2):
    sum = 0
    for i in range(len(vector1)):
        dist = (vector2[i]-vector1[i])**2
        sum += dist
    distance = math.sqrt(sum)
    return distance

def DeNoise(frame):  # deNoises the image. Consider not using this for better processing time, might have to adjust some threshold tho. Reminder, do test to test accuracy for difference preprocesses.
    deNoisedFrame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    cv2.normalize(deNoisedFrame, deNoisedFrame, 0, 255, cv2.NORM_MINMAX)
    return deNoisedFrame

def FindHandLookingThings(binaryImage, handThreshold=5000): #Finds hands in a binary image and returns a list of their contour
    cnt_list = []
    # binaryImage_grey = cv2.cvtColor(binaryImage,cv2.COLOR_BGR2GRAY)
    _, contours_list, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)  # Finds all contours
    for cnt in contours_list:  # goes through all contours
        if cv2.contourArea(cnt) > handThreshold:  # only take those above a certain pixel threshold
            cnt_list.append(cnt)  # put them in a new list
        else:
            cv2.drawContours(binaryImage, [cnt], -1, 0, -1)
    contour1 = cv2.drawContours(binaryImage, cnt_list, -1, (255, 0, 255), 3)  # draw them on the picture, for debugging
    # the list cnt_list now contains all contours which should also be hands
    return cnt_list

def FindLRHand(handContour, image): # the long one which finds the finger and makes a mask of the point direction
    center = (0, 0)
    # find and make the bounding box
    rect = cv2.minAreaRect(handContour[0])
    center = rect[0]
    rotation = rect[2]
    halfHeight = rect[1][0] / 2
    halfWidth = rect[1][1] / 2
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # for all the corners we want to find the halfway point
    # This is the center of the quardsected hand
    for corner in box:
        corner_center_center = (center + corner) / 2
        cornerRect = (corner_center_center, (halfHeight, halfWidth), rotation)
        box = cv2.boxPoints(cornerRect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    cv2.imwrite("Skraldespand/Chop.png", image)
    # with the drawn quardsection, we han threshold the image, this will remove "center lines"
    _, diff_binary = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)

    # we then find the new contours which are the four parts of the hand
    _, contours_list, hierarchy = cv2.findContours(diff_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # we want some information about the second largest area
    # we therefore construct list which we want to extract the smallest, such that the second smallest becomes the smallest
    # This might have to be reworked later, to adjust for hand sizes as only 3 and not 4 segments might be found
    contourArea = []
    centerList = []
    rotations = []
    handPieces = []
    for cnt in contours_list:  # this is to make sure we only look at areas of the right size
        if cv2.contourArea(cnt) > 100:
            handPieces.append(cnt)
            rectCenter, _, rotation = cv2.minAreaRect(cnt)
            centerList.append(rectCenter)
            contourArea.append(cv2.contourArea(cnt))
            rotations.append(rotation)
            print(rectCenter)

    # throw out the smallest area, this is not the finger
    if len(handPieces) > 3:
        smallestIndex = contourArea.index(min(contourArea))  # index for smallest area
        handPieces.pop(smallestIndex)
        contourArea.pop(smallestIndex)
        centerList.pop(smallestIndex)
        rotations.pop(smallestIndex)

    # the second smallest area IS the finger, unless there is only one area, handle later, simple if statement
    smallestIndex = contourArea.index(min(contourArea))  # second smallest/Finger contour
    centerFinger = centerList.pop(smallestIndex)
    fingerCenter, FingerDimenstions, FingerRotation = cv2.minAreaRect(handPieces[smallestIndex])  # finger contour
    cornersFitRect = cv2.boxPoints((fingerCenter, FingerDimenstions, FingerRotation))  # corners for finger rect

    # We now have the contour, boundingbox and center of the finger, we now want to take the boundingBox and extend it

    # We only want the box to extend in the pointed direction
    # Therefore by utilising our list of remaining hand contours, we can tke the two corners of the finger bounding box closest to the second furthest other peices center, and keep them whilst extending the others
    if Distance(centerFinger, centerList[0]) > Distance(centerFinger, centerList[1]):
        furthestCenter = centerList[0]
    else:
        furthestCenter = centerList[1]

    corner1Index = 0
    corner2Index = 0
    smallesDistance = math.inf
    index = -1
    for corner in cornersFitRect:  # find first corner
        index += 1
        if Distance(corner, furthestCenter) < smallesDistance:
            corner1Index = index
            smallesDistance = Distance(corner, furthestCenter)
    corner1 = cornersFitRect[corner1Index]
    smallesDistance = Distance(corner, furthestCenter)
    index = 0
    for corner in cornersFitRect:  # find second corner
        index += 1
        if index != corner1Index:  # dont find first corner again
            if Distance(corner, furthestCenter) < smallesDistance:
                corner2Index = index
                smallesDistance = Distance(corner, furthestCenter)
    corner2 = cornersFitRect[corner2Index]

    # resize the bounding box length
    if FingerDimenstions[0] < FingerDimenstions[1]:
        newFingerDimension = (FingerDimenstions[0], FingerDimenstions[1] * 20)
    else:
        newFingerDimension = (FingerDimenstions[0] * 20, FingerDimenstions[1])
    fingerSquareLong = fingerCenter, newFingerDimension, FingerRotation

    # remake the corners of the new bounding box
    box = cv2.boxPoints(fingerSquareLong)
    for i in range(4):
        if i == corner1Index:
            box[i] = corner1
        if i == corner2Index:
            box[i] = corner2

    # we now have the corners of a box, which extened from the finger bounding box
    # however, we also want the box to

    vectorDifference = box[corner2Index] - box[corner1Index]
    distance = Distance(box[corner1Index], box[corner2Index])
    vectorDifferenceNormal = vectorDifference / distance
    box[corner2Index] = box[corner2Index] + (vectorDifferenceNormal * 40)
    box[(corner2Index + 1) % 3] = box[(corner2Index + 1) % 3] + (vectorDifferenceNormal * 40)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    mask = np.zeros(image.shape[:2], dtype="uint8")
    mask = cv2.drawContours(mask, [box], 0, (255, 255, 255), -1)
    return center, mask

def ColourThreshold(diffImage):
    image = cv2.cvtColor(diffImage, cv2.COLOR_BGR2HSV)
    # getting the height and width of the image
    Huescale = 0.7
    # these are found by trial and error, and a bit of help from imagej
    MinHue = 0 * Huescale
    MaxHue = 50 * Huescale
    MinSat = 120
    MaxSat = 255
    MinVal = 80
    MaxVal = 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    binaryHSVImage = cv2.inRange(image, (MinHue, MinSat, MinVal), (MaxHue, MaxSat, MaxVal))
    return binaryHSVImage

def Closing(binImage):
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)) #kan blive nødvendige at gøre større
    binImageClosed = cv2.morphologyEx(binImage, cv2.MORPH_CLOSE, kernel_close)
    return binImageClosed

def FindPointDirectionMask(background_Denoised, frame):
    cv2.imwrite("Skraldespand/Before.png", frame)
    frame_Denoised = DeNoise(frame)

    diff_Colour = cv2.absdiff(background_Denoised, frame_Denoised)

    diff_Bin=ColourThreshold(diff_Colour)


    diff_Bin_Closed = Closing(diff_Bin)
    cv2.imwrite("Skraldespand/diff_Bin_Closed.png", diff_Bin_Closed)
    cv2.imwrite("Skraldespand/diff_Colour.png", diff_Colour)

    handContours = FindHandLookingThings(diff_Bin_Closed)
    handCenter, binaryMask = FindLRHand(handContours, diff_Bin_Closed)

    handInImage = True
    if len(handContours) < 1:
        handInImage = False
        return None, handInImage, handCenter


    frame_masked = cv2.bitwise_and(frame, frame, mask=binaryMask)
    cv2.imwrite("Skraldespand/PointingBoxWrite.png", frame_masked)
    return frame_masked, handInImage, handCenter