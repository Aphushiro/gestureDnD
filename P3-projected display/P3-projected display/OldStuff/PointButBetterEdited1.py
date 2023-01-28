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

def FindHandLookingThings(binaryImage, handThreshold=20000): #Finds hands in a binary image and returns a list of their contour
    handContour = None
    # binaryImage_grey = cv2.cvtColor(binaryImage,cv2.COLOR_BGR2GRAY)
    center=(binaryImage.shape[0]/2,binaryImage.shape[1]/2)
    print(center)
    _, contours_list, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  # Finds all contours
    ClosestDistanceToCenter = math.inf
    for cnt in contours_list:  # goes through all contours
        DistanceToCenter=Distance(cv2.minAreaRect(cnt)[0],center)
        if cv2.contourArea(cnt) > handThreshold:  # only take those above a certain pixel threshold
            DistanceToCenter = Distance(cv2.minAreaRect(cnt)[0], center)
            if DistanceToCenter<ClosestDistanceToCenter:
                ClosestDistanceToCenter=DistanceToCenter
                handContour=cnt
            else:
                cv2.drawContours(binaryImage, [cnt], -1, 0, -1)  # draws over the contour
        else:
            cv2.drawContours(binaryImage, [cnt], -1, 0, -1)#draws over the contour

    # the list cnt_list now contains all contours which should also be hands
    return handContour

def FindLRHand(handContour, image, frame): # the long one which finds the finger and makes a mask of the point direction
    # find and make the bounding box
    rect = cv2.minAreaRect(handContour)
    center = rect[0]
    rotation = rect[2]
    halfHeight = rect[1][0] / 2
    halfWidth = rect[1][1] / 2
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # for all the corners we want to find the halfway point
    # This is the center of the quardsected hand
    quardcenters=[]
    for corner in box:
        corner_center_center = (center + corner) / 2
        quardcenters.append(corner_center_center)
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

    # throw out the smallest area, this is not the finger
    while len(handPieces) > 3:
        smallestIndex = contourArea.index(min(contourArea))  # index for smallest area
        handPieces.pop(smallestIndex)
        contourArea.pop(smallestIndex)
        center = centerList.pop(smallestIndex)
        rotations.pop(smallestIndex)

    smallestIndex = contourArea.index(min(contourArea))  # second smallest/Finger contour
    centerFinger = centerList.pop(smallestIndex)
    cv2.circle(frame, (math.floor(centerFinger[0]), math.floor(centerFinger[1])), 10, (255, 255, 255), -1)
    fingerCenter, FingerDimenstions, FingerRotation = cv2.minAreaRect(handPieces[smallestIndex])  # finger contour
    listBecauseTupleStoopid = list(FingerDimenstions)

    if FingerDimenstions[0]>FingerDimenstions[1]:#0=longest
        NewLength=listBecauseTupleStoopid[0]*20
        listBecauseTupleStoopid[1]*=2
        lengthIndex = 0
    else:
        listBecauseTupleStoopid[0] *= 2
        NewLength=listBecauseTupleStoopid[1] * 20
        lengthIndex=1
    FingerDimenstions = tuple(listBecauseTupleStoopid)
    cornersFitRect = cv2.boxPoints((fingerCenter, FingerDimenstions, FingerRotation))  # corners for finger rect

    # We now have the contour, boundingbox and center of the finger, we now want to take the boundingBox and extend it

    # We only want the box to extend in the pointed direction
    # Therefore by utilising our list of remaining hand contours, we can tke the two corners of the finger bounding box closest to the second furthest other peices center, and keep them whilst extending the others
    if Distance(centerFinger,centerList[0])<Distance(centerFinger,centerList[1]):
        closestCenter = centerList[0]
    else:
        closestCenter = centerList[1]

    cv2.circle(frame, (math.floor(closestCenter[0]), math.floor(closestCenter[1])), 10, (255, 0, 0), -1)

    corner1Index = 0
    corner2Index = 0
    smallesDistance = math.inf
    index = -1
    for corner in cornersFitRect:  # find first corner
        index += 1
        if Distance(corner, closestCenter) < smallesDistance:
            corner1Index = index
            smallesDistance = Distance(corner, closestCenter)
    corner1 = cornersFitRect[corner1Index]
    smallesDistance = Distance(corner, closestCenter)
    index = -1
    for corner in cornersFitRect:  # find second corner
        index += 1
        if index != corner1Index:  # dont find first corner again
            if Distance(corner, closestCenter) < smallesDistance:
                corner2Index = index
                smallesDistance = Distance(corner, closestCenter)
    corner2 = cornersFitRect[corner2Index]

    cv2.circle(frame, (math.floor(corner1[0]),math.floor(corner1[1])), 10, (0,0,255), -1)
    cv2.circle(frame, (math.floor(corner2[0]),math.floor(corner2[1])), 10, (0,0,255), -1)

    # resize the bounding box length
    listBecauseTupleStoopid = list(FingerDimenstions)
    listBecauseTupleStoopid[lengthIndex]*=20

    FingerDimenstions = tuple(listBecauseTupleStoopid)

    fingerSquareLong = fingerCenter, FingerDimenstions, FingerRotation

    # remake the corners of the new bounding box
    box = cv2.boxPoints(fingerSquareLong)
    for i in range(4):
        if i == corner1Index:
            box[i] = corner1
        if i == corner2Index:
            box[i] = corner2

    # we now have the corners of a box, which extened from the finger bounding box
    # however, we also want the box to
    '''
    vectorDifference = box[corner2Index] - box[corner1Index]
    distance = Distance(box[corner1Index], box[corner2Index])
    vectorDifferenceNormal = vectorDifference / distance
    box[corner2Index] = box[corner2Index] + (vectorDifferenceNormal * 40)
    box[(corner2Index + 1) % 3] = box[(corner2Index + 1) % 3] + (vectorDifferenceNormal * 40)
    '''
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

def BinImageFromDiffColour(diff_Colour):
    _, diff_binary = cv2.threshold(diff_Colour, 160, 255, cv2.THRESH_BINARY)
    blue_img, green_img, red_img = cv2.split(diff_binary)
    cv2.imwrite("Skraldespand/blueBin.png",blue_img)
    cv2.imwrite("Skraldespand/greenBin.png", green_img)
    cv2.imwrite("Skraldespand/redBin.png", red_img)
    redMinusBlue=red_img-blue_img
    cv2.imwrite("Skraldespand/redminusblue.png", redMinusBlue)
    return redMinusBlue

def Closing(binImage):
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)) #kan blive nødvendige at gøre større
    binImageClosed = cv2.morphologyEx(binImage, cv2.MORPH_CLOSE, kernel_close)
    return binImageClosed


def BinPreProcessing(binImage):
    erodeKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilateKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    binImage = cv2.erode(binImage, erodeKernel)
    binImage = cv2.dilate(binImage, dilateKernel)
    binImage = cv2.dilate(binImage, dilateKernel)
    #binImage = cv2.erode(binImage, erodeKernel)

    return binImage

def FindPointDirectionMask(background_Denoised, frame):
    cv2.imwrite("Skraldespand/Before.png", frame)

    frame_Denoised = DeNoise(frame)

    diff_Colour = cv2.absdiff(background_Denoised, frame_Denoised)

    diff_Bin=BinImageFromDiffColour(diff_Colour)

    #diff_Bin_processed=Closing(diff_Bin)

    diff_Bin_processed = BinPreProcessing(diff_Bin)

    cv2.imwrite("Skraldespand/diff_Bin_Closed.png", diff_Bin_processed)
    cv2.imshow("colourdiff",diff_Colour)
    cv2.waitKey(0)
    cv2.imwrite("Skraldespand/diff_Colour.png", diff_Colour)

    handContours = FindHandLookingThings(diff_Bin_processed)

    cv2.imwrite("Skraldespand/afterNotHandRemoval.png",diff_Bin_processed)

    print(handContours)
    handInImage = True
    handCenter, binaryMask = FindLRHand(handContours, diff_Bin_processed, frame)



    handCenter, binaryMask = FindLRHand(handContours, diff_Bin_processed, frame)
    frame_masked = cv2.bitwise_and(frame, frame, mask=binaryMask)
    cv2.imwrite("Skraldespand/PointingBoxWrite.png", frame_masked)

    return frame_masked, handInImage, handCenter

#run test
background=cv2.imread("Background.png")
img=cv2.imread("Finger.png")
masked,_,_=FindPointDirectionMask(background,img)
cv2.imshow("finaleMasked",masked)
cv2.imshow("frameFinale",img)
cv2.waitKey(0)
