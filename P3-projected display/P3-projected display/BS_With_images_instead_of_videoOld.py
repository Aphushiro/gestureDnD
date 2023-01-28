import cv2
import math
import numpy as np

def Distance(vector1,vector2):
    sum = 0
    for i in range(len(vector1)):
        dist = (vector2[i]-vector1[i])**2
        sum += dist
    distance = math.sqrt(sum)
    return distance

def BinImageFromDiffColour(diff_Colour):
    _, diff_binary = cv2.threshold(diff_Colour, 160, 255, cv2.THRESH_BINARY)
    blue_img, green_img, red_img = cv2.split(diff_binary)
    cv2.imwrite("Skraldespand/blueBin.png",blue_img)
    cv2.imwrite("Skraldespand/greenBin.png", green_img)
    cv2.imwrite("Skraldespand/redBin.png", red_img)
    redMinusBlue=red_img-blue_img
    cv2.imwrite("Skraldespand/redminusblue.png", redMinusBlue)
    redMinusBlue=red_img-blue_img
    return redMinusBlue

def BinPreProcessing(binImage):
    erodeKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilateKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    binImage = cv2.erode(binImage, erodeKernel)
    binImage = cv2.dilate(binImage, dilateKernel)
    binImage = cv2.dilate(binImage, dilateKernel)
    cv2.imwrite("skraldespand/closed.png",binImage)
    #binImage = cv2.erode(binImage, erodeKernel)
    return binImage

def FindHandLookingThings(binaryImage, handThreshold=40000): #Finds hands in a binary image and returns a list of their contour
    handContour = []
    # binaryImage_grey = cv2.cvtColor(binaryImage,cv2.COLOR_BGR2GRAY)
    center=(binaryImage.shape[0]/2,binaryImage.shape[1]/2)
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

def DeNoise(frame):  # deNoises the image.
    deNoisedFrame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    cv2.normalize(deNoisedFrame, deNoisedFrame, 0, 255, cv2.NORM_MINMAX)
    return deNoisedFrame

def FindSquareFromContour(cnt):
    rect = cv2.minAreaRect(cnt)
    fitRectangleHeight = rect[1][0]
    fitRectangleWidth = rect[1][1]
    maxOfWandH = max(fitRectangleWidth, fitRectangleHeight)
    areaRect = (fitRectangleHeight * fitRectangleWidth)
    HWRelation = (max(fitRectangleHeight, fitRectangleWidth) / min(fitRectangleHeight, fitRectangleWidth))
    areaSquare = (maxOfWandH ** 2)
    return areaSquare, areaRect, HWRelation

def FindBloBAreaFitRectRelation(blobSize, fitRectArea):
    return blobSize / fitRectArea

def FindBloBAreaSquareRelation(blobSize, squareArea):
    return blobSize / squareArea

def colourThreshold(diffImage):
    image = cv2.cvtColor(diffImage, cv2.COLOR_BGR2HSV)
    # getting the height and width of the image
    Huescale = 0.7
    # these are found by trial and error, and a bit of help from imagej
    MinHue = 0 * Huescale
    MaxHue = 50 * Huescale
    MinSat = 40
    MaxSat = 255
    MinVal = 157
    MaxVal = 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    binaryHSVImage = cv2.inRange(image, (MinHue, MinSat, MinVal), (MaxHue, MaxSat, MaxVal))
    return binaryHSVImage

def GetFeaturesOne (image, background_Denoise):
    # denoise image
    image_Denoise = DeNoise(image)

    # subtracted image
    diff_colour = cv2.absdiff(background_Denoise, image_Denoise)
    cv2.imwrite("Skraldespand/featureOneColour.png", diff_colour)

    #binary image
    diff_binary = BinImageFromDiffColour(diff_colour)

    #Kernels for opening and closing
    diff_bin_closed=BinPreProcessing(diff_binary)

    #find contours
    contours = FindHandLookingThings(diff_bin_closed, 100)
    if len(contours)>1:
        contourArea = cv2.contourArea(contours)
        areaSquare, areaFitRect, HWRelation = FindSquareFromContour(contours)
        bloBAreaFitRectRelation = FindBloBAreaFitRectRelation(contourArea, areaFitRect)
        retCoord = (HWRelation, bloBAreaFitRectRelation)
        return retCoord, True
    else:
        return (0, 0), False

def GetFeaturesTwo (image, background_Denoise):
    # denoise image
    image_Denoise = DeNoise(image)

    # subtracted image
    diff_colour = cv2.absdiff(background_Denoise, image_Denoise)

    #binary image
    diff_binary = BinImageFromDiffColour(diff_colour)

    #Kernels for opening and closing
    diff_bin_closed=BinPreProcessing(diff_binary)

    #find contours
    contours = FindHandLookingThings(diff_bin_closed, 100)
    if len(contours)>1:
        contourArea = cv2.contourArea(contours)
        areaSquare, areaFitRect, HWRelation = FindSquareFromContour(contours)
        blobAreaSquareRelation = FindBloBAreaSquareRelation(contourArea, areaSquare)
        retCoord = (HWRelation, blobAreaSquareRelation)
        return retCoord, True
    else:
        return (0, 0), False