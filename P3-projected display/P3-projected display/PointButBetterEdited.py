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

def FindHandLookingThings(binaryImage, handThreshold=4000): #Finds hands in a binary image and returns a list of their contour
    handContour = []
    # binaryImage_grey = cv2.cvtColor(binaryImage,cv2.COLOR_BGR2GRAY)
    screenCenter=(binaryImage.shape[0]/2,binaryImage.shape[1]/2)
    center=(0, 0)
    _, contours_list, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  # Finds all contours
    ClosestDistanceToCenter = math.inf
    for cnt in contours_list:  # goes through all contours
        #print(cv2.contourArea(cnt))
        if cv2.contourArea(cnt) > handThreshold:  # only take those above a certain pixel threshold
            DistanceToCenter = Distance(cv2.minAreaRect(cnt)[0], screenCenter)
            if DistanceToCenter<ClosestDistanceToCenter:
                ClosestDistanceToCenter=DistanceToCenter
                handContour=cnt
                M = cv2.moments(handContour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                center = (cx, cy)
                print(center, DistanceToCenter, ClosestDistanceToCenter)
            else:
                cv2.drawContours(binaryImage, [cnt], -1, 0, -1)  # draws over the contour
        else:
            cv2.drawContours(binaryImage, [cnt], -1, 0, -1)#draws over the contour

    # the list cnt_list now contains all contours which should also be hands
    return handContour, center

def FindLRHand(handContour, image): # the long one which finds the finger and makes a mask of the point direction
    #find and make the bounding box of the hand
    rect = cv2.minAreaRect(handContour)
    center = rect[0]
    rotation = rect[2]
    length = max(rect[1][0],rect[1][1])#find the longest part of the hand, this is the direction of the hand


    #use the length, rotation and center to create two point in each end of the hand at the center
    x1 = center[0] + length * math.cos(math.radians(rotation+90))
    y1 = center[1] + length * math.sin(math.radians(rotation+90))
    x2 = center[0] + length * math.cos(math.radians(rotation - 90))
    y2 = center[1] + length * math.sin(math.radians(rotation - 90))

    #draw a line between the two points, this will cleave the hand in twine
    cv2.line(image,(math.floor(x1),math.floor(y1)),(math.floor(x2),math.floor(y2)),(0,0,255),10)

    #with the cleaft hand find the two parts
    _, contours_tuble, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Make sure there are only the two largest contours
    handpiecesArea=[]
    contours_list=[]
    for cnt in contours_tuble:
        handpiecesArea.append(cv2.contourArea(cnt))
        contours_list.append(cnt)
    while len(handpiecesArea)>2:
        index=handpiecesArea.index(min(handpiecesArea))
        handpiecesArea.pop(index)
        contours_list.pop(index)


    #find outh which half has the largest blob, this is the half with the finger
    if handpiecesArea[0]>handpiecesArea[1]:
        center, dimensions, rotation=cv2.minAreaRect(contours_list[0])
    else:
        center, dimensions, rotation = cv2.minAreaRect(contours_list[1])

    #make the dimensions of the finger box a lot longer and a bit wider
    dimensions = list(dimensions)
    if dimensions[0]>dimensions[1]:
        dimensions[0]*=20
        dimensions[1]*=2
    else:
        dimensions[1]*=20
        dimensions[0]*=2


    dimensions = tuple(dimensions)

    #make points from box information
    newbox=(center,dimensions,rotation)
    box = cv2.boxPoints(newbox)
    box = np.int0(box)

    #
    #cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    #make the mask
    mask = np.zeros(image.shape[:2], dtype="uint8")
    mask = cv2.drawContours(mask, [box], 0, (255, 255, 255), -1)
    return center, mask, length

    # we want some information about the second largest area
    # we therefore construct list which we want to extract the smallest, such that the second smallest becomes the smallest
    # This might have to be reworked later, to adjust for hand sizes as only 3 and not 4 segments might be found


def BinImageFromDiffColour(diff_Colour):
    _, diff_binary = cv2.threshold(diff_Colour, 100, 255, cv2.THRESH_BINARY)
    blue_img, green_img, red_img = cv2.split(diff_binary)
    #cv2.imwrite("Skraldespand/blueBin.png",blue_img)
    #cv2.imwrite("Skraldespand/greenBin.png", green_img)
    #cv2.imwrite("Skraldespand/redBin.png", red_img)
    redMinusBlue=red_img-blue_img
    #cv2.imwrite("Skraldespand/redminusblue.png", redMinusBlue)
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


def FindPointDirectionWithLine(contour, frame):
    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    vectorAngle = np.arctan(vy,vx) * (180/math.pi)
    box = ((int(x),int(y)),(4000,170),int(vectorAngle))
    box = cv2.boxPoints(box)
    box = np.int0(box)
    mask = np.zeros(frame.shape[:2], dtype="uint8")
    mask = cv2.drawContours(mask, [box], 0, (255, 255, 255), -1)
    return mask

def FindPointDirectionMask(background_Denoised, frame):
    #cv2.imwrite("Skraldespand/Before.png", frame)

    frame_Denoised = DeNoise(frame)

    diff_Colour = cv2.absdiff(background_Denoised, frame_Denoised)

    diff_Bin=BinImageFromDiffColour(diff_Colour)

    diff_Bin_processed = BinPreProcessing(diff_Bin)

    #cv2.imwrite("Skraldespand/diff_Bin_Closed.png", diff_Bin_processed)

    #cv2.imwrite("Skraldespand/diff_Colour.png", diff_Colour)
    handContours, handCenter = FindHandLookingThings(diff_Bin_processed)
    #cv2.imwrite("Skraldespand/afterNotHandRemoval.png", diff_Bin_processed)

    handInImage = True
    if len(handContours) < 1:
        handInImage = False
        noHandCenter = (0, 0)
        return None, handInImage, noHandCenter


    #handCenter, binaryMask, length = FindLRHand(handContours, diff_Bin_processed)

    binaryMask = FindPointDirectionWithLine(handContours, frame)
    frame_masked = cv2.bitwise_and(frame, frame, mask=binaryMask)
    cv2.imwrite("Skraldespand/PointingBoxWrite.png", frame_masked)

    return frame_masked, handInImage, handCenter#, length

#run test
'''
background=cv2.imread("Background.png")
img=cv2.imread("Finger.png")
masked,_=FindPointDirectionMask(background,img)
cv2.imshow("test",masked)
cv2.waitKey(0)
'''