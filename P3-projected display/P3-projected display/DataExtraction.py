import cv2
import numpy as np
import math

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


def FindHandLookingThings(binaryImage, handThreshold=20000): #Finds hands in a binary image and returns a list of their contour
    handContour = []
    # binaryImage_grey = cv2.cvtColor(binaryImage,cv2.COLOR_BGR2GRAY)
    center=(binaryImage.shape[0]/2,binaryImage.shape[1]/2)
    print(center)
    contours_list, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  # Finds all contours
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


def FindSquareFromContour(cnt, binaryImage):
    rect = cv2.minAreaRect(cnt)

    fitRectangleCenter = rect[0]
    fitRectangleHeight = rect[1][0]
    fitRectangleWidth = rect[1][1]
    maxOfWandH = max(fitRectangleWidth, fitRectangleHeight)
    areaRect = (fitRectangleHeight * fitRectangleWidth)
    print(fitRectangleHeight,fitRectangleWidth)
    HWRelation = (max(fitRectangleHeight, fitRectangleWidth) / min(fitRectangleHeight, fitRectangleWidth))
    areaSquare = (maxOfWandH ** 2)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    M = cv2.moments(cnt)
    cy = int(M['m10'] / M['m00'])
    cx = int(M['m01'] / M['m00'])
    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    return areaSquare, areaRect, fitRectangleCenter, HWRelation, (cy, cx)


def FindBloBAreaFitRectRelation(blobSize, fitRectArea):
    return blobSize / fitRectArea


def FindBloBAreaSquareRelation(blobSize, squareArea):
    return blobSize / squareArea

def roundDown(n, d=4):
    d = int('1' + ('0' * d))
    return math.floor(n * d) / d

# threshold
lowerThresh = 110
upperThresh = 255

HWRelationData                      = []
BloBFitRectRelationData             = []
BlobSquareRelationData              = []
CenterMassCenterRectRelationData    = []
BloBSizeData                        = []

#CheckListe
startNum = 0
stopNum = 180
gestureFolder = "Roll"


bgNum = 0

def ChangeBg (num):
    # Background declaration and denoise
    bgName = ["MeanBgBefore45", "MeanBg45-74", "MeanBg75-180"]
    background = cv2.imread(f"HandImages - don't share/zzData images/{bgName[num]}.png")  # Rename to new folder
    backgroundDenoise = DeNoise(background)
    return backgroundDenoise

background_Denoise = ChangeBg(bgNum)
bgNum += 1

for img in range(startNum, stopNum):
    print(f"Imnum: {img}")
    # read image
    image = cv2.imread(f"HandImages - don't share/zzData images/{gestureFolder}/img{img}.png")
    # denoise image
    image_Denoise = DeNoise(image)

    if img == 45 or img == 75:
        ChangeBg(bgNum)
        bgNum += 1

    # subtracted image
    diff_colour = cv2.absdiff(background_Denoise, image_Denoise)

    #binary image
    diff_binary = BinImageFromDiffColour(diff_colour)

    #Kernels for opening and closing
    diff_bin_closed=BinPreProcessing(diff_binary)

    #find contours
    coutours = FindHandLookingThings(diff_bin_closed, 100)
    print(len(coutours))
    if len(coutours)>1:
        contourArea = cv2.contourArea(coutours)
        areaSquare, areaFitRect, fitRectangleCenter, HWRelation, centerMass = FindSquareFromContour(coutours, diff_binary)
        blobAreaSquareRelation = FindBloBAreaSquareRelation(contourArea, areaSquare)
        bloBAreaFitRectRelation = FindBloBAreaFitRectRelation(contourArea, areaFitRect)

        cv2.circle(image, centerMass, 3, (0, 0, 255), -1)
        cv2.circle(image, (math.floor(fitRectangleCenter[0]), math.floor(fitRectangleCenter[1])), 3,(0, 255, 0), -1)
        centerDifference = math.dist(centerMass,(math.floor(fitRectangleCenter[0]), math.floor(fitRectangleCenter[1])))
        smolImg = cv2.resize(image, (1920//2, 1080//2))
        smolDiff = cv2.resize(diff_bin_closed, (1920 // 2, 1080 // 2))
        cv2.imshow("binary", smolDiff)
        cv2.imshow("image with centers and square", smolImg)
        key = cv2.waitKey(0)
        if key == ord("y"):
            HWRelationData.append(roundDown(HWRelation, 4))
            BloBFitRectRelationData.append(roundDown(bloBAreaFitRectRelation, 4))
            BlobSquareRelationData.append(roundDown(blobAreaSquareRelation, 4))
            CenterMassCenterRectRelationData.append(roundDown(centerDifference, 4))
            BloBSizeData.append(roundDown(contourArea, 4))

        if img == stopNum:
            break


print(f"{gestureFolder}Data1 = {HWRelationData}")
print(f"{gestureFolder}Data2 = {BloBFitRectRelationData}")
print(f"{gestureFolder}Data3 = {BlobSquareRelationData}")
print(f"{gestureFolder}Data4 = {CenterMassCenterRectRelationData}")