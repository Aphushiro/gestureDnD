import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2
import numpy as np

'''
--- Debugging and other methods we don't use ---

def PointWithinEllipsoidOfInterest(point=(0, 0, 0), center=(0, 0, 0), dimensions=(0, 0, 0)):
    isWithin = False
    x = point[0]
    y = point[1]
    z = point[2]

    h = center[0]
    k = center[1]
    l = center[2]

    a = dimensions[0]
    b = dimensions[1]
    c = dimensions[2]

    dimX = math.pow(x-h, 2)/math.pow(a, 2)
    dimY = math.pow(y-k, 2)/math.pow(b, 2)
    dimZ = math.pow(z-l, 2)/math.pow(c, 2)

    iterable = [dimX, dimY, dimZ]
    if math.fsum(iterable) <= 1:
        isWithin = True
    return isWithin

def PointWithinSphereOfInterest (point=(0, 0, 0), center=(0, 0, 0), radius = 1.0):
    isWithin = False
    Di = math.sqrt(math.pow(center[0]-point[0], 2) + math.pow(center[1]-point[1], 2) + math.pow(center[2]-point[2], 2))
    if Di < radius:
        isWithin = True
    return isWithin

def MeanBackground(nImages=100, port=0, ramp_frames=30, x=1920, y=1080):
    capture = cv2.VideoCapture(port)
    # Set Resolution
    capture.set(3, x)
    capture.set(4, y)

    # Adjust camera lighting
    for i in range(ramp_frames):
        temp = capture.read()
    _, frame = capture.read()
    background_mean = np.zeros(frame.shape, float)
    for i in range(nImages):
        _,frame=capture.read()
        background_mean=np.add(background_mean,frame)

    background_mean=np.divide(background_mean,nImages) #makes mean image >:(
    Background_mean = np.array(np.round(background_mean), dtype=np.uint8)
    capture.release()
    cv2.imwrite("Skraldespand/MeanBg.png", Background_mean)
    return Background_mean

def find_tokens(fName):
    low_yellow = np.array([125, 200, 170], dtype="uint8")
    upper_yellow = np.array([220, 255, 255], dtype="uint8")

    low_red = np.array([0, 0, 150], dtype="uint8")
    upper_red = np.array([150, 130, 255], dtype="uint8")

    cnt_list = []

    brik_area = 400
    board_for_brikkr = np.array(cv2.imread(fName))
    _, threshold = cv2.threshold(board_for_brikkr, 100, 255, cv2.THRESH_BINARY)

    mask_yellow = cv2.inRange(board_for_brikkr, low_yellow, upper_yellow)
    mask_red = cv2.inRange(board_for_brikkr, low_red, upper_red)
    detected_output_red = cv2.bitwise_and(threshold, threshold, mask=mask_red)
    detected_output_yellow = cv2.bitwise_and(threshold, threshold, mask=mask_yellow)

    cv2.imwrite("Red Mask.jpg", detected_output_red)
    cv2.imwrite("Yellow Mask.jpg", detected_output_yellow)

    src_red = cv2.imread("Red Mask.jpg")
    src_yellow = cv2.imread("Yellow Mask.jpg")

    beta = (1.0 - 0.5)
    dst = cv2.addWeighted(src_red, 0.5, src_yellow, beta, 0.0)

    denoise_output = cv2.fastNlMeansDenoisingColored(dst, None, 50, 10, 7, 21)
    cv2.imwrite("Tokens.jpg", denoise_output)
    binary_image = cv2.cvtColor(denoise_output, cv2.COLOR_BGR2GRAY)

    _, contours_list, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_list:
        if cv2.contourArea(cnt) > brik_area:
            cnt_list.append(cnt)
    return cnt_list
'''

# Draws input point and input ellipse using matplotlib
# for debugging feature extrations
def DrawFeatureComparison (point=(0, 0), center=(0, 0), angle=0, width=.1, height=.1):
    fig, ax = plt.subplots(1)

    plt.scatter(point[0], point[1], c="green")
    plt.scatter(center[0], center[1], c="black")
    circleArea = patches.Ellipse(center, width, height, angle, fill=False)
    ax.add_patch(circleArea)

    plt.xlim([0.8, 4])
    plt.ylim([0, 1.2])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Returns whether or not a point is within an ellipse
# Also returns distance to center of ellipse for use when point is in an overlap
def PointWithinRegion(point=(0, 0), center=(0, 0), angle=0, width=.1, height=.1):
    isWithin = False

    # Some test points
    x = point[0]
    y = point[1]

    cos_angle = np.cos(np.radians(180. - angle))
    sin_angle = np.sin(np.radians(180. - angle))

    xc = x - center[0]
    yc = y - center[1]

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle

    rad_cc = (xct ** 2 / (width / 2.) ** 2) + (yct ** 2 / (height / 2.) ** 2)
    toCenterDist = math.sqrt(math.pow(center[0]-point[0], 2) + math.pow(center[1]-point[1], 2))
    if rad_cc <= 1.:
        isWithin = True

    #DrawFeatureComparison(point, center, angle, width, height)
    return isWithin, toCenterDist

# Tests point generated from feature extraction with stage 2 settings
# including cancel, move and attack gesture
def TestFeatureOne (point=(0, 0)):
    dimRectAttCenter = (2.6165, 0.5990)
    dimRectMoveCenter = (1.3101, 0.5500)
    dimRectCancelCenter = (1.3205, 0.7467)

    inAttack, cDistAtt = PointWithinRegion(point, dimRectAttCenter, 0, 1.7, 0.5)
    inMove, cDistMov = PointWithinRegion(point, dimRectMoveCenter, 7, 1, 0.26)
    inCancel, cDistCan = PointWithinRegion(point, dimRectCancelCenter, 0, 1, 0.3)

    nextState = 2
    shortestDist = math.inf
    if not inAttack and not inMove and not inCancel:
        return 2, False

    if inAttack and cDistAtt < shortestDist:
        shortestDist = cDistAtt
        nextState = 3
    if inMove and cDistMov < shortestDist:
        shortestDist = cDistMov
        nextState = 2
    if inCancel and cDistCan < shortestDist:
        nextState = 1
    print(f"Hand is of state {nextState}")
    return nextState, True

# Same as TestFeatureOne(), but for stage 3
# This tests for cancel and roll
def TestFeatureTwo (point=(0, 0)):
    dimSquareCancelCenter = (1.3205, 0.5633)
    dimSquareRollCenter = (2.3069, 0.2746)

    inCancel, cDistCan = PointWithinRegion(point, dimSquareCancelCenter, -25, 1, 0.4)
    inRoll, cDistRol = PointWithinRegion(point, dimSquareRollCenter, -6, 1.4, 0.24)

    nextState = 3
    shortestDist = math.inf
    if not inCancel and not inRoll:
        return 3, False

    if inCancel and cDistCan < shortestDist:
        shortestDist = cDistCan
        nextState = 1
    if inRoll and cDistRol < shortestDist:
        nextState = 2
    print(f"Hand is of state {nextState}")
    return nextState, True

# This method simply returns an image from the webcam
# By default it also takes 30 ramp frames since the camera taks a while to 'warm up'
def CaptureWebcamImg (port=0, ramp_frames=30, x=1920, y=1080):
    camera = cv2.VideoCapture(port)

    # Set Resolution
    camera.set(3, x)
    camera.set(4, y)

    # Adjust camera lighting
    for i in range(ramp_frames):
        temp = camera.read()
    retval, im = camera.read()
    del(camera)
    print("Image complete")
    return im

# Warps the board from a region of interest int a 1500x1500 image
# This is used to get accurate coordinates for tokens
def WarpBoard(roi, frame):

    warpedImgSize = [1500, 1500] # Aspect ratio 3:3
    dst_points = np.array(((0, 0), (warpedImgSize[0], 0), (warpedImgSize[0], warpedImgSize[1]), (0, warpedImgSize[1])))

    cv2.imwrite("tempwarp.png", frame)
    imgToWarp = cv2.imread("tempwarp.png")
    src_points = np.array([(x*2, y*2) for x, y in roi.points])
    transform, _ = cv2.findHomography(src_points, dst_points)
    adjusted_image = cv2.warpPerspective(imgToWarp, transform, (warpedImgSize[0], warpedImgSize[1]))
    filename = "Board.jpg"
    cv2.imwrite(filename, adjusted_image)
    return filename

# Takes an warped image of the board and returns contours
# for each token.
def ThreshForTokens(fname="Board.jpg"):
    lower_green = np.array([50*0.7, 40, 30], dtype="uint8")
    upper_green = np.array([100*0.7, 255, 130], dtype="uint8")

    low_yellow = np.array([0, 85, 100], dtype="uint8")
    upper_yellow = np.array([115, 195, 255], dtype="uint8")

    low_red = np.array([0, 0, 80], dtype="uint8")
    upper_red = np.array([55, 55, 190], dtype="uint8")

    boardImg = cv2.imread(fname)
    boardImg = cv2.cvtColor(boardImg, cv2.COLOR_BGR2HSV)

    gMask = cv2.inRange(boardImg, lower_green, upper_green)
    yMask = cv2.inRange(boardImg, low_yellow, upper_yellow)
    rMask = cv2.inRange(boardImg, low_red, upper_red)

    #iMask = yMask>0
    #iMask += rMask>0
    iMask = gMask>0
    #print(iMask)
    threshold = np.zeros_like(boardImg, np.uint8)
    threshold[iMask] = boardImg[iMask]
    threshold = cv2.cvtColor(threshold, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(threshold, 70, 255, 0)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    cv2.imwrite("Skraldespand/TokenThresh.png", thresh)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    actualTokenCnt = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 2000:
            actualTokenCnt.append(cnt)
    return actualTokenCnt

# Finds positions of every token in a list of contours
def FindTokenPositions(cnt):
    if not cnt:
        print("No tokens found")
        return
    tokenCenters = []
    for i in range(len(cnt)):
        (x, y), radius = cv2.minEnclosingCircle(cnt[i])
        radius = int(radius)
        center = (int(x), int(y))
        tokenCenters.append(center)
    return tokenCenters

# Calculates cirularity of a contour
def CalulateCntCircularity (radius, cntArea, ):
    cntPerimeter = 2 * math.pi * radius
    cntCircularity = (4 * math.pi * cntArea) / math.pow(cntPerimeter, 2)
    return cntCircularity

# Finds a seleced token bassed on a list of every token center, center of the hand,
# and a list of contours for tokens found in a sliced image
def FindSelection (cnt, tokenCenters, handCenter):
    selection = (-1, -1)
    shortDist = math.inf

    sizeLower = 700
    sizeUpper = 7000
    if not cnt:
        print("No tokens found. Try again")
        return selection, 1
    for i in range(len(cnt)):
        (x, y), radius = cv2.minEnclosingCircle(cnt[i])
        #radius = int(radius)
        center = (int(x), int(y))
        cntArea = cv2.contourArea(cnt[i])
        cntCircularity = CalulateCntCircularity(radius, cntArea)
        print(cntCircularity, cntArea)
        if cntArea not in range(sizeLower, sizeUpper) and cntCircularity < 0.75:
            continue
        distance = math.sqrt(math.pow(center[0]-handCenter[0], 2) + math.pow(center[1]-handCenter[1], 2))
        if distance < shortDist:
            shortDist = distance
            selection = center
    if selection == (-1, -1):
        print("No real tokens found")
        return selection, 1

    closest = math.inf
    closestToSelection = (0, 0)
    for token in tokenCenters:
        distToToken = math.sqrt(math.pow(token[0]-selection[0], 2) + math.pow(token[1]-selection[1], 2))
        print(distToToken, closest)
        print(token, selection)
        if distToToken < closest:
            closestToSelection = token
            closest = distToToken
    return closestToSelection, 2

# Denoises an image
def DeNoise(frame):  # deNoises the image.
    deNoisedFrame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    cv2.normalize(deNoisedFrame, deNoisedFrame, 0, 255, cv2.NORM_MINMAX)
    return deNoisedFrame

# All states:
def StateZero(roi):
    nextState = 1

    img = CaptureWebcamImg()
    
    WarpBoard(roi, img)
    tokenContours = ThreshForTokens("Board.jpg")
    tokenCenters = FindTokenPositions(tokenContours)

    bgImage = DeNoise(img)
    return nextState, bgImage, tokenCenters

def StateOne(roi, bgImage, tokenCenters):
    import PointButBetterEdited as pbb
    pointCutoutImg, foundHand, handCenter = pbb.FindPointDirectionMask(bgImage, CaptureWebcamImg())
    print(f"Hand search done. "
          f"Found hand: {foundHand}")
    if not foundHand:
        nextState, nextBg, newTokenCenters = StateZero(roi)
        noSelection = (0, 0)
        return noSelection, nextState, nextBg, newTokenCenters
    else:
        WarpBoard(roi, pointCutoutImg)
        print("Warp done")
        blobContours = ThreshForTokens("Board.jpg")
        print("Token search done")
        newSelection, nextState = FindSelection(blobContours, tokenCenters, handCenter)
        return newSelection, nextState, bgImage, tokenCenters

def StateTwo(bgImage):
    import BS_With_images_instead_of_video as featEx
    image = CaptureWebcamImg()
    pointPos, foundCon = featEx.GetFeaturesOne(image, bgImage)
    if not foundCon:
        print("Found no hand")
        return 2, foundCon

    nextState, hasGesture = TestFeatureOne(pointPos)
    print(f"Feature extraction dat: {pointPos, nextState, hasGesture}")
    if hasGesture:
        return nextState, hasGesture
    print("Found no gesture")
    return 2, hasGesture

def StateThree(bgImage):
    import BS_With_images_instead_of_video as featEx
    image = CaptureWebcamImg()
    pointPos, foundCon = featEx.GetFeaturesTwo(image, bgImage)
    if not foundCon:
        print("Found no hand")
        return 3

    nextState, hasGesture = TestFeatureTwo(pointPos)
    print(f"Feature extraction dat: {pointPos, nextState, hasGesture}")
    if hasGesture:
        return nextState
    print("Found no gesture")
    return 3