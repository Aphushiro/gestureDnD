import cv2
import numpy as np

def FindPointingHand(bgImg, newImg):
    bgImgGray = cv2.cvtColor(bgImg, cv2.COLOR_BGR2GRAY)
    newImgGray = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(bgImgGray)
    mask[bgImgGray > 200] = 1

    bgImgMasked = cv2.bitwise_and(bgImgGray, bgImgGray, mask=mask)

    diffImg = cv2.absdiff(bgImgMasked, newImgGray)

    threshImg = cv2.threshold(diffImg, 25, 255, cv2.THRESH_BINARY)[1]

    dilatedImg = cv2.dilate(threshImg, None, iterations=5)

    contours = cv2.findContours(dilatedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        hull = cv2.convexHull(cnt)

        if cv2.contourArea(cnt) / cv2.contourArea(hull) <= 0.9:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX , cY = 0, 0
            x, y, w, h = cv2.boundingRect(cnt)
            return (cX, cY), (x, y, w, h)
    return None