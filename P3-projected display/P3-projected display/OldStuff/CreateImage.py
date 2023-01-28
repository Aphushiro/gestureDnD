import cv2
def CaptureWebcamImg (port=0, ramp_frames=30, x=1920, y=1080):
    camera = cv2.VideoCapture(port)

    # Set Resolution
    camera.set(3, x)
    camera.set(4, y)

    # Adjust camera lighting
    for i in range(ramp_frames):
        temp = camera.read()

    imNum = 0
    curFolder = ""
    while True:
        retval, frame = camera.read()
        cv2.imshow(f"Camera - {curFolder}", frame)
        key = cv2.waitKey(30)
        if key == ord("f"):
            image_dir = f"{curFolder}img{imNum}.png"
            cv2.imwrite(image_dir, frame)
            imNum += 1
        if key == ord("e"):
            break
    del(camera)

CaptureWebcamImg()