import numpy as np
import cv2
# from opencv.video import create_capture

# Use a classifier to detect a face in the video frame

def get_cam_frame(cam):
    ret, img = cam.read()
    # smaller frame size - things run a lot smoother than a full screen img
    img = cv2.resize(img, (800, 450))
    return img

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def main():
    # Camera 0 is usually the built in webcam camera... also most people only have 1 webcam on their laptop
    cam = cv2.VideoCapture(0)
    
    cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    
    while True:
        img = get_cam_frame(cam)
        dognose = cv2.imread('dog_nose.png')
        dogleftear = cv2.imread('dog_leftear.png')
        dogrightear = cv2.flip(cv2.imread('dog_leftear.png'), 180)
        final = img.copy()
        
        # classifier wants things in black and white
        bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bw = cv2.equalizeHist(bw)
        
        rects = detect(bw, cascade)
        # Mostly useful for debugging
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cx = (x2 + x1)/2
            cy = 53*(y2 + y1)/100
            nx1 = int(cx - (x2-x1)/7)
            ny1 = int(cy - (y2-y1)/10)
            nx2 = int(cx + (x2-x1)/7)
            ny2 = int(cy + (y2-y1)/10)
            dognose = cv2.resize(dognose, (nx2 - nx1, ny2 - ny1))
            final[ny1:ny2, nx1:nx2] = dognose
        
            lx2 = int(abs(cx - (3*(x2 - x1)/10)))
            ly1 = int(abs(cy - (y2-y1)/2))
            lx1 = int(abs(lx2 - (3*(x2-x1)/8)))
            ly2 = int(abs(ly1 + (3*(y2-y1)/16)))
            dogleftear = cv2.resize(dogleftear, (abs(lx2 - lx1), abs(ly2 - ly1)))

            final[ly1:ly2, lx1:lx2] = dogleftear
        
            rx1 = int(abs(cx + (3*(x2 - x1)/10)))
            ry1 = int(abs(ly1))
            ry2 = int(abs(ly2))
            rx2 = int(abs(rx1 + (3*(x2-x1)/8)))
            dogrightear = cv2.resize(dogrightear, (abs(rx2 - rx1), abs(ry2 - ry1)))
            final[ry1:ry2, rx1:rx2] = dogrightear
        
        cv2.imshow('face detect', final)
        
        # Esc key quits
        if 0xFF & cv2.waitKey(1) == 27:
            break
cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
