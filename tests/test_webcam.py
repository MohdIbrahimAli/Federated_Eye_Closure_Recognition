import cv2 as cv
import sys

cap = cv.VideoCapture(0)

if not cap.isOpened():
    sys.exit("Cannot open Camera")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot access frames.. Exiting...")
        break
    cv.imshow("Live Video",frame)

    if cv.waitKey(1) == ord("q"):
        break
cap.release()
cv.destroyAllWindows()
