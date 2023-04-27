import cv2 as cv
from cv2 import aruco
import numpy as np

marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50) # It is a dictionary which contains all the types of arucomarkers 

param_markers = aruco.DetectorParameters_create() # initialising the parameters 

cap = cv.VideoCapture(0) # start capturing from  the camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )
    if marker_corners:
        for ids, corners in zip(marker_IDs, marker_corners):
            cv.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            ) # cv.polylines will highlight the arucomarkers 
            corners = corners.reshape(4, 2) 
            corners = corners.astype(int) #it will give you the int values of the cordinates of corners
            top_right = corners[0].ravel() # the inputs will be in form of matrix data so ravel will put that in one list 
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()
            cv.putText(
                frame,
                f"id: {ids[0]}",
                top_right,
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (200, 100, 0),
                2,
                cv.LINE_AA,
            )
            # puttext will print(ids, "  ", corners)
    cv.imshow("frame", frame) # will show the dispaly on yuor screen 
    key = cv.waitKey(1) # waitkey is fuction which will detect the inputed key strokes 
    if key == ord("q"):
        break
cap.release()
cv.destroyAllWindows()