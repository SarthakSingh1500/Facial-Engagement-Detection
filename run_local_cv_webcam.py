# Run this file on CV2 in local machine to construct a Concentration Index (CI).
# Video image will show emotion on first line, and engagement on second. Engagement/concentration classification displays either 'Pay attention', 'You are engaged' and 'you are highly engaged' based on CI. Webcam is required.
# Analysis is in 'Util' folder.


from util.analysis_realtime import analysis
import cv2
import numpy as np
import time
# Initializing
cap = cv2.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0
ana = analysis()

# Capture every frame and send to detector
while True:
    _, frame = cap.read()
    bm = ana.detect_face(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = "FPS: " + str(fps)
    cv2.putText(frame, fps, (50, 150), font, 1, (0, 0, 255), 3)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
# Exit if 'q' is pressed
    if key == ord('q'):
        break

# Release the memory
cap.release()
cv2.destroyAllWindows()
