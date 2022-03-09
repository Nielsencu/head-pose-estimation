"""Human Head Pose Estimation Demo

1. Detect human face in the video frame.
2. Run facial landmark detection on the face image.
3. Estimate the pose by solving a PnP problem.

To find more details, please refer to:
https://github.com/yinguobing/head-pose-estimation
"""

from argparse import ArgumentParser

import cv2
import math
import time
import pandas as pd
import matplotlib.pyplot as plt


from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from blink_detection import eye_aspect_ratio

import time

from database import db

print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

# Parse arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
args = parser.parse_args()

# FPS
prev_frame_time, cur_frame_time = 0, 0

# Attention Metrics
attn = 100
attn_span = pd.DataFrame(columns=['timestamp', 'attention'])
looking_away = None

# Blink Detection
LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]
eyes_closed, eyes_already_closed = False, False
start_eyes_closed, time_eyes_closed = 0, 0

from imutils import face_utils
import dlib
import cv2
import time
from scipy.spatial import distance
import numpy as np

# let's go code an faces detector(HOG) and after detect the 
# landmarks on this detected face

# p = our pre-treined model directory, on my case, it's on the same script's diretory.
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

def shape_to_np(shape, dtype="f"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

if __name__ == '__main__':
    # Before estimation started, there are some startup works to do.

    # 1. Setup the video source from webcam or video file.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Video source not assigned, default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)

    # Get the frame size. This will be used by the pose estimator.
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # 2. Introduce a pose estimator to solve pose.
    pose_estimator = PoseEstimator(img_size=(height, width))

    # 3. Introduce a mark detector to detect landmarks.
    mark_detector = MarkDetector()

    # 4. Measure the performance with a tick meter.
    tm = cv2.TickMeter()

    # Initialize variables sent to database every 5 seconds
    poses = []
    avg_time_eyes_closed = 0
    avg_attention = 0
    count = 0
    last_sent = time.time()

    detector = dlib.get_frontal_face_detector()

    # Now, let the frames flow.
    while True:
        start = time.time()

        # Read a frame.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # If the frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Converting the image to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        # Get faces into webcam's image
        rects = detector(gray, 0)
        
        cur_time = pd.Timestamp.now(tz='Asia/Singapore').strftime('%d-%m-%Y %H:%M:%S.%f')

        if not(rects):
            attn = max(attn-0.1,0)
            looking_away = None
            eyes_closed = False
        else:
            # For each detected face, find the landmark.
            for (i, rect) in enumerate(rects):
                # Make the prediction and transfom it to numpy array
                shape = predictor(gray, rect)
                shape = shape_to_np(shape)

                # Draw on our image, all the finded cordinate points (x,y) 
                for (x, y) in shape:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                print(shape.dtype)
                # Try pose estimation with 68 points.
                pose = pose_estimator.solve_pose_by_68_points(shape)

                # All done. The best way to show the result would be drawing the pose on the frame in realtime.

                # Do you want to see the pose annotation cube?
                # pose_estimator.draw_annotation_box(
                #     frame, pose[0], pose[1], color=(0, 255, 0))

                # Do you want to see the head axes?
                # pose_estimator.draw_axes(frame, pose[0], pose[1])

                pose = pose[0]

                # Convert to angles
                pose = [ith_pose[0] * 180 / math.pi for ith_pose in pose]

                # Do you want to see the marks?
                #mark_detector.draw_marks(frame, marks, color=(0, 255, 0))
                #mark_detector.draw_marks(frame, [mark for i, mark in enumerate(marks) if i in LEFT_EYE_POINTS + RIGHT_EYE_POINTS], color=(0, 0, 255))

                # Do you want to see the facebox?
                # mark_detector.draw_box(frame, [facebox])

                #--- ATTENTION ALGORITHMS
                print(f'[{cur_time}] (x,y,z): ({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f})')

                # Attention Thresholds
                face_left_right_threshold = 20
                face_up_threshold = 30
                face_down_threshold = 20
                attn_change = 0.5
                EAR_threshold = 0.25

                left_Eye = shape[lStart:lEnd]
                right_Eye = shape[rStart:rEnd]
                left_EAR = eye_aspect_ratio(left_Eye)
                right_EAR = eye_aspect_ratio(right_Eye)
                ear = (left_EAR + right_EAR) / 2.0
                eyes_closed = ear < EAR_threshold
                print(f'LEFT: {left_EAR:.3f}, RIGHT: {right_EAR:.3f}, CLOSED: {eyes_closed}')

                if eyes_closed:
                    if eyes_already_closed:
                        time_eyes_closed = time.time() - start_eyes_closed
                    else:
                        start_eyes_closed = time.time()
                        eyes_already_closed = True
                else: # eyes opened
                    if eyes_already_closed:
                        eyes_already_closed = False
                        time_eyes_closed = time.time() - start_eyes_closed
                    else: # eyes not already closed
                        time_eyes_closed = 0

                # Add/Deduct Attention based on the Metrics
                if (-face_left_right_threshold < pose[0] < face_left_right_threshold) \
                    and (-face_down_threshold < pose[1] < face_up_threshold) \
                    and time_eyes_closed < 2:
                    attn = min(100, attn + attn_change / 2)
                    looking_away = False
                else:
                    attn = max(attn-attn_change,0)
                    looking_away = True

                print('-------------------------------------------------------------------------------')

        # Calculate FPS
        cur_frame_time = time.time()
        fps = 1/(cur_frame_time-prev_frame_time)
        prev_frame_time = cur_frame_time

        mark_detector.draw_text(frame, f'FPS: {int(fps)}', coords=(30,30))
        attentiveness = 'Please keep your face within the screen' if looking_away is None else f'Looking Away: {looking_away}'
        mark_detector.draw_text(frame, attentiveness, coords=(30,60))
        mark_detector.draw_text(frame, f'Attention: {attn:.2f}%', coords=(30,90))
        eyes_closed_text = f'{time_eyes_closed:.2f}s' if eyes_closed else ''
        mark_detector.draw_text(frame, f'Eyes Closed: {eyes_closed} {eyes_closed_text}', coords=(30,120))

        # poses.append(pose)
        # avg_attention += attn
        # avg_time_eyes_closed += time_eyes_closed
        # count +=1

        # # Save value every 5 seconds to database
        # time_now = time.time()
        # if time_now - last_sent > 5:
        #     last_sent = time_now
        #     # 
        #     pose = poses[2]
        #     avg_attention /= count
        #     avg_time_eyes_closed /= count
        #     print("Pushed to database")
        #     db.child("Meeting102").child("metrics").push({"time":str(cur_time), "attention": avg_attention, "X pose": pose[0], "Y pose": pose[1], "Z pose": pose[2], "eyes_closed" : time_eyes_closed })
        #     # Reset all variables
        #     poses = []
        #     avg_attention = 0
        #     avg_time_eyes_closed = 0
        #     count = 0

        attn_span = pd.concat([
            attn_span, 
            pd.DataFrame.from_dict({
                'timestamp': [cur_time],
                'attention': [attn],
                'eyes_closed': [time_eyes_closed],
            })
        ], ignore_index=True)

        # Show preview of the webcam/video feed
        app_name = 'Attention Tracker'
        cv2.imshow(app_name, frame)

        if cv2.waitKey(1) == 27 or cv2.getWindowProperty(app_name, cv2.WND_PROP_VISIBLE) < 1:
            print(f'\nShutting Down {app_name}...')
            break

cap.release()
cv2.destroyAllWindows()

# Save results into CSV and plot graphs
attn_span['timestamp'] = pd.to_datetime(attn_span['timestamp'])
attn_span = attn_span.set_index('timestamp')
attn_span.to_csv('attn.csv', index=True)
attn_span['attention'].plot.line(label='Attention (%)', legend=True)
attn_span['eyes_closed'].plot.line(label='Eyes Closed (s)', secondary_y=True, legend=True)
plt.show()
