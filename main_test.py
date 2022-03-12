'''Human Head Pose Estimation Demo

1. Detect human face in the video frame.
2. Run facial landmark detection on the face image.
3. Estimate the pose by solving a PnP problem.

To find more details, please refer to:
https://github.com/yinguobing/head-pose-estimation
'''

from argparse import ArgumentParser
import math
import time

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imutils import face_utils
import dlib

from src.pose_estimator import PoseEstimator
from src.landmark_metrics import eye_aspect_ratio, mouth_aspect_ratio
from src.database import Database
from src.eyes import update_eyes
from src.yawn import update_yawn
from src.message import Message

print(__doc__)
print('OpenCV version: {}'.format(cv2.__version__))

# Parse arguments from user input
parser = ArgumentParser()
parser.add_argument('--video', type=str, default=None,
                    help='Video file to be processed.')
parser.add_argument('--cam', type=int, default=None,
                    help='The webcam index.')
args = parser.parse_args()

# FPS
prev_frame_time, cur_frame_time = 0, 0

# Attention Metrics
attn = 100
attn_span = pd.DataFrame(columns=['timestamp', 'attention'])
looking_away = None

# Blink Detection
eyes_closed, eyes_already_closed = False, False
start_eyes_closed, time_eyes_closed = 0, 0

# Yawn detection
mouth_open, mouth_already_open = False, False
start_mouth_open, time_mouth_open = 0, 0

# Landmarks Detection
pretrained_landmarks = r'assets/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(pretrained_landmarks)

(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']
(mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_68_IDXS['mouth']

def draw_text(image, label, coords=(50,50)):
    cv2.putText(
        img=image,
        text=label,
        org=(coords[0], coords[1]),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 0, 255),
    )

if __name__ == '__main__':
    # Setup the video source from webcam or video file
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print('Video source not assigned, default webcam will be used')
        video_src = 0

    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print("Canot open camera")
        exit()

    # Get the frame size. This will be used by the pose estimator.
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Initialize pose estimator
    pose_estimator = PoseEstimator(img_size=(height, width))

    # Initialize variables sent to database every 5 seconds
    poses = []
    avg_time_eyes_closed = 0
    avg_time_mouth_open = 0
    avg_attention = 0
    count = 0
    yawn_duration = 0
    last_sent = time.time()

    # Initialize database connection
    db = Database('Meeting102')

    # Initialize real-time message sent to database
    message = Message()

    detector = dlib.get_frontal_face_detector()

    while True:
        start = time.time()

        # Read a frame
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # If the frame comes from webcam, flip it so it looks like a mirror
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Converting the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get faces into webcam's image
        rects = detector(gray, 0)

        cur_time = pd.Timestamp.now(tz='Asia/Singapore').strftime('%d-%m-%Y %H:%M:%S.%f')

        if not(rects):
            attn = max(attn-0.1,0)
            looking_away = None
            eyes_closed = False
            yawn = False
            pose = None
        else:
            # For each detected face, find the landmarks
            for (i, rect) in enumerate(rects):
                # Make the prediction and transfom it to numpy array
                shape = predictor(gray, rect)
                shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype='f')

                # Draw all landmark (x,y) coordinate points
                for (x, y) in shape:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

                # Estimate pose using the 68 facial landmarks
                pose = pose_estimator.solve_pose_by_68_points(shape)

                # View pose annotation cube
                # pose_estimator.draw_annotation_box(
                #     frame, pose[0], pose[1], color=(0, 255, 0))

                # View head axes
                # pose_estimator.draw_axes(frame, pose[0], pose[1])

                pose = pose[0]

                # Convert to angles
                pose = [ith_pose[0] * 180 / math.pi for ith_pose in pose]

                # View marks
                # draw_marks(frame, marks, color=(0, 255, 0))

                # View facebox
                # draw_box(frame, [facebox])

                #--- ATTENTION ALGORITHMS ---#
                print(f'[{cur_time}] (x,y,z): ({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f})')

                # Attention Thresholds
                face_left_right_threshold = 20
                face_up_threshold = 30
                face_down_threshold = 20
                attn_change = 0.5
                EAR_threshold = 0.25

                # Eye Aspect Ratio (EAR)
                left_eye = shape[left_eye_start:left_eye_end]
                right_eye = shape[right_eye_start:right_eye_end]
                left_EAR = eye_aspect_ratio(left_eye)
                right_EAR = eye_aspect_ratio(right_eye)
                EAR = (left_EAR + right_EAR) / 2.0
                eyes_closed = EAR < EAR_threshold
                print(f'LEFT: {left_EAR:.3f}, RIGHT: {right_EAR:.3f}, CLOSED: {eyes_closed}')

                # Mouth Aspect Ratio (MAR)
                mouth = shape[mouth_start:mouth_end]
                MAR = mouth_aspect_ratio(mouth)
                mouth_open = MAR > 0.3
                print(f'YAWN: {MAR}')

                time_eyes_closed, start_eyes_closed, eyes_already_closed = update_eyes(eyes_closed, time_eyes_closed, eyes_already_closed, start_eyes_closed)

                time_mouth_open, start_mouth_open, mouth_already_open = update_yawn(mouth_open, time_mouth_open, mouth_already_open, start_mouth_open)

                # Add/Deduct Attention based on the thresholds
                if (-face_left_right_threshold < pose[0] < face_left_right_threshold) \
                    and (-face_down_threshold < pose[1] < face_up_threshold) \
                    and time_eyes_closed < 2 \
                    and time_mouth_open < 1:
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

        # Display metrics on the screen
        draw_text(frame, f'FPS: {int(fps)}', coords=(30,30))
        attentiveness = 'Please keep your face within the screen' if looking_away is None else f'Looking Away: {looking_away}'
        draw_text(frame, attentiveness, coords=(30,60))
        draw_text(frame, f'Attention: {attn:.2f}%', coords=(30,90))
        eyes_closed_text = f'{time_eyes_closed:.2f}s' if eyes_closed else ''
        draw_text(frame, f'Eyes Closed: {eyes_closed} {eyes_closed_text}', coords=(30,120))
        mouth_opened_text = f'{time_mouth_open:.2f}s' if mouth_open else ''
        draw_text(frame, f'Yawn: {mouth_open} {mouth_opened_text}', coords=(30,150))

        # Update message
        message.update(pose, attn, time_eyes_closed, time_mouth_open)

        # Save value every 5 seconds to database
        time_now = time.time()
        if time_now - last_sent > 5:
            last_sent = time_now
            # Finalize message by taking average, or median for pose
            message.finalize(cur_time)
            # Send message to database
            db.send(message.as_dict())
            # Reset all variables in message
            message.reset()

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
# attn_span.to_csv(r'assets/attn.csv', index=True) # Temporarily avoid saving to csv
attn_span['attention'].plot.line(label='Attention (%)', legend=True)
attn_span['eyes_closed'].plot.line(label='Eyes Closed (s)', secondary_y=True, legend=True)
plt.show()