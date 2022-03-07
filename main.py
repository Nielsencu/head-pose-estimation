"""Demo code showing how to estimate human head pose.

There are three major steps:
1. Detect human face in the video frame.
2. Run facial landmark detection on the face image.
3. Estimate the pose by solving a PnP problem.

To find more details, please refer to:
https://github.com/yinguobing/head-pose-estimation
"""
from argparse import ArgumentParser

import cv2
import math


from mark_detector import MarkDetector
from pose_estimator import PoseEstimator

import time

from imutils import face_utils, resize

print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

# Parse arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
args = parser.parse_args()

# def eye_aspect_ratio(self, eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

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

        # Step 1: Get a face from current frame.
        facebox = mark_detector.extract_cnn_facebox(frame)

        # Any face found?
        if facebox is not None:

            # Step 2: Detect landmarks. Crop and feed the face area into the
            # mark detector.
            x1, y1, x2, y2 = facebox
            face_img = frame[y1: y2, x1: x2]

            # Run the detection.
            tm.start()
            marks = mark_detector.detect_marks(face_img)
            tm.stop()

            # Convert the locations from local face area to the global image.
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
            # (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

            # leftEye = shape[self.lStart:self.lEnd]
            # rightEye = shape[self.rStart:self.rEnd]

            # leftEAR = eye_aspect_ratio(leftEye)
            # rightEAR = eye_aspect_ratio(rightEye)

            # Try pose estimation with 68 points.
            pose = pose_estimator.solve_pose_by_68_points(marks)

            # All done. The best way to show the result would be drawing the
            # pose on the frame in realtime.

            # Do you want to see the pose annotation?
            # pose_estimator.draw_annotation_box(
            #     frame, pose[0], pose[1], color=(0, 255, 0))

            # Do you want to see the head axes?
            pose_estimator.draw_axes(frame, pose[0], pose[1])
            pose = pose[0]
            print(f'Rotation around x is {pose[0] * 180 / math.pi}')
            print(f'Rotation around y is {pose[1] * 180 / math.pi}')
            print(f'Rotation around z is {pose[2] * 180 / math.pi}')
            # Convert to angles
            pose = [i * 180 / math.pi for i in pose]

            # Do you want to see the marks?
            mark_detector.draw_marks(frame, marks, color=(0, 255, 0))
            x_threshold = 25
            face_up_y_threshold = 15
            face_down_y_threshold = 30
            attentive = 1 if (-x_threshold <= pose[0] <= x_threshold) and (-face_up_y_threshold <= pose[1] <= face_down_y_threshold) else 0
            mark_detector.draw_text(frame, f'attentive : {attentive}', facebox=facebox)

            # Do you want to see the facebox?
            #mark_detector.draw_box(frame, [facebox])
        

        # Show preview.
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) == 27:
            break
        print(time.time() - start)
