"""Detect if eyes are closed for blink detection"""
import math

def eye_aspect_ratio(landmarks, points):
    """Calculates a ratio that can indicate whether an eye is closed or not.
    It's the division of the width of the eye, by its height.

    Arguments:
        landmarks: Facial landmarks for the face region
        points (list): Points of an eye (from the 68 Multi-PIE landmarks)

    Returns:
        The computed eye aspect ratio, EAR
    """

    try:
        # EAR = (|p1-p5|+|p2-p4|) / (2|p0-p3|)
        eye_width = abs(math.dist(landmarks[points[1]], landmarks[points[5]])) + abs(math.dist(landmarks[points[2]], landmarks[points[4]]))
        eye_height = abs(math.dist(landmarks[points[0]], landmarks[points[3]]))
        ratio = eye_width/(2 * eye_height)
    except ZeroDivisionError:
        ratio = None

    return ratio