"""
Verify video file
"""

import cv2
import sys

video_path = "videos/barcelona_logo_projection_2x.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"ERROR: Could not open video: {video_path}")
    sys.exit(1)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = frame_count / fps if fps > 0 else 0

print(f"Video: {video_path}")
print(f"  Status: OK")
print(f"  Frames: {frame_count}")
print(f"  FPS: {fps}")
print(f"  Resolution: {width}x{height}")
print(f"  Duration: {duration:.2f} seconds")

# Try to read first frame
ret, frame = cap.read()
if ret:
    print(f"  First frame: OK (shape: {frame.shape})")
else:
    print(f"  First frame: ERROR - could not read")

cap.release()

