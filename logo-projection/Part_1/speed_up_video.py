"""
Speed up video by 2x
"""

import cv2
import os

def speed_up_video(input_path, output_path, speed_factor=2.0):
    """
    Speed up video by a factor
    """
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Original video: {fps} fps, {width}x{height}, {total_frames} frames")
    
    # New FPS (2x speed = 2x FPS)
    new_fps = fps * speed_factor
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    cap.release()
    out.release()
    
    print(f"Speed-up video saved: {output_path}")
    print(f"New video: {new_fps} fps, {total_frames} frames")

if __name__ == "__main__":
    input_video = "videos/barcelona_logo_projection.mp4"
    output_video = "videos/barcelona_logo_projection_2x.mp4"
    
    if os.path.exists(input_video):
        speed_up_video(input_video, output_video, speed_factor=2.0)
    else:
        print(f"Error: Input video not found: {input_video}")

