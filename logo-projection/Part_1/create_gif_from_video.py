"""
Convert existing video to GIF
"""
import cv2
import imageio
import os
import sys

def video_to_gif(video_path, gif_path=None, fps=30, scale=1.0):
    """
    Convert video file to GIF
    
    Args:
        video_path: Path to input video file
        gif_path: Path to output GIF file (default: same name with .gif extension)
        fps: Frames per second for GIF (default: 30)
        scale: Scale factor for GIF size (default: 1.0, use < 1.0 to reduce size)
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False
    
    if gif_path is None:
        gif_path = os.path.splitext(video_path)[0] + ".gif"
    
    print(f"Reading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return False
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Scale if needed
        if scale != 1.0:
            height, width = frame_rgb.shape[:2]
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
        
        frames.append(frame_rgb)
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    
    print(f"Writing GIF: {gif_path} ({len(frames)} frames)...")
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"GIF saved: {gif_path}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_gif_from_video.py <video_path> [gif_path] [fps] [scale]")
        print("Example: python create_gif_from_video.py videos/barcelona_logo_projection_1_5x.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    gif_path = sys.argv[2] if len(sys.argv) > 2 else None
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 30
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    
    video_to_gif(video_path, gif_path, fps, scale)

