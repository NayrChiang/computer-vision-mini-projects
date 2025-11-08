"""
Create web-compatible video from processed frames
Uses H.264 codec for better browser compatibility
"""

import os
import glob
import numpy as np
import cv2
from warp_pts import warp_pts
import utils


def main():
    # Load Penn logo image, and get its corners
    penn = cv2.imread("../data/barcelona/images/logos/penn_engineering_logo.png")
    penn_y, penn_x, _ = penn.shape
    penn_corners = np.array([[0, 0], [penn_x, 0], [penn_x, penn_y], [0, penn_y]])

    # Load all image paths, and the goal corners in each image
    img_files = sorted(glob.glob("../data/barcelona/images/barca_real/*.png"))
    goal_data = np.load("../data/barcelona/BarcaReal_pts.npy")

    print(f"Processing {len(goal_data)} frames...")
    
    # Process all images
    processed_imgs = []
    for i in range(len(goal_data)):
        goal = cv2.imread(img_files[i])
        goal_corners = goal_data[i]
        # Warping
        int_pts = utils.calculate_interior_pts(goal.shape, goal_corners)
        warped_pts = warp_pts(goal_corners, penn_corners, int_pts)
        projected_img = utils.inverse_warping(goal, penn, int_pts, warped_pts)
        processed_imgs.append(projected_img)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(goal_data)} frames")

    # Create video directory
    if not os.path.exists("videos"):
        os.mkdir("videos")
    
    output_path = "videos/barcelona_logo_projection_2x.mp4"
    height, width, _ = processed_imgs[0].shape
    fps = 60  # 2x speed (original 30 fps * 2)
    
    # Use H.264 codec for better browser compatibility
    # Try different codec options
    fourcc_options = [
        cv2.VideoWriter_fourcc(*'H264'),  # H.264 codec
        cv2.VideoWriter_fourcc(*'avc1'),  # Alternative H.264
        cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4
    ]
    
    fourcc = None
    for codec in fourcc_options:
        try:
            test_writer = cv2.VideoWriter('test_temp.mp4', codec, fps, (width, height))
            if test_writer.isOpened():
                fourcc = codec
                test_writer.release()
                os.remove('test_temp.mp4')
                break
        except:
            continue
    
    if fourcc is None:
        # Fallback to default
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("Warning: Using default codec, may have compatibility issues")
    else:
        print(f"Using codec: {fourcc}")
    
    # Create video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return
    
    print(f"Creating video: {output_path}")
    for i, img in enumerate(processed_imgs):
        # Ensure image is in correct format (BGR for OpenCV)
        if len(img.shape) == 3:
            out.write(img)
        else:
            print(f"Warning: Frame {i} has unexpected shape: {img.shape}")
        
        if (i + 1) % 10 == 0:
            print(f"Written {i + 1}/{len(processed_imgs)} frames")
    
    out.release()
    print(f"Video saved: {output_path}")
    
    # Verify video
    cap = cv2.VideoCapture(output_path)
    if cap.isOpened():
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\nVideo verification:")
        print(f"  Frames: {frame_count}")
        print(f"  FPS: {video_fps}")
        print(f"  Resolution: {video_width}x{video_height}")
        print(f"  Duration: {frame_count / video_fps:.2f} seconds")
        cap.release()
    else:
        print("Warning: Could not verify video file")
    
    # Also save individual frames for reference
    if not os.path.exists("part_1_results"):
        os.mkdir("part_1_results")
    
    save_ind = [0, 25, 50, 75, 100, 125]
    for ind in save_ind:
        if ind < len(processed_imgs):
            cv2.imwrite(f"part_1_results/frame_{ind}.png", processed_imgs[ind])


if __name__ == "__main__":
    main()

