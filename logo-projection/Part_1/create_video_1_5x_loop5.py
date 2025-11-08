"""
Create 1.5x speed video looped 5 times
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
    
    output_path = "videos/barcelona_logo_projection_1_5x_loop5.mp4"
    height, width, _ = processed_imgs[0].shape
    original_fps = 30
    speed_factor = 1.5
    fps = original_fps * speed_factor  # 45 fps for 1.5x speed
    loop_count = 5
    
    # Try different codec options for browser compatibility
    fourcc_options = [
        cv2.VideoWriter_fourcc(*'avc1'),  # H.264
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
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("Warning: Using default codec")
    else:
        print(f"Using codec: {fourcc}")
    
    # Create video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return
    
    print(f"Creating video: {output_path}")
    print(f"  Speed: {speed_factor}x ({fps} fps)")
    print(f"  Loops: {loop_count} times")
    print(f"  Total frames: {len(processed_imgs) * loop_count}")
    
    # Write frames loop_count times
    for loop in range(loop_count):
        print(f"\nLoop {loop + 1}/{loop_count}:")
        for i, img in enumerate(processed_imgs):
            out.write(img)
            if (i + 1) % 20 == 0:
                print(f"  Written {i + 1}/{len(processed_imgs)} frames")
    
    out.release()
    print(f"\nVideo saved: {output_path}")
    
    # Verify video
    cap = cv2.VideoCapture(output_path)
    if cap.isOpened():
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / video_fps if video_fps > 0 else 0
        print(f"\nVideo verification:")
        print(f"  Frames: {frame_count}")
        print(f"  FPS: {video_fps}")
        print(f"  Resolution: {video_width}x{video_height}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Expected frames: {len(processed_imgs) * loop_count}")
        print(f"  Expected duration: {len(processed_imgs) * loop_count / fps:.2f} seconds")
        cap.release()
    else:
        print("Warning: Could not verify video file")


if __name__ == "__main__":
    main()

