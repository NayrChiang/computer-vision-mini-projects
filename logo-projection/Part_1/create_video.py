"""
Create video from processed frames
"""

import os
import glob
import numpy as np
import cv2
import imageio
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

    # Create video
    if not os.path.exists("videos"):
        os.mkdir("videos")
    
    output_path = "videos/barcelona_logo_projection.mp4"
    height, width, _ = processed_imgs[0].shape
    fps = 30  # Original fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating video: {output_path}")
    for i, img in enumerate(processed_imgs):
        out.write(img)
        if (i + 1) % 10 == 0:
            print(f"Written {i + 1}/{len(processed_imgs)} frames")
    
    out.release()
    print(f"Video saved: {output_path}")
    
    # Also save as GIF
    print("Saving GIF...")
    gif_path = "videos/barcelona_logo_projection.gif"
    # Convert BGR to RGB for imageio
    gif_frames = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in processed_imgs]
    imageio.mimsave(gif_path, gif_frames, fps=30, loop=0)
    print(f"GIF saved: {gif_path}")
    
    # Also save individual frames for reference
    if not os.path.exists("part_1_results"):
        os.mkdir("part_1_results")
    
    save_ind = [0, 25, 50, 75, 100, 125]
    for ind in save_ind:
        if ind < len(processed_imgs):
            cv2.imwrite(f"part_1_results/frame_{ind}.png", processed_imgs[ind])


if __name__ == "__main__":
    main()

