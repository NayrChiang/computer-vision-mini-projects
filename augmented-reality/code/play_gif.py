"""
Play GIF animation
"""
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys

def play_gif(gif_path):
    """Play GIF as animation using matplotlib"""
    try:
        # Read GIF
        gif = imageio.get_reader(gif_path)
        frames = []
        for frame in gif:
            frames.append(frame)
        
        print(f"Loaded {len(frames)} frames from {gif_path}")
        print("Displaying animation... Close the window to stop.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.axis('off')
        
        # Display first frame
        im = ax.imshow(frames[0])
        
        def update(frame):
            im.set_array(frames[frame])
            return [im]
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update, frames=len(frames),
            interval=100,  # 100ms between frames (~10 fps)
            blit=True, repeat=True
        )
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    gif_path = "VR_res.gif"
    if len(sys.argv) > 1:
        gif_path = sys.argv[1]
    
    play_gif(gif_path)

