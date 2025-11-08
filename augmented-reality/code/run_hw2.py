"""
CIS 5800 Homework 2 - Augmented Reality with AprilTags
Converted from Jupyter notebook to Python script for local execution
"""

import numpy as np
import matplotlib.pyplot as plt
from render import Renderer
import imageio
import trimesh
import glob
from pyrender import Mesh
from solve_pnp import PnP
from solve_p3p import P3P
from est_Pw import est_Pw
from tqdm import tqdm
from est_pixel_world import est_pixel_world
from image_click import ImageClicker
import copy
import argparse
import sys


def test_pnp():
    """Test PnP function with sample data"""
    print("=" * 60)
    print("Testing PnP Function")
    print("=" * 60)
    
    Pc = np.array([[304.28405762, 346.36758423],
                   [449.04196167, 308.92901611],
                   [363.24179077, 240.77729797],
                   [232.29425049, 266.60055542]])
    Pw = np.array([[-0.07,  0.07,  0.  ],
                   [ 0.07,  0.07,  0.  ],
                   [ 0.07, -0.07,  0.  ],
                   [-0.07, -0.07,  0.  ]])
    
    K = np.eye(3)
    R, t = PnP(Pc, Pw, K)
    
    print(f"\nRotation matrix R:\n{np.round(R, 3)}")
    print(f"\nTranslation vector t:\n{np.round(t, 3)}")
    print(f"\nDeterminant of R: {round(np.linalg.det(R), 3)}")
    print(f"R is orthogonal: {np.allclose(R @ R.T, np.eye(3))}")
    print()


def test_p3p():
    """Test P3P function with sample data"""
    print("=" * 60)
    print("Testing P3P Function")
    print("=" * 60)
    
    Pc = np.array([[304.28405762, 346.36758423],
                   [449.04196167, 308.92901611],
                   [363.24179077, 240.77729797],
                   [232.29425049, 266.60055542]])
    Pw = np.array([[-0.07,  0.07,  0.  ],
                   [ 0.07,  0.07,  0.  ],
                   [ 0.07, -0.07,  0.  ],
                   [-0.07, -0.07,  0.  ]])
    
    K = np.eye(3)
    R, t = P3P(Pc, Pw, K)
    
    print(f"\nRotation matrix R:\n{np.round(R, 3)}")
    print(f"\nTranslation vector t:\n{np.round(t, 3)}")
    print(f"\nDeterminant of R: {round(np.linalg.det(R), 3)}")
    print(f"R is orthogonal: {np.allclose(R @ R.T, np.eye(3))}")
    print()


def test_pixel_world():
    """Test pixel to world coordinate conversion"""
    print("=" * 60)
    print("Testing Pixel to World Coordinate Conversion")
    print("=" * 60)
    
    pixels = np.array([[220, 330],
                       [550, 260]])
    R_wc = np.array([[ 0.876,  0.478, -0.064],
                     [-0.156,  0.407,  0.9  ],
                     [ 0.456, -0.779,  0.431]])
    t_wc = np.array([0.028, 0.044, 0.717])
    K = np.array([[823.8,   0.,  304.8],
                  [  0.,  822.8, 236.3],
                  [  0.,    0.,    1.  ]])
    
    Pw = est_pixel_world(pixels, R_wc, t_wc, K)
    print(f"\nWorld coordinates:\n{np.round(Pw, 3)}")
    print()


def run_reproject(pixels, R_wc, t_wc, K, image, obj_meshes=None, shift=None, renderer=None):
    """Project virtual objects onto the image"""
    assert(len(pixels) == len(obj_meshes))

    # estimate object center point
    if shift is None:
        shift = est_pixel_world(pixels, R_wc, t_wc, K)

    # modify mesh
    all_meshes = []
    for i in range(len(pixels)):
        obj_meshes[i].vertices = obj_meshes[i].vertices + shift[i:i+1, :]
        pyrender_mesh = Mesh.from_trimesh(obj_meshes[i], smooth=False)
        all_meshes.append(pyrender_mesh)

    # Render
    result_image, depth_map = renderer.render(all_meshes, R_wc, t_wc, image)
    return result_image, shift


def generate_ar_video(solver='PnP', click_points=None, debug=False):
    """Generate augmented reality video"""
    print("=" * 60)
    print(f"Generating AR Video using {solver} solver")
    print("=" * 60)
    
    # Process each frame
    print('Processing each frame ...')
    frames = glob.glob('../data/frames/*.jpg')
    frames.sort()
    images = [np.array(imageio.imread(f)) for f in frames]

    # extrinsic
    corners = np.load('../data/corners.npy')

    # intrinsics
    K = np.array([[823.8, 0.0, 304.8],
                  [0.0, 822.8, 236.3],
                  [0.0, 0.0, 1.0]])

    solvers_dict = {'PnP': PnP, 'P3P': P3P}

    # construct renderer
    renderer = Renderer(intrinsics=K, img_w=640, img_h=480)

    # load meshes
    axis_mesh = trimesh.creation.axis(origin_size=0.02)
    drill_mesh = trimesh.load("../data/models/drill.obj")
    fuze_mesh = trimesh.load("../data/models/fuze.obj")
    vr_meshes = [drill_mesh, fuze_mesh]

    tag_size = 0.14
    Pw = est_Pw(tag_size)

    # Use provided click points or default
    if click_points is None:
        click_points = np.array([[220, 330], [550, 260]])  # pre-defined locations

    final = np.zeros([len(images), 480, 640, 3], dtype=np.uint8)
    shift = None  # translations of the objects that will be rendered on the table
    
    for i, f in enumerate(tqdm(images)):
        # get pose with pnp
        Pc = corners[i]
        R_wc, t_wc = solvers_dict[solver](Pc, Pw, K)

        # render based on click points
        # Note that all pixels are in (x, y) format
        meshes = [copy.deepcopy(vr_meshes[m % len(vr_meshes)]) for m in range(click_points.shape[0])]
        if i == 0:
            reproject_image, shift = run_reproject(click_points, R_wc, t_wc, K, images[i], 
                                                  obj_meshes=meshes, renderer=renderer)
        else:
            reproject_image, shift = run_reproject(click_points, R_wc, t_wc, K, images[i],
                                                  obj_meshes=meshes, shift=shift, renderer=renderer)
        final[i] = reproject_image

        if debug and i == 0:
            pyrender_mesh = Mesh.from_trimesh(axis_mesh, smooth=False)
            axis_image, depth_map = renderer.render([pyrender_mesh], R_wc, t_wc, images[i])
            plt.subplot(1, 2, 1)
            plt.imshow(axis_image)
            plt.title('Axis Visualization')
            plt.subplot(1, 2, 2)
            plt.imshow(reproject_image)
            plt.title('AR Result')
            plt.tight_layout()
            plt.savefig('debug_frame0.png', dpi=150)
            print("Debug frame saved as debug_frame0.png")
            if not debug:  # Only show if not in batch mode
                plt.show()
        elif i == 0:
            # Save the first frame that will be included in your report
            plt.imsave('vis.png', reproject_image.astype(np.uint8))
            print(f"First frame saved as vis.png")

    print('Frame processing complete.')
    print('Saving GIF ...')
    with imageio.get_writer('VR_res.gif', mode='I') as writer:
        for i in range(len(final)):
            img = final[i]
            writer.append_data(img)

    print('GIF saved as VR_res.gif')
    print('Complete!')


def main():
    parser = argparse.ArgumentParser(description='CIS 5800 Homework 2 - AR with AprilTags')
    parser.add_argument('--solver', type=str, default='PnP', 
                       help="Algorithm to use, PnP or P3P", choices=["PnP", "P3P"])
    parser.add_argument('--click_points', action='store_true',
                       help="Whether to click points for placing the objects")
    parser.add_argument('--debug', action='store_true', 
                       help="Helper flag for debugging (shows first frame)")
    parser.add_argument('--test', action='store_true',
                       help="Run test functions instead of generating video")
    args = parser.parse_args()

    if args.test:
        # Run test functions
        test_pnp()
        test_p3p()
        test_pixel_world()
    else:
        # Generate AR video
        click_points = None
        if args.click_points:
            fig, ax = plt.subplots()
            ax.set_title('Click two points to place objects')
            frame = imageio.imread("../data/frames/frame0000.jpg")
            image = ax.imshow(frame)
            pixel_selector = ImageClicker(image)
            plt.show()
            click_points = np.stack(pixel_selector.points).astype(int)
        
        generate_ar_video(solver=args.solver, click_points=click_points, debug=args.debug)


if __name__ == "__main__":
    main()

