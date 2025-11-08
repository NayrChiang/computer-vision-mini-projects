# Computer Vision Mini Projects

Mini projects from CIS5800 Computer Vision course, demonstrating practical implementations of core computer vision techniques. Projects include homography estimation and image warping for logo projection, as well as camera pose estimation (PnP and P3P) for augmented reality applications.

## Project Overview

**Course**: CIS5800 - Computer Vision  
**Institution**: University of Pennsylvania

### Objective

Demonstrate practical understanding of projective geometry, homography estimation, and camera pose estimation through two mini projects:
1. **Logo Projection**: Using homography and inverse warping
2. **Augmented Reality**: Using PnP/P3P pose estimation

## Projects

### Project 1: Logo Projection using Homography Estimation

This project implements logo projection onto a goal post in a video sequence using homography estimation and inverse warping.

#### Technical Implementation

- **Homography Estimation**: 
  - Computed homography matrix H using SVD from 4 point correspondences between goal corners and logo corners
  - Applied Direct Linear Transform (DLT) algorithm to solve for the 3×3 homography matrix
  - Used SVD decomposition to find the null space vector, which gives the homography matrix
  - Normalized the homography matrix by dividing by H[2,2] to ensure proper scaling

- **Point Transformation**:
  - Implemented `warp_pts()` function to transform interior points using the computed homography
  - Applied inverse warping to project logo onto video frames

- **Results**:
  - Successfully implemented homography estimation using SVD-based DLT algorithm
  - Processed 129 video frames with accurate logo projection
  - Demonstrated robust implementation of core computer vision techniques

#### Key Features

- Homography estimation using SVD-based DLT algorithm
- Point correspondence matching between goal corners and logo corners
- Inverse warping for logo projection
- Frame-by-frame processing of video sequences

### Project 2: Augmented Reality with AprilTags

This project implements augmented reality by estimating camera pose and rendering 3D objects in real scenes using AprilTags.

#### Technical Implementation

- **Camera Pose Estimation**:
  - Implemented two methods:
    1. **PnP (Perspective-n-Point)**: With coplanar assumption via homography
    2. **P3P (Perspective-3-Point)**: With Procrustes problem solving
  - Used AprilTags for marker detection and pose estimation
  - Applied 6DoF pose estimation for accurate 3D object placement

- **3D Rendering**:
  - Rendered 3D objects (bottle and drill) in real scenes
  - Applied perspective transformation based on estimated camera pose
  - Implemented lighting and shading for realistic appearance

- **Results**:
  - Successfully implemented two camera pose estimation methods
  - Created augmented reality applications with accurate perspective and lighting
  - Enabled virtual object placement in real scenes

#### Key Features

- PnP problem solving with coplanar assumption
- P3P problem solving with Procrustes method
- 6DoF pose estimation
- 3D object rendering with perspective and lighting
- AprilTag marker detection

## Technologies Used

- **Programming**: Python 3.8-3.11
- **Libraries**: 
  - NumPy
  - OpenCV
  - PyRender
  - Trimesh
  - ImageIO
  - Matplotlib
- **Algorithms**:
  - Direct Linear Transform (DLT)
  - Singular Value Decomposition (SVD)
  - Perspective-n-Point (PnP)
  - Perspective-3-Point (P3P)
  - Procrustes problem solving

## Project Structure

```
computer-vision-mini-projects/
├── augmented-reality/
│   ├── code/
│   │   ├── main.py              # Main AR application
│   │   ├── est_homography.py     # Homography estimation
│   │   ├── est_pose.py          # Pose estimation (PnP/P3P)
│   │   └── [other implementation files]
│   ├── data/
│   │   ├── corners.npy          # Corner coordinates
│   │   ├── frames/              # Video frames
│   │   └── models/              # 3D models (.obj, .mtl)
│   ├── requirements.txt
│   └── README.md
├── logo-projection/
│   ├── data/
│   │   └── barcelona/           # Video data
│   ├── Part_1/
│   │   ├── [implementation files]
│   │   └── [output videos and images]
│   └── README.md
└── README.md
```

## Installation & Setup

### Prerequisites

- Python 3.8-3.11
- pip or conda

### Setup

We recommend using [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) or other virtual environment managers (venv, virtualenv, etc.).

#### Using Conda

```bash
conda create -n cis580 python=3.10
conda activate cis580
```

#### Using venv

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

Navigate to the project directory and install required packages:

```bash
cd augmented-reality
pip install -r requirements.txt
```

Required packages:
- `gradescope-utils==0.2.7`
- `subprocess32`
- `numpy`
- `tqdm`
- `imageio`
- `pyrender`
- `matplotlib`
- `trimesh`

## Usage

### Logo Projection

1. Navigate to the logo-projection directory
2. Run the main script:
```bash
python main.py
```

### Augmented Reality

1. Navigate to the augmented-reality code directory:
```bash
cd augmented-reality/code
python main.py
```

#### Command Line Options

- `--debug`: Run in debug mode (faster rendering for testing)
- `--solver PnP`: Use PnP algorithm for pose estimation
- `--solver P3P`: Use P3P algorithm for pose estimation
- `--click_point`: Render objects at different locations

#### Example Usage

```bash
# Run with PnP solver
python main.py --solver PnP

# Run with P3P solver
python main.py --solver P3P

# Run in debug mode
python main.py --debug

# Customize object placement
python main.py --click_point
```

## Key Achievements

- Successfully implemented homography estimation using SVD-based DLT algorithm for logo projection across 129 video frames
- Implemented two camera pose estimation methods: PnP with coplanar assumption and P3P with Procrustes problem solving
- Created augmented reality applications with accurate perspective and lighting, enabling virtual object placement in real scenes
- Demonstrated robust implementation of core computer vision techniques with visually accurate results

## Skills Demonstrated

- **Projective Geometry**: Understanding of homogeneous coordinates and projective transformations
- **Homography Estimation**: SVD-based DLT algorithm implementation
- **PnP Problem**: Perspective-n-Point pose estimation
- **P3P Problem**: Perspective-3-Point pose estimation with Procrustes
- **Computer Vision Algorithms**: Implementation of fundamental CV techniques
- **3D Rendering**: Integration of 3D objects in real scenes
- **Augmented Reality**: Real-time pose estimation and object rendering

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CIS5800 course instructors and TAs
- University of Pennsylvania for providing resources and facilities
