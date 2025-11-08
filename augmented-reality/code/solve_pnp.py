from est_homography import est_homography
import numpy as np


def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Homography Approach
    # Following slides: Pose from Projective Transformation
    Pw = np.array(Pw)
    Pw_re = Pw[:, :2]
    H = est_homography(Pw_re, Pc)
    H_prime = np.linalg.inv(K) @ H
    U, S, VT = np.linalg.svd(H_prime[:, :2])
    UV = U[:, :2] @ VT
    r1, r2 = UV[:, 0], UV[:, 1]
    r3 = np.cross(r1, r2)
    R = np.column_stack((r1, r2, r3))
    lam = np.mean(S)
    t = H_prime[:, 2] / lam

    ##### STUDENT CODE END #####

    return R.T, -R.T @ t
