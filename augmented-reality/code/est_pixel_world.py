import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    ##### STUDENT CODE START #####
    pixels = np.array(pixels)
    R_wc = np.array(R_wc)
    t_wc = np.array(t_wc)
    K = np.array(K)

    n = pixels.shape[0]
    Pw = np.zeros((n, 3))
    k1, k2, k3 = K[0], K[1], K[2]
    a3 = R_wc[2]
    b = np.array([0, 0, -t_wc[2]])
    for i in range(n):
        u = pixels[i, 0]
        v = pixels[i, 1]
        a1 = u * k3 - k1
        a2 = v * k3 - k2
        A = np.vstack((a1, a2, a3))
        P_c = np.linalg.solve(A, b)
        Pw[i, :] = R_wc @ P_c + t_wc
    ##### STUDENT CODE END #####

    return Pw
