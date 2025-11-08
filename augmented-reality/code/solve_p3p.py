import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####
    Pc = np.array(Pc)
    Pw = np.array(Pw)
    K = np.array(K)
    # Three Point Perspective Eq 1-9
    Pc_homog = np.hstack((Pc, np.ones((4, 1))))
    Pc_homog = np.linalg.inv(K) @ Pc_homog[0:3, :].T

    q1, q2, q3 = Pc_homog[:, 0], Pc_homog[:, 1], Pc_homog[:, 2]

    j1 = q1 / np.linalg.norm(q1)
    j2 = q2 / np.linalg.norm(q2)
    j3 = q3 / np.linalg.norm(q3)

    cos_alpha = np.dot(q2, q3) / (np.linalg.norm(q2) * np.linalg.norm(q3))
    cos_beta = np.dot(q1, q3) / (np.linalg.norm(q1) * np.linalg.norm(q3))
    cos_gamma = np.dot(q1, q2) / (np.linalg.norm(q1) * np.linalg.norm(q2))

    # Find abc distance
    a = np.linalg.norm(Pw[1, :] - Pw[2, :])
    b = np.linalg.norm(Pw[0, :] - Pw[2, :])
    c = np.linalg.norm(Pw[0, :] - Pw[1, :])
    a2, b2, c2 = a ** 2, b ** 2, c ** 2

    # Solve for roots
    A4 = ((a2 - c2) / b2 - 1) ** 2 - (4 * c2 / b2 * cos_alpha ** 2)
    A3 = 4 * ((((a2 - c2) / b2) * (1 - (a2 - c2) / b2) * cos_beta)
              - ((1 - (a2 + c2) / b2) * cos_alpha * cos_gamma)
              + (2 * c2 / b2 * cos_alpha ** 2 * cos_beta))
    A2 = 2 * (((a2 - c2) / b2) ** 2 - 1
              + 2 * ((a2 - c2) / b2) ** 2 * cos_beta ** 2
              + 2 * ((b2 - c2) / b2) * cos_alpha ** 2
              - 4 * ((a2 + c2) / b2) * cos_alpha * cos_beta * cos_gamma
              + 2 * ((b2 - a2) / b2) * cos_gamma ** 2)
    A1 = 4 * (-1 * ((a2 - c2) / b2) * (1 + (a2 - c2) / b2) * cos_beta
              + 2 * a2 / b2 * cos_gamma ** 2 * cos_beta
              - (1 - (a2 + c2) / b2) * cos_alpha * cos_gamma)
    A0 = (1 + (a2 - c2) / b2) ** 2 - 4 * a2 / b2 * cos_gamma ** 2

    # Solve for v
    coefficients = [A4, A3, A2, A1, A0]
    roots = np.roots(coefficients)
    v = np.real(roots[np.abs(np.imag(roots)) < 1e-4])
    u = ((-1 + (a2 - c2) / b2) * v ** 2
         - 2 * ((a2 - c2) / b2) * cos_beta * v
         + 1 + ((a2 - c2) / b2)) / (2 * (cos_gamma - v * cos_alpha))
    s1 = np.sqrt(c2 / (1 + u ** 2 - 2 * u * cos_gamma))
    s2 = u * s1
    s3 = v * s1

    p1, p2, p3 = s1[0] * j1, s2[0] * j2, s3[0] * j3
    Pc_3d = np.vstack((p1, p2, p3))
    R, t = Procrustes(Pc_3d, Pw[:3])
    # error_th = 1000000
    # for i in range(v.shape[0]):
    #     p1, p2, p3 = s1[i] * j1, s2[i] * j2, s3[i] * j3
    #     Pc_3d = np.vstack((p1, p2, p3))
    #     R_temp, t_temp = Procrustes(Pc_3d, Pw[0:3])
    #     Pc_error = K @ (R_temp.T @ Pw[3] - t_temp)
    #     Pc_error = Pc_error / Pc_error[2]
    #     error = np.linalg.norm(Pc_error[0:2] - Pc[3])
    #     if error < error_th:
    #         error_th = error
    #         R, t = R_temp, t_temp
        # Pw_error = R_temp.T @ ( np.linalg.inv(K) @ np.hstack((Pc[3], 1)) + t_temp)
        # Pw_error = Pw_error / Pw_error[2]
        # error = np.linalg.norm(Pw_error - Pw[3])
        # if error < error_th:
        #     error_th = error
        #     R, t = R_temp, t_temp

    ##### STUDENT CODE END #####

    return R, t
def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    ##### STUDENT CODE START #####
    # PnP Procrustes Slide p.27
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    H = X_centered.T @ Y_centered

    U, _, Vt = np.linalg.svd(H)
    duv = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, duv]]) @ U.T

    t = Y_mean - R @ X_mean
    ##### STUDENT CODE END #####

    return R, t


