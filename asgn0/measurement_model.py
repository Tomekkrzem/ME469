import numpy as np
from numpy import sin, cos

# def z_t(X_t, X_i):

#     """
#     :param X_t: Current robot state
#     :param X_i: Current landmark position
    
#     """

#     x_t = X_t[0]
#     y_t = X_t[1]
#     theta_t = X_t[2]

#     x_i = X_i[0]
#     y_i = X_i[1]

#     z_t = np.array([np.sqrt((x_t - x_i)**2 + (y_t - y_i)**2), np.atan2((y_i - y_t),(x_i - x_t)) - theta_t])

#     z_t_pos = np.array([z_t[0] * cos(z_t[1]) + x_t, z_t[0] * sin(z_t[1]) + y_t])

#     return z_t_pos

# X = np.array([[0,0,0],[1,1,1],[2,2,2],[2,2,2]])

# print(X.shape)

# theta = X[:,2]
# print(theta)



def z_t(X_t, lm_x, lm_y, lm_nums = []):

    """
    :param X_t: Current robot state
    :param X_i: Current landmark position
    
    """

    print(X_t)

    x_t = X_t[:,0]
    y_t = X_t[:,1]
    theta_t = X_t[:,2]
    zt_pos_out = []

    if lm_nums == []:

        # Compute the distance to the landmark using Euclidean distance, and the bearing taking arctan of the coordinates minus the heading
        zt = [np.sqrt((x_t - lm_x)**2 + (y_t - lm_y)**2), 
                        np.atan2((lm_y - y_t),(lm_x - x_t)) - theta_t]

        print(zt)

        zt[1] = (zt[1] + np.pi) % (2 * np.pi) - np.pi

        range_out = zt[0]
        bearing_out = zt[1]

    return range_out, bearing_out, zt_pos_out


PARTICLE_SIZE = 10

x_start = 1.29812900 
y_start = 1.88315210
theta_start = 2.82870000 

temp = np.array([x_start, y_start, theta_start])
test_vec = np.full((PARTICLE_SIZE,3), temp)

v_arr = 5
w_arr = 1e-12

u_t = np.array([v_arr,w_arr])

dt = 0.01


print(z_t(test_vec, 0.48704624, -4.95127346)[0])