import numpy as np
from numpy import cos,sin
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import os


def bezier_curve(c_points, num_points):

    n = len(c_points) - 1

    t = np.linspace(0, 1, num_points)

    output_curve = np.zeros((num_points, 2))

    for i in range(num_points):
        for j in range(n + 1):
            
            # Bezier Curve Equation Referenecd From "Path Planning based on Bezier Curve for Autonomous Ground Vehicles"
            output_curve[i] += c_points[j] * math.comb(n, j) * (t[i] ** j) * ((1 - t[i]) ** (n - j)) 
    
    return output_curve


def feed_forward_control(start, goal, u, T, dt):

    # Necessary Inputs:
    #   Starting Robot State (x,y,theta)
    #   Starting Robot Controls (v = 0, w = 0)
    #   Desired Trajectory
    #   Sampling Trajectory dt = 0.1

    # Preprocess Trajectory Using Bezier Curve
    smooth_T = bezier_curve(T, int(len(T)/dt))
    
    # Compose Feedforward States for Robot
    ffwd_points = np.zeros((len(smooth_T), 3))
    ffwd_points[0] = (start[0], start[1], start[2])

    for i in range(len(smooth_T) - 1):
        
        x,y = smooth_T[i+1]
        x_p, y_p = smooth_T[i]
        theta = np.atan2((y - y_p),(x - x_p))
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        ffwd_points[i+1] = (x,y, theta)

    thresh = 1e-4

    # Kp_v = 3
    # Kp_w = 5
    # Kd_v = 9
    # Kd_w = 4

    Kp_v = 0.08
    Kp_w = 10
    Kd_v = 1
    Kd_w = 1

    # Current State
    xt = start
    # Initilialize Errors
    e_v = None
    e_w = None

    robot_path = [[xt[0],xt[1]]]

    # Loop Until Robot Reaches Completes Trajectory
    for p in ffwd_points:

        # Check if the Current Position is the Goal Position
        if np.sqrt((goal[0]-xt[0])**2 + (goal[1]-xt[1])**2) < thresh:
            return robot_path, ffwd_points

        # Compute Derivative (i.e. Change between Current and Previous State)
        if e_v is not None and e_w is not None:
            d_v = e_v - prev_e_v
            d_w = e_w - prev_e_w
        else:
            d_v = 0
            d_w = 0

        # Error Computation Referenced From "Nonlinear Model Predictive Control for Mobile Robot Using Varying-Parameter Convergent Differential Neural Network"
        
        R = np.array([[cos(theta), -sin(theta), 0],
                      [sin(theta), cos(theta), 0],
                      [0 , 0, 1]])
        # Compute Error Between Desired State and Actual State
        diff_x_xT = np.array([p[0] - xt[0],
                              p[1] - xt[1],
                              p[2] - xt[2]])

        Xe = np.matmul(R, diff_x_xT)

        e_v = np.sqrt(Xe[0]**2 + Xe[1]**2)
        e_w = Xe[2]

        # Compute Velocity
        v = Kp_v * e_v + Kd_v * d_v
        w = Kp_w * e_w + Kd_w * d_w

        # Update Previous Errors
        prev_e_v = e_v
        prev_e_w = e_w

        # Compute Next State Position
        xt = x_t(xt,[v,w],dt)

        robot_path.append([xt[0],xt[1]])

    return robot_path, ffwd_points


# Sampling Function Referenced from Probabilistic Robotics Table 5.4 (sample_normal_distribution)
def sample(b):

    return b/6 * np.random.uniform(-1,1,12).sum()

def x_t(X_t_p, u_t, dt):

    """
    :param x: Starting x-coordinate
    :param y: Starting y-coordinate
    :param theta: Starting heading position of robot
    :param v: Translational Speed (Constant)
    :param w: Rotational Speed (Constant)
    :param t: Duration time of the commands (Constant)
    
    """

    v_hat = u_t[0] + sample(a1 * abs(u_t[0]) + a2 * abs(u_t[1]))
    w_hat = u_t[1] + sample(a3 * abs(u_t[0]) + a4 * abs(u_t[1]))
    gamma = sample(a5 * abs(u_t[0]) + a6 * abs(u_t[1]))

    if w_hat == 0:
        x_dt = np.array([v_hat * cos(X_t_p[2]) * dt, v_hat * sin(X_t_p[2]) * dt, 0])

    else:
        # Instantaneous Center of Rotation (ICR)
        xc = X_t_p[0] - (v_hat/w_hat)*sin(X_t_p[2])
        yc = X_t_p[1] + (v_hat/w_hat)*cos(X_t_p[2])

        R = np.array([[cos(w_hat*dt), -sin(w_hat*dt), 0],
                      [sin(w_hat*dt),  cos(w_hat*dt), 0],
                      [0        ,          0, 1]])

        x_dt = np.matmul(R,np.array([X_t_p[0] - xc, 
                                     X_t_p[1] - yc,
                                     w_hat * dt + gamma * dt]))

        X_t_p = np.array([xc,yc,X_t_p[2]])


    x_t = X_t_p + x_dt
    
    x_t[2] = (x_t[2] + np.pi) % (2 * np.pi) - np.pi     

    return x_t

a1 = 0.1
a2 = 0.1
a3 = 0.1
a4 = 0.1
a5 = 0.1
a6 = 0.1

control_path, ffwd = feed_forward_control((4.9,-0.1,-np.pi/2),(2.4,0.2,1),(0,0),test_traj,0.1)

cx =[]
cy = []

for c in control_path:
    cx.append(c[0])
    cy.append(c[1])

plt.plot(ffwd[:, 0], ffwd[:, 1])
plt.plot(cx, cy)
plt.show()