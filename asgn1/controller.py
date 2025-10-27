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


def potential_field(Grid, res, goal, sqr_size):
        
        C = 0.8

        width, height, potential_grid = Grid([-2.0,5.0], [-6.0,6.0], res, obstacle_locations)

        obstacles = [(o[0], o[1], sqr_size, sqr_size) for o in obstacle_locations]

        Px, Py = np.meshgrid(width, height)

        P_goal = C * np.sqrt((Px - goal[0])**2 + (Py - goal[1])**2)

        P_obs = np.zeros((len(height),len(width)))

        for o in obstacles:
                    
            # Extract Obstacle Properties
            ox, oy, w, h = o

            # If Resouliton is Fine
            if res < 1: 

                # Round the Center of the Obstacle and Transform it to the Real Center of the Obstacle
                ox = np.ceil(ox * 10) / 10 - 0.05
                oy = np.ceil(oy * 10) / 10 - 0.05

            else: 
                ox = np.floor(ox)
                oy = np.floor(oy)

            dist_to_obs = np.sqrt((np.maximum(np.abs(Px - ox) - w/2, 0))**2 + (np.maximum(np.abs(Py - oy) - h/2, 0))**2)

            P_obs += 1 / (dist_to_obs + 0.1)

        potential_grid = P_goal + P_obs

        return width, height, potential_grid
    
def potential_feild_path(PF, res, start, goal):

    width, height, pf = PF

    neighbor_dirs = [(-res,-res),(-res,0),(-res,res),(0,res),(res,res),(res,0),(res,-res),(0,-res)]

    if res < 1: 
        xg = (np.floor(goal[0] * 10) / 10, np.floor(goal[1] * 10) / 10)
        xt = (np.floor(start[0] * 10) / 10, np.floor(start[1] * 10) / 10)
    # If the Resolution is Coarses Floor the Start and Goal Position to Zero Decimals
    else: 
        xg = (np.floor(goal[0]), np.floor(goal[1]))
        xt = (np.floor(start[0]), np.floor(start[1]))

    p_path = [(float(xt[0]),float(xt[1]))]

    thresh = 0.2

    while np.sqrt((xg[0] - xt[0])**2 + (xg[1] - xt[1])**2) > thresh:

        map_x = int(np.where(width == xt[0])[0][0])
        map_y = int(np.where(height == xt[1])[0][0])

        min_pot = pf[map_y][map_x]


        # For each Neighbor Direction
        for n in neighbor_dirs:
            
            # If Resolution is Fine Floor the Neighbor X and Y Coordinate Values to One Decimal
            if res < 1: 
                nx = round((np.floor(xt[0] * 10) / 10) + n[0],1)
                ny = round((np.floor(xt[1] * 10) / 10) + n[1],1)
                
            # If Resolution is Coarse Floor the Neighbor X and Y Coordinate Values to Zero Decimals
            else:
                nx = np.floor(xt[0] + n[0])
                ny = np.floor(xt[1] + n[1])

            if not (width[0] <= nx <= width[-1] and height[-1] <= ny <= height[0]):
                continue
            
            # Extract Grid Indices of Neighbor Node
            map_nx = int(np.where(width == nx)[0][0])
            map_ny = int(np.where(height == ny)[0][0])

            temp_pot = pf[map_ny][map_nx]

            if temp_pot < min_pot:
                xt = (nx,ny)
                p_path.append((float(xt[0]),float(xt[1])))

    pcx =[]
    pcy = []

    for c in p_path:
        pcx.append(c[0])
        pcy.append(c[1])

    # Grid Width and Length for Plot Creation
    grid_width = pf.shape[1]
    grid_height = pf.shape[0]

    plt.figure(figsize=(6,10))
    # Display 2D Grid
    plt.imshow(pf, origin='upper', extent=[-2, grid_width * res - 2 , -6, grid_height * res - 6])
    plt.plot(pcx,pcy, linewidth=3)

    # Label Major Values on Axes (i.e. -6, -5.5, -5, etc.)
    ax = plt.gca()

    # Check that the Value to Label is a Multiple of 0.5 
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}" if abs(x*2 - round(x*2)) < 1e-6 else ""))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}" if abs(y*2 - round(y*2)) < 1e-6 else ""))

    # Create Grid Lines
    x_ticks = np.arange(-2, grid_width * res - 2, res)
    y_ticks = np.arange(-6, grid_height * res - 6, res)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.grid(True, color='gray', linewidth = res * 1.5)

    # Display Plot
    plt.title("Test")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.show()

    return p_path