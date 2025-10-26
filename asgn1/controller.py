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


def potential_field(self):
        
        C = 0.8

        Px, Py = np.meshgrid(self.width, self.height)

        P_goal = C * np.sqrt((Px - self.xg[0])**2 + (Py - self.xg[1])**2)

        P_obs = np.zeros((len(self.height),len(self.width)))

        for o in self.obstacles:
                    
            # Extract Obstacle Properties
            ox, oy, w, h = o

            # If Resouliton is Fine
            if self.res < 1: 

                # Round the Center of the Obstacle and Transform it to the Real Center of the Obstacle
                ox = np.ceil(ox * 10) / 10 - 0.05
                oy = np.ceil(oy * 10) / 10 - 0.05

            else: 
                ox = np.floor(ox)
                oy = np.floor(oy)

            dist_to_obs = np.sqrt((np.maximum(np.abs(Px - ox) - w/2, 0))**2 + (np.maximum(np.abs(Py - oy) - h/2, 0))**2)

            P_obs += 1 / (dist_to_obs + 0.1)

        self.potential_grid = P_goal + P_obs

        px_grad, py_grad = np.gradient(self.potential_grid)

        return self.potential_grid
    
    def potential_feild_path(self):

        pf = self.potential_field()
        Flag = 0
        t=0
        res = self.res
        neighbor_dirs = [(-res,-res),(-res,0),(-res,res),(0,res),(res,res),(res,0),(res,-res),(0,-res)]

        if res < 1: 
            self.xg = (np.floor(self.xg[0] * 10) / 10, np.floor(self.xg[1] * 10) / 10)
            self.xt = (np.floor(self.xt[0] * 10) / 10, np.floor(self.xt[1] * 10) / 10)
        # If the Resolution is Coarses Floor the Start and Goal Position to Zero Decimals
        else: 
            self.xg = (np.floor(self.xg[0]), np.floor(self.xg[1]))
            self.xt = (np.floor(self.xt[0]), np.floor(self.xt[1]))

        p_path = [(float(self.xt[0]),float(self.xt[1]))]

        print(self.xg)
        thresh = 0.2

        while np.sqrt((self.xg[0] - self.xt[0])**2 + (self.xg[1] - self.xt[1])**2) > thresh:
            if Flag == 0:
                
                map_x = int(np.where(self.width == self.xt[0])[0][0])
                map_y = int(np.where(self.height == self.xt[1])[0][0])

                min_pot = pf[map_y][map_x]


                # For each Neighbor Direction
                for n in neighbor_dirs:
                    
                    # If Resolution is Fine Floor the Neighbor X and Y Coordinate Values to One Decimal
                    if res < 1: 
                        nx = round((np.floor(self.xt[0] * 10) / 10) + n[0],1)
                        ny = round((np.floor(self.xt[1] * 10) / 10) + n[1],1)
                        
                    # If Resolution is Coarse Floor the Neighbor X and Y Coordinate Values to Zero Decimals
                    else:
                        nx = np.floor(self.xt[0] + n[0])
                        ny = np.floor(self.xt[1] + n[1])

                    if not (self.width[0] <= nx <= self.width[-1] and self.height[-1] <= ny <= self.height[0]):
                        continue
                    
                    # Extract Grid Indices of Neighbor Node
                    map_nx = int(np.where(self.width == nx)[0][0])
                    map_ny = int(np.where(self.height == ny)[0][0])

                    temp_pot = pf[map_ny][map_nx]

                    if temp_pot < min_pot:
                        self.xt = (nx,ny)
                        p_path.append((float(self.xt[0]),float(self.xt[1])))

        pcx =[]
        pcy = []

        for c in p_path:
            pcx.append(c[0])
            pcy.append(c[1])

        # Grid Width and Length for Plot Creation
        grid_width = self.grid.shape[1]
        grid_height = self.grid.shape[0]

        plt.figure(figsize=(6,10))
        # Display 2D Grid
        plt.imshow(pf, origin='upper', extent=[-2, grid_width*Res2 - 2 , -6, grid_height*Res2 - 6])
        plt.plot(pcx,pcy, linewidth=3)

        # Label Major Values on Axes (i.e. -6, -5.5, -5, etc.)
        ax = plt.gca()

        # Check that the Value to Label is a Multiple of 0.5 
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}" if abs(x*2 - round(x*2)) < 1e-6 else ""))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}" if abs(y*2 - round(y*2)) < 1e-6 else ""))

        # Create Grid Lines
        x_ticks = np.arange(-2, grid_width * Res2 - 2, Res2)
        y_ticks = np.arange(-6, grid_height * Res2 - 6, Res2)
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
        plt.grid(True, color='gray', linewidth = Res2 * 1.5)

        # Display Plot
        plt.title("Test")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.show()

        return p_path