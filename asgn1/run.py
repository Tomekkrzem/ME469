import numpy as np
import pandas as pd
import os

landmark_dir = os.path.dirname(os.path.abspath(__file__))
landmark_fp = os.path.join(landmark_dir, "datasets/ds1_Landmark_Groundtruth.dat")
landmark_data = pd.read_table(landmark_fp, sep=r'\s+', skiprows=3).to_numpy()

obstacle_locations = np.column_stack((landmark_data[:,1].T,landmark_data[:,2].T))

def build_grid(x_range,y_range,res,obstacles):

    width = np.linspace(x_range[0],x_range[1], int((x_range[1] - x_range[0])/res) + 1)
    height = np.linspace(y_range[0],y_range[1], int((y_range[1] - y_range[0])/res) + 1)
    grid_vals = np.zeros((len(width),len(height)))

    for o in obstacles:
        x,y = o

        x = round(x,0)
        y = round(y,0) 

        x_indx = int(np.where(width == x)[0][0])
        y_indx = int(np.where(height == y)[0][0])

        grid_vals[x_indx][y_indx] = 1

    return grid_vals

def A_star (start,goal):
    open_set = set()
    closed_set = set()

    open_set.add(start)
    f = 0

    while len(open_set) != 0:

        None

    return open_set


grid = build_grid([-2,5],[-6,6],1,obstacle_locations)