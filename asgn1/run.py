import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import os

landmark_dir = os.path.dirname(os.path.abspath(__file__))
landmark_fp = os.path.join(landmark_dir, "datasets/ds1_Landmark_Groundtruth.dat")
landmark_data = pd.read_table(landmark_fp, sep=r'\s+', skiprows=3).to_numpy()

obstacle_locations = np.column_stack((landmark_data[:,1].T,landmark_data[:,2].T))

# Builds a Graph Populated with Obstacles
def build_grid(x_range, y_range, res, obstacles = []):

    # If Resolution is Smaller than 1 then Round to 1 Decimals
    if res < 1:
        rnd = 1
    # Else Round to 0 Decimals
    else: 
        rnd = 0

    # X-Value Linespace
    width = np.linspace(x_range[0],x_range[1], int((x_range[1] - x_range[0])/res) + 1)
    # Y-Value Linespace
    height = np.flip(np.linspace(y_range[0],y_range[1], int((y_range[1] - y_range[0])/res) + 1))

    # Truncate the Values in the Linespaces
    width = np.round(width,decimals=rnd)
    height = np.round(height,decimals=rnd)
    
    # Initializing 2D Grid of Empty Positions
    grid_vals = np.zeros((len(height),len(width)))

    if rnd == 1:

        obstacle_arr = np.ones((7,7))

    # Add Obstacles to Grid
    for o in obstacles:

        # Retrieve Obstacle Position
        x,y = o

        # Truncate the Position Coordiantes
        x = round(x,rnd)
        y = round(y,rnd) 

        if rnd == 1:

            offset = 0.3

            # Compute Corresponding Map Position of Obstacle
            x_indx = int(np.where(width == round(x - offset,rnd))[0][0])
            y_indx = int(np.where(height == round(y - offset,rnd))[0][0])
            
            grid_vals[y_indx - 7 : y_indx,
                      x_indx : x_indx + 7] = obstacle_arr
            
            x_indx_c = int(np.where(width == round(x,rnd))[0][0])
            y_indx_c = int(np.where(height == round(y,rnd))[0][0])

            grid_vals[y_indx_c][x_indx_c] = 1

        else: 
            
            # Compute Corresponding Map Position of Obstacle
            x_indx = int(np.where(width == x)[0][0])
            y_indx = int(np.where(height == y)[0][0])

            # Add Obstacles to Map by Setting Value at Associated Position to 1
            grid_vals[y_indx][x_indx] = 1

    return width, height, grid_vals

def H(node_s, node_g):
    return np.sqrt((node_g[0] - node_s[0])**2 + (node_g[1] - node_s[1])**2)

    # return max((abs(node_g[0]-node_s[0]),abs(node_g[1]-node_s[1])))


def A_star (map_vals, res, start, goal):

    if res < 1:
        rnd = 1
    else: 
        rnd = 0
    

    goal = (round(goal[0], rnd), round(goal[1], rnd))
    start = (round(start[0], rnd), round(start[1], rnd))


    width, height, Map = map_vals

    # Initialize Open Set
    open_set = [(0 + H(start,goal), 0, start)]

    # Initialize Closed Set
    closed_set = []

    # Initialize Parent-Child Dictionary for Path Creation
    parent = {start: None}

    # Initialize Timepath Cost Dictionary
    g_dict = {start:0}

    # 8 Neighbor Directions to Check
    nieghbor_dirs = [(-res,-res),(-res,0),(-res,res),(0,res),(res,res),(res,0),(res,-res),(0,-res)]

    while len(open_set) != 0:
        
        open_set = sorted(open_set)

        h, g, curr_node = open_set.pop(0)

        if curr_node == goal:

            path = [goal]

            child = goal

            while parent[child]:

                path.append(parent[child])
                child = parent[child]

            path.reverse()

            closed_set.append(curr_node)

            return closed_set, path

        if curr_node in closed_set:
            continue

        closed_set.append(curr_node)

        for n in nieghbor_dirs:

            nx = round(curr_node[0] + n[0],rnd)
            ny = round(curr_node[1] + n[1],rnd)

            neighbor_node = (nx,ny)
            
            if neighbor_node in closed_set:
                continue
            
            # Bounds the Search
            if not (-2 <= nx < round(len(width)*Res -2,rnd) and -6 <= ny < round(len(height)*Res -6,rnd)):
                continue

            # Check if Neighbor is an Obstalce
            map_x = int(np.where(width == nx)[0][0])
            map_y = int(np.where(height == ny)[0][0])

            if Map[map_y][map_x] == 1:
                g_temp = g + 1000
            else: 
                g_temp = g + 1

            if neighbor_node not in g_dict or g_temp < g_dict[neighbor_node]:

                g_dict.update({neighbor_node:g_temp})

                h = H(neighbor_node,goal)

                f = g_temp + h

                open_set.append((f,g_temp,neighbor_node))
                parent[neighbor_node] = curr_node

    return closed_set, []

Res = 1

w,h,grid = build_grid([-2.0,5.0], [-6.0,6.0], Res, obstacle_locations)

c_set, path = A_star([w,h,grid], Res, (0.5,-1.5),(0.5,1.5))
# c_set, path = A_star([w,h,grid], Res, (4.5,3.5),(4.5,-1.5))
# c_set, path = A_star([w,h,grid], Res, (-0.5,-5.5),(1.5,-3.5))

# c_set, path = A_star([w,h,grid], Res, (2.45,-3.55),(0.95,-1.55))
# c_set, path = A_star([w,h,grid], Res, (4.95,-0.05),(2.45,0.25))
# c_set, path = A_star([w,h,grid], Res, (-0.55,-1.45),(1.95,3.95))
print(path)
print(grid)

for p in path:
    x_indx = int(np.where(w == p[0])[0][0])
    y_indx = int(np.where(h == p[1])[0][0])
    grid[y_indx][x_indx] = 2


cmap = colors.ListedColormap(['white', 'black', 'lime'])

plt.figure(figsize=(8,12))
plt.imshow(grid, cmap=cmap, origin='upper', extent=[-2, len(w)*Res - 2 , -6, len(h)*Res - 6])

x_ticks = np.arange(-2, len(w)*Res - 2, Res)
y_ticks = np.arange(-6, len(h)*Res - 6, Res)
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.grid(True, color='gray', linewidth = Res*1.5)

plt.title("World")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.show()