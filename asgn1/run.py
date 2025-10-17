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

    # If Resolution is Finer
    if res < 1:
        # X-Value Linespace
        width = np.linspace(x_range[0],x_range[1], int((x_range[1] - x_range[0])/res) + 1)
        # Y-Value Linespace
        height = np.flip(np.linspace(y_range[0],y_range[1], int((y_range[1] - y_range[0])/res) + 1))

        # Truncate the Values in the Linespaces
        width = np.floor(np.round(width,decimals = 1) * 10) / 10
        height = np.floor(np.round(height,decimals = 1) * 10) / 10

        # Initializing 2D Grid of Empty Positions
        grid_vals = np.zeros((len(height),len(width)))

        # Create Inflation Array for Obstacles
        obstacle_arr = np.ones((7,7))

    # Else If Resolution is More Coarse
    else: 
        # X-Value Linespace
        width = np.linspace(x_range[0],x_range[1]-1, int((x_range[1] - x_range[0])/res))
        # Y-Value Linespace
        height = np.flip(np.linspace(y_range[0],y_range[1]-1, int((y_range[1] - y_range[0])/res)))

        # Truncate the Values in the Linespaces
        width = np.floor(width)
        height = np.floor(height)

        # Initializing 2D Grid of Empty Positions
        grid_vals = np.zeros((len(height),len(width)))

    # Add Obstacles to Grid
    for o in obstacles:

        # Retrieve Obstacle Position
        x,y = o

        if res < 1:

            # Truncate the Position Coordiantes
            x = np.floor(x * 10) / 10
            y = np.floor(y * 10) / 10

            # Obstacle Inflation Offset
            offset = 0.3

            # Determine Left X-Coordinate Index of Inflation Obstacle
            x_indx = int(np.where(width == round(x - offset, 1))[0][0])

            # Determine Top Y-Coordinate Index of Inflated Obstacle
            y_indx = int(np.where(height == round(y - offset, 1))[0][0])

            # Add Inflated Obstacle by Replacing Values in Grid with Corresponding X and Y Coordinates
            grid_vals[y_indx - 6 : y_indx + 1,
                      x_indx : x_indx + 7] = obstacle_arr

        else: 
            
            # Truncate the Position Coordiantes
            x = np.floor(x)
            y = np.floor(y)

            # Compute Corresponding Map Position of Obstacle
            x_indx = int(np.where(width == x)[0][0])
            y_indx = int(np.where(height == y)[0][0])

            # Add Obstacles to Map by Setting Value at Associated Position to 1
            grid_vals[y_indx][x_indx] = 1

    return width, height, grid_vals

# Heuristic Function for A*
def Heuristic(node_s, node_g):
    
    # Euclidean Distance
    return np.sqrt((node_g[0] - node_s[0])**2 + (node_g[1] - node_s[1])**2)

    # # Chebyshev Distance
    # return np.max((node_g[0] - node_s[0], node_g[1] - node_s[1]))



def A_star (map_vals, res, start, goal):

    if res < 1: 
        goal = (np.floor(goal[0] * 10) / 10, np.floor(goal[1] * 10) / 10)
        start = (np.floor(start[0] * 10) / 10, np.floor(start[1] * 10) / 10)
    else: 
        goal = (np.floor(goal[0]), np.floor(goal[1]))
        start = (np.floor(start[0]), np.floor(start[1]))

    width, height, Map = map_vals

    # Initialize Open Set
    open_set = [(0 + Heuristic(start,goal), 0, start)]

    # Initialize Closed Set
    closed_set = []

    # Initialize Parent-Child Dictionary for Path Creation
    parent = {start: None}

    # Initialize Timepath Cost Dictionary
    g_dict = {start:0}

    # 8 Neighbor Directions to Check
    neighbor_dirs = [(-res,-res),(-res,0),(-res,res),(0,res),(res,res),(res,0),(res,-res),(0,-res)]

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

        for n in neighbor_dirs:
            if res < 1: 
                nx = round((np.floor(curr_node[0] * 10) / 10) + n[0],1)
                ny = round((np.floor(curr_node[1] * 10) / 10) + n[1],1)
            else:
                nx = np.floor(curr_node[0] + n[0])
                ny = np.floor(curr_node[1] + n[1])

            neighbor_node = (nx,ny)

            if neighbor_node in closed_set:
                continue
            
            # Bounds the Search
            if not (width[0] <= nx <= width[-1] and height[-1] <= ny <= height[0]):
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

                h = Heuristic(neighbor_node,goal)

                f = g_temp + h

                open_set.append((f,g_temp,neighbor_node))
                parent[neighbor_node] = curr_node

    return closed_set, []

def LRTA_Cost(map_vals, goal, res, s, a, s_prime, H):

    w, h, Map = map_vals

    if s_prime is None:

        if res < 1: 
            nx = round((np.floor(s[0] * 10) / 10) + a[0],1)
            ny = round((np.floor(s[1] * 10) / 10) + a[1],1)
        else:
            nx = np.floor(s[0] + a[0])
            ny = np.floor(s[1] + a[1])

        if not (w[0] <= nx <= w[-1] and h[-1] <= ny <= h[0]):
            return 1000

        map_x = int(np.where(w == nx)[0][0])
        map_y = int(np.where(h == ny)[0][0])

        if Map[map_y][map_x] == 1:
            a_cost = 1000
        else: 
            a_cost = 1

        s = (nx, ny)

        return a_cost + Heuristic(s, goal)

    else: 

        if res < 1: 
            nx = round((np.floor(s_prime[0] * 10) / 10) + a[0],1)
            ny = round((np.floor(s_prime[1] * 10) / 10) + a[1],1)
        else:
            nx = np.floor(s_prime[0] + a[0])
            ny = np.floor(s_prime[1] + a[1])

        if not (w[0] <= nx <= w[-1] and h[-1] <= ny <= h[0]):
            return 1000

        map_x = int(np.where(w == nx)[0][0])
        map_y = int(np.where(h == ny)[0][0])

        if Map[map_y][map_x] == 1:
            a_cost = 1000
        else: 
            a_cost = 1

        return a_cost + H[s_prime]   

def Online_A_Star(map_vals, res, start, goal):

    if res < 1: 
        goal = (np.floor(goal[0] * 10) / 10, np.floor(goal[1] * 10) / 10)
        start = (np.floor(start[0] * 10) / 10, np.floor(start[1] * 10) / 10)
    else: 
        goal = (np.floor(goal[0]), np.floor(goal[1]))
        start = (np.floor(start[0]), np.floor(start[1]))

    s_prime = start

    s = None
    a = None

    resulting_states = dict()
    H = dict()

    parent = {start: None}

    path = [start]

    # 8 Neighbor Directions to Check
    neighbor_dirs = [(-res,-res),(-res,0),(-res,res),(0,res),(res,res),(res,0),(res,-res),(0,-res)]

    while True:
        
        if s_prime == goal:

            # path = [goal]

            # child = goal

            # while parent[child]:

            #     path.append(parent[child])
            #     child = parent[child]

            # path.reverse()

            return path

        if s_prime not in H:
            H[s_prime] = Heuristic(s_prime, goal)

        if s is not None:
            
            resulting_states[(s, a)] = s_prime

            H[s] = min(LRTA_Cost(map_vals, goal, res, s, n, resulting_states.get((s, n)), H) for n in neighbor_dirs)
            
        a_list = []

        for n in neighbor_dirs:

            next_s = resulting_states.get((s_prime, n))

            cost = LRTA_Cost(map_vals, goal, res, s_prime, n, next_s, H)

            a_list.append((cost,n))

        a = min(a_list)[1]

        s = s_prime

        s_prime = (round((np.floor(s_prime[0] * 10) / 10) + a[0],1),
                   round((np.floor(s_prime[1] * 10) / 10) + a[1],1))

        path.append(s_prime)


# Resolution of Grid
Res = 0.1

# Grid Representation Variables
w,h,grid = build_grid([-2.0,5.0], [-6.0,6.0], Res, obstacle_locations)

# STEP 3 PATHS
# c_set, path = A_star([w,h,grid], Res, (0.5,-1.5),(0.5,1.5))
# c_set, path = A_star([w,h,grid], Res, (4.5,3.5),(4.5,-1.5))
# c_set, path = A_star([w,h,grid], Res, (-0.5,5.5),(1.5,-3.5))

path = Online_A_Star([w,h,grid], Res, (-1.5,-5.5),(4.9,-2.5))

# STEP 7 PATHS
# c_set, path = A_star([w,h,grid], Res, (2.45,-3.55),(0.95,-1.55))
# c_set, path = A_star([w,h,grid], Res, (4.95,-0.05),(2.45,0.25))
# c_set, path = A_star([w,h,grid], Res, (-0.55,1.45),(1.95,3.95))


for i,p in enumerate(path):

    x_indx = int(np.where(w == p[0])[0][0])
    y_indx = int(np.where(h == p[1])[0][0])

    if i == 0:
        grid[y_indx][x_indx] = 3
    elif i == len(path) - 1:
        grid[y_indx][x_indx] = 4
    else:
        grid[y_indx][x_indx] = 2

cmap = colors.ListedColormap(['white', 'black', 'lime', 'red', 'blue'])

grid_width = grid.shape[1]
grid_height = grid.shape[0]

plt.figure(figsize=(6,10))
plt.imshow(grid, cmap=cmap, origin='upper', extent=[-2, grid_width*Res - 2 , -6, grid_height*Res - 6])

x_ticks = np.arange(-2, grid_width*Res - 2, Res)
y_ticks = np.arange(-6, grid_height*Res - 6, Res)
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.grid(True, color='gray', linewidth = Res*1.5)

plt.title("World")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.show()