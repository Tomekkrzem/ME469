import numpy as np
from numpy import cos,sin
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import os

# Importing Landmark Dataset for Obstacle Locations
landmark_dir = os.path.dirname(os.path.abspath(__file__))
landmark_fp = os.path.join(landmark_dir, "datasets/ds1_Landmark_Groundtruth.dat")
landmark_data = pd.read_table(landmark_fp, sep=r'\s+', skiprows=3).to_numpy()

# Extracting Landmark Locations
obstacle_locations = np.column_stack((landmark_data[:,1].T,landmark_data[:,2].T))

# Builds a Graph Populated with Obstacles
def build_grid(x_range, y_range, res, obstacles = []):
    """
    :param x_range: X Range of Grid
    :param y_range: Y Range of Grid
    :param res: Resolution
    :param obstacles: Obstacle in World
    """

    # If Resolution is Finer
    if res < 1:
        # X-Value Linespace
        width = np.linspace(x_range[0],x_range[1], int((x_range[1] - x_range[0])/res) + 1)
        # Y-Value Linespace
        height = np.flip(np.linspace(y_range[0],y_range[1], int((y_range[1] - y_range[0])/res) + 1))

        # Truncate the Values in the Linespaces
        width = np.floor(np.round(width,decimals = 1) * 10) / 10
        height = np.ceil(np.round(height,decimals = 1) * 10) / 10

        # Initializing 2D Grid of Empty Positions
        grid_vals = np.zeros((len(height),len(width)))

        # Create Inflation Array for Obstacles
        obstacle_arr = np.ones((7,7))

    # Else If Resolution is More Coarse
    else: 
        # X-Value Linespace
        width = np.linspace(x_range[0],x_range[1]-1, int((x_range[1] - x_range[0])/res))
        # Y-Value Linespace (Had to Be Flipped for Correct Orientation)
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

        # If Resouliton is Fine
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

        # If Resouliton is Coarse
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

def potential_field(Grid, res, goal, sqr_size, C):
    """
    :param Grid: Build Grid Function
    :param res: Resolution
    :param goal: Goal Position
    :param sqr_size: Obstacle Size
    :param C: Attractive Potential Constant
    """
        
    # Create a New Grid for the Potential Field
    width, height, potential_grid = Grid([-2.0,5.0], [-6.0,6.0], res, obstacle_locations)

    # Represent Each Obstacle by its Center (x,y), and its Width and Height
    obstacles = [(o[0], o[1], sqr_size, sqr_size) for o in obstacle_locations]

    # Create a Mesgrid of the World for Vectorized Computation 
    Px, Py = np.meshgrid(width, height)

    # If the Resolution is Fine Floor the Goal Position Values to One Decimal
    if res < 1: 
        goal = (np.floor(goal[0] * 10) / 10, np.floor(goal[1] * 10) / 10)

    # If the Resolution is Coarses Floor the Goal Position to Zero Decimals
    else: 
        goal = (np.floor(goal[0]), np.floor(goal[1]))
    
    # Compute the Attractive Potential in Each Cell of the Grid
    P_goal = C * np.sqrt((Px - goal[0])**2 + (Py - goal[1])**2)

    # Initialize a 2D Array for the Repulsive Potential 
    P_obs = np.zeros((len(height),len(width)))

    # Loop Through All Obstacles
    for o in obstacles:
                
        # Extract Obstacle Properties
        ox, oy, w, h = o

        # If Resouliton is Fine
        if res < 1: 

            # Round the Center of the Obstacle and Transform it to the Real Center of the Obstacle
            ox = np.ceil(ox * 10) / 10 - 0.05
            oy = np.ceil(oy * 10) / 10 - 0.05

            # Compute the Distance to the Closet Edge of Each Obsacle
            # This is Necessary to Account for the Width and Height of the Obstacles
            dist_to_obs = np.sqrt((np.maximum(np.abs(Px - ox) - w/2, 0))**2 + (np.maximum(np.abs(Py - oy) - h/2, 0))**2)

        # If Resouliton is Coarse
        else: 
            # Round the Center of the Obstacle
            ox = np.floor(ox)
            oy = np.floor(oy)

            # Compute the Distance to the Center of Each Obstacle
            dist_to_obs = np.sqrt((Px - ox)**2 + (Py - oy)**2)

        # Compute the Contribution of Each Obstacle to the Repulsive Potential
        P_obs += (C-0.3) / (dist_to_obs + 0.1)

    # Update the Potential Field Grid by Adding the Attractive and Repulsive Potentials in Each Cell
    potential_grid = P_goal + P_obs

    return width, height, potential_grid
    
def potential_feild_path(PF, res, start, goal, C, fig_title):
    """
    :param map_vals: Potential Field Funciton
    :param res: Resolution
    :param start: Start Position
    :param goal: Goal Position
    :param C: Attractive Potential Constant
    :param fig_title: Figure Title
    """

    # Compute the Potential Field Given the Resolution, Goal Position, Obstacle Size, and Constant for Attractive Potential
    width, height, pf = PF(build_grid, res, goal, 0.7, C)

    # Initialize Timeout Counter
    t = 0   

    # Establish Neighbor Directions
    neighbor_dirs = [(-res,-res),(-res,0),(-res,res),(0,res),(res,res),(res,0),(res,-res),(0,-res)]

    # If the Resolution is Fine Floor the Start and Goal Position Values to One Decimal
    if res < 1: 
        xg = (np.floor(goal[0] * 10) / 10, np.floor(goal[1] * 10) / 10)
        xt = (np.floor(start[0] * 10) / 10, np.floor(start[1] * 10) / 10)

    # If the Resolution is Coarses Floor the Start and Goal Position to Zero Decimals
    else: 
        xg = (np.floor(goal[0]), np.floor(goal[1]))
        xt = (np.floor(start[0]), np.floor(start[1]))

    # Add the Start Position to the Path
    p_path = [(float(xt[0]),float(xt[1]))]

    # Establish a Distance Threshold
    thresh = 0.2

    while np.sqrt((xg[0] - xt[0])**2 + (xg[1] - xt[1])**2) > thresh:

        # Get Potential Value at Current Postition from Potential Field
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
                nx = xt[0] + n[0]
                ny = xt[1] + n[1]

            # If Out of World Bounds Continue
            if not (width[0] <= nx <= width[-1] and height[-1] <= ny <= height[0]):
                continue
            
            # Extract Grid Indices of Neighbor Node
            map_nx = int(np.where(width == nx)[0][0])
            map_ny = int(np.where(height == ny)[0][0])

            # Compute Temporary Potential Value at Neighbor Position
            temp_pot = pf[map_ny][map_nx]
            
            # If Temporary Potential is Smaller than Current Potential Move to That Position
            if temp_pot < min_pot:
                xt_temp = (nx,ny)
        
        # Update Current State
        xt = xt_temp

        # Add Current State to Path
        p_path.append((float(xt[0]),float(xt[1])))

        # Increment Timeout Counter
        t += 1

        # If Timeout Counter Reaches 200, the Path was Unable to Converge
        if t == 200:
            break
    
    # Extract X and Y Coordinates from Path
    pcx =[]
    pcy = []

    for c in p_path:
        pcx.append(c[0] + res/2)
        pcy.append(c[1] + res/2)

    # Plot Potential Field Path
    # Grid Width and Length for Plot Creation
    grid_width = pf.shape[1]
    grid_height = pf.shape[0]

    plt.figure(figsize=(6,8))
    # Display 2D Grid
    plt.imshow(pf, origin='upper', cmap='plasma', extent=[-2, grid_width * res - 2 , -6, grid_height * res - 6])
    plt.plot(pcx,pcy, c='lime', linewidth=5)

    # Label Major Values on Axes (i.e. -6, -5.5, -5, etc.)
    ax = plt.gca()

    # Plot Start and Goal Points
    plt.scatter(path[0][0] + res/2, path[0][1] + res/2, c='r', edgecolor='k', label = f'Start {s}', zorder=3)
    plt.scatter(path[-1][0] + res/2, path[-1][1] + res/2, c='lime', edgecolor='k', label = f'Goal {g}', zorder=3)

    # Check that the Value to Label is a Multiple of 0.5 (Matplotlib Documentation)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}" if abs(x*2 - round(x*2)) < 1e-6 else ""))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}" if abs(y*2 - round(y*2)) < 1e-6 else ""))

    # Create Grid Lines
    x_ticks = np.arange(-2, grid_width * res - 2, res)
    y_ticks = np.arange(-6, grid_height * res - 6, res)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.grid(True, color='gray', linewidth = res * 1.5)

    # Display Plot
    plt.title(f"Potential Field Path (Res = {res})")
    plt.xlabel("X [m]", fontsize=12)
    plt.ylabel("Y [m]", fontsize=12)
    plt.legend(fontsize=12)
    # plt.savefig('asgn1/' + fig_title + '_PF', bbox_inches='tight')
    plt.show()

    return p_path

# Heuristic Function for A*
def Heuristic(node_s, node_g):
    """
    :param node_s: Current State Position
    :param node_g: Goal Position
    """
    
    # Euclidean Distance
    return np.sqrt((node_g[0] - node_s[0])**2 + (node_g[1] - node_s[1])**2)


# A* Start Algorithm
def A_star (map_vals, res, start, goal):
    """
    :param map_vals: Grid Values (width, height, grid)
    :param res: Resolution
    :param start: Start Position
    :param goal: Goal Position
    """

    # If the Resolution is Fine Floor the Start and Goal Position Values to One Decimal
    if res < 1: 
        goal = (np.floor(goal[0] * 10) / 10, np.floor(goal[1] * 10) / 10)
        start = (np.floor(start[0] * 10) / 10, np.floor(start[1] * 10) / 10)
    # If the Resolution is Coarses Floor the Start and Goal Position to Zero Decimals
    else: 
        goal = (np.floor(goal[0]), np.floor(goal[1]))
        start = (np.floor(start[0]), np.floor(start[1]))

    # Extract Map Properties
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

    # Loop While there is a Node in the Open Set
    while len(open_set) != 0:
        
        # Sort the Open Set by the Total Cost (f) to Find the Lowest Cost Node
        open_set = sorted(open_set)

        # Extract the Total Cost, Action Cost and Current Node Position from the First Element in the Open Set
        h, g, curr_node = open_set.pop(0)

        # If Current Node is Goal
        if curr_node == goal:

            # Add Goal to Path
            path = [goal]
            
            # Set Child as Goal
            child = goal
            
            # Check all Children until No Children are Present
            while parent[child]:
                    
                # Add Node to Path
                path.append(parent[child])
                # Get Next Child
                child = parent[child]

            # Reverse the Path to Go from Start to Goal
            path.reverse()

            # Add Goal Node to Closed Set
            closed_set.append(curr_node)

            return path

        # If Node is in Closed Set, Skip
        if curr_node in closed_set:
            continue

        # Add Current Node to Closed Set
        closed_set.append(curr_node)

        # For each Neighbor Direction
        for n in neighbor_dirs:
            
            # If Resolution is Fine Floor the Neighbor X and Y Coordinate Values to One Decimal
            if res < 1: 
                nx = round((np.floor(curr_node[0] * 10) / 10) + n[0],1)
                ny = round((np.floor(curr_node[1] * 10) / 10) + n[1],1)
                
            # If Resolution is Coarse Floor the Neighbor X and Y Coordinate Values to Zero Decimals
            else:
                nx = np.floor(curr_node[0] + n[0])
                ny = np.floor(curr_node[1] + n[1])

            # Set Neighbor Node Position
            neighbor_node = (nx,ny)

            # If Neighbor Node in Closed Set, Skip
            if neighbor_node in closed_set:
                continue
            
            # Bounds the Search
            if not (width[0] <= nx <= width[-1] and height[-1] <= ny <= height[0]):
                continue
            
            # Extract Grid Indices of Neighbor Node
            map_x = int(np.where(width == nx)[0][0])
            map_y = int(np.where(height == ny)[0][0])

            # If Neighbor Node is an Obstacle Update Action Cost by 1000
            if Map[map_y][map_x] == 1:
                g_temp = g + 1000
                
            # Else Update Action Cost by 1
            else: 
                g_temp = g + 1

            # If Neighbor Node not in Cost Dictionary or Action Cost is Less than Action Cost of Neighbor Node
            if neighbor_node not in g_dict or g_temp < g_dict[neighbor_node]:
                
                # Add Neighbor Node Action Cost to Cost Dictionary
                g_dict.update({neighbor_node:g_temp})   

                # Compute Heuristic Cost of Neighbor Node
                h = Heuristic(neighbor_node,goal)

                # Compute Total Cost
                f = g_temp + h

                # Add Costs and Neighbor Node to Open Set
                open_set.append((f,g_temp,neighbor_node))
                
                # Update Parent of Neighbor Node to be Current Node
                parent[neighbor_node] = curr_node

    return []


# Online A* Cost Function - LRTA_Cost Function Referenced from Articial Intelligence Ch. 4.5.3
def Online_A_Star_Cost(map_vals, goal, res, s, a, s_prime, H):
    """
    :param map_vals: Grid Values (width, height, grid)
    :param goal: Goal Position
    :param res: Resolution
    :param s: Previous State
    :param a: Previous Action
    :param s_prime: Current State
    :param H: Cost Table
    """

    # Extract Map Properties
    w, h, Map = map_vals

    # If Next State is Unknown
    if s_prime is None:

        # If Resolution is Fine Floor the Neighbor X and Y Coordinate Values to One Decimal
        if res < 1: 
            # Use Current Position (s) as State
            nx = round((np.floor(s[0] * 10) / 10) + a[0],1)
            ny = round((np.floor(s[1] * 10) / 10) + a[1],1)
        # If Resolution is Coarse Floor the Neighbor X and Y Coordinate Values to Zero Decimals
        else:
            # User Current Position (s) as State
            nx = np.floor(s[0] + a[0])
            ny = np.floor(s[1] + a[1])

        # If Search is Out of Bounds Treat as Obstacle
        if not (w[0] <= nx <= w[-1] and h[-1] <= ny <= h[0]):
            return 1000

        # Extract Grid Indices of Neighbor State
        map_x = int(np.where(w == nx)[0][0])
        map_y = int(np.where(h == ny)[0][0])

        # If Neighbor State is an Obstacle Update Action Cost by 1000
        if Map[map_y][map_x] == 1:
            a_cost = 1000
        else: 
            a_cost = 1

        # Current State is Neighbor State
        s = (nx, ny)

        # Return Cost of Neighbor State
        return a_cost + Heuristic(s, goal)

    # If Next State is Known
    else: 
        if res < 1: 
            # Use Next Position (s_prim) as State 
            nx = round((np.floor(s_prime[0] * 10) / 10) + a[0],1)
            ny = round((np.floor(s_prime[1] * 10) / 10) + a[1],1)
        else:
            # Use Next Position (s_prim) as State 
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

# Online A* Function - LRTA_Agent Function Referenced from Articial Intelligence Ch. 4.5.3
def Online_A_Star(map_vals, res, start, goal):
    """
    :param map_vals: Grid Values (width, height, grid)
    :param res: Resolution
    :param start: Start Position
    :param goal: Goal Position
    """

    # If the Resolution is Fine Floor the Start and Goal Position Values to One Decimal
    if res < 1: 
        goal = (np.floor(goal[0] * 10) / 10, np.floor(goal[1] * 10) / 10)
        start = (np.floor(start[0] * 10) / 10, np.floor(start[1] * 10) / 10)
    # If the Resolution is Coarses Floor the Start and Goal Position to Zero Decimals
    else: 
        goal = (np.floor(goal[0]), np.floor(goal[1]))
        start = (np.floor(start[0]), np.floor(start[1]))

    # Initialize Current State
    s_prime = start

    # Initialize Previous State to None
    s = None
    # Initialize Previous Action to None
    a = None

    # Initialize Resulting States Dictionary (s, a) <- s'
    resulting_states = dict()
    
    # Initialize Heuristic Table H[s] = h(s)
    H = dict()

    # Initialize Path
    path = [start]

    # 8 Neighbor Directions to Check
    neighbor_dirs = [(-res,-res),(-res,0),(-res,res),(0,res),(res,res),(res,0),(res,-res),(0,-res)]

    # While Goal is Not Found
    while True:
        
        # If Current State is Goal
        if s_prime == goal:

            # Return the Path
            return path

        # If Current Path is Not in Heuristic Table Add it 
        if s_prime not in H:
            H[s_prime] = Heuristic(s_prime, goal)

        # If Previous State Exists
        if s is not None:
            
            # Update Resulting States
            resulting_states[(s, a)] = s_prime

            # Update Heuristic Table with the Smallest Cost of the Previous State
            H[s] = min(Online_A_Star_Cost(map_vals, goal, res, s, n, resulting_states.get((s, n)), H) for n in neighbor_dirs)
        
        # Action Cost List
        a_list = []

        # For Each Neighbor Direction
        for n in neighbor_dirs: 
            
            # Next State is the State of Current State + Action if it Exists in Resulting States. Otherwise it is None
            next_s = resulting_states.get((s_prime, n))

            # Compute Cost of Next State
            cost = Online_A_Star_Cost(map_vals, goal, res, s_prime, n, next_s, H)

            # Add it to Action Cost List
            a_list.append((cost,n))

        # Action Taken is the One with Least Cost
        a = min(a_list)[1]
        
        # Previous State is now the Current State
        s = s_prime

        # Compute Next State By Performing Action
        s_prime = (round((np.floor(s_prime[0] * 10) / 10) + a[0],1),
                   round((np.floor(s_prime[1] * 10) / 10) + a[1],1))

        # Add Next State to Path
        path.append(s_prime)

# Grid Resolutions
Res = 1
Res2 = 0.1

# Coarse Grid Representation Variables
c_w, c_h, c_grid = build_grid([-2.0,5.0], [-6.0,6.0], Res, obstacle_locations)
# Fine Grid Representation Variables
f_w, f_h, f_grid = build_grid([-2.0,5.0], [-6.0,6.0], Res2, obstacle_locations)

s_g_list_step = [[Res,(0.5,-1.5),(0.5,1.5),0],       # STEP 3 START-GOAL POSITIONS
                 [Res,(4.5,3.5),(4.5,-1.5),0],
                 [Res,(-0.5,5.5),(1.5,-3.5),0],
                  
                 [Res,(0.5,-1.5),(0.5,1.5),1],       # STEP 5 START-GOAL POSITIONS
                 [Res,(4.5,3.5),(4.5,-1.5),1],
                 [Res,(-0.5,5.5),(1.5,-3.5),1],
                  
                 [Res2,(2.45,-3.55),(0.95,-1.55),1], # STEP 7 START-GOAL POSITIONS
                 [Res2,(4.95,-0.05),(2.45,0.25),1],
                 [Res2,(-0.55,1.45),(1.95,3.95),1]]

# Question Number
q_count = 3
# Question Part Number
count = 1

# For Each Start-Goal Pair
for s_g in s_g_list_step:
    
    # Extract Resolution, Start and Goal
    Res, s, g, Alg_type = s_g
    
    # Grid Representation Variables
    if Res < 1:
        w = f_w
        h = f_h
        grid = f_grid.copy()
    else:
        w = c_w
        h = c_h
        grid = c_grid.copy()

    # If Alg_type Is 1 Use Online A*
    if Alg_type:
        path = Online_A_Star([w,h,grid], Res, s, g)
        
        if Res < 1:
            q_count = 7
        else: 
            q_count = 5
        
        # Update Plot Title and Figure Title
        plot_title = f"Online A* (Res = {Res})"
        fig_title = f"Question_{q_count}_{count}"

        # Run the Potential Field Algorithm
        potential_feild_path(potential_field, Res, s, g, 0.5, fig_title)

        count += 1
        if count == 4:
            count = 1
    
    # If Alg_type is 0 Use A_Star
    else:
        path = A_star([w,h,grid], Res, s, g)

        plot_title = f"A* (Res = {Res})"
        fig_title = f"Question_{q_count}_{count}"
        
        potential_feild_path(potential_field, Res, s, g, 0.5, fig_title)

        count += 1
        if count == 4:
            count = 1
    
    # Update Path Positons in Grid
    for i,p in enumerate(path):
        
        # Find Corresponding Grid Index of Path Position
        x_indx = int(np.where(w == p[0])[0][0])
        y_indx = int(np.where(h == p[1])[0][0])

        # If Position is Start Color it Red
        if i == 0:
            grid[y_indx][x_indx] = 3    # 3 = Red
            
        # If Position is Goal Color it Red
        elif i == len(path) - 1:
            grid[y_indx][x_indx] = 4    # 4 = Blue
            
        # Otherwise Color the Path green
        else:
            grid[y_indx][x_indx] = 2

    # Color Map for Grid Position Values (i.e. 0,1,2,3,4)
    cmap = colors.ListedColormap(['white', 'black', 'orange', 'red', 'lime'])

    # Grid Width and Length for Plot Creation
    grid_width = grid.shape[1]
    grid_height = grid.shape[0]

    plt.figure(figsize=(6,8))
    # Display 2D Grid
    plt.imshow(grid, cmap=cmap, origin='upper', extent=[-2, grid_width*Res - 2 , -6, grid_height*Res - 6])
    
    # Plot Start and Goal Points
    plt.scatter(path[0][0] + Res/2, path[0][1] + Res/2, c='r', edgecolor='k', label = f'Start {s}', zorder=3)
    plt.scatter(path[-1][0] + Res/2, path[-1][1] + Res/2, c='lime', edgecolor='k', label = f'Goal {g}', zorder=3)

    # Label Major Values on Axes (i.e. -6, -5.5, -5, etc.)
    ax = plt.gca()

    # Check that the Value to Label is a Multiple of 0.5 (Matplotlib Documentation)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}" if abs(x*2 - round(x*2)) < 1e-6 else ""))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}" if abs(y*2 - round(y*2)) < 1e-6 else ""))

    # Create Grid Lines
    x_ticks = np.arange(-2, grid_width*Res - 2, Res)
    y_ticks = np.arange(-6, grid_height*Res - 6, Res)
    plt.xticks(x_ticks, fontsize=10)
    plt.yticks(y_ticks, fontsize=10)
    plt.grid(True, color='gray', linewidth = Res*1.5)

    # Display Plot
    plt.title(plot_title)
    plt.xlabel("X [m]", fontsize=12)
    plt.ylabel("Y [m]", fontsize=12)
    plt.legend(fontsize=12)
    # plt.savefig('asgn1/' + fig_title, bbox_inches='tight')
    plt.show()


#--------------------------------------------------------------------------------------------------------------------------#
class Controller:

    def __init__(self, Grid, res, start, goal, sqr_size, o_locs, noise, dt, plot_traj, offset=0.3):
        
        # Create a Grid for this Controller
        self.width, self.height, self.grid = Grid([-2.0,5.0], [-6.0,6.0], res, o_locs)

        # Resolution of Grid
        self.res = res

        # Trajectory
        self.Traj = Online_A_Star([self.width,self.height,self.grid], res, (start[0],start[1]),(goal[0],goal[1]))
        
        # Add Trajectory Points in Grid if Specified
        if plot_traj:
            for i,p in enumerate(self.Traj):

                # Find Corresponding Grid Index of Path Position
                x_indx = int(np.where(self.width == p[0])[0][0])
                y_indx = int(np.where(self.height == p[1])[0][0])

                # If Position is Start Color it Red
                if i == 0:
                    self.grid[y_indx][x_indx] = 3    # 3 = Red
                    
                # If Position is Goal Color it Blue
                elif i == len(self.Traj) - 1:
                    self.grid[y_indx][x_indx] = 4    # 4 = Blue

                # Otherwise Color the Path Green
                else:
                    self.grid[y_indx][x_indx] = 2
        
        # Otherwise Just Add Start and End Points in Grid
        else: 
            # Find Corresponding Grid Index of Start Point
            x_indx = int(np.where(self.width == self.Traj[0][0])[0][0])
            y_indx = int(np.where(self.height == self.Traj[0][1])[0][0])

            # Color Start Point Red
            self.grid[y_indx][x_indx] = 3    # 3 = Red
                
            # Find Corresponding Grid Index of Start Point
            x_indx = int(np.where(self.width == self.Traj[-1][0])[0][0])
            y_indx = int(np.where(self.height == self.Traj[-1][1])[0][0])

            # Color Goal Point Blue
            self.grid[y_indx][x_indx] = 4    # 4 = Blue

        # Simulation Time Step
        self.dt = dt
        
        # Starting State
        self.xt = start

        # Goal State
        self.xg = goal

        # Initilialize Errors
        self.e_v = None        
        self.e_w = None        

        # Initialize Previous State Errors
        self.prev_e_v = None
        self.prev_e_w = None

        # Initialize Previous State Velocities 
        self.v_prev = 0
        self.w_prev = 0
        
        # Initialize Summation for Integral Term in PID Controller
        self.e_v_intg = 0
        self.e_w_intg = 0

        # Add Initial State to Path
        self.robot_path = [[self.xt[0], self.xt[1], self.xt[2]]]

        # Obstacle Clearance Offset for Trajectory Smoothing
        self.obst_off = offset

        # Represent Each Obstacle by its Center (x,y), and its Width and Height
        self.obstacles = [(o[0], o[1], sqr_size, sqr_size) for o in o_locs]

        # Movement Noise
        self.noise = noise

    def bezier_curve(self, c_points, num_points, check_collisions):
        """
        :param c_points: Control Points Obtained from Input Trajectory
        :param num_points: Number of Output Trajectory Points
        :param check_collisions: Boolean to Check Collisions
        """

        # Degree of Bezier Curve
        n = len(c_points) - 1

        # Trajectory Discretization
        t = np.linspace(0, 1, num_points)

        # Initialize Output Curve
        output_curve = np.zeros((num_points, 2))

        for i in range(num_points):
            for j in range(n + 1):
                
                # Bezier Curve Equation Referenecd From "Nerding out with bezier curves"
                output_curve[i] += c_points[j] * math.comb(n, j) * (t[i] ** j) * ((1 - t[i]) ** (n - j)) 

            # If User Specifies Collision Checking
            if check_collisions:

                # Loop Through All Obstacles
                for o in self.obstacles:
                    
                    # Select Current Trajectory Point
                    p = output_curve[i]

                    # Extract Obstacle Properties
                    ox, oy, w, h = o

                    # If Resouliton is Finer
                    if self.res < 1: 

                        # Round the Center of the Obstacle and Transform it to the Real Center of the Obstacle
                        ox = np.ceil(ox * 10) / 10 - 0.05
                        oy = np.ceil(oy * 10) / 10 - 0.05

                    # If Resouliton is Coarser
                    else: 

                        # Round the Center of the Obstacle and Transform it to the Real Center of the Obstacle
                        ox = np.ceil(ox)
                        oy = np.ceil(oy)

                    # X-Distance Between Point and Obstacle Center
                    x_to_cx_dist = p[0] - ox
                    # Y-Distance Between Point and Obstacle Center
                    y_to_cy_dist = p[1] - oy

                    # If Trajectory Position is Inside Obstacle
                    if abs(x_to_cx_dist) <= w/2 and abs(y_to_cy_dist) <= h/2:
                        
                        # If Collision is Closer to Obstacle Center in X-Direction
                        if w/2 - abs(x_to_cx_dist) < h/2 - abs(y_to_cy_dist):

                            # Offset the X-Position of the Trajectory Position
                            output_curve[i][0] = ox + np.sign(x_to_cx_dist) * (w/2 + self.obst_off)

                        # If Collision is Closer to Obstacle Center in X-Direction
                        else:

                            # Offset the Y-Position of the Trajectory Position
                            output_curve[i][1] = oy + np.sign(y_to_cy_dist) * (h/2 + self.obst_off)

        return output_curve

    def Offline_PID_Controller(self):

        # Preprocess Trajectory Using Bezier Curve
        smooth_T = self.bezier_curve(np.array(self.Traj), int(len(self.Traj)/(self.dt*10)), 1)
        smooth_T = self.bezier_curve(smooth_T, int(len(smooth_T)/(self.dt)), 0)

        # Compose Feedforward States for Robot
        ffwd_points = np.zeros((len(smooth_T), 3))
        ffwd_points[0] = self.xt

        # Loop Through All Trajectory Points
        for i in range(len(smooth_T) - 1):
            
            # Compute Heading of Trajectory Points
            x,y = smooth_T[i+1]
            x_p, y_p = smooth_T[i]
            theta = np.atan2((y - y_p),(x - x_p))

            # Wrap Angles
            theta = (theta + np.pi) % (2 * np.pi) - np.pi

            ffwd_points[i+1] = (x,y, theta)

        # Proportional Terms for Velocity Control
        Kp_v = 0.5
        Kp_w = 1
        # Intergral Terms for Velocity Control
        Ki_v = 0.001
        Ki_w = 0.0005
        # Derivative Terms for Velocity Control
        Kd_v = 0.4
        Kd_w = 0.5

        # Loop Until Robot Reaches Completes Trajectory
        for i,p in enumerate(ffwd_points):

            # Compute Linear Error as Distance Between Trajectory Point and Robot Position
            x_e = p[0] - self.xt[0]
            y_e = p[1] - self.xt[1]
            self.e_v = np.sqrt(x_e**2 + y_e**2)

            # Compute Angular Error as the Difference between Desired Heading to Converge to Trajectory and Robot Heading
            theta = self.xt[2]
            theta_desired = np.atan2(y_e, x_e)
            self.e_w = theta_desired - theta

            # Wrap Angles
            self.e_w = (self.e_w + np.pi) % (2 * np.pi) - np.pi 

            # Compute Error Derivative
            if self.prev_e_v is not None and self.prev_e_w is not None: 
                d_ev = self.e_v - self.prev_e_v 
                d_ew = self.e_w - self.prev_e_w
            else: 
                d_ev = 0 
                d_ew = 0

            # Compute Velocity
            v = Kp_v * self.e_v + Ki_v * self.e_v_intg + Kd_v * d_ev 
            w = Kp_w * self.e_w + Ki_w * self.e_w_intg + Kd_w * d_ew 
            w = (w + np.pi) % (2 * np.pi) - np.pi

            # Limit the Accelerations
            dv = np.clip(v - self.v_prev, -0.288 * self.dt, 0.288 * self.dt)
            dw = np.clip(w - self.w_prev, -5.579 * self.dt, 5.579 * self.dt)

            # Update Velocites with Limited Accelerations
            v = self.v_prev + dv
            w = self.w_prev + dw

            # Update Previous Errors
            self.prev_e_v = self.e_v
            self.prev_e_w = self.e_w

            # Update Previous Velocity
            self.v_prev = v
            self.w_prev = w

            # Compute Next State Position
            self.xt = self.x_t(self.xt,[v,w],self.dt)

            # Update Error Integral
            self.e_v_intg += self.e_v
            self.e_w_intg += self.e_w

            # Construct Robot Path
            self.robot_path.append([self.xt[0]+0.05,self.xt[1]+0.05, self.xt[2]])

        return self.robot_path, ffwd_points
    
    def Online_PID_Controller(self):

        # Proportional Terms for Velocity Control
        Kp_v = 0.02
        Kp_w = 1
        # Intergral Terms for Velocity Control
        Ki_v = 0.0005
        Ki_w = 0.005
        # Derivative Terms for Velocity Control
        Kd_v = 0.9
        Kd_w = 1

        # Point Convergence Threshold
        thresh = 0.08

        # Resolution for Online Path Planning
        res = self.res

        # If the Resolution is Fine Floor the Start and Goal Position Values to One Decimal
        if res < 1: 
            goal = (np.floor(self.xg[0] * 10) / 10, np.floor(self.xg[1] * 10) / 10)
        # If the Resolution is Coarses Floor the Start and Goal Position to Zero Decimals
        else: 
            goal = (np.floor(self.xg[0]), np.floor(self.xg[1]))

        # Initialize Current State
        s_prime = (self.xt[0], self.xt[1])

        # Initialize Previous State to None
        s = None
        # Initialize Previous Action to None
        a = None

        # Initialize Resulting States Dictionary (s, a) <- s'
        resulting_states = dict()
        
        # Initialize Heuristic Table H[s] = h(s)
        H = dict()

        # 8 Neighbor Directions to Check
        neighbor_dirs = [(-res,-res),(-res,0),(-res,res),(0,res),(res,res),(res,0),(res,-res),(0,-res)]

        # Loop Until Robot is Within A Threshold Position Around Goal
        while np.sqrt((self.xg[0] - self.xt[0])**2 + (self.xg[1] - self.xt[1])**2) > thresh:

            # If Current Path is Not in Heuristic Table Add it 
            if s_prime not in H:
                H[s_prime] = Heuristic(s_prime, goal)

            # If Previous State Exists
            if s is not None:
                
                # Update Resulting States
                resulting_states[(s, a)] = s_prime

                # Update Heuristic Table with the Smallest Cost of the Previous State
                H[s] = min(Online_A_Star_Cost([self.width,self.height,self.grid], goal, res, s, n, resulting_states.get((s, n)), H) for n in neighbor_dirs)
            
            # Action Cost List
            a_list = []

            # For Each Neighbor Direction
            for n in neighbor_dirs: 
                
                # Next State is the State of Current State + Action if it Exists in Resulting States. Otherwise it is None
                next_s = resulting_states.get((s_prime, n))

                # Compute Cost of Next State
                cost = Online_A_Star_Cost([self.width,self.height,self.grid], goal, res, s_prime, n, next_s, H)

                # Add it to Action Cost List
                a_list.append((cost,n))

            # Action Taken is the One with Least Cost
            a = min(a_list)[1]
            
            # Previous State is now the Current State
            s = s_prime

            # Compute Next State By Performing Action
            s_prime = (round((np.floor(s_prime[0] * 10) / 10) + a[0],1),
                    round((np.floor(s_prime[1] * 10) / 10) + a[1],1))
            
            # Loop While the Robot Position Is Not Within a Threshold of a Control Point
            while np.sqrt((s_prime[0] - self.xt[0])**2 + (s_prime[1] - self.xt[1])**2) > thresh:

                # Compute Linear Error as Distance Between Trajectory Point and Robot Position
                x_e = s_prime[0] - self.xt[0]
                y_e = s_prime[1] - self.xt[1]
                self.e_v = np.sqrt(x_e**2 + y_e**2)

                # Compute Angular Error as the Difference between Desired Heading to Converge to Trajectory and Robot Heading
                theta = self.xt[2]
                theta_desired = np.atan2(y_e, x_e)
                self.e_w = theta_desired - theta

                # Wrap Angles
                self.e_w = (self.e_w + np.pi) % (2 * np.pi) - np.pi 

                # Compute Error Derivative
                if self.prev_e_v is not None and self.prev_e_w is not None: 
                    d_ev = self.e_v - self.prev_e_v 
                    d_ew = self.e_w - self.prev_e_w
                else: 
                    d_ev = 0 
                    d_ew = 0

                # Compute Velocity
                v = Kp_v * self.e_v + Ki_v * self.e_v_intg + Kd_v * d_ev 
                w = Kp_w * self.e_w + Ki_w * self.e_w_intg + Kd_w * d_ew 
                w = (w + np.pi) % (2 * np.pi) - np.pi

                # Limit the Accelerations
                if (v - self.v_prev) < -0.288 * self.dt:
                    dv = -0.288 * self.dt
                elif (v - self.v_prev) > 0.288 * self.dt:
                    dv = 0.288 * self.dt
                else: 
                    dv = v - self.v_prev

                if (w - self.w_prev) < -5.579 * self.dt:
                    dw = -5.579 * self.dt
                elif (w - self.w_prev) > 5.579 * self.dt:
                    dw = 5.579 * self.dt
                else: 
                    dw = w - self.w_prev

                # Update Velocites with Limited Accelerations
                v = self.v_prev + dv
                w = self.w_prev + dw

                # Update Previous Errors
                self.prev_e_v = self.e_v
                self.prev_e_w = self.e_w

                # Update Previous Velocity
                self.v_prev = v
                self.w_prev = w

                # Temporary State Calculation
                temp_xt = self.x_t(self.xt,[v,w],self.dt)
                
                # Initialize Collision Flag
                collided = False

                # Loop Through All Obstacles
                for o in self.obstacles:

                    # Extract Obstacle Properties
                    ox, oy, w, h = o

                    # If Resouliton is Finer
                    if self.res < 1: 

                        # Round the Center of the Obstacle and Transform it to the Real Center of the Obstacle
                        ox = np.ceil(ox * 10) / 10 - 0.05
                        oy = np.ceil(oy * 10) / 10 - 0.05

                    # If Resouliton is Coarser
                    else: 

                        # Round the Center of the Obstacle and Transform it to the Real Center of the Obstacle
                        ox = np.ceil(ox)
                        oy = np.ceil(oy)

                    # X-Distance Between Point and Obstacle Center
                    x_to_cx_dist = temp_xt[0] - ox
                    # Y-Distance Between Point and Obstacle Center
                    y_to_cy_dist = temp_xt[1] - oy

                    # If Trajectory Position is Inside Obstacle
                    if abs(x_to_cx_dist) <= w/2 and abs(y_to_cy_dist) <= h/2:
                        
                        # Update Collision Flag
                        collided = True

                        # If Collision is Closer to Obstacle Center in X-Direction
                        if w/2 - abs(x_to_cx_dist) < h/2 - abs(y_to_cy_dist):
       
                            self.xt[0] = self.xt[0]
                            self.xt[1] = temp_xt[1]
                            self.xt[2] = temp_xt[2]

                        # If Collision is Closer to Obstacle Center in Y-Direction
                        else:
                            
                            self.xt[0] = temp_xt[0]
                            self.xt[1] = self.xt[1]
                            self.xt[2] = temp_xt[2]

                        # Don't Check Other Obstacles
                        break
                
                # If No Collision is Detected 
                if not collided:
                    
                    # Compute Next State Position
                    self.xt = temp_xt

                # Update Error Integral
                self.e_v_intg += self.e_v
                self.e_w_intg += self.e_w

                if res < 1:
                    # Construct Robot Path
                    self.robot_path.append([self.xt[0]+0.05, self.xt[1]+0.05, self.xt[2]])
                else:
                    self.robot_path.append([self.xt[0], self.xt[1], self.xt[2]])
        
        return self.robot_path

    # Sampling Function Referenced from Probabilistic Robotics Table 5.4 (sample_normal_distribution)
    def sample(self,b):

        return b/6 * np.random.uniform(-1,1,12).sum()

    # Motion Model Referenced Probabilistic Robotics Table 5.3 (sample_motion_model_velocity)
    # Additionally referenced "CS W4733 NOTES - Differential Drive Robots" for Instantaneous Center of Rotation + Rotation Matrix
    def x_t(self, x_t_p, u_t, dt):
        """
        :param X_t_p: Current robot state [x, y, theta]
        :param u_t: Control input [v, w] 
        :param dt: Time step
        """
        a1 = a2 = a3 = a4 = a5 = a6 = self.noise

        # Add motion noise to translational and rotational velocities
        v_hat = u_t[0] + self.sample(a1 * abs(u_t[0]) + a2 * abs(u_t[1]))
        w_hat = u_t[1] + self.sample(a3 * abs(u_t[0]) + a4 * abs(u_t[1]))
        gamma = self.sample(a5 * abs(u_t[0]) + a6 * abs(u_t[1]))

        # Straight-line motion when rotational speed is zero
        if w_hat == 0:
            x_dt = np.array([v_hat * cos(x_t_p[2]) * dt, v_hat * sin(x_t_p[2]) * dt, 0])

        # Circular motion: compute Instantaneous Center of Rotation (ICR)
        else:
            # Instantaneous Center of Rotation (ICR)
            xc = x_t_p[0] - (v_hat/w_hat)*sin(x_t_p[2])
            yc = x_t_p[1] + (v_hat/w_hat)*cos(x_t_p[2])

            # Rotation matrix for turning around ICR
            R = np.array([[cos(w_hat*dt), -sin(w_hat*dt), 0],
                        [sin(w_hat*dt),  cos(w_hat*dt), 0],
                        [0        ,          0, 1]])

            # Apply rotation and translation relative to ICR
            x_dt = np.matmul(R,np.array([x_t_p[0] - xc, 
                                         x_t_p[1] - yc,
                                         w_hat * dt + gamma * dt]))

            # Shift reference to ICR
            x_t_p = np.array([xc,yc,x_t_p[2]])

        # Compute new state by adding displacement
        x_t = x_t_p + x_dt
        
        # Wrap Angles
        x_t[2] = (x_t[2] + np.pi) % (2 * np.pi) - np.pi     

        return x_t

# Grid Resolutions
Res = 1
Res2 = 0.1

# Coarse Grid Representation Variables
c_w, c_h, c_grid = build_grid([-2.0,5.0], [-6.0,6.0], Res, obstacle_locations)
# Fine Grid Representation Variables
f_w, f_h, f_grid = build_grid([-2.0,5.0], [-6.0,6.0], Res2, obstacle_locations)


s_g_list      = [[Res2,(2.45,-3.55),(0.95,-1.55),9], # STEP 7 START-GOAL POSITIONS
                 [Res2,(4.95,-0.05),(2.45,0.25),9],
                 [Res2,(-0.55,1.45),(1.95,3.95),9],
                  
                 [Res2,(2.45,-3.55),(0.95,-1.55),10], # STEP 7 START-GOAL POSITIONS
                 [Res2,(4.95,-0.05),(2.45,0.25),10],
                 [Res2,(-0.55,1.45),(1.95,3.95),10],
                 
                 [Res2,(0.5,-1.5),(0.5,1.5),11],       # STEP 3 START-GOAL POSITIONS
                 [Res2,(4.5,3.5),(4.5,-1.5),11],
                 [Res2,(-0.5,5.5),(1.5,-3.5),11],
                 
                 [Res,(0.5,-1.5),(0.5,1.5),12],       # STEP 3 START-GOAL POSITIONS
                 [Res,(4.5,3.5),(4.5,-1.5),12],
                 [Res,(-0.5,5.5),(1.5,-3.5),12]]


# Question Part Number
count = 1

# For Each Start-Goal Pair
for s_g in s_g_list:
    
    # Extract Resolution, Start and Goal
    Res, s, g, Alg_type = s_g

    # Question 9 Plots
    if Alg_type == 9:
        # Initialize Controller Values
        robot = Controller(build_grid, Res, (s[0],s[1],-np.pi/2), (g[0],g[1]), 0.8, obstacle_locations, 0.5, 0.1, True, 0.2)

        # Extract Control Path and Trajectory
        control_path, ffwd = robot.Offline_PID_Controller()

        plot_title = f"Offline Controller (Step 7 Paths) (Res = {Res})"
        fig_title = f"Question_9_{count}"
        
        count += 1
        if count == 4:
            count = 1

    # Question 10 Plots
    elif Alg_type == 10:
        # Initialize Controller Values
        robot = Controller(build_grid, Res, (s[0],s[1],-np.pi/2), (g[0],g[1]), 0.7, obstacle_locations, 0.5, 0.1, False, 0.2)

        # Extract Control Path
        control_path = robot.Online_PID_Controller()

        plot_title = f"Online Controller (Step 7 Paths) (Res = {Res})"
        fig_title = f"Question_10_{count}"
        
        count += 1
        if count == 4:
            count = 1

    # Question 11a Plots
    elif Alg_type == 11:
        # Initialize Controller Values
        robot = Controller(build_grid, Res, (s[0],s[1],-np.pi/2), (g[0],g[1]), 0.7, obstacle_locations, 0.5, 0.1, False, 0.2)
        
        # Extract Control Path
        control_path = robot.Online_PID_Controller()

        plot_title = f"Online Controller (Step 3 Paths) (Res = {Res})"
        fig_title = f"Question_11_1_{count}"
        
        count += 1
        if count == 4:
            count = 1

    # Question 11b Plots
    elif Alg_type == 12:
        # Initialize Controller Values
        robot = Controller(build_grid, Res, (s[0],s[1],-np.pi/2), (g[0],g[1]), 0.7, obstacle_locations, 0.5, 0.1, False, 0.2)

        # Extract Control Path
        control_path = robot.Online_PID_Controller()

        plot_title = f"Online Controller (Step 3 Paths) (Res = {Res})"
        fig_title = f"Question_11_2_{count}"
        
        count += 1
        if count == 4:
            count = 1

    # Initialize X, Y and Theta Arrays
    cx =[]
    cy = []
    ctheta = []

    # Extract X, Y and Theta Coordinates from Path
    for c in control_path:
        cx.append(c[0])
        cy.append(c[1])
        ctheta.append(c[2])

    # Color Map for Grid Position Values (i.e. 0,1,2,3,4)
    cmap = colors.ListedColormap(['white', 'black', 'orange', 'red', 'lime'])

    # Grid Width and Length for Plot Creation
    grid_width = robot.grid.shape[1]
    grid_height = robot.grid.shape[0]

    # Display 2D Grid
    plt.figure(figsize=(6,8))
    plt.imshow(robot.grid, cmap=cmap, origin='upper', extent=[-2, grid_width*Res - 2 , -6, grid_height*Res - 6])

    # If Resolution is Fine
    if Res < 1:
        
        # Shift Start and Goal Points by Res/2
        plt.scatter(cx[0] + Res/2, cy[0] + Res/2, c='r', edgecolor='k', label = f'Start {s}', zorder=3)
        plt.scatter(robot.xg[0] + Res/2, robot.xg[1] + Res/2, c='lime', edgecolor='k', label = f'Goal {g}', zorder=3)

    # If Resolution is Coarse
    else:  

        # Simply Plot Start and Goal Points
        plt.scatter(cx[0], cy[0], c='r', edgecolor='k', label = f'Start {s}', zorder=3)
        plt.scatter(robot.xg[0], robot.xg[1], c='lime', edgecolor='k', label = f'Goal {g}', zorder=3)

    # Add Arrows to Represent Position and Heading of Robot
    plt.plot(cx, cy, 'c',linewidth=2, zorder=1, label='Robot Path')
    num_arrow = 60
    arrow_theta = np.array(ctheta)
    x_arrow = np.cos(arrow_theta)
    y_arrow = np.sin(arrow_theta)

    # Plotting Orientated Robot Chassis
    plt.quiver(cx[::num_arrow],cy[::num_arrow],x_arrow[::num_arrow],y_arrow[::num_arrow],scale=40, color='blue', width=0.01, zorder=2)

    # Label Major Values on Axes (i.e. -6, -5.5, -5, etc.)
    ax = plt.gca()

    # Check that the Value to Label is a Multiple of 0.5 (Matplotlib Documentation)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}" if abs(x*2 - round(x*2)) < 1e-6 else ""))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}" if abs(y*2 - round(y*2)) < 1e-6 else ""))

    # Create Grid Lines
    x_ticks = np.arange(-2, grid_width*Res - 2, Res)
    y_ticks = np.arange(-6, grid_height*Res - 6, Res)
    plt.xticks(x_ticks, fontsize=10)
    plt.yticks(y_ticks, fontsize=10)
    plt.grid(True, color='gray', linewidth = Res*1.5)

    # Display Plot
    plt.title(plot_title)
    plt.xlabel("X [m]", fontsize=12)
    plt.ylabel("Y [m]", fontsize=12)
    plt.legend(fontsize=12)
    # plt.savefig('asgn1/' + fig_title, bbox_inches='tight')
    plt.show()