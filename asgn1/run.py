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

# A* Start Algorithm
def A_star (map_vals, res, start, goal):

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

            return closed_set, path

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

    return closed_set, []

# Online A* Cost Function - LRTA_Cost Function Referenced from Articial Intelligence Ch. 4.5.3
def Online_A_Star_Cost(map_vals, goal, res, s, a, s_prime, H):

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
    w,h,grid = build_grid([-2.0,5.0], [-6.0,6.0], Res, obstacle_locations)
    
    # If Alg_type Is 1 Use Online A*
    if Alg_type:
        path = Online_A_Star([w,h,grid], Res, s, g)
        
        if Res < 1:
            q_count = 7
        else: 
            q_count = 5
        
        plot_title = f"Online A* (Res = {Res})"
        fig_title = f"Question_{q_count}_{count}"
        
        count += 1
        if count == 4:
            count = 1
    
    # If Alg_type is 0 Use A_Star
    else:
        c_set, path = A_star([w,h,grid], Res, s, g)
                
        plot_title = f"A* (Res = {Res})"
        fig_title = f"Question_{q_count}_{count}"
        
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
    cmap = colors.ListedColormap(['white', 'black', 'lime', 'red', 'blue'])

    # Grid Width and Length for Plot Creation
    grid_width = grid.shape[1]
    grid_height = grid.shape[0]

    plt.figure(figsize=(6,10))
    # Display 2D Grid
    plt.imshow(grid, cmap=cmap, origin='upper', extent=[-2, grid_width*Res - 2 , -6, grid_height*Res - 6])

    # Label Major Values on Axes (i.e. -6, -5.5, -5, etc.)
    ax = plt.gca()

    # Check that the Value to Label is a Multiple of 0.5 
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}" if abs(x*2 - round(x*2)) < 1e-6 else ""))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}" if abs(y*2 - round(y*2)) < 1e-6 else ""))

    # Create Grid Lines
    x_ticks = np.arange(-2, grid_width*Res - 2, Res)
    y_ticks = np.arange(-6, grid_height*Res - 6, Res)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.grid(True, color='gray', linewidth = Res*1.5)

    # Display Plot
    plt.title(plot_title)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.savefig('asgn1/' + fig_title)
    plt.show()