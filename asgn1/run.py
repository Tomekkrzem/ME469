import numpy as np
from numpy import cos,sin
import math
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


# # Grid Resolutions
# Res = 1
# Res2 = 0.1

# # Coarse Grid Representation Variables
# c_w, c_h, c_grid = build_grid([-2.0,5.0], [-6.0,6.0], Res, obstacle_locations)
# # Fine Grid Representation Variables
# f_w, f_h, f_grid = build_grid([-2.0,5.0], [-6.0,6.0], Res2, obstacle_locations)

# s_g_list_step = [[Res,(0.5,-1.5),(0.5,1.5),0],       # STEP 3 START-GOAL POSITIONS
#                  [Res,(4.5,3.5),(4.5,-1.5),0],
#                  [Res,(-0.5,5.5),(1.5,-3.5),0],
                  
#                  [Res,(0.5,-1.5),(0.5,1.5),1],       # STEP 5 START-GOAL POSITIONS
#                  [Res,(4.5,3.5),(4.5,-1.5),1],
#                  [Res,(-0.5,5.5),(1.5,-3.5),1],
                  
#                  [Res2,(2.45,-3.55),(0.95,-1.55),1], # STEP 7 START-GOAL POSITIONS
#                  [Res2,(4.95,-0.05),(2.45,0.25),1],
#                  [Res2,(-0.55,1.45),(1.95,3.95),1]]

# # Question Number
# q_count = 3
# # Question Part Number
# count = 1

# # For Each Start-Goal Pair
# for s_g in s_g_list_step:
    
#     # Extract Resolution, Start and Goal
#     Res, s, g, Alg_type = s_g
    
#     # Grid Representation Variables
#     if Res < 1:
#         w = f_w
#         h = f_h
#         grid = f_grid.copy()
#     else:
#         w = c_w
#         h = c_h
#         grid = c_grid.copy()

#     # If Alg_type Is 1 Use Online A*
#     if Alg_type:
#         path = Online_A_Star([w,h,grid], Res, s, g)
        
#         if Res < 1:
#             q_count = 7
#         else: 
#             q_count = 5
        
#         plot_title = f"Online A* (Res = {Res})"
#         fig_title = f"Question_{q_count}_{count}"
        
#         count += 1
#         if count == 4:
#             count = 1
    
#     # If Alg_type is 0 Use A_Star
#     else:
#         c_set, path = A_star([w,h,grid], Res, s, g)
                
#         plot_title = f"A* (Res = {Res})"
#         fig_title = f"Question_{q_count}_{count}"
        
#         count += 1
#         if count == 4:
#             count = 1
    
#     # Update Path Positons in Grid
#     for i,p in enumerate(path):
        
#         # Find Corresponding Grid Index of Path Position
#         x_indx = int(np.where(w == p[0])[0][0])
#         y_indx = int(np.where(h == p[1])[0][0])

#         # If Position is Start Color it Red
#         if i == 0:
#             grid[y_indx][x_indx] = 3    # 3 = Red
            
#         # If Position is Goal Color it Red
#         elif i == len(path) - 1:
#             grid[y_indx][x_indx] = 4    # 4 = Blue
            
#         # Otherwise Color the Path green
#         else:
#             grid[y_indx][x_indx] = 2

#     # Color Map for Grid Position Values (i.e. 0,1,2,3,4)
#     cmap = colors.ListedColormap(['white', 'black', 'lime', 'red', 'blue'])

#     # Grid Width and Length for Plot Creation
#     grid_width = grid.shape[1]
#     grid_height = grid.shape[0]

#     plt.figure(figsize=(6,10))
#     # Display 2D Grid
#     plt.imshow(grid, cmap=cmap, origin='upper', extent=[-2, grid_width*Res - 2 , -6, grid_height*Res - 6])

#     # Label Major Values on Axes (i.e. -6, -5.5, -5, etc.)
#     ax = plt.gca()

#     # Check that the Value to Label is a Multiple of 0.5 
#     ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}" if abs(x*2 - round(x*2)) < 1e-6 else ""))
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}" if abs(y*2 - round(y*2)) < 1e-6 else ""))

#     # Create Grid Lines
#     x_ticks = np.arange(-2, grid_width*Res - 2, Res)
#     y_ticks = np.arange(-6, grid_height*Res - 6, Res)
#     plt.xticks(x_ticks)
#     plt.yticks(y_ticks)
#     plt.grid(True, color='gray', linewidth = Res*1.5)

#     # Display Plot
#     plt.title(plot_title)
#     plt.xlabel("X [m]")
#     plt.ylabel("Y [m]")
#     plt.savefig('asgn1/' + fig_title)
#     plt.show()


#--------------------------------------------------------------------------------------------------------------------------#
class Controller:

    def __init__(self, start):
        
        # Starting State
        self.xt = start

        # Initilialize Errors
        self.e_v = None         # Linear Error
        self.e_w = None         # Angular Error

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
        self.robot_path = [[self.xt[0], self.xt[1]]]

def bezier_curve(c_points, num_points, check_collisions):

    n = len(c_points) - 1

    t = np.linspace(0, 1, num_points)

    output_curve = np.zeros((num_points, 2))
    offset = 0.3

    obstacles = [(o[0], o[1], 0.8, 0.8) for o in obstacle_locations]

    for i in range(num_points):
        for j in range(n + 1):
            
            # Bezier Curve Equation Referenecd From "Path Planning based on Bezier Curve for Autonomous Ground Vehicles"
            output_curve[i] += c_points[j] * math.comb(n, j) * (t[i] ** j) * ((1 - t[i]) ** (n - j)) 

        if check_collisions:
            # Collision Checking
            for o in obstacles:
                
                p = output_curve[i]
                ox, oy, w, h = o

                # X-Distance Between Point and Obstacle Center
                x_to_cx_dist = p[0] - ox
                # Y-Distance Between Point and Obstacle Center
                y_to_cy_dist = p[1] - oy

                if abs(x_to_cx_dist) <= w/2 and abs(y_to_cy_dist) <= h/2:
                    if w/2 - abs(x_to_cx_dist) < h/2 - abs(y_to_cy_dist):
                        output_curve[i][0] = ox + np.sign(x_to_cx_dist) * (w/2 + offset)
                    else:
                        output_curve[i][1] = oy + np.sign(y_to_cy_dist) * (h/2 + offset)

    return output_curve


def feed_forward_control(start, goal, T, dt):

    # Preprocess Trajectory Using Bezier Curve
    smooth_T = bezier_curve(T, int(len(T)/(dt*10)),1)
    smooth_T = bezier_curve(smooth_T, int(len(smooth_T)/(dt)), 0)

    # Compose Feedforward States for Robot
    ffwd_points = np.zeros((len(smooth_T), 3))
    ffwd_points[0] = (start[0], start[1], start[2])

    for i in range(len(smooth_T) - 1):

        x,y = smooth_T[i+1]
        x_p, y_p = smooth_T[i]
        theta = np.atan2((y - y_p),(x - x_p))
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        ffwd_points[i+1] = (x,y, theta)

    Kp_v = 0.5
    Kp_w = 1
    Ki_v = 0.0001
    Ki_w = 0.0005
    Kd_v = 0.5
    Kd_w = 1

    # Current State
    xt = start

    # Initilialize Errors
    e_v = None
    e_w = None

    v_prev = 0
    w_prev = 0
    
    prev_e_v = None
    prev_e_w = None

    e_v_intg = 0
    e_w_intg = 0

    robot_path = [[xt[0],xt[1]]]

    # Loop Until Robot Reaches Completes Trajectory
    for i,p in enumerate(ffwd_points):

        # Compute Error Between Desired State and Actual State
        x_e = p[0] - xt[0]
        y_e = p[1] - xt[1]
        theta = xt[2]

        e_v = np.sqrt(x_e**2 + y_e**2)
 
        theta_desired = np.atan2(p[1] - xt[1], p[0] - xt[0])
        e_w = theta_desired - theta
        e_w = (e_w + np.pi) % (2 * np.pi) - np.pi

        # Compute Derivative (i.e. Change between Current and Previous State)
        if prev_e_v is not None and prev_e_w is not None: 
            d_ev = e_v - prev_e_v 
            d_ew = e_w - prev_e_w
        else: 
            d_ev = 0 
            d_ew = 0

        # Compute Velocity
        v = Kp_v * e_v + Ki_v * e_v_intg + Kd_v * d_ev 
        w = Kp_w * e_w + Ki_w * e_w_intg + Kd_w * d_ew 
        w = (w + np.pi) % (2 * np.pi) - np.pi

        # Limit the Accelerations
        dv = np.clip(v - v_prev, -0.288 * dt, 0.288 * dt)
        dw = np.clip(w - w_prev, -5.579 * dt, 5.579 * dt)

        # update velocities respecting limits
        v = v_prev + dv
        w = w_prev + dw

        # Update Previous Errors
        prev_e_v = e_v
        prev_e_w = e_w

        # Update Previous Velocity
        v_prev = v
        w_prev = w

        # Compute Next State Position
        xt = x_t(xt,[v,w],dt)

        # Update Error Integral
        e_v_intg += e_v
        e_w_intg += e_w

        robot_path.append([xt[0],xt[1]])

    return robot_path, ffwd_points


# Sampling Function Referenced from Probabilistic Robotics Table 5.4 (sample_normal_distribution)
def sample(b):

    return b/6 * np.random.uniform(-1,1,12).sum()

def x_t(X_t_p, u_t, dt):

    a1 = a2 = a3 = a4 = a5 = a6 = 0

    """
    :param X_t_p: Current robot state [x, y, theta]
    :param u_t: Control input [v, w] 
    :param dt: Time step
    """
    # Add motion noise to translational and rotational velocities
    v_hat = u_t[0] + sample(a1 * abs(u_t[0]) + a2 * abs(u_t[1]))
    w_hat = u_t[1] + sample(a3 * abs(u_t[0]) + a4 * abs(u_t[1]))
    gamma = sample(a5 * abs(u_t[0]) + a6 * abs(u_t[1]))

    # Straight-line motion when rotational speed is zero
    if w_hat == 0:
        x_dt = np.array([v_hat * cos(X_t_p[2]) * dt, v_hat * sin(X_t_p[2]) * dt, 0])

    # Circular motion: compute Instantaneous Center of Rotation (ICR)
    else:
        # Instantaneous Center of Rotation (ICR)
        xc = X_t_p[0] - (v_hat/w_hat)*sin(X_t_p[2])
        yc = X_t_p[1] + (v_hat/w_hat)*cos(X_t_p[2])

        # Rotation matrix for turning around ICR
        R = np.array([[cos(w_hat*dt), -sin(w_hat*dt), 0],
                      [sin(w_hat*dt),  cos(w_hat*dt), 0],
                      [0        ,          0, 1]])

        # Apply rotation and translation relative to ICR
        x_dt = np.matmul(R,np.array([X_t_p[0] - xc, 
                                     X_t_p[1] - yc,
                                     w_hat * dt + gamma * dt]))

        # Shift reference to ICR
        X_t_p = np.array([xc,yc,X_t_p[2]])

    # Compute new state by adding displacement
    x_t = X_t_p + x_dt
    
    # Wrap Angles
    x_t[2] = (x_t[2] + np.pi) % (2 * np.pi) - np.pi     

    return x_t

Res2 = 0.1

w,h,grid = build_grid([-2.0,5.0], [-6.0,6.0], Res2, obstacle_locations)



# path = Online_A_Star([w,h,grid], Res2, (2.45,-3.55),(0.95,-1.55))
# path = Online_A_Star([w,h,grid], Res2, (4.95,-0.05),(2.45,0.25))
# path = Online_A_Star([w,h,grid], Res2, (-0.55,1.45),(1.95,3.95))

# x_path = [p[0] for p in path]
# y_path = [p[1] for p in path]

# for i,p in enumerate(path):

#         # Find Corresponding Grid Index of Path Position
#         x_indx = int(np.where(w == p[0])[0][0])
#         y_indx = int(np.where(h == p[1])[0][0])

#         # If Position is Start Color it Red
#         if i == 0:
#             grid[y_indx][x_indx] = 3    # 3 = Red
            
#         # If Position is Goal Color it Red
#         elif i == len(path) - 1:
#             grid[y_indx][x_indx] = 4    # 4 = Blue
            
#         # Otherwise Color the Path green
#         else:
#             grid[y_indx][x_indx] = 2

# control_path, ffwd = feed_forward_control((2.45,-3.55,-np.pi/2),(0.95,-1.55,np.pi/2),np.array(path),0.1)
# control_path, ffwd = feed_forward_control((4.95,-0.05,-np.pi/2),(2.45,0.25,np.pi/2),np.array(path),0.1)
# control_path, ffwd = feed_forward_control((-0.55,1.45,-np.pi/2),(1.95,3.95,np.pi/2),np.array(path),0.1)

# cx =[]
# cy = []

# for c in control_path:
#     cx.append(c[0])
#     cy.append(c[1])




# Color Map for Grid Position Values (i.e. 0,1,2,3,4)
cmap = colors.ListedColormap(['white', 'black', 'lime', 'red', 'blue'])

# Grid Width and Length for Plot Creation
grid_width = grid.shape[1]
grid_height = grid.shape[0]

plt.figure(figsize=(6,10))
# Display 2D Grid
plt.imshow(grid, cmap=cmap, origin='upper', extent=[-2, grid_width*Res2 - 2 , -6, grid_height*Res2 - 6])

# Label Major Values on Axes (i.e. -6, -5.5, -5, etc.)
ax = plt.gca()

# Check that the Value to Label is a Multiple of 0.5 
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}" if abs(x*2 - round(x*2)) < 1e-6 else ""))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}" if abs(y*2 - round(y*2)) < 1e-6 else ""))

# Create Grid Lines
x_ticks = np.arange(-2, grid_width*Res2 - 2, Res2)
y_ticks = np.arange(-6, grid_height*Res2 - 6, Res2)
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.grid(True, color='gray', linewidth = Res2*1.5)

# Display Plot
plt.title("Test")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")

plt.plot(ffwd[:, 0], ffwd[:, 1])
plt.plot(cx, cy)
plt.show()

