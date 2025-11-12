import numpy as np
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

class Grid:

    def __init__(self, X, Y, res, O):

        self.res = res

        self.x_range = X
        self.y_range = Y

        self.obstacles = O

        # If Resolution is Fine
        if self.res < 1:
            # X-Value Linespace
            self.width = np.linspace(self.x_range[0],self.x_range[1], int((self.x_range[1] - self.x_range[0])/self.res) + 1)
            # Y-Value Linespace
            self.height = np.flip(np.linspace(self.y_range[0],self.y_range[1], int((self.y_range[1] - self.y_range[0])/self.res) + 1))

            # Truncate the Values in the Linespaces
            self.width = np.floor(np.round(self.width,decimals = 1) * 10) / 10
            self.height = np.ceil(np.round(self.height,decimals = 1) * 10) / 10

            # Initialize 2D Grid of Empty Positions
            self.Grid = np.zeros((len(self.height),len(self.width)))

            # Create Inflation Array for Obstacles
            self.obst_arr = np.ones((7,7))

        # Else If Resolution is Coarse
        else: 
            # X-Value Linespace
            self.width = np.linspace(self.x_range[0],self.x_range[1]-1, int((self.x_range[1] - self.x_range[0])/self.res))
            # Y-Value Linespace (Had to Be Flipped for Correct Orientation)
            self.height = np.flip(np.linspace(self.y_range[0],self.y_range[1]-1, int((self.y_range[1] - self.y_range[0])/self.res)))

            # Truncate the Values in the Linespaces
            self.width = np.floor(self.width)
            self.height = np.floor(self.height)

            # Initializing 2D Grid of Empty Positions
            self.Grid = np.zeros((len(self.height),len(self.width)))


    # Builds a Graph Populated with Obstacles
    def Build_Grid(self):

        # Add Obstacles to Grid
        for o in self.obstacles:

            # Retrieve Obstacle Position
            x,y = o

            # If Resouliton is Fine
            if self.res < 1:

                # Truncate the Position Coordiantes
                x = np.floor(x * 10) / 10
                y = np.floor(y * 10) / 10

                # Obstacle Inflation Offset
                offset = 0.3

                # Determine Left X-Coordinate Index of Inflation Obstacle
                x_indx = int(np.where(self.width == round(x - offset, 1))[0][0])

                # Determine Top Y-Coordinate Index of Inflated Obstacle
                y_indx = int(np.where(self.height == round(y - offset, 1))[0][0])

                # Add Inflated Obstacle by Replacing Values in Grid with Corresponding X and Y Coordinates
                self.Grid[y_indx - 6 : y_indx + 1,
                        x_indx : x_indx + 7] = self.obst_arr

            # If Resouliton is Coarse
            else: 
                
                # Truncate the Position Coordiantes
                x = np.floor(x)
                y = np.floor(y)

                # Compute Corresponding Map Position of Obstacle
                x_indx = int(np.where(self.width == x)[0][0])
                y_indx = int(np.where(self.height == y)[0][0])

                # Add Obstacles to Map by Setting Value at Associated Position to 1
                self.Grid[y_indx][x_indx] = 1

    def Plot_Grid(self, grid, Title):
        
        self.Build_Grid()

        # Color Map for Grid Position Values (i.e. 0,1,2,3,4)
        cmap = colors.ListedColormap(['white', 'black', 'orange', 'red', 'lime'])

        g_H, g_W = self.Grid.shape

        # Display 2D Grid
        plt.figure(figsize=(6,8))
        plt.imshow(grid, cmap=cmap, origin='upper', extent=[-2, g_W * self.res - 2 , -6, g_H * self.res - 6])

        # Label Major Values on Axes (i.e. -6, -5.5, -5, etc.)
        ax = plt.gca()

        # Check that the Value to Label is a Multiple of 0.5 (Matplotlib Documentation)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}" if abs(x*2 - round(x*2)) < 1e-6 else ""))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}" if abs(y*2 - round(y*2)) < 1e-6 else ""))

        # Create Grid Lines
        x_ticks = np.arange(-2, g_W * self.res - 2, self.res)
        y_ticks = np.arange(-6, g_H * self.res - 6, self.res)
        plt.xticks(x_ticks, fontsize=10)
        plt.yticks(y_ticks, fontsize=10)
        plt.grid(True, color='gray', linewidth = self.res * 1.5)

        # Display Plot
        plt.title("Grid World")
        plt.xlabel("X [m]", fontsize=12)
        plt.ylabel("Y [m]", fontsize=12)
        plt.legend(fontsize=12)
        plt.savefig('asgn2/' + Title, bbox_inches='tight')
        plt.show()


class Q_Learning:

    def __init__(self, num_rows, num_cols, start_pos, goal_pos, num_episodes, alpha, gamma, epsilon, res, GW):
        """
        :param num_rows: Number of Rows in Grid
        :param num_cols: Number of Cols in Grid
        :param num_episodes: Number of Training Episodes
        :param goal_pos: Goal Position
        :parma start_pos: Start Position
        :param alpha: Learning Rate
        :param gamma: Discount Factor
        :param epsilon: Exploration Probability 
        :param res: Resolution of Gridworld
        """

        # Action Set
        self.A = [(-res,0),(res,0),(0,res),(0,-res),(-res,res),(res,res),(res,-res),(-res,-res)]
        self.num_actions = len(self.A)

        self.res = res

        # Initialize all State-Action Pairs in Q-Table with Zeros
        self.Q_table = np.zeros((num_rows, num_cols, self.num_actions))

        # If the Resolution is Fine Floor the Start and Goal Position Values to One Decimal
        if res < 1: 
            self.xg = (np.floor(goal_pos[0] * 10) / 10, np.floor(goal_pos[1] * 10) / 10)
            self.start = (np.floor(start_pos[0] * 10) / 10, np.floor(start_pos[1] * 10) / 10)

        # If the Resolution is Coarses Floor the Start and Goal Position to Zero Decimals
        else: 
            self.xg = (np.floor(goal_pos[0]), np.floor(goal_pos[1]))
            self.start = (np.floor(start_pos[0]), np.floor(start_pos[1]))

        # Initialize Learning Rate, Discount Factor and Exploration Probability
        self.a = alpha
        self.g = gamma
        self.e = epsilon

        # Initialize Number of Episodes
        self.episodes = num_episodes

        self.Gridworld = GW

    def get_Grid_Indices(self, xt):

        _, width, height = self.Gridworld

        x_indx = int(np.where(width == xt[0])[0][0])
        y_indx = int(np.where(height == xt[1])[0][0])

        return x_indx, y_indx

    def Reward(self, xt):

        grid, _, _ = self.Gridworld

        x_indx, y_indx = self.get_Grid_Indices(xt)

        goal_dist = math.hypot(xt[0] - self.xg[0], xt[1] - self.xg[1])
        r = -goal_dist * 0.1
        
        if grid[y_indx, x_indx] == 1:
            r = -10
        elif xt == self.xg:
            r = 10

        return r

    def Train_Q_Learning(self):
        
        _, width, height = self.Gridworld

        for episode in range(self.episodes):
            
            print(episode)

            curr_xt = self.start

            for _ in range(3000):
                # Extract Grid Indices of Neighbor Node
                x_indx, y_indx = self.get_Grid_Indices(curr_xt)

                # Select Action 
                if np.random.rand() < self.e:

                    A_indx = np.random.randint(0, self.num_actions)

                    action = self.A[A_indx]

                else: 

                    A_indx = np.argmax(self.Q_table[y_indx, x_indx])

                    action = self.A[A_indx]

                # If Resolution is Fine Floor the Neighbor X and Y Coordinate Values to One Decimal
                if self.res < 1: 
                    nx = float(round((np.floor(curr_xt[0] * 10) / 10) + action[0],1))
                    ny = float(round((np.floor(curr_xt[1] * 10) / 10) + action[1],1))
                    
                # If Resolution is Coarse Floor the Neighbor X and Y Coordinate Values to Zero Decimals
                else:
                    nx = int(np.floor(curr_xt[0] + action[0]))
                    ny = int(np.floor(curr_xt[1] + action[1]))


                if not (width[0] <= nx <= width[-1] and height[-1] <= ny <= height[0]):
                    next_xt = curr_xt 
                    r = -10        

                else:
                    next_xt = (nx, ny)
                    r = self.Reward(next_xt)
            
                nx_indx, ny_indx = self.get_Grid_Indices(next_xt)

                self.Q_table[y_indx, x_indx, A_indx] += self.a * (r + self.g * np.max(self.Q_table[ny_indx, nx_indx]) - self.Q_table[y_indx, x_indx, A_indx])

                if next_xt == self.xg:
                    break
                
                curr_xt = next_xt

    def Optimal_Path(self):

        path = [self.start]

        curr_xt = self.start

        closed_set = []

        _, width, height = self.Gridworld

        for step in range(100):

            if curr_xt == self.xg:
                break

            closed_set.append(curr_xt)

            x_indx, y_indx = self.get_Grid_Indices(curr_xt)

            A_indx = np.argmax(self.Q_table[y_indx, x_indx])

            action = self.A[A_indx]

            # If Resolution is Fine Floor the Neighbor X and Y Coordinate Values to One Decimal
            if self.res < 1: 
                nx = float(round((np.floor(curr_xt[0] * 10) / 10) + action[0],1))
                ny = float(round((np.floor(curr_xt[1] * 10) / 10) + action[1],1))
                
            # If Resolution is Coarse Floor the Neighbor X and Y Coordinate Values to Zero Decimals
            else:
                nx = int(np.floor(curr_xt[0] + action[0]))
                ny = int(np.floor(curr_xt[1] + action[1]))

            if (width[0] <= nx <= width[-1] and height[-1] <= ny <= height[0]):

                next_xt = (nx, ny)

            else:

                next_xt = curr_xt

            path.append(next_xt)
            curr_xt = next_xt

        return path


G = Grid([-2.0,5.0], [-6.0,6.0], 0.1, obstacle_locations)
Gh, Gw = G.Grid.shape
G.Build_Grid()
QL = Q_Learning(Gh, Gw, (0.5,-1.5),(0.5,1.5), 4000, 0.8, 0.95, 0.3, 0.1,[G.Grid, G.width, G.height])
QL.Train_Q_Learning()
path = QL.Optimal_Path()

# Update Path Positons in Grid
for i,p in enumerate(path):
    
    # Find Corresponding Grid Index of Path Position
    x_indx = int(np.where(G.width == p[0])[0][0])
    y_indx = int(np.where(G.height == p[1])[0][0])

    # If Position is Start Color it Red
    if i == 0:
        G.Grid[y_indx][x_indx] = 3    # 3 = Red
        
    # If Position is Goal Color it Red
    elif i == len(path) - 1:
        G.Grid[y_indx][x_indx] = 4    # 4 = Blue
        
    # Otherwise Color the Path green
    else:
        G.Grid[y_indx][x_indx] = 2

G.Plot_Grid(G.Grid, "Question5a")

G1 = Grid([-2.0,5.0], [-6.0,6.0], 0.1, obstacle_locations)
Gh, Gw = G1.Grid.shape
G1.Build_Grid()
QL1 = Q_Learning(Gh, Gw, (-0.55,1.45),(1.95,3.95), 4000, 0.8, 0.95, 0.3, 0.1,[G1.Grid, G1.width, G1.height])
QL1.Train_Q_Learning()
path = QL1.Optimal_Path()

# Update Path Positons in Grid
for i,p in enumerate(path):
    
    # Find Corresponding Grid Index of Path Position
    x_indx = int(np.where(G1.width == p[0])[0][0])
    y_indx = int(np.where(G1.height == p[1])[0][0])

    # If Position is Start Color it Red
    if i == 0:
        G1.Grid[y_indx][x_indx] = 3    # 3 = Red
        
    # If Position is Goal Color it Red
    elif i == len(path) - 1:
        G1.Grid[y_indx][x_indx] = 4    # 4 = Blue
        
    # Otherwise Color the Path green
    else:
        G1.Grid[y_indx][x_indx] = 2

G1.Plot_Grid(G1.Grid, "Question5b")

G2 = Grid([-2.0,5.0], [-6.0,6.0], 0.1, obstacle_locations)
Gh, Gw = G2.Grid.shape
G2.Build_Grid()
QL2 = Q_Learning(Gh, Gw, (4.95,-0.05),(2.45,0.25), 4000, 0.8, 0.95, 0.3, 0.1,[G2.Grid, G2.width, G2.height])
QL2.Train_Q_Learning()
path = QL2.Optimal_Path()

# Update Path Positons in Grid
for i,p in enumerate(path):
    
    # Find Corresponding Grid Index of Path Position
    x_indx = int(np.where(G2.width == p[0])[0][0])
    y_indx = int(np.where(G2.height == p[1])[0][0])

    # If Position is Start Color it Red
    if i == 0:
        G2.Grid[y_indx][x_indx] = 3    # 3 = Red
        
    # If Position is Goal Color it Red
    elif i == len(path) - 1:
        G2.Grid[y_indx][x_indx] = 4    # 4 = Blue
        
    # Otherwise Color the Path green
    else:
        G2.Grid[y_indx][x_indx] = 2

G2.Plot_Grid(G2.Grid, "Question5c")