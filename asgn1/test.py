import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import os

res = 0.1


def LRTA_Cost(map_vals, goal, res, s, a, s_prime, H):

    w, h, Map = map_vals

    if s_prime is None:

        if res < 1: 
            nx = round((np.floor(s[0] * 10) / 10) + a[0],1)
            ny = round((np.floor(s[1] * 10) / 10) + a[1],1)
        else:
            nx = np.floor(s[0] + a[0])
            ny = np.floor(s[1] + a[1])

        print(ny)

        map_x = int(np.where(w == nx)[0][0])
        map_y = int(np.where(h == ny)[0][0])

        if Map[map_y][map_x] == 1:
            a_cost = 1000
        else: 
            a_cost = 1

        s = (nx, ny)

        return Heuristic(s, goal)

    else: 

        if res < 1: 
            nx = round((np.floor(s_prime[0] * 10) / 10) + a[0],1)
            ny = round((np.floor(s_prime[1] * 10) / 10) + a[1],1)
        else:
            nx = np.floor(s_prime[0] + a[0])
            ny = np.floor(s_prime[1] + a[1])

        map_x = int(np.where(w == nx)[0][0])
        map_y = int(np.where(h == ny)[0][0])

        if Map[map_y][map_x] == 1:
            a_cost = 1000
        else: 
            a_cost = 1

        return a_cost + H[s_prime]   

def Heuristic(node_s, node_g):
    
    # Euclidean Distance
    return np.sqrt((node_g[0] - node_s[0])**2 + (node_g[1] - node_s[1])**2)


def Online_A_Star(map_vals, goal, res, start):

    s_prime = start

    s = None
    a = None

    resulting_states = dict()
    H = dict()

    parent = {start: None}

    # 8 Neighbor Directions to Check
    neighbor_dirs = [(-res,-res),(-res,0),(-res,res),(0,res),(res,res),(res,0),(res,-res),(0,-res)]

    while True:
        
        if s_prime == goal:
            path = [goal]

            child = goal

            while parent[child]:

                path.append(parent[child])
                child = parent[child]

            path.reverse()

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

        parent[s_prime] = s

        print(s_prime)

x_range = (0,4)
y_range = (0,4)

width = np.linspace(x_range[0],x_range[1], int((x_range[1] - x_range[0])/res) + 1)
# Y-Value Linespace
height = np.flip(np.linspace(y_range[0],y_range[1], int((y_range[1] - y_range[0])/res) + 1))

# Truncate the Values in the Linespaces
width = np.floor(np.round(width,decimals = 1) * 10) / 10
height = np.floor(np.round(height,decimals = 1) * 10) / 10

# Initializing 2D Grid of Empty Positions
grid_vals = np.zeros((len(height),len(width)))

map_vals = (width, height, grid_vals)

start = (1,2)
goal = (2,2)

print(Online_A_Star(map_vals,goal,res,start))




