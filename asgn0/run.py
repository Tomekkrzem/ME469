import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import sin, cos
import os

# GLOBAL VARIABLES
x_start = 1.29812900 
y_start = 1.88315210
theta_start = 2.82870000 


# FILEPATHS
# Filepath for Control Data
control_dir = os.path.dirname(os.path.abspath(__file__))
control_fp = os.path.join(control_dir, "ds0_Control.dat")

# Filepath for Robot Groundtruth Data
gt_dir = os.path.dirname(os.path.abspath(__file__))
groundtruth_fp = os.path.join(gt_dir, "ds0_Groundtruth.dat")

# Filepath for Groundtruth Landmark Positions
lm_dir = os.path.dirname(os.path.abspath(__file__))
landmark_fp = os.path.join(lm_dir, "ds0_Landmark_Groundtruth.dat")

# Filepath for Measurement Data
measurement_dir = os.path.dirname(os.path.abspath(__file__))
measurement_fp = os.path.join(measurement_dir, "ds0_Measurement.dat")

# Filepath for Barcode Data
barcode_dir = os.path.dirname(os.path.abspath(__file__))
barcode_fp = os.path.join(barcode_dir, "ds0_Barcodes.dat")



# POST-PROCESSED DATASETS
# Post-processed Control Data with 3 columns for duration time, linear velocity and angular velocity
control_data = pd.read_table(control_fp, sep=r'\s+', skiprows=3).to_numpy()

# Post-processed  Groundtruth Data with 4 columns for duration time, x-position, y-position and orientation
groundtruth_data = pd.read_table(groundtruth_fp, sep=r'\s+', skiprows=3).to_numpy()

# Post-processed  Landmark Data with 5 columns for subject landmark, x-positon, y-position, xstdev and ystdev
landmark_data = pd.read_table(landmark_fp, sep=r'\s+', skiprows=3).to_numpy()

# Post-processed Measurement Data with 4 columns for time, subject, range, and bearing
measurement_data = pd.read_table(measurement_fp, sep=r'\s+', skiprows=3).to_numpy()

# Post-processed Barcode Data with 2 columns for index, id
barcode_data = pd.read_table(barcode_fp, sep=r'\s+', skiprows=3).to_numpy()


lm_locations = [np.array([lm[1],lm[2]]) for lm in landmark_data]

bardcodes = pd.DataFrame(barcode_data, columns=["index","id"])
barcode_to_index = (bardcodes.groupby("id")["index"].apply(list).to_dict())

# Reducing dataset to have arrays of [id,range,bearing] for each timestep
# This step puts arrays values with the same timestep into a combined array 
id_extraction = pd.DataFrame(measurement_data, columns=["timestamp","id","range","bearing"])

timestamps = id_extraction["timestamp"].drop_duplicates().tolist()
id_vectors = (
    id_extraction.groupby("timestamp")[["id", "range", "bearing"]]   
      .apply(lambda g: list(map(tuple, g.values)))   
      .to_dict()
)


# Initializing arrays for reformatting the two velocities and timestep values
dt_arr = []
id_arr = dict()
v_arr = []
w_arr = []
time_index = 0
curr_time = 0

for i in range(len(control_data)-1):
    # Determine timestep between Controls at t+1 and t where t = 0. This omits the first control command
    time_stamp = (control_data[i+1][0] - control_data[i][0]).item()
    
    if curr_time < len(timestamps):
        if timestamps[curr_time] - control_data[i][0] < 0.02:

            id_arr[time_index] = id_vectors[timestamps[curr_time]]

            curr_time += 1

    time_index += 1

    # Updating the timestep and velocity arrays with data from Control Data
    dt_arr.append(time_stamp)
    v_arr.append(control_data[i+1][1].item())
    w_arr.append(control_data[i+1][2].item())

# Initializing arrays for groundtruth positions to plot
X_arr_true = []
Y_arr_true = []
theta_arr_true = []

for i in range(len(groundtruth_data)-1):
    # Populate arrays with data from the Groundtruth data
    X_arr_true.append(groundtruth_data[i+1][1].item())
    Y_arr_true.append(groundtruth_data[i+1][2].item())
    theta_arr_true.append(groundtruth_data[i+1][3].item())

# Motion Model Referenced Probabilistic Robotics Table 5.3 (sample_motion_model_velocity)
# Additionally referenced "CS W4733 NOTES - Differential Drive Robots" for Instantaneous Center of Rotation + Rotation Matrix
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


# Sampling Function Referenced from Probabilistic Robotics Table 5.4 (sample_normal_distribution)
def sample(b):

    return b/6 * np.random.uniform(-1,1,12).sum()

# Measurement Model Mostly Referenced from Probabilistic Robotics Table 6.4 (landmark_model_known_correspondence)
def z_t(X_t, lm_x, lm_y, lm_nums = []):

    """
    :param X_t: Current robot state
    :param X_i: Current landmark position
    
    """
    x_t, y_t, theta_t = X_t
    zt_pos_out = []

    if lm_nums == []:

        # Compute the distance to the landmark using Euclidean distance, and the bearing taking arctan of the coordinates minus the heading
        zt = [np.sqrt((x_t - lm_x)**2 + (y_t - lm_y)**2), 
                        np.atan2((lm_y - y_t),(lm_x - x_t)) - theta_t]

        zt[1] = (zt[1] + np.pi) % (2 * np.pi) - np.pi

        range_out = zt[0]
        bearing_out = zt[1]


    if lm_nums != []:
        zt_pos_out = []
        for n in lm_nums:

            # Extract x and y coordinates from the current landmark position
            x_i = lm_locations[n-6][0]
            y_i = lm_locations[n-6][1]

            # Compute the distance to the landmark using Euclidean distance, and the bearing taking arctan of the coordinates minus the heading
            zt = [np.sqrt((x_t - x_i)**2 + (y_t - y_i)**2), 
                            np.atan2((y_i - y_t),(x_i - x_t)) - theta_t]

            zt[1] = (zt[1] + np.pi) % (2 * np.pi) - np.pi

            range_out = zt[0]
            bearing_out = zt[1]
            
            # Compute the predicted landmark position 
            z_t_pos = [zt[0] * cos(zt[1]) + x_t, 
                                zt[0] * sin(zt[1]) + y_t]         

            zt_pos_out.append(z_t_pos)
            
            print(f'\nPredicted Distance to Landmark {n}: {round(range_out,4)} [m]\nPredicted Bearing to Landmark {n}: {round(bearing_out,4)} [rad]')
            print(f'Predicted Position of Landmark {n}: {(round(zt_pos_out[0][0],4).item(), round(zt_pos_out[0][1],4).item())} [m]')

    return range_out, bearing_out, zt_pos_out


# Gaussian PDF Referenced from Probabilistic Robotics Eqn. 2.3
def gaussian_distribution(x, mean, stdev):

    return 1 / (stdev * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x-mean)**2 / (stdev)**2))


# Algorithm Referenced from Probabilistic Robotics Table 4.3 (Particle_filter)
def particle_filter(X_t_p, W_p, u_t, dt, count):
    
    Xt = np.zeros((PARTICLE_SIZE,3))
    w = np.ones(PARTICLE_SIZE)

    norm_factor = 0

    if count in id_arr:

        landmark_list = []

        for id in id_arr[count]:
            
            indx = barcode_to_index[id[0]]

            lm_x = lm_locations[indx[0]-6][0]
            lm_y = lm_locations[indx[0]-6][1]

            actual_range = id[1]
            actual_bearing = id[2]

            landmark_list.append([lm_x, lm_y, actual_range, actual_bearing])

        for m in range(PARTICLE_SIZE):

            xt = x_t(X_t_p[m], u_t, dt)
            Xt[m] = xt

            wt = 1.0

            for l in range(len(landmark_list)):
                zt = z_t(xt, landmark_list[l][0], landmark_list[l][1])

                range_prob = gaussian_distribution(zt[0], actual_range, 0.4)
                bearing_prob = gaussian_distribution(zt[1], actual_bearing, 0.25)

                wt *= range_prob * bearing_prob

            norm_factor += wt
            w[m] = wt

    else:

        for m in range(PARTICLE_SIZE):

            xt = x_t(X_t_p[m], u_t, dt)
            Xt[m] = xt
            w[m] = W_p[m]

            norm_factor += w[m]

    if norm_factor <= 0:
        w = w / len(w)
    else:
        w = w / norm_factor     

    Xt, w = low_variance_sampler(Xt, w)

    return Xt, w


# Algorithm Referenced from Probabilistic Robotics Table 4.4 (Low_variance_sampler)
def low_variance_sampler(Xt, w):

    r = np.random.uniform(0, 1 / PARTICLE_SIZE)
    c = w[0]
    i = 0 

    Xt_sampled = np.zeros((PARTICLE_SIZE,3))
    W_sampled = np.zeros(PARTICLE_SIZE) / PARTICLE_SIZE

    for m in range(PARTICLE_SIZE):

        u = r + m * (1/PARTICLE_SIZE)

        while u > c:
            i += 1

            c = c + w[i]

        Xt_sampled[m] = Xt[i] 
        W_sampled[m] = w[i]  

    return Xt_sampled, W_sampled



# QUESTION 2 CODE:

a1 = 0
a2 = 0
a3 = 0
a4 = 0
a5 = 0
a6 = 0


# Initialize X and Y arrays for plotting
X_arr = [0]
Y_arr = [0]

# Calculate next robot state for v = 0.5 m/s, w = 0 rad/s, and t = 1s
x_o, y_o, theta_o = x_t(np.array([0,0,0]), np.array([0.5,0]), 1)
# Update X and Y arrays
X_arr.append(x_o)
Y_arr.append(y_o)

# Calculate next robot state for v = 0 m/s, w = -1/2pi rad/s, and t = 1s
x_o, y_o, theta_o = x_t(np.array([x_o, y_o, theta_o]), np.array([0,-1/(2*np.pi)]), 1)
# Update X and Y arrays
X_arr.append(x_o)
Y_arr.append(y_o)

# Calculate next robot state for v = 0.5 m/s, w = 0 rad/s, and t = 1s
x_o, y_o, theta_o = x_t(np.array([x_o, y_o, theta_o]), np.array([0.5,0]), 1)
# Update X and Y arrays
X_arr.append(x_o)
Y_arr.append(y_o)

# Calculate next robot state for v = 0 m/s, w = -1/2pi rad/s, and t = 1s
x_o, y_o, theta_o = x_t(np.array([x_o, y_o, theta_o]), np.array([0,1/(2*np.pi)]), 1)
# Update X and Y arrays
X_arr.append(x_o)
Y_arr.append(y_o)

# Calculate next robot state for v = 0.5 m/s, w = 0 rad/s, and t = 1s
x_o, y_o, theta_o = x_t(np.array([x_o, y_o, theta_o]), np.array([0.5,0]), 1)
# Update X and Y arrays
X_arr.append(x_o)
Y_arr.append(y_o)

# Plot output for Question 2
plt.figure(figsize=(12,8))
plt.plot(X_arr,Y_arr)
plt.xlabel("x-position")
plt.ylabel("y-position")
plt.title("Question 2: Motion Model Test Trajectory")
plt.savefig("Question_2_Plot.png")
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------------------#
# QUESTION 3 CODE:

# Initializing arrays for reformatting the two velocities and timestep values
time_arr = []
v_arr = []
w_arr = []

for i in range(len(control_data)-1):
    # Determine timestep between Controls at t+1 and t where t = 0. This omits the first control command
    time_dur = (control_data[i+1][0] - control_data[i][0]).item()

    # Updating the timestep and velocity arrays with data from Control Data
    time_arr.append(time_dur)
    v_arr.append(control_data[i+1][1].item())
    w_arr.append(control_data[i+1][2].item())

# Initializing arrays for groundtruth positions to plot
X_arr_true = []
Y_arr_true = []
theta_arr_true = []

for i in range(len(groundtruth_data)-1):
    # Populate arrays with data from the Groundtruth data
    X_arr_true.append(groundtruth_data[i+1][1].item())
    Y_arr_true.append(groundtruth_data[i+1][2].item())
    theta_arr_true.append(groundtruth_data[i+1][3].item())

# Control array of control velocities for all timesteps
u_t0 = [v_arr, w_arr]

# Starting position and orientation of robot
x_start = 1.29812900 
y_start = 1.88315210
theta_start = 2.82870000 

# Initial robot state to be treated as prior state input into motion model
x_t_c = np.array([x_start, y_start, theta_start])

# Populate arrays with inital robot state
X_arr = [x_start]
Y_arr = [y_start]
theta_arr = [theta_start]   

# Loop through all the control commands
for j in range(len(time_arr)):
    # Calculate current robot state from prior robot state, control velocities and timesteps
    x_t_c = x_t(x_t_c, np.array([u_t0[0][j], u_t0[1][j]]), time_arr[j])

    # Update arrays with current robot state
    X_arr.append(x_t_c[0])
    Y_arr.append(x_t_c[1])
    theta_arr.append(x_t_c[2])


# CODE FOR PLOTTING QUESTION 3:

# Number of points to skip before next robot marker is to be plotted
num_arrow = 150

plt.figure(figsize=(12,8))
# Dead reckoned trajectory
plt.plot(X_arr,Y_arr, 'c--', linewidth=1,  label= "Dead Reckoned")

# Updating orientation of robot chassis marker for plotting
arrow_theta = np.array(theta_arr)
x_arrow = np.cos(arrow_theta)
y_arrow = np.sin(arrow_theta)

# Plotting orientated robot chassis
plt.quiver(X_arr[::num_arrow],Y_arr[::num_arrow],x_arrow[::num_arrow],y_arrow[::num_arrow],scale=90,color='blue', width=0.005)

# Groundtruh trajectory
plt.plot(X_arr_true,Y_arr_true, color='yellow', linestyle = 'dashed', linewidth=1, label = "Ground Truth")
arrow_theta2 = np.array(theta_arr_true)
x_arrow2 = np.cos(arrow_theta2)
y_arrow2 = np.sin(arrow_theta2)
plt.quiver(X_arr_true[::num_arrow],Y_arr_true[::num_arrow],x_arrow2[::num_arrow],y_arrow2[::num_arrow],scale=90,color='orange', width=0.005)

plt.xlabel("x-position")
plt.ylabel("y-position")
plt.title("Question 3: Ground Truth vs. Dead Reckoned Trajectories")
plt.legend()
plt.savefig("Question_3_Plot.png")
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------------------#
# QUESTION 6 CODE:

# Provided robot positions for measuring landmarks
r_pos_1 = np.array([2,3,0])
r_pos_2 = np.array([0,3,0])
r_pos_3 = np.array([1,-2,0])

# Landmark coordinates 
l6_pos = np.array([landmark_data[0][1],landmark_data[0][2]])
l13_pos = np.array([landmark_data[7][1],landmark_data[7][2]])
l17_pos = np.array([landmark_data[11][1],landmark_data[11][2]])

# Predicted Landmark positions and %-error from measurement model
z_t_1 = z_t(r_pos_1,0,0,[6])
error1 = ((z_t_1[2][0] - l6_pos) / l6_pos) * 100
print(f'Actual Position of Landmark 6: {(round(l6_pos[0],4).item(), round(l6_pos[1],4).item())} [m]')
print(f'Percent Error between Actual and Predicted Landmark 6 Position: {(round(error1[0],4).item(),round(error1[1],4).item())}')

z_t_2 = z_t(r_pos_2,0,0,[13])
error2 = ((z_t_2[2][0] - l13_pos) / l13_pos) * 100
print(f'Actual Position of Landmark 13: {(round(l13_pos[0],4).item(), round(l13_pos[1],4).item())} [m]')
print(f'Percent Error between Actual and Predicted Landmark 13 Position: {(round(error2[0],4).item(),round(error2[1],4).item())}')

z_t_3 = z_t(r_pos_3,0,0,[17])
error3 = ((z_t_3[2][0] - l17_pos) / l17_pos) * 100
print(f'Actual Position of Landmark 13: {(round(l17_pos[0],4).item(), round(l17_pos[1],4).item())} [m]')
print(f'Percent Error between Actual and Predicted Landmark 17 Position: {(round(error3[0],4).item(),round(error3[1],4).item())}')

# CODE FOR PLOTTING QUESTION 6:

plt.figure(figsize=(12,8))
# Plotting robot chassis in space for landmark 6 prediction
plt.plot(r_pos_1[0],r_pos_1[1], 'D:r', ms = 15, label = "Robot Position for Landmark 6")

# Updating orientation of robot chassis marker for plotting
x_arw = np.cos(r_pos_1[2])
y_arw = np.sin(r_pos_1[2])
plt.quiver(r_pos_1[0],r_pos_1[1],x_arw,y_arw,scale=20,color='red', width=0.005)

# Plotting actual landmark 6 position
plt.plot(l6_pos[0],l6_pos[1], '*:r',  ms = 13, label = "Actual Landmark 6 Position")
# Plotting predicted landmark 6 position
plt.plot(z_t_1[2][0][0], z_t_1[2][0][1], 'o:r', mec = 'k', ms = 7, label = "Predicted Landmark 6 Position")

plt.plot(r_pos_2[0],r_pos_2[1], 'D:c', ms = 15, label = "Robot Position for Landmark 13")
plt.plot(l13_pos[0],l13_pos[1], '*:c', ms = 13, label = "Actual Landmark 13 Position")
plt.plot(z_t_2[2][0][0], z_t_2[2][0][1], 'o:c', mec = 'k', ms = 7, label = "Predicted Landmark 13 Position")
x_arw2 = np.cos(r_pos_2[2])
y_arw2 = np.sin(r_pos_2[2])
plt.quiver(r_pos_2[0],r_pos_2[1],x_arw2,y_arw2,scale=20,color='cyan', width=0.005)

plt.plot(r_pos_3[0],r_pos_3[1], 'D:g', ms = 15, label = "Robot Position for Landmark 17")
plt.plot(l17_pos[0],l17_pos[1], '*:g', ms = 13, label = "Actual Landmark 17 Position")
plt.plot(z_t_3[2][0][0], z_t_3[2][0][1], 'o:g', mec = 'k', ms = 7, label = "Predicted Landmark 17 Position")
x_arw3 = np.cos(r_pos_3[2])
y_arw3 = np.sin(r_pos_3[2])
plt.quiver(r_pos_3[0],r_pos_3[1],x_arw3,y_arw3,scale=20,color='green', width=0.005)

plt.xlabel("x-position")
plt.ylabel("y-position")
plt.title("Question 6: Ground Truth Landmarks Locations vs. Predicted Landmark Locations")
plt.legend()
plt.savefig("Question_6_Plot.png")
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------------------#
# QUESTION 8 CODE:

# QUESTION 8-1 
PARTICLE_SIZE = 300

a1 = 0.4
a2 = 0.4
a3 = 1.25
a4 = 1.25
a5 = 0.6
a6 = 0.6

X_arr = [0]
Y_arr = [0]

Xtp_S2 = np.full((PARTICLE_SIZE,3), np.array([0, 0, 0])) 
W_S2 = np.ones(len(Xtp_S2)) / PARTICLE_SIZE

# Calculate next robot state for v = 0.5 m/s, w = 0 rad/s, and t = 1s
Xtp_S2 = particle_filter(Xtp_S2, W_S2, np.array([0.5,0]),1,0)[0]
# Update X and Y arrays
X_arr.append(np.mean(Xtp_S2, axis=0)[0])
Y_arr.append(np.mean(Xtp_S2, axis=0)[1])

# Calculate next robot state for v = 0 m/s, w = -1/2pi rad/s, and t = 1s
Xtp_S2 = particle_filter(Xtp_S2, W_S2, np.array([0,-1/(2*np.pi)]),1,0)[0]
# Update X and Y arrays
X_arr.append(np.mean(Xtp_S2, axis=0)[0])
Y_arr.append(np.mean(Xtp_S2, axis=0)[1])

# Calculate next robot state for v = 0.5 m/s, w = 0 rad/s, and t = 1s
Xtp_S2 = particle_filter(Xtp_S2, W_S2, np.array([0.5,0]),1,0)[0]
# Update X and Y arrays
X_arr.append(np.mean(Xtp_S2, axis=0)[0])
Y_arr.append(np.mean(Xtp_S2, axis=0)[1])

# Calculate next robot state for v = 0 m/s, w = -1/2pi rad/s, and t = 1s
Xtp_S2 = particle_filter(Xtp_S2, W_S2, np.array([0,1/(2*np.pi)]),1,0)[0]
# Update X and Y arrays
X_arr.append(np.mean(Xtp_S2, axis=0)[0])
Y_arr.append(np.mean(Xtp_S2, axis=0)[1])

# Calculate next robot state for v = 0.5 m/s, w = 0 rad/s, and t = 1s
Xtp_S2 = particle_filter(Xtp_S2, W_S2, np.array([0.5,0]),1,0)[0]
# Update X and Y arrays
X_arr.append(np.mean(Xtp_S2, axis=0)[0])
Y_arr.append(np.mean(Xtp_S2, axis=0)[1])

# Plot output for Question 8-1
plt.figure(figsize=(12,8))
plt.plot(X_arr,Y_arr)
plt.xlabel("x-position")
plt.ylabel("y-position")
plt.title("Question 8: Particle Filter Test Trajectory")
plt.savefig("Question_8-1_Plot.png")
plt.show()




# QUESTION 8-2 

# Point Mass Representation of Starting Point
Xtp = np.full((PARTICLE_SIZE,3), np.array([x_start, y_start, theta_start])) 

# Populate arrays with inital robot state for plotting
X_arr = [x_start]
Y_arr = [y_start]
theta_arr = [theta_start]   

# Uniform distribution of weights for all particles
W_list = np.ones(len(Xtp)) / PARTICLE_SIZE

for i in range(len(dt_arr)):
    # Calculate current robot state from prior robot state, control velocities and timesteps
    Xtp, W_list = particle_filter(Xtp, W_list,[v_arr[i], w_arr[i]], dt_arr[i], i)

    Xt_best = np.mean(Xtp, axis=0)

    # Update arrays with current robot state
    X_arr.append(Xt_best[0])
    Y_arr.append(Xt_best[1])
    theta_arr.append(Xt_best[2])


# CODE FOR PLOTTING QUESTION 8-2:

# Number of points to skip before next robot marker is to be plotted
num_arrow = 150

plt.figure(figsize=(12,8))
# Dead reckoned trajectory
plt.plot(X_arr,Y_arr, 'c--', linewidth=1,  label= "Dead Reckoned")
# Updating orientation of robot chassis marker for plotting
arrow_theta = np.array(theta_arr)
x_arrow = np.cos(arrow_theta)
y_arrow = np.sin(arrow_theta)

# Plotting orientated robot chassis
plt.quiver(X_arr[::num_arrow],Y_arr[::num_arrow],x_arrow[::num_arrow],y_arrow[::num_arrow],scale=90,color='blue', width=0.005)

# Groundtruh trajectory
plt.plot(X_arr_true,Y_arr_true, color='yellow', linestyle = 'dashed', linewidth=1, label = "Ground Truth")
arrow_theta2 = np.array(theta_arr_true)
x_arrow2 = np.cos(arrow_theta2)
y_arrow2 = np.sin(arrow_theta2)
plt.quiver(X_arr_true[::num_arrow],Y_arr_true[::num_arrow],x_arrow2[::num_arrow],y_arrow2[::num_arrow],scale=90,color='orange', width=0.005)

plt.scatter(x_start,y_start,label="START")
plt.scatter(X_arr[-1],Y_arr[-1],label="Dead-Reckoned End")
plt.scatter(X_arr_true[-1],Y_arr_true[-1],label="Groundtruth END")

plt.xlabel("x-position")
plt.ylabel("y-position")
plt.title("Question 8: Ground Truth vs. Dead Reckoned Trajectories for Particle Filter")
plt.legend()
plt.savefig("Question_8-2_Plot.png")
plt.show()