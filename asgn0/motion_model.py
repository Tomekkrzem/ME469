import numpy as np
from numpy import sin, cos
import random

# def x_t(x, y, theta, v, w, dt):

#     """
#     :param x: Starting x-coordinate
#     :param y: Starting y-coordinate
#     :param theta: Starting heading position of robot
#     :param v: Translational Speed (Constant)
#     :param w: Rotational Speed (Constant)
#     :param t: Duration time of the commands (Constant)
    
#     """

#     x_arr = [x]
#     y_arr = [y]
#     theta_arr = [theta]

#     q_in = np.array([x,y,theta])

#     vx = v * cos(theta)
#     vy = v * sin(theta)

#     # OLD SOLUTION 3
#     if w == 0:
#         q = np.array([vx*dt, vy*dt, 0])
#     else:
#         R = np.array([[cos(w*dt), -sin(w*dt), 0],
#                       [sin(w*dt),  cos(w*dt), 0],
#                       [0        ,          0, 1]])

#         q = np.matmul(R,np.array([vx*dt, vy*dt, w*dt]))

#     q_out = q_in + q

#     # if w == 0:
#     #     q_dot = np.array([vx, vy, 0])
#     # else:
#     #     R = np.array([[cos(w*dt), -sin(w*dt), 0],
#     #                   [sin(w*dt),  cos(w*dt), 0],
#     #                   [0        ,          0, 1]])

#     #     q_dot = np.matmul(R,np.array([vx, vy, w]))

#     # q_out = rk4(q_dot, q_in, dt)

#     x = q_out[0]
#     y = q_out[1]
#     theta = q_out[2]

#     # OLD SOLUTION 1
#     # x = x + v * cos(theta) * dt
#     # y = y + v * sin(theta) * dt
#     # theta = theta + w * dt

#     # OLD SOLUTION 2
#     # x = rk4(v * cos(theta), x, dt)
#     # y = rk4(v * sin(theta), y, dt)
#     # theta = rk4(w, theta, dt)

#     x_arr.append(x)
#     y_arr.append(y)
#     theta_arr.append(theta)


#     return x_arr, y_arr, theta_arr

# def rk4(v, pos, dt):

#     k1 = dt * v
#     k2 = dt * (k1/2 + v)
#     k3 = dt * (k2/2 + v)
#     k4 = dt * (k3 + v)

#     return pos + (1/6.) * (k1 + 2.0*k2 + 2.0*k3 + k4)

# def x_t(x, y, theta, v, w, dt):

#     """
#     :param x: Starting x-coordinate
#     :param y: Starting y-coordinate
#     :param theta: Starting heading position of robot
#     :param v: Translational Speed (Constant)
#     :param w: Rotational Speed (Constant)
#     :param t: Duration time of the commands (Constant)
    
#     """

#     a1 = 1
#     a2 = 1
#     a3 = 1
#     a4 = 1
#     a5 = 1
#     a6 = 1

#     q_in = np.array([x,y,theta])

#     v_hat = v + sample(a1 * abs(v) + a2 * abs(w))
#     w_hat = w + sample(a3 * abs(v) + a4 * abs(w))
#     gamma = sample(a5 * abs(v) + a6 * abs(w))

#     vx = v_hat * cos(theta)
#     vy = v_hat * sin(theta)

#     if w_hat == 0:
#         q = np.array([vx*dt, vy*dt, 0])
#     else:
#         R = np.array([[cos(w_hat*dt), -sin(w_hat*dt), 0],
#                       [sin(w_hat*dt),  cos(w_hat*dt), 0],
#                       [0        ,          0, 1]])

#         q = np.matmul(R,np.array([vx*dt, vy*dt, w_hat*dt + gamma*dt]))

#     q_out = q_in + q

#     x = q_out[0]
#     y = q_out[1]
#     theta = q_out[2]

#     return x, y, theta

# def sample(b):

#     return b/6 * sum([random.uniform(-1,1) for i in range(12)])

# print(sample(0))

PARTICLE_SIZE = 100
a1 = 0.25
a2 = 0.25
a3 = 1.5
a4 = 1.5
a5 = 0.75
a6 = 0.75

# a1 = 0
# a2 = 0
# a3 = 0
# a4 = 0
# a5 = 0
# a6 = 0


# def x_t(x_t_p, u_t, dt):

#     """
#     :param x: Starting x-coordinate
#     :param y: Starting y-coordinate
#     :param theta: Starting heading position of robot
#     :param v: Translational Speed (Constant)
#     :param w: Rotational Speed (Constant)
#     :param t: Duration time of the commands (Constant)
    
#     """

#     x_dt = np.zeros_like(x_t_p)
#     xcs = np.zeros_like(x_t_p[0])
#     ycs = np.zeros_like(x_t_p[1])

#     thetas = x_t_p[:, 2]
#     print(thetas)

#     v_hat = u_t[0] + sample(a1 * abs(u_t[0]) + a2 * abs(u_t[1])) * np.ones(PARTICLE_SIZE)
#     w_hat = u_t[1] + sample(a3 * abs(u_t[0]) + a4 * abs(u_t[1])) * np.ones(PARTICLE_SIZE)
#     gamma = sample(a5 * abs(u_t[0]) + a6 * abs(u_t[1])) * np.ones(PARTICLE_SIZE)

#     straight = np.abs(w_hat) < 1e-6
#     print(v_hat[straight])

#     print([v_hat[straight] * cos(thetas[straight]) * dt])

#     x_dt[straight] = np.array([[v_hat[straight] * cos(thetas[straight]) * dt],
#                                [v_hat[straight] * sin(thetas[straight]) * dt],
#                                [np.zeros(PARTICLE_SIZE)]])

#     # if w_hat == 0:
#     #     x_dt = np.array([v_hat * cos(x_t_p[2]) * dt, v_hat * sin(x_t_p[2]) * dt, 0])

#     rotation = np.abs(w_hat) >= 1e-6

#     print(v_hat[rotation])
    

#     # Instantaneous Center of Rotation for all Particles
#     xcs[rotation] = x_t_p[rotation, 0] - (v_hat[rotation] / w_hat[rotation]) * sin(thetas[rotation]) * dt
#     ycs[rotation] = x_t_p[rotation, 1] - (v_hat[rotation] / w_hat[rotation]) * cos(thetas[rotation]) * dt

#     R = np.array([[cos(w_hat[rotation] * dt), -sin(w_hat[rotation] * dt), 0],
#                   [sin(w_hat[rotation] * dt),  cos(w_hat[rotation] * dt), 0],
#                   [0                        ,                          0, 1]])
    
#     x_dt = np.matmul(R,np.array([x_t_p[rotation, 0] - xcs[rotation], 
#                                  x_t_p[rotation, 1] - ycs[rotation],
#                                  w_hat[rotation] * dt + gamma[rotation] * dt]))

#     x_t_p = np.array([xcs,ycs,thetas])



#     # else:
#     #     # Instantaneous Center of Rotation
#     #     xc = x_t_p[0] - (v_hat/w_hat)*sin(x_t_p[2])
#     #     yc = x_t_p[1] + (v_hat/w_hat)*cos(x_t_p[2])

#     #     R = np.array([[cos(w_hat*dt), -sin(w_hat*dt), 0],
#     #                   [sin(w_hat*dt),  cos(w_hat*dt), 0],
#     #                   [0        ,          0, 1]])

#     #     x_dt = np.matmul(R,np.array([x_t_p[0] - xc, 
#     #                                  x_t_p[1] - yc,
#     #                                  w_hat * dt + gamma * dt]))

#     #     x_t_p = np.array([xc,yc,x_t_p[2]])


#     x_t = x_t_p + x_dt
    
#     x_t[:, 2] = (x_t[:,2] + np.pi) % (2 * np.pi) - np.pi

#     return x_t

def sample(b, size):
    return b / 6 * np.random.uniform(-1, 1, (size, 12)).sum(axis=1)

def x_t_particles_with_R(x_t_p, u_t, dt):
    """
    :param X: array containing particles [x, y, theta]
    :param u_t: control input [v, w]
    :param dt: timestep
    
    :return: updated particles (N,3)
    """
    thetas = x_t_p[:, 2]
    x_dt = np.zeros_like(x_t_p)
    xcs = np.zeros_like(x_t_p[0])
    ycs = np.zeros_like(x_t_p[1])

    v_noise = a1 * abs(u_t[0]) + a2 * abs(u_t[1])
    w_noise = a3 * abs(u_t[0]) + a4 * abs(u_t[1])
    gamma_noise = a5 * abs(u_t[0]) + a6 * abs(u_t[1])

    v_hat = u_t[0] + sample(v_noise, PARTICLE_SIZE)
    w_hat = u_t[1] + sample(w_noise, PARTICLE_SIZE)
    gamma = sample(gamma_noise, PARTICLE_SIZE)

    straight = np.abs(w_hat) < 1e-6

    x_dt[straight,0] = v_hat[straight] * np.cos(thetas[straight]) * dt
    x_dt[straight,1] = v_hat[straight] * np.sin(thetas[straight]) * dt
    x_dt[straight,2] = 0

    # Rotational motion
    rotation = np.abs(w_hat) > 1e-6

    x_dt[rotation,0] = -(v_hat[rotation] / w_hat[rotation]) * sin(thetas[rotation]) + (v_hat[rotation] / w_hat[rotation]) * sin(thetas[rotation] + w_hat[rotation]*dt)
    x_dt[rotation,1] = (v_hat[rotation] / w_hat[rotation]) * cos(thetas[rotation]) - (v_hat[rotation] / w_hat[rotation]) * cos(thetas[rotation] + w_hat[rotation]*dt)
    x_dt[rotation,2] = w_hat[rotation]*dt + gamma[rotation]*dt

    x_t = x_t_p + x_dt

    x_t[:, 2] = (x_t[:, 2] + np.pi) % (2*np.pi) - np.pi

    return x_t


x_start = 1.29812900 
y_start = 1.88315210
theta_start = 2.82870000 

temp = np.array([x_start, y_start, theta_start])
test_vec = np.full((PARTICLE_SIZE,3), temp)

v_arr = 5
w_arr = 1e-12

u_t = np.array([v_arr,w_arr])

dt = 0.01

x = x_t_particles_with_R(test_vec,u_t,dt)
print(x_t_particles_with_R(x,u_t,dt))
