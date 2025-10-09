import matplotlib.pyplot as plt
import numpy as np
from motion_model import motion_model

X_arr = []
Y_arr = []

x_o, y_o, theta_o = motion_model(0,0,0 , 0.5, 0, 1)
X_arr = X_arr + x_o
Y_arr = Y_arr + y_o

x_o, y_o, theta_o = motion_model(x_o[-1], y_o[-1], theta_o[-1] , 0, -1/(2*np.pi), 1)
X_arr = X_arr + x_o
Y_arr = Y_arr + y_o

x_o, y_o, theta_o = motion_model(x_o[-1], y_o[-1], theta_o[-1] , 0.5, 0, 1)
X_arr = X_arr + x_o
Y_arr = Y_arr + y_o

x_o, y_o, theta_o = motion_model(x_o[-1], y_o[-1], theta_o[-1] , 0, 1/(2*np.pi), 1)
X_arr = X_arr + x_o
Y_arr = Y_arr + y_o

x_o, y_o, theta_o = motion_model(x_o[-1], y_o[-1], theta_o[-1] , 0.5, 0, 1)
X_arr = X_arr + x_o
Y_arr = Y_arr + y_o

plt.plot(X_arr,Y_arr)
plt.show()