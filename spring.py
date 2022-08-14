# uncomment the next line if running in a notebook
# %matplotlib inline
from dis import dis
import numpy as np
import matplotlib.pyplot as plt

# mass, spring constant, initial position and velocity
m = 1
k = 1
init_disp = 0
init_vel = 1

# simulation time, timestep and time (global varaible)
t_max = 100
dt = 0.1
t_array = np.arange(0, t_max, dt)

# initialise empty lists to record trajectories
x_list = []
v_list = []


def Euler(disp_list, vel_list,x0, v0):
    # Euler integration
    x = x0
    v = v0

    for t in t_array:
        # append current state to trajectories
        disp_list.append(x)
        vel_list.append(v)

        # calculate new position and velocity
        a = -k * x / m
        x = x + dt * v
        v = v + dt * a

    return disp_list, vel_list

def Verlet(disp_list, vel_list,x0, v0):
    #Verlet method for numerical integration 

    x = x0
    v = v0

    for t in t_array:
        # Append current state to trajectories
        disp_list.append(x)
        vel_list.append(v)

        # Record the current x value
        x_cur = x
        # calculate new position and velocity
        if t == 0:
            # Use Euler's method to find step 1
            a = -k * x / m
            x = x + dt * v
            v = v + dt * a
        else:
            # The x and v here are for next iteration
            F = -k * x
            x = 2 * x - x_prev + (dt**2)*F/m
            v = (x - x_cur)/dt
        # Update the x_prev for next iteration = x_cur for this iteration
        x_prev = x_cur

    return disp_list,vel_list


if __name__ == "__main__":
    # convert trajectory lists into arrays, so they can be sliced (useful for Assignment 2)
    """
    euler_results = Euler(x_list,v_list,init_disp,init_vel)
    x_array = np.array(euler_results[0])
    v_array = np.array(euler_results[1])
    """
    
    
    verlet_results = Verlet(x_list,v_list,init_disp,init_vel)
    x_array = np.array(verlet_results[0])
    v_array = np.array(verlet_results[1])

    # plot the position-time graph
    plt.figure(1)
    plt.clf()
    plt.xlabel('time (s)')
    plt.grid()
    plt.plot(t_array, x_array, label='x (m)')
    plt.plot(t_array, v_array, label='v (m/s)')
    plt.legend()
    plt.show()
