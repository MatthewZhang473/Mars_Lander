import numpy as np
from matplotlib import pyplot as plt

G = 6.6743 * (10**(-11))
M_Mars = 6.42 * (10**(23))
mass = 1




def F_G(vector : np.ndarray) -> np.ndarray: 
    dist = np.linalg.norm(vector)
    return -(G*M_Mars*mass/(dist**3)) * vector

def F_spring(vector: np.ndarray) -> np.ndarray:
    k = 1
    return -k*vector

def Euler(disp_list : list, 
           vel_list : list,
           x0 : np.ndarray, 
           v0 : np.ndarray, 
           force_func,
           m : float):

    """
    Euler method for numerical integration 
    """
    x = x0
    v = v0
    for t in t_array:
        # Append current state to trajectories
        disp_list.append(x)
        vel_list.append(v)

        a = force_func(x) / m
        print()
        x = x + dt * v
        v = v + dt * a

    return disp_list,vel_list

def Verlet(disp_list : list, 
           vel_list : list,
           x0 : np.ndarray, 
           v0 : np.ndarray, 
           force_func,
           m : float):

    """
    Verlet method for numerical integration 
    """
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
            a = force_func(x) / m
            x = x + dt * v
            v = v + dt * a
        else:
            # The x and v here are for next iteration
            F = force_func(x)
            x = 2 * x - x_prev + (dt**2)*F/m
            v = (x - x_cur)/dt
        # Update the x_prev for next iteration = x_cur for this iteration
        x_prev = x_cur

    return disp_list,vel_list

    

def plot(x_array : list, y_array : list):
    # plot the position-time graph
    plt.figure(1)
    plt.clf()
    plt.xlabel('time (s)')
    plt.grid()
    plt.plot(x_array, y_array, label='x (m)')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # simulation time, timestep and time (global varaible)
    t_max = 10
    dt = 0.0001
    t_array = np.arange(0, t_max, dt)

    # initialise empty lists to record trajectories
    x_list = []
    v_list = []

    init_disp = np.array([10000,0,0])
    init_vel = np.array([0,0,0])
    
    verlet_results = Verlet(x_list,v_list,init_disp,init_vel,F_G,mass)
    H_array = [posi[0] for posi in verlet_results[0]]
    X_array = [posi[0] for posi in verlet_results[0]]
    Y_array = [posi[1] for posi in verlet_results[0]]
    V_array = [velo[0] for velo in verlet_results[1]]
    R_array = [np.linalg.norm(posi) for posi in verlet_results[0]]

    plot(t_array,H_array)
    #plot(X_array,Y_array)
    #plot(t_array,V_array)
    #plot(t_array, X_array)
    #plot(t_array,R_array)
    

    """
    # Test for force functions
    test_x_array = np.array([i for i in range(-51,51,2)])
    test_y_array = np.array([F_G(x) for x in test_x_array])
    plot(test_x_array, test_y_array)
    """