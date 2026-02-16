
from display.GUI_quadcopter import plot_quad_3d
from control import LQG
from model.quadcopter import Quadcopter
import numpy as np
import json
from estimator.extendKalmanFilter import extendKalmanFilter
from model.sensor import Sensor
from utils.check_distance import distance

control_frequency = 200 # Hz for attitude control loop
dt = 1.0 / control_frequency # time step
time = [0.0]
k = [0]
# variables to plot
F_t = list()  # Thrust
M_t = list()  # Torque
t_s = list()  # simulation time
k_s = list() # simulation steps
state_his = list() # observations
extendkalmanstate = list() # EKF estiamtion
truestate = list() # quadcopter system state



def attitudeControl(quad, extendkalmanfilter, sensor,time,k, targetpoints,dyaw):
    F, M = LQG.controller(quad, extendkalmanfilter,sensor,targetpoints,dyaw,dt)
    q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11 = quad.state
    state_his.append(list(sensor.Y_obs3().reshape(1, 12)[0]))
    quad.update(dt, F, M)
    time[0] += dt
    k[0] +=1
    print("k",k)
    # save variables to graph later
    m1,m2,m3 = M
    k0,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11 = extendkalmanfilter.state_est
    print("time", time[0])
    extendkalmanstate.append((k0,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11))
    truestate.append((q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11))
    F_t.append(F)
    M_t.append((m1[0],m2[0],m3[0]))
    t_s.append(time[0])
    k_s.append(k[0])



def main():

    waypoints = np.array([[0.5, 1, 0], [4, 3, 3], [3, 5, 4], [6, 4, 5], [4, 3, 4], [2, 1, 5]])
    yaw = np.array([[0], [0], [0], [0], [0], [0]])

    # initial position
    pos = waypoints[0]
    targetpoints = waypoints[1]
    dyaw = yaw[1]
    ii = 1
    # initial attitude
    attitude = (0,0,0)
    # create extend kalman filter object
    extendkalmanfilter = extendKalmanFilter()
    # create quadcopter object
    quadcopter = Quadcopter(pos, attitude)
    # create sensor object
    sensor = Sensor()
    def control_loop(i):
        nonlocal ii
        nonlocal targetpoints
        nonlocal dyaw
        if distance(quadcopter.position(), targetpoints) < 0.1:
            ii = ii + 1
            if ii < len(waypoints):
                targetpoints = waypoints[ii]
                dyaw = yaw[ii]
            else: ii = ii -1

        else:
            targetpoints = targetpoints
        for _ in range(4):
            attitudeControl(quadcopter, extendkalmanfilter,sensor,time,k, targetpoints,dyaw)
        return quadcopter.world_frame()
    # plot quadcopter trajectory in real time
    plot_quad_3d(waypoints, control_loop)
    # saving data for state estimation
    if(True):
        # saving thrust F and Moment M
        with open('controlleroutput.txt', 'w') as outfile:
            data = {}
            data["F"] = F_t
            data["M"] = M_t
            json.dump(data, outfile)
        # saving observations
        with open('state_his.txt', 'w') as outfile:
            data = {}
            data["state_his"] = state_his
            json.dump(data, outfile)
        # saving extend kalman filter state estimation
        with open('extendkalmanstate.txt', 'w') as outfile:
            data = {}
            data = extendkalmanstate
            json.dump(data, outfile)
        # saving state from quadcopter system
        with open('realstate.txt', 'w') as outfile:
            data = {}
            data = truestate
            json.dump(data, outfile)
        # saving simulation time
        with open('times.txt', 'w') as outfile:
            data = {}
            data = t_s
            json.dump(data, outfile)
        # saving simulation steps
        with open('ks.txt', 'w') as outfile:
            data = {}
            data = k_s
            json.dump(data, outfile)
    print("Closing.")


if __name__ == "__main__":
    main()
