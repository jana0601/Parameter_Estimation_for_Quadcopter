This project is called Quadcopter, which contains simulations for the quadcopter used in my thesis (2019-2020). According to different research purposes, there are six types of simulators. I implemented all of those. Due to the differences in the three observations, the observations used in extended Kalman filter also need to be changed accordingly. So we have 6 folders.
For offline parameter estimation:
Quadcopter_EKF_offline_est
Quadcopter_fullsensor_offline_est
Quadcopter_partsensor_offline_est
For online parameter estimation:
Quadcopter_EKF_online_est
Quadcopter_fullsensor_online_est
Quadcopter_partsensor_online_est

## Simulation
![Drone](Quadcopter_EKF_off.png)

## Install Python requirements
```
pip3 install -r requirements.txt
```

There is a quadcopter simulator under each folder. Every simulator includes GUI, LQG controller, extended Kalman filter, sensor and EM parameter estimation.
The logic behind it is based on:
[1] A. Gibiansky, “Quadcopter Dynamics, Simulation, and Control”
[2] E. E. Holmes , An EM algorithm for maximum likelihood estimation given corrupted observations. National Marine Fisheries Service.
To run this simulator, you can open the folder sim and run runsim.py.
## sim.runsim.py
This file is the main file of this project. Waypoints are given by runsim.py and then the LQG controller will be called to calculate thrust and torque. In LQG controller, The extended Kalman filter will be called to get the state estimation. This is because sensor can not observe all the  state without noise. At the same time, GUI_quadcopter.py will be called to plot the trajectory in real time.
In offline mode, the observations will only be saved to data files. Then the result_ploy.py can use the observations to estimate parameter mass and inertia matrix by calling em_algo.py. The results of EKF can also be plotted by result_ploy.py .
In online mode, the observations will only be saved to data list, and em_algo.py will be called to do parameter estimation every 4 steps. In the end, result_ploy.py will be called to plot the estimated parameter history and the results of EKF.
## sim.result_plot.py
In offline mode, this file will call em_algo.py and plot the estimated parameter mass and inertia matrix. The results of EKF can also be plotted by this file.
In online mode, this file will plot the estimated parameter mass, inertia matrix and the results of EKF.
## sim.dataset.plotforpaper.py
This file has been used to plot 20 rounds simulation results.
## control.LQG.py
This file will be invoked by the main file runsim.py. The desired waypoint has been sent to LQG controller and LQG controller can calculate the thrust and torque. 
## model.quadcopter.py
This file is the core part of quadcopter. The quadcopter.py will be invoked by the main file runsim.py to update the quadcopter system. 
## model.sensor.py
This file is about sensor module. This sensor module has three types observations.
## model.params.py
This file has all the parameters used in this project.
## estimator.extendKalmanFilter.py
This file holds a extended Kalman filter class. Extended Kalman filter can estimate and recover the state from noisy observations. This file will be called in runsim.py and LQG.py.
## utils.check_distance.py
This file holds a static function to calculate the distance between two points in 3D.
## utils.utilities.py
This file holds static functions. Functions include rotation matrix, inverse euler angle rates and etc.
## display.GUI_quadcopter.py
This file will be invoked by the main file runsim.py to plot the quadcopter’s trajectory in real time. 
## EM_parameter_estimation.em_algo.py
This file holds a function to estimate parameter mass and inertia matrix. In offline parameter estimation mode, this function will be called by result_plot.py . In online mode, this function will be called by em_est.py.
## EM_parameter_estimation.em_est.py
This file is only available in online estimate mode. The em_algo.py will be called at here to perform parameter estimation. 
