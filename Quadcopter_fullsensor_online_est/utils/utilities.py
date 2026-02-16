"""
author: Peter Huang
email: hbd730@gmail.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""


import numpy as np
from math import sin, cos, asin, atan2, sqrt,tan
import model.params as params

def RotToRPY_ZXY(R):
    phi = asin(R[1,2])
    theta = atan2(-R[0,2]/cos(phi),R[2,2]/cos(phi))
    psi = atan2(-R[1,0]/cos(phi),R[1,1]/cos(phi))
    return np.array([phi, theta, psi])

def RotToRPY_ZYX(R):
    """
        Euler angle convention is ZYX, which means first apply
        rotaion of psi-degrees around Z axis, then rotation of
        theta-degrees around new Y axis, and then rotation of
        phi-degrees around new X axis.

        ** The rotation R received should be from body to world frame. **
    """
    theta = np.arcsin(-1.0 * R[2, 0])
    phi = np.arctan2(R[2, 1] / np.cos(theta), R[2, 2] / np.cos(theta))
    psi = np.arctan2(R[1, 0] / np.cos(theta), R[0, 0] / np.cos(theta))

    return np.array([[phi], [theta], [psi]])


def RPYToRot_ZYX(phi, theta, psi):
    """
    phi, theta, psi = roll, pitch , yaw
    The euler angle convention used is ZYX. This means: first a rotation of psi-degrees
    around Z axis, then rotation of theta-degrees around Y axis, and finally rotation of
    phi-degress around X axis
    """
    return np.array([[cos(theta)*cos(psi), cos(theta)*sin(psi), -sin(theta)],
                     [-cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi), cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi), sin(phi)*cos(theta)],
                     [sin(phi)*sin(psi) + cos(phi)*sin(theta)*cos(psi), -sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi), cos(phi)*cos(theta)]])


def Eulerangleratesmatrix(phi,theta,psi):
    peta = np.array([
        [1.0, sin(phi) * tan(theta), cos(phi) * tan(theta)],
        [0.0, cos(phi), -1.0 * sin(phi)],
        [0.0, sin(phi) / cos(theta), cos(phi) / cos(theta)]])
    return peta

def state_dot(state,F, M,mass):
    x, y, z, xdot, ydot, zdot, roll,pitch,yaw, p, q, r = state
    wRb = RPYToRot_ZYX(roll, pitch, yaw)  # from body to world
    accel = -params.g * params.e3 + (F / mass) * np.dot(wRb, params.e3)
    omega = np.array([p, q, r])
    eulerdot = np.dot(Eulerangleratesmatrix(roll, pitch, yaw), omega.reshape(3, 1))
    pqrdot = params.invI.dot(M - np.cross(omega, params.I.dot(omega)))
    state_dot = np.zeros(12)
    state_dot[0] = xdot
    state_dot[1] = ydot
    state_dot[2] = zdot
    state_dot[3] = accel[0][0]
    state_dot[4] = accel[1][0]
    state_dot[5] = accel[2][0]
    state_dot[6] = eulerdot[0]
    state_dot[7] = eulerdot[1]
    state_dot[8] = eulerdot[2]
    state_dot[9] = pqrdot[0]
    state_dot[10] = pqrdot[1]
    state_dot[11] = pqrdot[2]
    state_dot = state_dot.reshape(1,12)

    return state_dot

# Set of functions that generate the states and inputs of the quadrotor
# given 4 flat outputs, namely: position (3-vector) and yaw angle.
#
def state_dot_(state,F, M,inertial,invI,mass):
    x, y, z, xdot, ydot, zdot, roll,pitch,yaw, p, q, r = state
    wRb = RPYToRot_ZYX(roll, pitch, yaw)  # from body to world
    accel = -params.g * params.e3 + (F / mass) * np.dot(wRb, params.e3)
    omega = np.array([p, q, r])
    eulerdot = np.dot(Eulerangleratesmatrix(roll, pitch, yaw), omega.reshape(3, 1))
    #print("inv",invI)
    #print("Iner",Iner)
    #print("np.cross(omega,Iner.dot(omega))",np.cross(omega,Iner.dot(omega)))
    #print("invI.dot(M.flatten() - np.cross(omega,Iner.dot(omega)))",invI.dot(M.flatten() - np.cross(omega,inertial.dot(omega))))
    pqrdot = invI.dot(M - np.cross(omega,inertial.dot(omega)))

    state_dot = np.zeros(12)
    state_dot[0] = xdot
    state_dot[1] = ydot
    state_dot[2] = zdot
    state_dot[3] = accel[0][0]
    state_dot[4] = accel[1][0]
    state_dot[5] = accel[2][0]
    state_dot[6] = eulerdot[0]
    state_dot[7] = eulerdot[1]
    state_dot[8] = eulerdot[2]
    state_dot[9] = pqrdot[0]
    state_dot[10] = pqrdot[1]
    state_dot[11] = pqrdot[2]
    state_dot = state_dot.reshape(1,12)[0]

    return state_dot



def get_x(sigma1):
    return sigma1


def get_y(sigma2):
    return sigma2


def get_z(sigma3):
    return sigma3


def get_psi(sigma4):
    return sigma4


def get_u1(t,mass):
    u1 = mass * np.linalg.norm(t)
    return u1


def get_zb(t):
    zb = t / (np.linalg.norm(t))
    return zb


def get_u1_dot(z_b, j,mass):
    u1_dot = mass * z_b.T * j
    return u1_dot


def get_t_vector(sigma1, sigma2, sigma3):
    # construct the t vector at each point in time
    t_vec = np.array([[sigma1], [sigma2], [sigma3 + params.g]])
    return t_vec


def get_xc(sigma4):
    x_c = np.array([[np.cos(sigma4)], [np.sin(sigma4)], [0.0]])
    return x_c


def get_yc(sigma4):
    y_c = np.array([[-1.0 * np.sin(sigma4)], [np.cos(sigma4)], [0.0]])
    return y_c


def get_xb(y_c, z_b):
    a = np.cross(y_c, z_b, axis=0)
    return a / np.linalg.norm(a)


def get_yb(z_b, x_b):
    a = np.cross(z_b, x_b, axis=0)
    return a / np.linalg.norm(a)


def get_wx(y_b, j, u_1,mass):
    w_x = -1.0 * mass * (np.dot(y_b.T , j)[0][0]) / u_1
    return w_x


def get_wy(x_b, j, u_1,mass):
    w_y = mass * (np.dot(x_b.T, j)[0][0])/ u_1
    return w_y


def get_wz(psi_rate, x_c, x_b, w_y, y_c, z_b):
    """
        Will compute as wz = (a + b)/c
    """
    a = psi_rate * np.dot(x_c.T, x_b)[0][0]
    b = w_y * np.dot(y_c.T,z_b)[0][0]
    c = np.linalg.norm(np.cross(y_c, z_b, axis=0))
    w_z = (a + b) / c
    return w_z

# This correspond to [u1, u2, u3]
def get_ux(w_dot_, w_,I):
    u_x = I * w_dot_ + np.array(np.cross(w_, I * w_, axis=0))
    return u_x


def get_ua(u_1, z_b,mass):
    """
        ua = -g*z_w +u1*z_b/m
    """
    u_a = -params.g * np.array([[0.0], [0.0], [1.0]]) + u_1 * z_b / mass
    return u_a


def get_ub(w_, M,I,invI):
    u_b = invI * (-1.0 * np.cross(w_, I * w_, axis=0) + M)
    return u_b


def get_uc(w_, ori):
    phi_ = ori[0][0]
    theta_ = ori[1][0]
    psi_ = ori[2][0]

    peta = Eulerangleratesmatrix(phi_,theta_,psi_)
    u_c = np.dot(peta , w_.reshape(3,1))

    return u_c


def compute_ref(trajectory):
    """
        Compute all reference states and inputs from the given desired trajectory point using
        differential flatness property.
    """
    # first convert all input np.array to np.matrices to simplify
    # computation.
    # This should be changed to use np.arrays only as np.matrix is not recommended anymore

    # extract all quantities from given trajectory
    pos_traj = trajectory[0]  # 3-vector
    vel_traj = trajectory[1]  # 3-vector
    acc_traj = trajectory[2]  # 3-vector
    jerk_traj = trajectory[3]  # 3-vector

    yaw_traj = trajectory[5]  # scalar
    yaw_dot_traj = trajectory[6]  # scalar


    # convert all vectors from np.array to np.matrix for compatibility and
    # ease
    pos_traj = np.array(pos_traj)
    vel_traj = np.array(vel_traj)
    acc_traj = np.array(acc_traj)
    jerk_traj = np.array(jerk_traj)

    yaw_traj = np.array(yaw_traj)
    yaw_dot_traj = np.array(yaw_dot_traj)

    #print("acc",acc_traj[0])
    t_vec = get_t_vector(acc_traj[0], acc_traj[1], acc_traj[2])  # get_t_vector(sigma1, sigma2,sigma3)

    u_1 = get_u1(t_vec,params.mass)
    z_b = get_zb(t_vec)
    y_c = get_yc(yaw_traj)
    x_b = get_xb(y_c, z_b)  # get_xb(y_c,z_b)
    y_b = get_yb(z_b, x_b)  # get_yb(z_b,x_b)
    j_ = np.array([[jerk_traj[0]], [jerk_traj[1]], [jerk_traj[2]]])
    w_x = get_wx(y_b, j_, u_1,params.mass)  # get_wx(y_b,j,u_1)
    w_y = get_wy(x_b, j_, u_1,params.mass)  # get_wy(x_b,j,u_1)
    x_c = get_xc(yaw_traj)
    w_z = get_wz(yaw_dot_traj, x_c, x_b, w_y, y_c, z_b)  # get_wz(psi_rate,x_c,x_b,w_y,y_c,z_b)


    # make angular velocity vector w
    w_ = np.array([w_x, w_y, w_z])

    # get rotation matrix from base frame to world frame
    # for current desired trajectory point.
    # This matrix represents the orientation of the quadrotor
    R_ = np.concatenate((x_b, y_b, z_b), axis=1)



    # Get roll pitch yaw angles assuming ZXY Euler angle convention
    # This means: first rotate psi degrees around Z axis,
    # then theta degrees around Y axis, and lastly phi degrees around X axis

    or_ = RotToRPY_ZYX(R_)  # assuming ZYX Eugler angle convention, so sent matrix should be
    # body to world frame

    # compute u_a input for system reference
    # can be computed as follows or simply the received acc_traj
    # vector after conversion to matrix. Both are exactly the same quantity
    u_a = get_ua(u_1, z_b,params.mass)  # get_ua(u_1,z_b)

    # compute u_b input for system reference

    # compute u_c input for system reference
    u_c = get_uc(w_, or_)  # get_uc(w_,ori)

    # we need to return back the 1) reference state vector and 2) reference inputs
    # The reference state vector is : (x,y,z, v_x, v_y, v_z, phi, theta, psi, p, q, r)
    # where x,y,z: are position coordinates
    # v_x, v_y, v_z: are velocities
    # phi, theta, psi: are orientation euler angles as described above
    # p, q, r: are angular velocities in body frame X,Y,Z axis respectively

    # we send the received pos_traj, and vel_traj vectors as the reference pos and vel vectors
    # because that is the result from the differential flatness output selection
    # return [pos_traj.T, vel_traj.T, or_, w_, acc_traj.T, w_dot_, R_, u_c, u_1, u_x]

    return [pos_traj.reshape(3,1), vel_traj.reshape(3,1), or_.reshape(3,1), w_.reshape(3,1), u_a.reshape(3,1), u_c.reshape(3,1)]

