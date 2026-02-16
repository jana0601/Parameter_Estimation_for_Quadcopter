import numpy as np
'''
Calculate the distance between two points in space.
'''
def distance(quad_pos, waypoint):
    x,y,z = quad_pos
    x_ref,y_ref,z_ref = waypoint

    d1 = np.sqrt(np.square(x - x_ref) + np.square(y - y_ref) + np.square(z - z_ref))

    return d1