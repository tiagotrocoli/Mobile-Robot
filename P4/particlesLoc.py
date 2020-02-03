# --------------
# USER INSTRUCTIONS
#
# Now you will put everything together.
#
# First make sure that your sense and move functions
# work as expected for the test cases provided at the
# bottom of the previous two programming assignments.
# Once you are satisfied, copy your sense and move
# definitions into the robot class on this page, BUT
# now include noise.
#
# A good way to include noise in the sense step is to
# add Gaussian noise, centered at zero with variance
# of self.bearing_noise to each bearing. You can do this
# with the command random.gauss(0, self.bearing_noise)
#
# In the move step, you should make sure that your
# actual steering angle is chosen from a Gaussian
# distribution of steering angles. This distribution
# should be centered at the intended steering angle
# with variance of self.steering_noise.
#
# Feel free to use the included set_noise function.
#
# Please do not modify anything except where indicated
# below.

from math import *
import random

# --------
# 
# some top level parameters
#

max_steering_angle = pi / 4.0 # You do not need to use this value, but keep in mind the limitations of a real car.
forward_noise  = 0.1 # Noise parameter: should be included in sense function.
turn_noise = 0.1 # Noise parameter: should be included in move function.
sense_noise = 5.0 # Noise parameter: should be included in move function.

tolerance_xy = 15.0 # Tolerance for localization in the x and y directions.
tolerance_orientation = 0.25 # Tolerance for orientation.


# --------
# 
# the "world" has 4 landmarks.
# the robot's initial coordinates are somewhere in the square
# represented by the landmarks.
#
# NOTE: Landmark coordinates are given in (y, x) form and NOT
# in the traditional (x, y) format!

X_LEFT_LIM = 6.0
X_RIGHT_LIM = -7.0
Y_BOTTOM_LIM = 7.0
Y_UPPER_LIM = -6.0

landmarks  = [[-5.0, -5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, 5.0]] # position of 4 landmarks in (y, x) format.
world_sizeX = X_LEFT_LIM - X_RIGHT_LIM # world is NOT cyclic. Robot is allowed to travel "out of bounds"
world_sizeY = Y_BOTTOM_LIM - Y_UPPER_LIM
# ------------------------------------------------
# 
# this is the robot class
#

class particle:

    # --------
    # init: 
    #    creates robot and initializes location/orientation 
    #

    def __init__(self):
        self.x = (2*random.random()-1) * world_sizeX /2 # initial x position
        self.y = (2*random.random()-1) * world_sizeY /2 # initial y position
        self.orientation = random.random() * 2.0 * pi # initial orientation
        self.ROBOT_WIDTH = 0.381
        self.WHEEL_RADIUS = 0.195/2.0
        self.forward_noise = 0.0
        self.turn_noise    = 0.0
        self.sense_noise   = 0.0
    
    # --------
    # set: 
    #    sets a robot coordinate
    #

    def set(self, new_x, new_y, new_orientation):
        # if new_x < 0 or new_x >= world_sizeX:
        #     raise ValueError( 'X coordinate out of bound')
        # if new_y < 0 or new_y >= world_sizeY:
        #     raise ValueError( 'Y coordinate out of bound')
        # if new_orientation < 0 or new_orientation >= 2 * pi:
        #     raise ValueError( 'Orientation must be in [0..2pi]')
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)
    
    
    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.forward_noise = float(new_f_noise)
        self.turn_noise    = float(new_t_noise)
        self.sense_noise   = float(new_s_noise)

    # --------
    # measurement_prob
    #    computes the probability of a measurement
    #  

    def measurement_prob(self, measurement):
        
        # calculates how likely a measurement should be
        
        prob = 1.0
        for i in range(len(landmarks)):
            dist = sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            prob *= self.Gaussian(dist, self.sense_noise, measurement[i])
        return prob
    
    def Gaussian(self, mu, sigma, x):
        
        # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))
    
    def __repr__(self): #allows us to print robot attributes.
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), 
                                                str(self.orientation))
    
    ############# ONLY ADD/MODIFY CODE BELOW HERE ###################
       
    # --------
    # move: 
    #   
    def move(self, left_radius, right_radius):       
        
        # turn, and add randomness to the turning command
        delta_ori = (right_radius - left_radius)/self.ROBOT_WIDTH 
        orientation = self.orientation + delta_ori
        orientation %= 2 * pi
        
        # move, and add randomness to the motion command
        dist = (right_radius + left_radius)/2 
        x = self.x + (cos(self.orientation + delta_ori/2) * dist)
        y = self.y + (sin(self.orientation + delta_ori/2) * dist)
        
        # set particle
        res = particle()
        res.set(x, y, orientation)
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        return res
                      
    # --------
    # sense: 
    #    
    def sense(self):
        Z = []
        for i in range(len(landmarks)):
            dist = sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            dist += random.gauss(0.0, self.sense_noise)
            Z.append(dist)
        return Z


# --------
#
# extract position from a particle set
# 

def get_position(p):
    x = 0.0
    y = 0.0
    orientation = 0.0
    for i in range(len(p)):
        x += p[i].x
        y += p[i].y
        # orientation is tricky because it is cyclic. By normalizing
        # around the first particle we are somewhat more robust to
        # the 0=2pi problem
        orientation += (((p[i].orientation - p[0].orientation + pi) % (2.0 * pi)) 
                        + p[0].orientation - pi)
    return [x / len(p), y / len(p), orientation / len(p)]

# --------
#
# The following code checks to see if your particle filter
# localizes the robot to within the desired tolerances
# of the true position. The tolerances are defined at the top.


def particle_filter(motion, measurement, N=500): # I know it's tempting, but don't change N!
    # --------
    #
    # Make particles

    p = []
    for i in range(N):
        r = particle()
        r.set_noise(forward_noise, turn_noise, sense_noise)
        p.append(r)

    # Update particles
    #     

    # motion update (prediction)
    p2 = []
    for i in range(N):
        p2.append(p[i].move(motion[0],motion[1]))
    p = p2

    # measurement update
    w = []
    for i in range(N):
        w.append(p[i].measurement_prob(measurement))

    # resampling
    p3 = []
    index = int(random.random() * N)
    beta = 0.0
    mw = max(w)
    for i in range(N):
        beta += random.random() * 2.0 * mw
        while beta > w[index]:
            beta -= w[index]
            index = (index + 1) % N
        p3.append(p[index])
    p = p3
    
    return get_position(p)