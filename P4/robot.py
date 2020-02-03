import sys, time, threading
import signal
import numpy as np
from math import *
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance
sys.path.insert(0, '../lib')
import vrep
from fuzzy import fuzzification, trangular_fuzzifier, defuzzification
from particlesLoc import particle, particle_filter
# Maximum distance for sonar sensors to detect
NO_DETECTION_DIST = 5

# Time interval in which the position of the robot will be checked/calculated.
INTERVAL_CHECK_POSITION = 0.3

# robot stops when reach a timeout
ROBOT_TIMEOUT = 120
# Plot limits

X_LEFT_LIM = 5.5530
X_RIGHT_LIM = -6.8010
Y_BOTTOM_LIM = 6.9940
Y_UPPER_LIM = -5.0660

# X_LEFT_LIM = 6.0
# X_RIGHT_LIM = -7.0
# Y_BOTTOM_LIM = 7.0
# Y_UPPER_LIM = -6.0


landmarks  = [[-5.0, -5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, 5.0]] # position of 4 landmarks in (y, x) format.
world_sizeX = X_LEFT_LIM - X_RIGHT_LIM # world is NOT cyclic. Robot is allowed to travel "out of bounds"
world_sizeY = Y_BOTTOM_LIM - Y_UPPER_LIM

forward_noise  = 0.1 # Noise parameter: should be included in sense function.
turn_noise = 0.1 # Noise parameter: should be included in move function.
sense_noise = 5.0 # Noise parameter: should be included in move function.


position_ground_truth = []
position_odometry = []
position_particle = []

points_cloud_ground_truth = []
points_cloud_odometry = []
laserXY = []
def difference_radians(rad1, rad2):
    """ difference between two radians 
    """
    
    try:
        if rad2 < 0:
            rad2 = 2*np.pi + rad2
            
        if rad1 < 0:
            rad1 = 2*np.pi + rad1
        
        if 0 <= rad2 <= np.pi and np.pi <= rad1 <= 2*np.pi or rad2 > rad1:
            rad = np.pi - abs(abs(rad2 - rad1) - np.pi)
        else:
            rad = -(np.pi - abs(abs(rad2 - rad1) - np.pi))
        return rad
    
    except Exception as e:
        raise e

def localToGlobal(xObject, yObject, theta, x, y):

    tTrans = [[1, 0, xObject],
              [0, 1, yObject],
              [0, 0, 1]]

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    tRot = [[cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]]

    coord = [x, y, 1]

    localToGlobalMatrix = np.dot(tTrans,tRot)
    resMatrix = np.dot(localToGlobalMatrix, coord)

    return resMatrix[0],resMatrix[1]

def display_path(points, title, lims=True):
	plt.figure(dpi=1080)
	plt.scatter(*zip(*points))
	print(len(points))
	if lims == True:
		plt.xlim(X_LEFT_LIM, X_RIGHT_LIM)
		plt.ylim(Y_BOTTOM_LIM, Y_UPPER_LIM)
	plt.title(title)
	plt.show()

def display_map(points, title, lims=True):
    plt.figure(dpi=1080)
    plt.scatter(*zip(*points), s=5, label='Map')
    plt.title(title)
    if lims == True:
        plt.xlim(X_LEFT_LIM, X_RIGHT_LIM)
        plt.ylim(Y_BOTTOM_LIM, Y_UPPER_LIM)
    plt.xlabel("x(m)")
    plt.ylabel("y(m)")
    plt.legend()
    plt.show()

def display_every(truth_points, odo_points, particles, lims=True):
	plt.figure(dpi=1080)
	plt.plot(*zip(*truth_points),label = 'ground truth')
	plt.plot(*zip(*odo_points), label = 'odometry')
	plt.plot(*zip(*particles), label = 'particle filter localization')
	if lims == True:
		plt.xlim(X_LEFT_LIM, X_RIGHT_LIM)
		plt.ylim(Y_BOTTOM_LIM, Y_UPPER_LIM)
	plt.xlabel("x(m)")
	plt.ylabel("y(m)")
	plt.legend()
	plt.show()

class Robot():
	def __init__(self):
		self.ROBOT_WIDTH = 0.381
		self.WHEEL_RADIUS = 0.195/2.0
		self.SERVER_IP = "127.0.0.1"
		self.SERVER_PORT = 25000
		self.clientID = self.start_sim()
		self.us_handle, self.vision_handle, self.laser_handle = self.start_sensors()
		self.motors_handle = self.start_motors()
		self.robot_handle = self.start_robot()
		self.DISTANCE_BETWEEN_WHEELS, self.ROBOT_INIT_XY, self.ROBOT_INIT_THETA= self.robot_constant_init()
		self.ROBOT_VELOCITY = 0.5
		self.x, self.y, self.orientation = self.get_position_orientation(self.robot_handle,-1)

	def start_sim(self):
		"""
			Function to start the simulation. The scene must be running before running this code.
		    Returns:
		        clientID: This ID is used to start the objects on the scene.
		"""
		vrep.simxFinish(-1)
		clientID = vrep.simxStart(self.SERVER_IP, self.SERVER_PORT, True, True, 2000, 5)
		if clientID != -1:
			print("Connected to remoteApi server.")
		else:
			vrep.simxFinish(clientID)
			sys.exit("\033[91m ERROR: Unable to connect to remoteApi server. Consider running scene before executing script.")

		return clientID

	def get_connection_status(self):
		"""
			Function to inform if the connection with the server is active.
			Returns:
				connectionId: -1 if the client is not connected to the server.
				Different connection IDs indicate temporary disconections in-between.
		"""
		return vrep.simxGetConnectionId(self.clientID)

	def start_sensors(self):
		"""
			Function to start the sensors.
		    Returns:
		        us_handle: List that contains each ultrassonic sensor handle ID.
				vision_handle: Contains the vision sensor handle ID.
				laser_handle: Contains the laser handle ID.
		"""
		#Starting ultrassonic sensors
		us_handle = []
		sensor_name=[]
		for i in range(0,16):
			sensor_name.append("Pioneer_p3dx_ultrasonicSensor" + str(i+1))

			res, handle = vrep.simxGetObjectHandle(self.clientID, sensor_name[i], vrep.simx_opmode_oneshot_wait)
			if(res != vrep.simx_return_ok):
				print ("\033[93m "+ sensor_name[i] + " not connected.")
			else:
				print ("\033[92m "+ sensor_name[i] + " connected.")
				us_handle.append(handle)

		#Starting vision sensor
		res, vision_handle = vrep.simxGetObjectHandle(self.clientID, "Vision_sensor", vrep.simx_opmode_oneshot_wait)
		if(res != vrep.simx_return_ok):
			print ("\033[93m Vision sensor not connected.")
		else:
			print ("\033[92m Vision sensor connected.")

		#Starting laser sensor
		res, laser_handle = vrep.simxGetObjectHandle(self.clientID, "fastHokuyo", vrep.simx_opmode_oneshot_wait)
		if(res != vrep.simx_return_ok):
			print ("\033[93m Laser not connected.")
		else:
			print ("\033[92m Laser connected.")

		return us_handle, vision_handle, laser_handle

	def start_motors(self):
		"""
			Function to start the motors.
		    Returns:
		        A dictionary that contains both motors handle ID.
		"""

		res, left_handle = vrep.simxGetObjectHandle(self.clientID, "Pioneer_p3dx_leftMotor", vrep.simx_opmode_oneshot_wait)
		if(res != vrep.simx_return_ok):
			print("\033[93m Left motor not connected.")
		else:
			print("\033[92m Left motor connected.")

		res, right_handle = vrep.simxGetObjectHandle(self.clientID, "Pioneer_p3dx_rightMotor", vrep.simx_opmode_oneshot_wait)
		if(res != vrep.simx_return_ok):
			print("\033[93m Right motor not connected.")
		else:
			print("\033[92m Right motor connected.")

		return {"left": left_handle, "right":right_handle}

	def start_robot(self):
		"""
			Function to start the robot.
			Returns:
				robot_handle: Contains the robot handle ID.
		"""
		res, robot_handle = vrep.simxGetObjectHandle(self.clientID, "Pioneer_p3dx", vrep.simx_opmode_oneshot_wait)
		if(res != vrep.simx_return_ok):
			print("\033[93m Robot not connected.")
		else:
			print("\033[92m Robot connected.")

		return robot_handle

	def robot_constant_init(self):
		# Distance between the wheels
		err_code, lw = vrep.simxGetObjectPosition(self.clientID, self.motors_handle["left"], -1, vrep.simx_opmode_streaming)
		while err_code != vrep.simx_return_ok:
			err_code, lw = vrep.simxGetObjectPosition(self.clientID, self.motors_handle["left"], -1, vrep.simx_opmode_buffer)
		err_code, rw = vrep.simxGetObjectPosition(self.clientID, self.motors_handle["right"], -1, vrep.simx_opmode_streaming)
		while err_code != vrep.simx_return_ok:
			err_code, rw = vrep.simxGetObjectPosition(self.clientID, self.motors_handle["right"], -1, vrep.simx_opmode_buffer)

		DISTANCE_BETWEEN_WHEELS = distance.euclidean(lw[:2], rw[:2])

		# Initial Position of the Robot
		err_code, robot_pos = vrep.simxGetObjectPosition(self.clientID, self.robot_handle, -1, vrep.simx_opmode_streaming)
		while err_code != vrep.simx_return_ok:
			err_code, robot_pos = vrep.simxGetObjectPosition(self.clientID, self.robot_handle, -1, vrep.simx_opmode_buffer)
			
		# Only X and Y
		ROBOT_INIT_XY = robot_pos[:2]
			
		# Initial Orientation of the Robot
		err_code, robot_ori = vrep.simxGetObjectOrientation(self.clientID, self.robot_handle, -1, vrep.simx_opmode_streaming)
		while err_code != vrep.simx_return_ok:
			err_code, robot_ori = vrep.simxGetObjectOrientation(self.clientID, self.robot_handle, -1, vrep.simx_opmode_buffer)
			
		# Only Gamma
		ROBOT_INIT_THETA = robot_ori[2]%(np.pi*2)

		position_odometry.append(self.get_position_orientation(self.robot_handle, -1))

		return DISTANCE_BETWEEN_WHEELS, ROBOT_INIT_XY, ROBOT_INIT_THETA

	def stop(self):
		"""
			Sets the motors velocities to 0.
		"""
		vrep.simxSetJointTargetVelocity(self.clientID, self.motors_handle["left"], 0, vrep.simx_opmode_streaming)
		vrep.simxSetJointTargetVelocity(self.clientID, self.motors_handle["right"], 0, vrep.simx_opmode_streaming)
		time.sleep(0.5)

	def set_left_velocity(self, vel):
		"""
			Sets the velocity on the left motor.
			Args:
				vel: The velocity to be applied in the motor (rad/s)
		"""
		vrep.simxSetJointTargetVelocity(self.clientID, self.motors_handle["left"], vel, vrep.simx_opmode_streaming)

	def set_right_velocity(self, vel):
		"""
			Sets the velocity on the right motor.
			Args:
				vel: The velocity to be applied in the motor (rad/s)
		"""
		vrep.simxSetJointTargetVelocity(self.clientID, self.motors_handle["right"], vel, vrep.simx_opmode_streaming)

	def set_velocity(self, V, W):
		"""
			Sets a linear and a angular velocity on the robot.
			Args:
				V: Linear velocity (m/s) to be applied on the robot along its longitudinal axis.
				W: Angular velocity (rad/s) to be applied on the robot along its axis of rotation, positive in the counter-clockwise direction.
		"""
		left_vel = (V - W*(self.ROBOT_WIDTH/2))/self.WHEEL_RADIUS
		right_vel = (V + W*(self.ROBOT_WIDTH/2))/self.WHEEL_RADIUS
		vrep.simxSetJointTargetVelocity(self.clientID, self.motors_handle["left"], left_vel, vrep.simx_opmode_streaming)
		vrep.simxSetJointTargetVelocity(self.clientID, self.motors_handle["right"], right_vel, vrep.simx_opmode_streaming)

	def set_velocity_one_motor(self, motor, v):
		""" set velocity to an specific motor """
		
		try:
			motor = motor.lower()

			if motor == "l" or motor == "left":
				vrep.simxSetJointTargetVelocity(self.clientID, self.motors_handle["left"], v, vrep.simx_opmode_streaming)
			elif motor == "r" or motor == "right":
				vrep.simxSetJointTargetVelocity(self.clientID, self.motors_handle["right"],v, vrep.simx_opmode_streaming)
		except Exception as e:
			raise e

	def set_velocity_both_motors(self, v):
		""" set the same velocity to both motors """
		
		self.set_left_velocity(v)
		self.set_right_velocity(v)

	def set_pos_ori_robot(self, x = 0.0, y = 0.0, gamma = 0.0):
		""" set both the position (x, y) and the orientation (gamma) of the robot """
		vrep.simxSetObjectPosition(self.clientID, self.robot_handle, -1, [x, y, +1.3879e-01],
								vrep.simx_opmode_oneshot)
		vrep.simxSetObjectOrientation(self.clientID, self.robot_handle, -1, [0.0000e+00, 0.0000e+00, gamma],
									vrep.simx_opmode_oneshot)
		return x, y, gamma	

	def get_wheels_velocity(self):
		"""
			Gives the current velocity of both wheels.
			Returns:
				position: Array with current angular velocity of left and right wheels.
		"""
		res, l_vel, a_vel = vrep.simxGetObjectVelocity(self.clientID, self.robot_handle, vrep.simx_opmode_streaming)
		while(res != vrep.simx_return_ok):
			res, l_vel, a_vel = vrep.simxGetObjectVelocity(self.clientID, self.robot_handle, vrep.simx_opmode_streaming)

		V = np.sqrt(l_vel[0]**2 + l_vel[1]**2)
		W = a_vel[2]
		left_vel = (V - W*(self.ROBOT_WIDTH/2))/self.WHEEL_RADIUS
		right_vel = (V + W*(self.ROBOT_WIDTH/2))/self.WHEEL_RADIUS
		
		return left_vel, right_vel
		# return V,W

	def get_current_velocity(self):
		"""
			Gives the current velocity of both wheels.
			Returns:
				position: Array with current angular velocity of left and right wheels.
		"""
		res, l_vel, a_vel = vrep.simxGetObjectVelocity(self.clientID, self.robot_handle, vrep.simx_opmode_streaming)
		while(res != vrep.simx_return_ok):
			res, l_vel, a_vel = vrep.simxGetObjectVelocity(self.clientID, self.robot_handle, vrep.simx_opmode_streaming)

		V = np.sqrt(l_vel[0]**2 + l_vel[1]**2)
		W = a_vel[2]

		return V,W

	def get_current_position(self, object_handle, parent_handle):
		"""
			Gives the current handle position on the environment.
			Returns:
				position: Array with the (x,y,z) coordinates.
		"""
		res, position = vrep.simxGetObjectPosition(self.clientID, object_handle, parent_handle, vrep.simx_opmode_streaming)
		while(res != vrep.simx_return_ok):
			res, position = vrep.simxGetObjectPosition(self.clientID, object_handle, parent_handle, vrep.simx_opmode_streaming)

		return position

	def get_current_orientation(self, object_handle, parent_handle):
		"""
			Gives the current object orientation on the environment.
			Returns:
				orientation: Array with the euler angles (alpha, beta and gamma).
		"""
		res, orientation = vrep.simxGetObjectOrientation(self.clientID, object_handle, parent_handle, vrep.simx_opmode_streaming)
		while(res != vrep.simx_return_ok):
			res, orientation = vrep.simxGetObjectOrientation(self.clientID, object_handle, parent_handle, vrep.simx_opmode_streaming)

		return orientation

	def get_position_orientation(self, object_handle, parent_handle):
		position = self.get_current_position(object_handle, parent_handle)
		orientation = self.get_current_orientation(object_handle, parent_handle)
		return position[0],position[1],orientation[2]

	def get_joint_position(self, object_handle):
		""" get joint position of an object """
		
		try:
			err_code, angle = vrep.simxGetJointPosition(self.clientID, object_handle, vrep.simx_opmode_streaming)
			while err_code != vrep.simx_return_ok:
				err_code, angle = vrep.simxGetJointPosition(self.clientID, object_handle, vrep.simx_opmode_buffer)

			return angle
		except Exception as e:
			raise e

	def read_sonic_sensors(self):
		"""
			Reads the distances from the 16 ultrassonic sensors.
			Returns:
				distances: List with the distances in meters.
		"""
		distances = []
		noDetectionDist = 5.0 #Here we define the maximum distance as 5 meters

		for sensor in self.us_handle:
			res, status, distance,_,_ = vrep.simxReadProximitySensor(self.clientID, sensor, vrep.simx_opmode_streaming)
			while(res != vrep.simx_return_ok):
				res, status, distance,_,_ = vrep.simxReadProximitySensor(self.clientID, sensor, vrep.simx_opmode_buffer)

			if(status != 0):
				distances.append(distance[2])
			else:
				distances.append(noDetectionDist)

		return distances

	def read_vision_sensor(self):
		"""
			Reads the image raw data from vrep vision sensor.
			Returns:
				resolution: Tuple with the image resolution.
				image: List with the image data.
		"""
		res, resolution, image = vrep.simxGetVisionSensorImage(self.clientID, self.vision_handle, 0, vrep.simx_opmode_streaming)
		while(res != vrep.simx_return_ok):
			res, resolution, image = vrep.simxGetVisionSensorImage(self.clientID, self.vision_handle, 0, vrep.simx_opmode_buffer)

		return resolution, image

	def read_laser(self):
		"""
			Gets the 572 points read by the laser sensor. Each reading contains 3 values (x, y, z) of the point relative to the sensor position.
			Returns:
				laser: List with 1716 values of x, y and z from each point.
		"""
		res, laser = vrep.simxGetStringSignal(self.clientID,"LasermeasuredDataAtThisTime", vrep.simx_opmode_streaming)
		laser = vrep.simxUnpackFloats(laser)
		while(res != vrep.simx_return_ok):
			res, laser = vrep.simxGetStringSignal(self.clientID,"LasermeasuredDataAtThisTime", vrep.simx_opmode_buffer)
			laser = vrep.simxUnpackFloats(laser)

		return laser

	def odometry_new_pos(self, ini_pos, left_rad, right_rad):
		""" calculate new odometry given old position and wheels radial displacement """

		try:
			delta_s =self. WHEEL_RADIUS * (left_rad + right_rad) / 2
			delta_theta = self.WHEEL_RADIUS * (right_rad - left_rad) / self.DISTANCE_BETWEEN_WHEELS

			new_pos = [0.0,0.0,0.0]
			new_pos[0] = ini_pos[0] + (delta_s * np.cos(ini_pos[2] + (delta_theta/2)))
			new_pos[1] = ini_pos[1] + (delta_s * np.sin(ini_pos[2] + (delta_theta/2)))
			new_pos[2] = ini_pos[2] + delta_theta

			return new_pos
		except Exception as e:
			raise e

	def collect_odometry(self, ini_left_angle, ini_right_angle):
		""" checks odometry and stores in the global array """
		
		try:
			ini_pos = position_odometry[-1]
			new_left_angle = self.get_joint_position(self.motors_handle["left"]) 
			new_right_angle = self.get_joint_position(self.motors_handle["right"]) 

			left_angle_diff = difference_radians(ini_left_angle, new_left_angle)
			right_angle_diff = difference_radians(ini_right_angle, new_right_angle)

			robot_new_odometry = self.odometry_new_pos(ini_pos, left_angle_diff, right_angle_diff)
			
			position_odometry.append(robot_new_odometry)

			return new_left_angle, new_right_angle
		except Exception as e:
			raise e

	def collect_position_particle_filter(self, ini_left_angle, ini_right_angle):
			
			ini_pos = [self.x, self.y, self.orientation]

			new_left_angle = self.get_joint_position(self.motors_handle["left"]) + random.gauss(0.0,0.5)
			new_right_angle = self.get_joint_position(self.motors_handle["right"]) + random.gauss(0.0,0.5)

			left_angle_diff = difference_radians(ini_left_angle, new_left_angle)
			right_angle_diff = difference_radians(ini_right_angle, new_right_angle)

			robot_new_odometry = self.odometry_new_pos(ini_pos, left_angle_diff, right_angle_diff)
			
			self.x, self.y, self.orientation = robot_new_odometry

			motion = [left_angle_diff, right_angle_diff]
			measurement = self.sense(sense_noise)
			self.x, self.y, self.orientation = particle_filter(motion, measurement)

			position_particle.append(robot_new_odometry)

	def collect_ground_truth(self):
		""" checks the ground truth and stores in the global array""" 
		
		try:
			robot_pos_gt = self.get_position_orientation(self.robot_handle,-1)
			position_ground_truth.append(robot_pos_gt)
			return robot_pos_gt
		except Exception as e:
			raise e
	
	def collect_laser(self, disp = False):
    
		# Use laser2D to create a map
		## x, y, angle need to be changed later from localization
		x, y, angle = self.collect_ground_truth() # robot location from ground truth

		laser = self.read_laser()

		size = int(np.shape(laser)[0])
		for i in range(0,size,24):
			pointX, pointY = localToGlobal(x, y, angle, laser[i], laser[i+1])
			laserXY.append([pointX, pointY])


		if disp is True:
			display_map(laserXY, '2D Map from laser - ground truth')

		return laserXY
		
	def spin(self, angle, direction='l', odo_current_position=[0,0,0], v=0, odometry=True, ground_truth=True, method='timer'):
		""" turns the robot to an specific angle 
			angle in radians
		"""
		if method == 'odo':
			try:
				direction = direction.lower()

				if direction in ("left", "l"):
					m_0, m_1 = "l", "r"
					motor_handle = self.motors_handle["left"]
				elif direction in ("right", "r"):
					m_0, m_1 = "r", "l"
					motor_handle = self.motors_handle["right"]

				# initial radian position of the wheel
				ini_angle = self.get_joint_position(motor_handle)

				wheels_radians_required = (self.DISTANCE_BETWEEN_WHEELS / 2) * angle / (self.WHEEL_RADIUS)

				self.set_velocity_one_motor(m_0, -self.ROBOT_VELOCITY)
				self.set_velocity_one_motor(m_1, self.ROBOT_VELOCITY)

				while wheels_radians_required > 0:
					new_angle = self.get_joint_position(motor_handle)

					diff_radians = difference_radians(ini_angle, new_angle)
					wheels_radians_required -= diff_radians

					if ground_truth is True:
						self.collect_ground_truth()

					if odometry is True:
						if m_1 == 'l':
							odo_new_position = self.odometry_new_pos(odo_current_position, diff_radians, -diff_radians)
						elif m_1 == 'r':
							odo_new_position = self.odometry_new_pos(odo_current_position, -diff_radians, diff_radians)

						odo_current_position = odo_new_position
						position_odometry.append(odo_current_position)

					ini_angle = new_angle

					time.sleep(INTERVAL_CHECK_POSITION)


				self.set_velocity_both_motors(v)

				return odo_current_position

			except Exception as e:
				raise e

		elif method == 'timer':
			direction = direction.lower()

			if direction in ("left", "l"):
				m_0 = 1
				# motor_handle = self.motors_handle["left"]
			elif direction in ("right", "r"):
				m_0 = -1
				# motor_handle = self.motors_handle["right"]

			W = 0.4
			wheels_radians_required = (self.ROBOT_WIDTH / 2) * angle / (self.WHEEL_RADIUS)
			T = wheels_radians_required / W
			# print(T)
			start = time.time()
			self.set_velocity_one_motor('l', -m_0*W)
			self.set_velocity_one_motor('r', m_0*W)
			end = time.time()
			while (end - start)<T:
				end = time.time()
			self.set_velocity_both_motors(0)
			# time.sleep(0.2)

	def sense(self, sense_noise):
		Z = []
		for i in range(len(landmarks)):
			dist = sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
			dist += random.gauss(0.0, sense_noise)
			Z.append(dist)
		return Z

	def isWall(self):
		distances = self.read_sonic_sensors()
		if(distances[2] < 0.3 or distances[3] < 0.2 or distances[4] < 0.2 or distances[5] < 0.3):
			return True
		return False


	def follow_wall_fuzzy(self, duration=ROBOT_TIMEOUT, stop_sim_when_finished=False, signal_ex=False):

		def handler_timeout(signum, frame):
			""" just a timeout handler :) """

			raise Exception
		def foo():
			ini_left_angle = self.get_joint_position(self.motors_handle["left"])
			ini_right_angle = self.get_joint_position(self.motors_handle["right"])
			time.sleep(0.2)
			self.collect_ground_truth()
			self.collect_odometry(ini_left_angle, ini_right_angle)
			self.collect_position_particle_filter(ini_right_angle, ini_right_angle)
			# # self.collect_laser()
			threading.Timer(10e-4, foo).start()


		try:
			vrep.simxStartSimulation(self.clientID,vrep.simx_opmode_blocking)
			
			if signal_ex is True:
				signal.signal(signal.SIGALRM, handler_timeout)
				signal.alarm(4*ROBOT_TIMEOUT)

			self.set_velocity_both_motors(1)
			foo()

			start = time.time()
			while (time.time()-start)<duration:

				distances = self.read_sonic_sensors()

				while(self.isWall()):
					self.stop()
					self.spin(np.pi/8,'r')
					distances = self.read_sonic_sensors()

					self.set_left_velocity(0.5)
					self.set_right_velocity(-0.5)
					# time.sleep(0.5)

				x1 = distances[15]
				x2 = distances[0]
				delta_sx = x1 + x2 - 1
				delta_x  = x2 - x1

				while((np.abs(delta_sx) > 0.001 or np.abs(delta_x) > 0.001 ) and (time.time()-start)<duration):

					delta_vl, delta_vr = fuzzification(delta_sx, delta_x)
					dv_l = defuzzification(delta_vl)
					dv_r = defuzzification(delta_vr)

					self.set_left_velocity(2 + dv_l)
					self.set_right_velocity(2 + dv_r)
					# time.sleep(0.5)

					distances = self.read_sonic_sensors()

					while(self.isWall()):
						self.stop()
						self.spin(np.pi/8,'r')
						distances = self.read_sonic_sensors()

						self.set_left_velocity(0.5)
						self.set_right_velocity(-0.5)
						# time.sleep(0.5)
						
					x1 = distances[15]
					x2 = distances[0]
					
					delta_sx = x1 + x2 - 1
					delta_x  = x2 - x1
			
			# self.set_velocity_both_motors(0)
			# display_every(np.asarray(position_ground_truth)[:,:2],np.asarray(position_odometry)[:,:2],np.asarray(position_particle)[:,:2])

			if stop_sim_when_finished is True:
				self.set_velocity_both_motors(0)
				display_path(np.asarray(position_ground_truth)[:,:2], "Wall following Fuzzy-ground truth")
				display_path(np.asarray(position_odometry)[:,:2], "Wall following Fuzzy-odometry")
				display_path(np.asarray(position_particle)[:,:2], "Wall following Fuzzy-particle filter")
				# display_map(laserXY, "2D map from laser - with wall following fuzzy path")
				# display_every(np.asarray(position_ground_truth)[:,:2],np.asarray(position_odometry)[:,:2],np.asarray(position_particle)[:,:2])
				# vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_blocking)
		except KeyboardInterrupt:
			#print(e)
			self.set_velocity_both_motors(0)
			display_path(np.asarray(position_ground_truth)[:,:2], "Wall following Fuzzy-ground truth")
			display_path(np.asarray(position_odometry)[:,:2], "Wall following Fuzzy-odometry")
			display_path(np.asarray(position_particle)[:,:2], "Wall following Fuzzy-particle filter")
			# display_map(laserXY, "2D map from laser - with wall following fuzzy path")
			# display_every(np.asarray(position_ground_truth)[:,:2],np.asarray(position_odometry)[:,:2],np.asarray(position_particle)[:,:2])
			# vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_blocking)

		except Exception as e:
			print(e)
			self.set_velocity_both_motors(0)
			display_path(np.asarray(position_ground_truth)[:,:2], "Wall following Fuzzy-ground truth")
			display_path(np.asarray(position_odometry)[:,:2], "Wall following Fuzzy-odometry")
			display_path(np.asarray(position_particle)[:,:2], "Wall following Fuzzy-particle filter")
			# display_map(laserXY, "2D map from laser - with wall following fuzzy path")
			# display_every(np.asarray(position_ground_truth)[:,:2],np.asarray(position_odometry)[:,:2],np.asarray(position_particle)[:,:2])
			# vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_blocking)