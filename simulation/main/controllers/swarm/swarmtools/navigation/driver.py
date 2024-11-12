import time
import random
import numpy as np
import math
from rich.pretty import pprint
import csv
import matplotlib.pyplot as plt
from datetime import datetime

MAX_SPEED = 6.28
MAP_WIDTH = 250
MAP_HEIGHT = 220
RESOLUTION = 0.1
WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.18
GPS_DEVICE_NAME = "gps"


##* KALAMN FILTER STUFF
Q = np.array([[0.0319165, 0.0], 
              [0.0,       0.0]])

R = np.array([[0.0339805,       0.0], 
              [0.0, 1]])

H = np.array([[1, 0], 
              [0, 0]]) # 2x2

A = np.array([[1, 0], 
              [0, 0]]) # 2x2

I = np.eye(2)
#* 1. Initialize system estimation
X_hat = np.array([[0.0], 
                  [0.0]]) # 2x1

P_hat = np.array([[0.0, 0.0], 
                  [0.0, 0.0]]) # 2x2

##* KALAMN FILTER STUFF


Kp = 4.0  # Proportional gain
Ki = 0.0  # Integral gain
theta_integral = 0.0  # Integral of the heading error
distance_integral = 0.0 
L = 0.16  # Wheelbase (distance between the wheels)
v = 1.0  # Linear velocity (adjust as needed)
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
FILENAME = f"log_data_{timestamp}.csv"
# with open(FILENAME, mode="a", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerow(
#         [
#             "sim_time",
#             "err_dirty",
#             "err_estimated",
#         ]
#     )
with open(FILENAME, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "sim_time",
            "position",
        ]
    )


def log_data(
    position: float,
    sim_time,
    filename=FILENAME,
):
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                sim_time,
                position,
            ]
        )
        
def log_data2(
    err_dirty: float,
    err_estimated: float,

    sim_time,
    filename=FILENAME,
):
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                sim_time,
                err_dirty,
                err_estimated,

            ]
        )


class Driver:
    def __init__(self, robot):
        self.timestep = int(robot.getBasicTimeStep())
        self.robot = robot
        self.robot_name = robot.getName()
        self.robot_position = {
            "dirty_x": np.float64(0.0),
            "clean_x": np.float64(0.0),
            "dirty_x": np.float64(0.0),
            "clean_vel": np.float64(0.0),
            "dirty_vel": np.float64(0.0),
            "y": np.float64(0.0),
            "theta": np.float64(0.0),
            "imu_theta": np.float64(0.0),
            "imu_x": np.float64(0.0),
            "imu_y": np.float64(0.0),
            "imu_v_x": np.float64(0.0),
            "imu_v_y": np.float64(0.0),
        }
        self.alive = True

        # Init motors
        self.left_motor = robot.getDevice("left_wheel_motor")
        self.right_motor = robot.getDevice("right_wheel_motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        # Encoder
        self.left_encoder = self.robot.getDevice("left_wheel_sensor")
        self.right_encoder = self.robot.getDevice("right_wheel_sensor")
        self.left_encoder.enable(self.timestep)
        self.right_encoder.enable(self.timestep)
        self.prev_left_encoder = np.float64(self.left_encoder.getValue())
        self.prev_right_encoder = np.float64(self.right_encoder.getValue())
        self.current_x = 0.0
        self.prev_position = np.float64(self.left_encoder.getValue()) * WHEEL_RADIUS

        # IMU
        self.accelerometer = self.robot.getDevice("accelerometer")
        self.accelerometer.enable(self.timestep)
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(self.timestep)
        self.imu_prev_w_z = np.float64(self.gyro.getValues()[2])
        self.imu_prev_a_x = np.float64(self.accelerometer.getValues()[0])
        self.imu_prev_a_y = np.float64(self.accelerometer.getValues()[1])
        self.time_prev = 0

        # PID config
        self.waypoint_threshold = 0.01
        if self.robot_name == "TurtleBot3Burger_3":
            self.sorted_waypoints = [(0.75, 0)]
        else:
            self.sorted_waypoints = []
        self.v_linear = 2
        self.Kp_linear = 40
        self.Ki_linear = 2
        self.Kp_angular = 50.0
        self.Ki_angular = 10
        self.linear_integral = 0.0
        self.angular_integral = 0.0
        self.prev_v_clean = 0.0

        # Enable sensory devices
        self.gps = robot.getDevice(GPS_DEVICE_NAME)
        self.gps.enable(self.timestep)
        # Lidar
        self.lidar = self.robot.getDevice("lidar_sensor")
        self.lidar.enable(self.timestep)
        self.map = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.float64)

    def get_pretty_position(self):
        return f"[helper]({self.robot_name}) Robot X position:{self.robot_position["dirty_x"]:6.3f}    Robot Y position: {self.robot_position['y']:6.3f}    Robot Theta position: {self.robot_position['theta']:6.3f} ||| X+IMU_THETA position:{self.robot_position['imu_x']:6.3f}    Y+IMU_THETA position: {self.robot_position['imu_y']:6.3f}    IMU Theta position: {self.robot_position['imu_theta']:6.3f}"

    # motion
    def move_forward(self, coeff=1):
        self.left_motor.setVelocity(coeff * MAX_SPEED)
        self.right_motor.setVelocity(coeff * MAX_SPEED)

    def move_backward(self):
        self.left_motor.setVelocity(-MAX_SPEED)
        self.right_motor.setVelocity(-MAX_SPEED)

    # Positive Theta
    def anti_clockwise_spin(self, coeff=0.5):
        self.left_motor.setVelocity(-coeff * MAX_SPEED)
        self.right_motor.setVelocity(coeff * MAX_SPEED)

    # Negetive Theta
    def clockwise_spin(self, coeff=0.5):
        self.left_motor.setVelocity(coeff * MAX_SPEED)
        self.right_motor.setVelocity(-coeff * MAX_SPEED)

    def stop(self):
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def pi_controller(self, linear_v, current_vector, target_vector):
        """
        PI Controller for differential-drive robot to follow waypoints.
        """
        dx = target_vector[0] - current_vector
        
        
        v_pi = self.Kp_linear * dx + self.Ki_linear * self.linear_integral
        v_pi = max(min(v_pi, MAX_SPEED), -MAX_SPEED)

        v_clean = v_pi

        return v_clean

    def update_plot(self):
        self.ax.clear()
        self.ax.set_title(f"Robot Position in Real-Time {self.robot_name}")
        self.ax.set_xlabel("t Time")
        self.ax.set_ylabel("X Position")
        # self.ax.set_xlim(-2.5, 2.5)
        # self.ax.set_ylim(-2.5, 2.5)
        self.ax.plot(self.times,self.x_positions, "bo-", markersize=1)
        # for waypoint in self.sorted_waypoints:
        #     self.ax.plot(
        #         waypoint[0], waypoint[1], "ro", markersize=5
        #     )  # Red color for waypoints
        plt.draw()
        plt.pause(0.001)

    def sara_pid(self):
        # Initialization code here
        self.x_positions = []
        self.times = []
        self.out_positions = []
        self.out_velocities = []
        self.sensor_positions = []
        self.sensor_velocities = []
        self.prev_position = self.robot_position["dirty_x"]
        self.prev_v_clean = 0.0
        self.time_prev = self.robot.getTime()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(f"Robot Position in Real-Time {self.robot_name}")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("X Position")
        plt.ion()  # Turn on interactive mode for live updates

        # Set motors to velocity control mode
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))

        while self.robot.step(self.timestep) != -1:

            # Define target position based on current waypoint using pop
            if self.sorted_waypoints:
                target_vector = self.sorted_waypoints[0]
            else:
                # All waypoints have been reached; stop the robot
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)
                print("All waypoints reached. Stopping the robot.")
                self.alive = False
                self.stop()
                quit()
                break


            # # Check if waypoint is reached
            # if distance < self.waypoint_threshold:
            #     print(f"Waypoint ({target_vector[0]}, {target_vector[1]}) reached.")
            #     self.sorted_waypoints.pop(0)  # Remove the reached waypoint
            #     continue

            # PID Controller
            v_clean = self.pi_controller(
                self.v_linear, self.robot_position["dirty_x"], target_vector=target_vector
            )

            current_time = self.robot.getTime()
            self.time_prev = current_time

            self.robot_position["clean_vel"] = v_clean
            mean = 0
            std_dev = 1
            #! DISTURBANCE 1
            #! DISTURBANCE 1
            noise = np.random.normal(mean, std_dev) * 0.125
            # noise = 0  # Set to zero if no noise is desired
            v_dirty = v_clean + noise
            #! DISTURBANCE 1
            #! DISTURBANCE 1
            self.robot_position["dirty_vel"] = v_dirty

            # Set motor velocities
            self.left_motor.setVelocity(v_dirty)
            self.right_motor.setVelocity(v_dirty)
            clean_position = self.robot_position["clean_x"]
            clean_velocity = self.robot_position["clean_vel"]
            dirty_position = self.robot_position["dirty_x"]
            dirty_velocity = self.robot_position["dirty_vel"]



            
            # print(self.get_pretty_position())
            self.x_positions.append(dirty_position)
            self.times.append(current_time)
            self.update_plot()


    def check_encoder_not_null_and_init(self):
        # Wait until valid encoder values are available
        print(f"[localisation]({self.robot.getName()}) Waiting for encoder != nan")
        while (
            math.isnan(self.prev_left_encoder)
            or math.isnan(self.prev_right_encoder)
            or math.isnan(self.imu_prev_w_z)
            or math.isnan(self.imu_prev_a_x)
            or math.isnan(self.imu_prev_a_y)
        ):
            self.robot.step(
                self.timestep
            )  # Step the simulation until we get valid readings
            self.prev_left_encoder = self.left_encoder.getValue()
            self.prev_right_encoder = self.right_encoder.getValue()
            self.imu_prev_w_z = np.float64(self.gyro.getValues()[2])
            self.imu_prev_a_x = np.float64(self.accelerometer.getValues()[0])
            self.imu_prev_a_y = np.float64(self.accelerometer.getValues()[1])

        print(
            f"[localisation]({self.robot.getName()}) Valid Initial Left Encoder: {self.prev_left_encoder}, Valid Initial Right Encoder: {self.prev_right_encoder}"
        )
        # when encoder is live then trigger the set
        self.robot_position["clean_x"], self.robot_position["y"], current_z = (
            self.gps.getValues()
        )  # init the coords even when using wheel odom
        self.robot_position["imu_x"], self.robot_position["imu_y"], current_z = (
            self.gps.getValues()
        )
        print(
            f"[localisaton]({self.robot.getName()}) INIT WITH GPS AT: Robot X position: {self.robot_position["dirty_x"]:6.3f}    Robot Y position: {self.robot_position['y']:6.3f}    Robot Theta position: {self.robot_position['theta']:6.3f}"
        )
        # pprint(_object=self.robot_position)
        return True

    # Writing to the robot_position
    def update_odometry_o1(self):
        def kalman_filter_2x2(sensor_data):
            global K, Q, R, H, I, X_p, P_p, X_hat, P_hat
            #* 2. Predict system state
            X_p = A @ X_hat
            P_p = A @ P_hat @ A.T + Q
            
            #* 3. Compute Kalman Gain
            K = P_p @ H.T @ np.linalg.inv(H @ P_p @ H.T + R)
            # print("np.linalg.inv\(H @ P_p @ H.T + R)")
            # print(np.linalg.inv(H @ P_p @ H.T + R))
            # print("P_p")
            # print(P_p)
            # print("H.T")
            # print(H.T)
            # print("K")
            # print(K)
            
            #* 4. Estimate system state
            X_hat = X_p + K @ (sensor_data - H @ X_p)
            P_hat = P_p - K @ H @ P_p
            print(f"estimated: {X_hat[0][0]}, predicted: {X_p[0][0]}, sensor_data: {sensor_data}, ground_truth: {self.robot_position["clean_x"]}")
            log_data2(
                err_dirty=sensor_data - self.robot_position["clean_x"],
                err_estimated=X_hat[0][0] - self.robot_position["clean_x"],
                sim_time=self.robot.getTime(),
            )
            return X_hat[0][0]
        
        temp_left = np.float64(self.left_encoder.getValue())
        delta_left = temp_left - self.prev_left_encoder
        self.prev_left_encoder = temp_left
        delta_center = delta_left * WHEEL_RADIUS
        self.robot_position["clean_x"] += delta_center
        #! DISTURBANCE 2
        #! DISTURBANCE 2
        #! DISTURBANCE 2
        mean = 0
        std_dev = 0.75
        # self.robot_position["dirty_x"] = self.robot_position["clean_x"] # no noise
        self.robot_position["dirty_x"] = kalman_filter_2x2((self.robot_position["clean_x"] + np.random.normal(mean, std_dev) * 0.1250))
        # self.robot_position["dirty_x"] = (self.robot_position["clean_x"] + np.random.normal(mean, std_dev) * 0.1250)
        #! DISTURBANCE 2
        #! DISTURBANCE 2
        #! DISTURBANCE 2
        #! DISTURBANCE 2
        log_data(
                position=self.robot_position["dirty_x"],
                sim_time=self.robot.getTime(),
                filename="withKalaman_new.csv",
            )

        




    def run_odometry_service(self):
        self
        while self.alive:
            # print(self.get_pretty_position())
            # self.robot_position["dirty_x"], self.robot_position["y"], current_z = self.get_position_gps()
            # self.time_prev = self.robot.getTime()
            self.update_odometry_o1()
            time.sleep(0.01)
