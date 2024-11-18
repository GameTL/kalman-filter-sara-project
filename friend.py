import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
delta_t = 0.1  # Time step (seconds)
t_total = 100  # Total simulation time (seconds)
num_steps = int(t_total / delta_t)

# Fryer parameters
tau = 5.0  # Time constant representing latent heat effect
T_initial = 20.0  # Initial temperature (°C)
dTdt_initial = 0.0  # Initial rate of change of temperature

# PID controller parameters
Kp = 2.0
Ki = 0.1
Kd = 1.0

# Kalman filter matrices
# State transition matrix A
A = np.array([[1, delta_t],
              [0, 1 - (delta_t / tau)]])

# Control input matrix B
B = np.array([[0],
              [delta_t / tau]])

# Observation matrix H
H = np.array([[1, 0]])

# Process noise covariance Q (2x2 matrix)
Q_std = 0.05  # Standard deviation for process noise
Q = np.array([[Q_std**2, 0],
              [0, Q_std**2]])

# Measurement noise covariance R (scalar, since we measure only temperature)
R_std = 0.1  # Standard deviation for measurement noise
R = np.array([[R_std**2]])

# Initial state estimate and covariance
x_est = np.array([[T_initial],
                  [dTdt_initial]])
P_est = np.eye(2)

# Target temperature
T_target = 180.0  # Desired temperature (°C)

# Initialize arrays to store simulation data
time = np.arange(0, t_total, delta_t)
T_true_history = []
T_est_history = []
U_history = [0.0]  # Start with initial control input U=0.0
error_history = []

# Initialize PID controller variables
integral = 0.0
previous_error = 0.0

# Initial true state
x_true = np.array([[T_initial],
                   [dTdt_initial]])

for k in range(num_steps):
    # Simulate process noise
    w_k = np.random.multivariate_normal([0, 0], Q).reshape((2, 1))
    
    # Update the true state with control input from the previous time step
    x_true = A @ x_true + B * U_history[-1] + w_k
    
    # Simulate measurement with measurement noise
    v_k = np.random.normal(0, R_std)
    y_k = H @ x_true + v_k

    # Time update (Predict)
    x_pred = A @ x_est + B * U_history[-1]
    P_pred = A @ P_est @ A.T + Q

    # Measurement update (Update)
    # Kalman gain
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    # Update estimate with measurement
    x_est = x_pred + K @ (y_k - H @ x_pred)
    P_est = (np.eye(2) - K @ H) @ P_pred

    # PID control
    error = T_target - x_est[0, 0]
    integral += error * delta_t
    derivative = (error - previous_error) / delta_t
    U = Kp * error + Ki * integral + Kd * derivative
    U = max(0, U)  # Heater power cannot be negative
    previous_error = error

    # Append control input for the next iteration
    U_history.append(U)

    # Store data for plotting
    T_true_history.append(x_true[0, 0])
    T_est_history.append(x_est[0, 0])
    error_history.append(error)

# Adjust U_history length to match other histories
U_history = U_history[:-1]  # Remove the extra control input

# Plotting results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time, T_true_history, label='True Temperature')
plt.plot(time, T_est_history, label='Estimated Temperature', linestyle='--')
plt.title('Temperature vs. Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time, U_history)
plt.title('Heater Power Input vs. Time')
plt.ylabel('Heater Power (U)')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time, error_history)
plt.title('Temperature Error vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Error (°C)')
plt.grid(True)

plt.tight_layout()
plt.show()