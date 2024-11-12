import numpy as np


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


def kalman_filter_2x2(sensor_data):
    global K, Q, R, H, I, X_p, P_p, X_hat, P_hat
    #* 2. Predict system state
    X_p = A @ X_hat
    P_p = A @ P_hat @ A.T + Q
    
    #* 3. Compute Kalman Gain
    K = P_p @ H.T @ np.linalg.inv(H @ P_p @ H.T + R)
    print("np.linalg.inv\(H @ P_p @ H.T + R)")
    print(np.linalg.inv(H @ P_p @ H.T + R))
    print("P_p")
    print(P_p)
    print("H.T")
    print(H.T)
    print("K")
    print(K)
    
    #* 4. Estimate system state
    X_hat = X_p + K @ (sensor_data - H @ X_p)
    P_hat = P_p - K @ H @ P_p


def main():
    # Example sensor data (mock data for demonstration purposes)
    sensor_readings = [1.0, 2.0, 1.5, 2.1, 1.9, 2.3, 2.0]

    # Initialize the Kalman filter and process each sensor reading
    for reading in sensor_readings:
        kalman_filter_2x2(reading)
        print()
        print(f'{X_hat=}')
        print(
            f"Sensor input: {reading:.10f}, Estimated state: {X_hat[0][0]:.10f}, Predicted state (x̂): {X_p[0][0]:.10f}"
        )
        print(f"Predicted covariance matrix (P̂):")
        print(P_hat)
        print(f"Kalman Gain (K):")
        print(K)


if __name__ == "__main__":
    main()
