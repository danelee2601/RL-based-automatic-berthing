import numpy as np
import matplotlib.pyplot as plt
from Berthing_Simulator_PythonCode.Berthing_Simulator_rev04 import BerthingSimulator

# Initial setting
# =====================================================
L = 175  # [m]
d = 8.5  # [m]
B = 25.4  # [m]
C_b = 0.559
m = (L * B * d) * C_b * 1024  # [kg]
x_G = 6.13  # [m]
x_P_prime = -0.43

P = 6.86  # [m]
D = 6.5  # [m]

A_R = 32.5  # [m^2]
aspect_ratio_of_rudder = 1.827
H_R = 7.70  # [m]

# initial coordinates
coordinate_x = 0.0  # [m]
coordinate_y = 0.0  # [m]
pos_termination = np.array([1.5 * L, 1.5 * L])
heading_angle = 0.0  # 230 * (np.pi/180)  # [rad.]

# initial velocities
u = 5  # [m/s]
v = 0  # [m/s]
r = 0  # [rad/s]

# initial n(rps)
n = 0.7

# initial rudder angle
rudder_angle = 0.0  # [rad.]

tau = 0  # quantity of trim [m]
t_p = 0.2  # thrust deduction factor for a propeller  # typically, 0.2 from Triantafyllou(2004)

water_depth_type = 'deep water'
H = 13  # water depth [m]

V_c = 0  # velocity of current [m/s]
current_angle = 0  # [rad.]

min_max_rudder_angle = (-35 * (np.pi / 180), 35 * (np.pi / 180))  # [rad.]
rudder_rate = 5 * (np.pi / 180)  # 2.5*(np.pi/180)  # [rad/s]

min_max_rps = (-0.5, 1.0)
rps_rate = 0.017 * 3.2  # 0.017

simulationDuration = 3000 * 1e1000  # [s]
# =====================================================

# Import the class
env = BerthingSimulator(coordinate_x, coordinate_y, pos_termination, heading_angle, m, L, d, B, C_b, x_G, u, v, r,
                        x_P_prime, n, P, D, rudder_angle, A_R, aspect_ratio_of_rudder, H_R, tau=tau, t_p=t_p,
                        water_depth_type=water_depth_type, H=H, V_c=V_c, current_angle=current_angle,
                        min_max_rudder_angle=min_max_rudder_angle, rudder_rate=rudder_rate, min_max_rps=min_max_rps,
                        simulationDuration=simulationDuration)

# run simulation
control_rudder_angles = [-10, -20, -30]
turning_test_simulationTimes = [2300, 1200, 1000] * 2

hist = {'x_coord': [], 'y_coord': [], 'rudder_angle': []}
for control_rudder_angle, turning_test_simulationTime in zip(control_rudder_angles, turning_test_simulationTimes):
    for key in hist.keys():
        hist[key] = []

    for i in range(turning_test_simulationTime):
        obs = env.step(control_rudder_angle * (np.pi / 180), 1, step_size=0.5)
        hist['x_coord'].append(obs[0])
        hist['y_coord'].append(obs[1])
        hist['rudder_angle'].append(env.rudder_angle)

    # plot
    plt.plot(hist['x_coord'], hist['y_coord'], label=control_rudder_angle)

    # reset
    env.reset(coordinate_x, coordinate_y, heading_angle, u, n)

plt.legend()
plt.grid()
plt.show()
