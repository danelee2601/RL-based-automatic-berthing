import numpy as np
from .berthing_simulator.Berthing_Simulator_rev04 import BerthingSimulator

def decalre_Berthing_env():
    # Initial setting
    # =====================================================
    L = 175  # [m]
    d = 8.5  # [m]
    B = 25.4  # [m]
    C_b = 0.559
    m = (L * B * d) * C_b * 1024  # [kg]
    x_G = 7  # [m] # [m] -3.15 ->교수님 논문참고, Berthing_Simulator_rev04.py 의 경우, x_G : 7 이 잘맞음.
    x_P_prime = -0.48

    P = 6.86  # [m]
    D = 6.5  # [m]

    A_R = 32.5  # [m^2]
    aspect_ratio_of_rudder = 1.827
    H_R = 7.70  # [m]

    # initial coordinates
    coordinate_x = 0.0 * L  # [m]
    coordinate_y = 0.0 * L  # [m]
    heading_angle = 0.0 * (np.pi / 180)  # [rad.]

    # initial velocities
    u = 1.0  # [m/s]
    v = 0.0  # [m/s]
    r = 0.0  # [rad/s]

    # initial n(rps)
    n = 0.5

    # initial rudder angle
    rudder_angle = 0.0

    tau = 0  # quantity of trim [m]
    t_p = 0.2  # thrust deduction factor for a propeller  # typically, 0.2 from Triantafyllou(2004)

    water_depth_type = 'deep water'
    H = 13  # water depth [m]

    V_c = 0  # velocity of current [m/s]
    current_angle = 0  # [rad.]

    min_max_rudder_angle = (-35 * (np.pi / 180), 35 * (np.pi / 180))  # [rad.]
    rudder_rate = 3.0 * (np.pi / 180)  # by Sohn(1992)

    min_max_rps = (-1.0, 1.0)

    simulationDuration = 3000  # [s]

    pos_termination = (1.5*L, 1.5*L)
    # =====================================================

    # Import the class
    env = BerthingSimulator(coordinate_x, coordinate_y, pos_termination, heading_angle, m, L, d, B, C_b, x_G, u, v, r,
                            x_P_prime, n, P, D, rudder_angle, A_R, aspect_ratio_of_rudder, H_R, tau=tau, t_p=t_p,
                            water_depth_type=water_depth_type, H=H, V_c=V_c, current_angle=current_angle,
                            min_max_rudder_angle=min_max_rudder_angle, rudder_rate=rudder_rate, min_max_rps=min_max_rps,
                            simulationDuration=simulationDuration, verbose=False)
    return env