import os
import numpy as np
import pandas as pd


class BerthingSimulator(object):
    fpath = os.path.dirname(__file__)

    def __init__(self, coordinate_x, coordinate_y, pos_termination, heading_angle, m, L, d, B, C_b, x_G, u, v, r,
                 x_P_prime, n, P, D, rudder_angle, A_R, aspect_ratio_of_rudder, H_R, tau=0.0, t_p=0.2,
                 water_depth_type='shallow water', H=13, V_c=0, current_angle=0, seawater_density=1024,
                 seawater_kinematic_viscosity=1.19 * 10e-6, g=9.81,
                 min_max_rudder_angle=(-35 * (np.pi / 180), 35 * (np.pi / 180)), rudder_rate=2.5 * (np.pi / 180),
                 min_max_rps=(-0.5, 1.0), simulationDuration=3000, verbose=True):
        """
        :param coordinate_x: initial coordinate x [m]
        :param coordinate_y: initial coordinate y [m]
        :param heading_angle: initial heading angle[rad.]
        :param m: a ship's mass [kg]
        :param L: L_bp [m]
        :param d: draft [m]
        :param B: Breadth [m]
        :param C_b:
        :param x_G: x coordinate of ship's center of gravity from the origin of the center(middle) of a ship.
        :param u: initial velocity in the surge direction [m/s^2]
        :param v: initial velocity in the sway direction [m/s^2]
        :param r: initial velocity in the yaw direction [m/s^2]
        :param x_P_prime: non-dimensional x coordinate of a propeller from the origin(0,0) of the middle of a ship
        :param n: initial rps
        :param P: Pitch of a propeller
        :param D: Diameter of a propeller
        :param initial rudder_angle: [rad]
        :param A_R: a profile area of movable part of a rudder  # [m^2]
        :param aspect_ratio_of_rudder: (span of a rudder) / (chord length of a rudder)
        :param H_R: rudder height (= span of a rudder)
        :param tau: quantity of trim [m] = (draft in after perpendicular - draft in fore perpendicular)  defined in Kijima(1990)
        :param t_p: thrust deduction factor for a propeller  # typically, 0.2 from Triantafyllou(2004)
        :param water_depth_type: 'shallow (water)' or 'deep (water)'
        :param H: water depth [m]
        :param V_c: velocity of current
        :param current_angle: [rad]
        :param seawater_density: typically, 1024 [kg/m^3]
        :param seawater_kinematic_viscosity: 1.19*10e-6 [m^2/s], presented from Takashiro(1980)
        :param g: gravity, 9.81 [m/s^2]
        :param min_max_rudder_angle: (min_rudder_angle, max_rudder_angle) [rad.]
        :param rudder_rate: [rad/s]
        :param min_max_rps: (min_rps, max_rps)
        :param simulationDuration: [s]

        # Possible Further Improvement :
            1. adding wind force
            2. adding wave force
            3. adding tug boat effect/force
        """
        self.simulationDuration = simulationDuration
        self.step_count = 0
        self.Global_Ep = 0
        self.small_val = 1e-6
        self.verbose = verbose

        self.m = m  # ship's mass [m]
        self.L = L  # L_bp [m]
        self.d = d  # draft [m]
        self.B = B  # breadth [m]
        self.C_b = C_b  # block coefficient
        self.seawater_density = seawater_density  # typically, 1024 [kg/m^3]
        self.seawater_kinematic_viscosity = seawater_kinematic_viscosity  # [m^2/s]  # from Takashiro(1980)
        self.g = g  # [m/s^2]
        self.x_G = x_G  # x coordinate of the center of gravity of a ship given that the center of a ship is the origin (0, 0) [m]
        self.tau = tau  # quantity of trim [m] = (draft in after perpendicular - draft in fore perpendicular)  defined in Kijima(1990)
        self.water_depth_type = water_depth_type  # 'shallow (water)' || 'deep (water)'
        self.H = H
        if 'shallow' in self.water_depth_type:
            self.h = self.d / self.H

            self.Inoue_chart_shallow_water_effect = \
                pd.read_excel(os.path.join(self.fpath, 'Inoue_chart_shallow_water_effect/Inoue_chart_shallow_water_effect.xlsx'))
            Inoue_chart_closest_value_idx = \
                self.find_closest_value_idx(self.h, self.Inoue_chart_shallow_water_effect['X'].values)
            self.C_DH_over_C_D_infinite = self.Inoue_chart_shallow_water_effect['Y'].values[
                Inoue_chart_closest_value_idx]

            self.Shallow_water_effect_on_X_vr_prime = \
                pd.read_excel(os.path.join(self.fpath, 'Shallow_water_effect_on_X_vr_prime/Shallow_water_effect_on_X_vr_prime.xlsx'))
            self.Shallow_water_effect_on_a_H = \
                pd.read_excel(os.path.join(self.fpath, 'Shallow_water_effect_on_a_H/Shallow_water_effect_on_a_H.xlsx'))
            self.Shallow_water_effect_on_x_H_prime = \
                pd.read_excel(os.path.join(self.fpath, 'Shallow_water_effect_on_x_H_prime/Shallow_water_effect_on_x_H_prime.xlsx'))

        self.k_yy = 0.25 * self.L  # typically, k_yy is (0.24 ~ 0.26) * L, 부유체운동조종론 p.7
        self.k_zz = self.k_yy  # inertia radius, k_zz is similar to k_yy when a ship's height and breadth are similar
        self.I_zz = self.m * self.k_zz ** 2  # moment of inertia(관성모멘트), 부유체운동조종론 p.8

        # Estimating equations for added mass by Clarke(1983), 부유체운동조종(p.173)
        self.m_prime = self.m / ((1 / 2) * self.seawater_density * self.L ** 2 * self.d)  # non-dimensional mass
        self.m_x_prime = 0.04 * self.m_prime  # non-dimensional added mass in the surge direction  # (0.03~0.05)*m'
        self.m_y_prime = np.pi * (self.d / self.L) * (1.0 + 0.16 * self.C_b * (self.B / self.d) - 5.1 * (
                    self.B / self.L) ** 2)  # non-dimensional added mass in the sway direction
        self.J_zz_prime = np.pi * (self.d / self.L) * (1 / 12 + 0.017 * self.C_b * (self.B / self.d) - 0.33 * (
                    self.B / self.L))  # non-dimensional added mass in the yaw direction

        if 'shallow' in self.water_depth_type:
            K_0 = 1 + (0.0775 / (self.h - 1) ** 2) - (0.0110 / (self.h - 1) ** 3)
            K_1 = - (0.0643 / (self.h - 1)) + (0.0724 / (self.h - 1) ** 2) - (0.0113 / (self.h - 1) ** 3)
            K_2 = 0.0342 / (self.h - 1)

            # Shallow water effect for added mass and added moment of inertia by Sheng(1981)
            self.m_y_prime = (K_0 + (2 / 3) * K_1 * (self.B / self.d) + (8 / 15) * K_2 * (
                        self.B / self.d) ** 2) * self.m_y_prime
            self.J_zz_prime = (K_0 + (2 / 5) * K_1 * (self.B / self.d) + (24 / 105) * K_2 * (
                        self.B / self.d) ** 2) * self.J_zz_prime

        self.m_x = self.m_x_prime * (
                    (1 / 2) * self.seawater_density * self.L ** 2 * self.d)  # added mass in the surge direction
        self.m_y = self.m_y_prime * (
                    (1 / 2) * self.seawater_density * self.L ** 2 * self.d)  # added mass in the sway direction
        self.J_zz = self.J_zz_prime * (
                    (1 / 2) * self.seawater_density * self.L ** 4 * self.d)  # added mass in the yaw direction

        # Initial velocities and heading angle of a ship  # These must be updated every step.
        self.u = u  # [m/s]
        self.init_u = np.copy(u)
        self.v = v  # [m/s]
        self.V = np.sqrt(self.u ** 2 + self.v ** 2)  # [m/s]  # defined by Yasukawa and Yoshimura(2015)
        self.r = r  # [rad/s]

        self.coordinate_x = coordinate_x  # initial coordinate x [m]
        self.coordinate_y = coordinate_y  # initial coordinate x [m]
        self.pos_termination = pos_termination
        self.heading_angle = heading_angle  # initial heading angle [rad.]

        self.imag_line_ang = 20 * (np.pi / 180)  # [rad]
        self.d1, self.d2 = self.get_d1_d2(pos_xy=(self.coordinate_x, self.coordinate_y),
                                          pos_termination=self.pos_termination)

        self.u_prime = self.u / (self.V + self.small_val)
        self.v_prime = self.v / (self.V + self.small_val)
        self.r_prime = self.r * self.L / (self.V + self.small_val)

        # Initialize current speed and angle as zero
        self.V_c = V_c  # [m/s]
        self.current_angle = current_angle  # [rad.]

        self.alpha = 0  # for Munk moment and some additional added mass, but we ignore it, setting it as zero.

        self.available_C_b_list_for_C_R = np.array(
            [0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84,
             0.86])  # C_R is estimated by an estimating graph by Takashiro(1980)
        if not np.min(self.available_C_b_list_for_C_R) <= self.C_b <= np.max(self.available_C_b_list_for_C_R):
            if self.verbose:
                print(
                    "[Error] available C_b for the estimated residual resistance coefficient by Takashiro(1980) is {}\n".format(
                        self.available_C_b_list_for_C_R))
            assert False
        if np.min(self.available_C_b_list_for_C_R) <= self.C_b <= np.max(self.available_C_b_list_for_C_R):
            if not self.C_b in self.available_C_b_list_for_C_R:

                closest_available_C_b_idx = self.find_closest_value_idx(self.C_b, self.available_C_b_list_for_C_R)
                self.C_b_for_C_R = self.available_C_b_list_for_C_R[closest_available_C_b_idx]
                if self.verbose:
                    print(
                        "[Warning] Because there is no C_b for the estimated residual resistance coefficient(C_R) by Takashiro(1980), we changed your input C_b({}) to {}. (NOTE:C_b is changed for estimating C_R only. In other equations, your input C_b is used as it was input.)\n".format(
                            self.C_b, self.C_b_for_C_R))

        self.str_C_b = self.C_b_to_str_C_b(self.C_b_for_C_R)
        self.residual_resistance_coefficient_estimate = \
            pd.read_excel(os.path.join(self.fpath, 'residual_resistance_coefficient_estimate/C_b_{}.xlsx'.format(self.str_C_b)))  # residual resistance coefficient_estimate graph by Takashiro(1980)

        self.t_p = t_p  # thrust deduction factor for a propeller  # typically, 0.2 from Triantafyllou(2004)
        self.x_P_prime = x_P_prime  # non-dimensional x coordinate of a propeller from the origin(0,0) of the middle point of a ship.

        self.w_P0 = 0.5 * self.C_b - 0.05  # by D.W taylor, presented by Kijima(1990)

        self.w_P = self.get_w_P(
            self.u)  # initial effective wake fraction  # must be updated after the velocities are updated
        self.u_P = self.get_u_P(self.u,
                                self.w_P)  # initial u_P            # must be updated after the velocities are updated

        self.available_pitch_ratio_for_C_T_star_list = np.array([0.8, 1.0, 1.2])
        self.available_pitch_ratio_for_POW_list = np.array([0.8, 1.0, 1.2])

        self.n = n  # initial rps
        self.init_n = np.copy(n)
        self.P = P  # pitch [m]
        self.D = D  # diameter of a propeller [m]
        self.pitch_ratio = self.P / self.D
        self.pitch_ratio = self.adjust_pitch_ratio(self.pitch_ratio)
        self.Y_P_star_curve = pd.read_excel(os.path.join(self.fpath, 'Y_P_star/Y_P_star.xlsx'))
        self.N_P_star_curve = pd.read_excel(os.path.join(self.fpath, 'N_P_star/N_P_star.xlsx'))
        self.C_T_star_curve = pd.read_excel(os.path.join(self.fpath, 'C_T_star/C_T_star_pitch_ratio_{}.xlsx'.format(self.pitch_ratio)))
        self.POW_curve = pd.read_excel(os.path.join(self.fpath, 'POW_curve/POW_curve_pitch_ratio_{}.xlsx'.format(self.pitch_ratio)))

        self.hydrodynamic_pitch_angle = 0  # initialized

        self.t_R = 1 - 0.28 * self.C_b - 0.55  # coefficient for additional drag for a rudder
        self.a_H_curve = \
            pd.read_excel(os.path.join(self.fpath, 'a_H_curve/a_H_curve.xlsx'))  # ratio of additional lateral force for a rudder  # from Kijima(1990)
        self.x_H_prime_curve = \
            pd.read_excel(os.path.join(self.fpath, 'x_H_prime_curve/x_H_prime_curve.xlsx'))  # non-dimensional distance between the center of gravity of a ship and center of additional lateral force  # from Kijima(1990)
        self.x_H_curve = pd.concat((self.x_H_prime_curve['X'], self.x_H_prime_curve['Y'] * self.L), axis=1)
        self.x_R_prime = -0.5  # non-dimensional distance between the center of gravity of ship and center of lateral force  # typically, -0.5.  # from 부유체운동조종 p.212
        self.x_R = self.x_R_prime * self.L
        self.rudder_angle = rudder_angle  # initial rudder angle [rad.]  # delta
        self.A_R = A_R  # a profile area of movable part of a rudder  # [m^2]
        self.aspect_ratio_of_rudder = aspect_ratio_of_rudder  # = (span of a rudder) / (chord length of a rudder)
        self.H_R = H_R  # rudder height (= span of a rudder)
        a_H_closest_value_idx = self.find_closest_value_idx(self.C_b, self.a_H_curve['X'].values)
        self.a_H = self.a_H_curve['Y'].values[a_H_closest_value_idx]
        x_H_closest_value_idx = self.find_closest_value_idx(self.C_b, self.x_H_curve['X'].values)
        self.x_H = self.x_H_curve['Y'].values[x_H_closest_value_idx]
        if self.C_b > np.max(self.a_H_curve['X'].values):
            if self.verbose:
                print("[Warning] The input C_b is greater than the maximum range of the estimated a_H curve.")
                print("The input C_b : {} || the maximum range of the estimated a_H curve : {}\n".format(
                    self.C_b, np.max(self.a_H_curve['X'].values)))
        if self.C_b > np.max(self.x_H_curve['X'].values):
            if self.verbose:
                print("[Warning] The input C_b is greater than the maximum range of the estimated x_H curve.")
                print("The input C_b : {} || the maximum range of the estimated x_H curve : {}\n".format(
                    self.C_b, np.max(self.x_H_curve['X'].values)))
        if 'shallow' in self.water_depth_type:
            Shallow_water_effect_on_a_H_closest_value_idx = self.find_closest_value_idx(
                self.h, self.Shallow_water_effect_on_a_H['X'].values)
            a_H_H_over_a_H_infinite = \
                self.Shallow_water_effect_on_a_H['Y'].values[Shallow_water_effect_on_a_H_closest_value_idx]
            self.a_H = a_H_H_over_a_H_infinite * self.a_H
            if self.h > np.max(self.Shallow_water_effect_on_a_H['X'].values):
                if self.verbose:
                    print(
                        "[Warning] draft(d)/water_depth(H) exceeds the maximum range of the shallow water effect on a_H curve.")
                    print(
                        "Input draft(d)/water_depth(H) : {} || the maximum range of the shallow water effect on a_H curve : {}\n".format(
                            self.h, self.Shallow_water_effect_on_a_H['X'].values))

            Shallow_water_effect_on_x_H_prime_closest_value_idx = self.find_closest_value_idx(self.h,
                                                                                              self.Shallow_water_effect_on_x_H_prime[
                                                                                                  'X'].values)
            x_H_prime_H_over_X_H_infinite_prime = self.Shallow_water_effect_on_x_H_prime['Y'].values[
                Shallow_water_effect_on_x_H_prime_closest_value_idx]
            self.x_H = x_H_prime_H_over_X_H_infinite_prime * self.x_H  # Since x_H_prime_H_over_X_H_infinite_prime is a ratio, it's directly applicable to x_H.
            if self.h > np.max(self.Shallow_water_effect_on_x_H_prime['X'].values):
                if self.verbose:
                    print(
                        "[Warning] draft(d)/water_depth(H) exceeds the maximum range of the shallow water effect on x_H_prime curve.")
                    print(
                        "Input draft(d)/water_depth(H) : {} || the maximum range of the shallow water effect on x_H_prime curve : {}\n".format(
                            self.h, self.Shallow_water_effect_on_x_H_prime['X'].values))

        # Hydrodynamic forces (you can add hydrodynamic forces for Tug, Wind)
        # Mathematical model at low speed defined by Son(손경호)(1992)
        # For the hydrodynamic coefficients in the sway, yaw directions, they are estimated by Inoue et al.(1981), 이성욱교수님 졸업논문 p.17
        self.X_H = self.get_X_H(
            self.u)  # initial hydrodynamic force on hull in the surge direction                             # must be updated after the velocities are updated
        self.X_P = self.get_X_P(self.u,
                                self.u_P)  # initial hydrodynamic force on propeller in the surge direction                      # must be updated after the velocities are updated
        self.X_R = self.get_X_R(self.u, self.w_P,
                                self.u_P)  # initial hydrodynamic force on rudder in the surge direction       # must be updated after the velocities are updated
        self.Y_H = self.get_Y_H_for_diff_v_star(
            self.v)  # initial hydrodynamic force on hull in the sway direction                              # must be updated after the velocities are updated
        self.Y_P = self.get_Y_P()  # initial hydrodynamic force on propeller in the sway direction                               # must be updated after the velocities are updated
        self.Y_R = self.get_Y_R_for_diff_v_star(
            self.v)  # initial hydrodynamic force on rudder in the sway direction                            # must be updated after the velocities are updated
        self.N_H = self.get_N_H_for_diff_r(
            self.r)  # initial hydrodynamic force on hull in the yaw direction                               # must be updated after the velocities are updated
        self.N_P = self.get_N_P()  # initial hydrodynamic force on propeller in the yaw direction                                # must be updated after the velocities are updated
        self.N_R = self.get_N_R_for_diff_r(
            self.r)  # initial hydrodynamic force on rudder in the yaw direction                             # must be updated after the velocities are updated

        self.diff_u_star = self.get_diff_u_star(
            self.u)  # initial diff_u_star  # Must not be updated during the Runge-Kutta calculation
        self.diff_v_star = self.get_diff_v_star(
            self.v)  # initial diff_v_star  # Must not be updated during the Runge-Kutta calculation
        self.diff_r = self.get_diff_r(
            self.r)  # initial diff_r       # Must not be updated during the Runge-Kutta calculation

        self.min_max_rudder_angle = min_max_rudder_angle  # [rad.]
        self.rudder_rate = rudder_rate

        self.min_max_rps = min_max_rps

        # initial hydrodynamic coefficients
        # self.X_uu_prime = 0
        # self.X_vr_prime = 0
        # self.Y_v_prime = 0
        # self.Y_ur_prime = 0
        # self.N_v_prime = 0
        # self.N_uv_prime = 0
        # self.N_r_prime = 0
        # self.Y_vv_prime = 0
        # self.Y_vr_prime = 0
        # self.Y_urr_prime = 0
        # self.N_rr_prime = 0
        # self.N_vvr_prime = 0
        # self.N_uvrr_prime = 0

        self.q_a = 0.0  # used in Runge-Kutta Gill
        self.q_b = 0.0

    def adjust_pitch_ratio(self, pitch_ratio):
        if pitch_ratio not in self.available_pitch_ratio_for_C_T_star_list:
            pitch_ratio_closest_value_idx = self.find_closest_value_idx(self.pitch_ratio,
                                                                        self.available_pitch_ratio_for_C_T_star_list)
            pitch_ratio_for_C_T_star = self.available_pitch_ratio_for_C_T_star_list[pitch_ratio_closest_value_idx]
            if (self.step_count == 0) and (self.Global_Ep == 0):
                if self.verbose:
                    print(
                        "[Warning] The input pitch_ratio(propeller_pitch / propeller_diameter)(={}) changed to {}.".format(
                            pitch_ratio, pitch_ratio_for_C_T_star))
                    print("Because the available_pitch_ratio_for_C_T_star_list : {}".format(
                        self.available_pitch_ratio_for_C_T_star_list))
                    print("If you'd like to, you can add the interpolation code for the C_T_star curve.\n")
        else:
            pitch_ratio_for_C_T_star = pitch_ratio

        if pitch_ratio not in self.available_pitch_ratio_for_POW_list:
            pitch_ratio_closest_value_idx = self.find_closest_value_idx(pitch_ratio,
                                                                        self.available_pitch_ratio_for_POW_list)
            pitch_ratio_for_POW = self.available_pitch_ratio_for_POW_list[pitch_ratio_closest_value_idx]
            if (self.step_count == 0) and (self.Global_Ep == 0):
                if self.verbose:
                    print(
                        "[Warning] The input pitch_ratio(propeller_pitch / propeller_diameter)(={}) changed to {}.".format(
                            pitch_ratio, pitch_ratio_for_POW))
                    print("Because the available_pitch_ratios_for_POW : {}".format(
                        self.available_pitch_ratio_for_POW_list))
                    print("If you'd like to, you can add the interpolation code for the POW curve.\n")
        else:
            pitch_ratio_for_POW = pitch_ratio

        if pitch_ratio_for_C_T_star != pitch_ratio_for_POW:
            if self.verbose:
                print("[Error] pitch_ratio_for_C_T_star(={}) != pitch_ratio_for_POW(={})\n".format(
                    pitch_ratio_for_C_T_star, pitch_ratio_for_POW))
            assert False
        else:
            return pitch_ratio_for_C_T_star

    def get_C_R(self, V):
        F_n = V / np.sqrt(self.L * self.g)  # Froude number
        closest_value_idx = self.find_closest_value_idx(F_n, self.residual_resistance_coefficient_estimate['X'].values)
        C_R = self.residual_resistance_coefficient_estimate['Y'].values[closest_value_idx]
        return C_R  # estimated residual resistance coefficient by Takashiro(1980)

    def get_X_H(self, u):
        V = np.sqrt(u ** 2 + self.v ** 2)
        u_prime = u / (V + self.small_val)

        X_uu_prime = -0.00957  # -0.00957 | -0.0076862(교수님논문)

        C_m = 1.675 * self.C_b - 0.505  # by Hasegawa(1980), 부유체운동조종 p.221
        X_vr_prime = (C_m - 1) * self.m_y_prime

        if 'shallow' in self.water_depth_type:
            if self.h >= np.max(self.Shallow_water_effect_on_X_vr_prime['X'].values) and (
                    self.step_count == 0 and self.Global_Ep == 0):
                if self.verbose:
                    print(
                        "[Warning] draft(d)/water_depth(H) exceeds the max range for the estimated graph of X_vr_prime.")
                    print("draft(d)/water_depth(H):{} || max range for the estimated graph of X_vr_prime:{}\n".format(
                        self.h, np.max(self.Shallow_water_effect_on_X_vr_prime['X'].values)))
            X_vr_H_prime_over_X_vr_infinite_prime_closest_value_idx = self.find_closest_value_idx(self.h,
                                                                                                  self.Shallow_water_effect_on_X_vr_prime[
                                                                                                      'X'].values)
            X_vr_H_prime_over_X_vr_infinite_prime = self.Shallow_water_effect_on_X_vr_prime['Y'].values[
                X_vr_H_prime_over_X_vr_infinite_prime_closest_value_idx]
            X_vr_prime = X_vr_H_prime_over_X_vr_infinite_prime * X_vr_prime

        if V != 0:
            # X_H = (1/2) * self.seawater_density * self.L * self.d * V**2 * ( X_uu_prime*u_prime*np.abs(u_prime) + X_vr_prime*self.v_prime*self.r_prime )  # -> original according to the 교수님졸업논문
            """
            # [Experiment]. coz I think that X_H should be negative when it goes forward. Force on Hull that is supposed to be the force in the opposing direction to u to slown down the ship speed. This sounds logical.
            # [Result] without "-1 *", ship speed diverges || with "-1 *", ship speed converges.
            """
            X_H = (1 / 2) * self.seawater_density * self.L * self.d * V ** 2 * (
                        X_uu_prime * u_prime * np.abs(u_prime) + X_vr_prime * self.v_prime * self.r_prime)
        else:
            X_H = 0

        return X_H

    def get_Y_H_for_diff_v_star(self, v):
        V = np.sqrt(self.u ** 2 + v ** 2)
        v_prime = v / (V + self.small_val)

        Lambda = (2 * self.d) / self.L

        Y_v_prime = - ((1 / 2) * np.pi * Lambda + 1.4 * self.C_b * (self.B / self.L)) * (
                    1 + (2 / 3) * (self.tau / self.d))
        Y_ur_prime = (1 / 4) * np.pi * Lambda * (1 + 0.8 * (self.tau / self.d))
        Y_vv_prime = -6.49 * (1 - self.C_b) * (self.d / self.B) + 0.0795
        Y_vr_prime = 1.82 * (1 - self.C_b) * (self.d / self.B) - 0.447
        Y_urr_prime = -0.4664 * (1 - self.C_b) * (self.d / self.B)

        # Shallow water effect by Kijima(1990) for Y_v_prime, Y_ur_prime
        # Shallow water effect by Inoue(1984) for Y_vv_prime, Y_urr_prime, Y_vr_prime
        if 'shallow' in self.water_depth_type:

            c_for_Y_v_prime = 0.40 * self.C_b * self.B / self.d  # c for Y_v_prime_s
            fh_for_Y_v_prime = 1 / ((
                                                1 - self.h) ** c_for_Y_v_prime) - self.h  # correction factor to consider the shallow water effect
            Y_v_prime = fh_for_Y_v_prime * Y_v_prime

            if self.verbose:
                print("(deep water) Y_ur_prime: ", Y_ur_prime)
            c1_for_Y_ur_prime = -5.5 * (self.C_b * self.B / self.d) ** 2 + 26 * self.C_b * self.B / self.d - 31.5
            c2_for_Y_ur_prime = 37 * (self.C_b * self.B / self.d) ** 2 - 185 * self.C_b * self.B / self.d - 230
            c3_for_Y_ur_prime = -38 * (self.C_b * self.B / self.d) ** 2 + 197 * self.C_b * self.B / self.d - 250
            fh_for_Y_ur_prime = 1 + c1_for_Y_ur_prime * self.h + c2_for_Y_ur_prime * self.h ** 2 + c3_for_Y_ur_prime * self.h ** 3
            # Y_ur_prime = fh_for_Y_ur_prime*(Y_ur_prime - (self.m_prime + self.m_x_prime)) + (self.m_prime + self.m_x_prime)  # 맞겠지?.. 교수님졸업논문 p.18
            # Y_ur_prime = fh_for_Y_ur_prime * Y_ur_prime  # 이건 내 추측이긴 한데.  # 이걸쓰면 turning circle 의 tactical diameter 가 줄어들고, shallow water 에서는 tactical diameter 가 작은게 옳다.
            if self.verbose:
                print("(shallow water) Y_ur_prime: ", Y_ur_prime)

            Y_vv_prime = 1.80 * self.C_DH_over_C_D_infinite * Y_vv_prime
            Y_urr_prime = 0.80 * self.C_DH_over_C_D_infinite * Y_urr_prime
            Y_vr_prime = 1.20 * self.C_DH_over_C_D_infinite * Y_vr_prime

        if V != 0:
            Y_H = (1 / 2) * self.seawater_density * self.L * self.d * V ** 2 * (
                        Y_v_prime * v_prime + Y_ur_prime * self.u_prime * self.r_prime + Y_vv_prime * v_prime * np.abs(
                    v_prime) + Y_vr_prime * v_prime * np.abs(
                    self.r_prime) + Y_urr_prime * self.u_prime * self.r_prime * np.abs(self.r_prime))
        else:
            Y_H = 0

        return Y_H

    def get_Y_H_for_diff_r(self, r):
        V = np.sqrt(self.u ** 2 + self.v ** 2)
        v_prime = self.v / (V + self.small_val)

        r_prime = r * self.L / (V + self.small_val)

        Lambda = (2 * self.d) / self.L

        Y_v_prime = - ((1 / 2) * np.pi * Lambda + 1.4 * self.C_b * (self.B / self.L)) * (
                    1 + (2 / 3) * (self.tau / self.d))
        Y_ur_prime = (1 / 4) * np.pi * Lambda * (1 + 0.8 * (self.tau / self.d))
        Y_vv_prime = -6.49 * (1 - self.C_b) * (self.d / self.B) + 0.0795
        Y_vr_prime = 1.82 * (1 - self.C_b) * (self.d / self.B) - 0.447
        Y_urr_prime = -0.4664 * (1 - self.C_b) * (self.d / self.B)

        # Shallow water effect by Kijima(1990) for Y_v_prime, Y_ur_prime
        # Shallow water effect by Inoue(1984) for Y_vv_prime, Y_urr_prime, Y_vr_prime
        if 'shallow' in self.water_depth_type:
            c_for_Y_v_prime = 0.40 * self.C_b * self.B / self.d  # c for Y_v_prime
            fh_for_Y_v_prime = 1 / ((
                                                1 - self.h) ** c_for_Y_v_prime) - self.h  # correction factor to consider the shallow water effect
            Y_v_prime = fh_for_Y_v_prime * Y_v_prime

            c1_for_Y_ur_prime = -5.5 * (self.C_b * self.B / self.d) ** 2 + 26 * self.C_b * self.B / self.d - 31.5
            c2_for_Y_ur_prime = 37 * (self.C_b * self.B / self.d) ** 2 - 185 * self.C_b * self.B / self.d - 230
            c3_for_Y_ur_prime = -38 * (self.C_b * self.B / self.d) ** 2 + 197 * self.C_b * self.B / self.d - 250
            fh_for_Y_ur_prime = 1 + c1_for_Y_ur_prime * self.h + c2_for_Y_ur_prime * self.h ** 2 + c3_for_Y_ur_prime * self.h ** 3
            # Y_ur_prime = fh_for_Y_ur_prime*(Y_ur_prime - (self.m_prime + self.m_x_prime)) + (self.m_prime + self.m_x_prime)  # 맞겠지?.. 교수님졸업논문 p.18
            # Y_ur_prime = fh_for_Y_ur_prime * Y_ur_prime  # 이건 내 추측이긴 한데.  # 이걸쓰면 turning circle 의 tactical diameter 가 줄어들고, shallow water 에서는 tactical diameter 가 작은게 옳다.

            Y_vv_prime = 1.80 * self.C_DH_over_C_D_infinite * Y_vv_prime
            Y_urr_prime = 0.80 * self.C_DH_over_C_D_infinite * Y_urr_prime
            Y_vr_prime = 1.20 * self.C_DH_over_C_D_infinite * Y_vr_prime

        if V != 0:
            Y_H = (1 / 2) * self.seawater_density * self.L * self.d * V ** 2 * (
                        Y_v_prime * v_prime + Y_ur_prime * self.u_prime * r_prime + Y_vv_prime * v_prime * np.abs(
                    v_prime) + Y_vr_prime * v_prime * np.abs(r_prime) + Y_urr_prime * self.u_prime * r_prime * np.abs(
                    r_prime))
        else:
            Y_H = 0

        return Y_H

    def get_N_H_for_diff_v_star(self, v):
        V = np.sqrt(self.u ** 2 + v ** 2)
        v_prime = v / (V + self.small_val)

        r_prime = self.r * self.L / (V + self.small_val)

        Lambda = (2 * self.d) / self.L
        l_v_prime = Lambda / ((1 / 2) * np.pi * Lambda + 1.4 * self.C_b * (self.B / self.L))

        N_v_prime = 0
        N_uv_prime = - Lambda * (1 - (0.27 / l_v_prime) * (self.tau / self.d))
        N_r_prime = - (0.54 * Lambda - Lambda ** 2) * (1 + 0.3 * (self.tau / self.d))
        N_rr_prime = -1.70 * np.abs(self.C_b * (self.B / self.L) - 0.157) ** 1.5 - 0.010
        N_vvr_prime = -3.25 * self.C_b * (self.B / self.L) + 0.35 - 10 ** -7 * (self.L / (self.C_b * self.L)) ** 6
        N_uvrr_prime = 0.444 * self.C_b * (self.d / self.B) - 0.064

        # Shallow water effect by Kijima(1990) for N_uv_prime, N_r_prime
        # Shallow water effect by Inoue(1984) for N_rr_prime, N_uvrr_prime, N_vvr_prime
        if 'shallow' in self.water_depth_type:
            c_for_N_uv_prime = 0.425 * self.C_b * self.B / self.d  # c for N_uv_prime
            fh_for_N_uv_prime = 1 / ((
                                                 1 - self.h) ** c_for_N_uv_prime) - self.h  # correction factor to consider the shallow water effect
            N_uv_prime = fh_for_N_uv_prime * N_uv_prime

            c_for_N_r_prime = -7.14 * Lambda + 1.5
            fh_for_N_r_prime = 1 / ((
                                                1 - self.h) ** c_for_N_r_prime) - self.h  # correction factor to consider the shallow water effect
            N_r_prime = fh_for_N_r_prime * N_r_prime

            N_rr_prime = 0.80 * self.C_DH_over_C_D_infinite * N_rr_prime
            N_uvrr_prime = 0.80 * self.C_DH_over_C_D_infinite * N_uvrr_prime
            N_vvr_prime = 0.65 * self.C_DH_over_C_D_infinite * N_vvr_prime

        if V != 0:
            N_H = (1 / 2) * self.seawater_density * self.L ** 2 * self.d * V ** 2 * (
                        N_v_prime * v_prime + N_uv_prime * self.u_prime * v_prime + N_r_prime * r_prime + N_vvr_prime * v_prime ** 2 * r_prime + N_uvrr_prime * self.u_prime * v_prime * r_prime ** 2 + N_rr_prime * r_prime * np.abs(
                    r_prime))
        else:
            N_H = (1 / 2) * self.seawater_density * self.L ** 4 * self.d * N_rr_prime * self.r * np.abs(self.r)
        return N_H

    def get_N_H_for_diff_r(self, r):
        r_prime = r * self.L / (self.V + self.small_val)

        Lambda = (2 * self.d) / self.L
        l_v_prime = Lambda / ((1 / 2) * np.pi * Lambda + 1.4 * self.C_b * (self.B / self.L))

        N_v_prime = 0
        N_uv_prime = - Lambda * (1 - (0.27 / l_v_prime) * (self.tau / self.d))
        N_r_prime = - (0.54 * Lambda - Lambda ** 2) * (1 + 0.3 * (self.tau / self.d))
        N_rr_prime = -1.70 * np.abs(self.C_b * (self.B / self.L) - 0.157) ** 1.5 - 0.010
        N_vvr_prime = -3.25 * self.C_b * (self.B / self.L) + 0.35 - 10 ** -7 * (self.L / (self.C_b * self.L)) ** 6
        N_uvrr_prime = 0.444 * self.C_b * (self.d / self.B) - 0.064

        # Shallow water effect by Kijima(1990) for N_uv_prime, N_r_prime
        # Shallow water effect by Inoue(1984) for N_rr_prime, N_uvrr_prime, N_vvr_prime
        if 'shallow' in self.water_depth_type:
            c_for_N_uv_prime = 0.425 * self.C_b * self.B / self.d  # c for N_uv_prime
            fh_for_N_uv_prime = 1 / ((
                                                 1 - self.h) ** c_for_N_uv_prime) - self.h  # correction factor to consider the shallow water effect
            N_uv_prime = fh_for_N_uv_prime * N_uv_prime

            c_for_N_r_prime = -7.14 * Lambda + 1.5
            fh_for_N_r_prime = 1 / ((
                                                1 - self.h) ** c_for_N_r_prime) - self.h  # correction factor to consider the shallow water effect
            N_r_prime = fh_for_N_r_prime * N_r_prime

            N_rr_prime = 0.80 * self.C_DH_over_C_D_infinite * N_rr_prime
            N_uvrr_prime = 0.80 * self.C_DH_over_C_D_infinite * N_uvrr_prime
            N_vvr_prime = 0.65 * self.C_DH_over_C_D_infinite * N_vvr_prime

        if self.V != 0:
            N_H = (1 / 2) * self.seawater_density * self.L ** 2 * self.d * self.V ** 2 * (
                        N_v_prime * self.v_prime + N_uv_prime * self.u_prime * self.v_prime + N_r_prime * r_prime + N_vvr_prime * self.v_prime ** 2 * r_prime + N_uvrr_prime * self.u_prime * self.v_prime * r_prime ** 2 + N_rr_prime * r_prime * np.abs(
                    r_prime))
        else:
            N_H = (1 / 2) * self.seawater_density * self.L ** 4 * self.d * N_rr_prime * r * np.abs(r)
        return N_H

    def get_w_P(self, u):
        if u > 0:
            w_P = self.w_P0 * np.exp(-4.0 * (self.v_prime + self.x_P_prime * self.r_prime) ** 2)
        else:
            w_P = 0
        return w_P

    def get_u_P(self, u, w_P):
        return u * (1 - w_P)

    def get_X_P(self, u, u_P, verbose=True):
        if not ((u >= 0) and (self.n >= 0)):  # 제 1상한(1st quadrant)이 아니라면., 교수님 졸업논문 p.7
            self.hydrodynamic_pitch_angle = self.get_hydrodynamic_pitch_angle(u, u_P, verbose)

            C_T_star_closest_idx = self.find_closest_value_idx(self.hydrodynamic_pitch_angle,
                                                               self.C_T_star_curve['X'].values)
            C_T_star = self.C_T_star_curve['Y'].values[C_T_star_closest_idx]

            X_P = (1 / 8) * self.seawater_density * np.pi * self.D ** 2 * (1 - self.t_p) * C_T_star * (
                        u_P ** 2 + (0.7 * np.pi * self.n * self.D) ** 2)
        else:
            J = u_P / (self.n * self.D + self.small_val)  # advance ratio
            K_T = self.get_K_T(J)  # thrust coefficient
            X_P = (1 - self.t_p) * K_T * self.seawater_density * self.n ** 2 * self.D ** 4
            if self.verbose:
                print("제 1상한, J:{}, K_T:{}, X_P:{}".format(J, K_T, X_P)) if verbose else None
        return X_P

    def get_K_T(self, J):
        K_T_closest_value_idx = self.find_closest_value_idx(J, self.POW_curve['X'].values)
        K_T = self.POW_curve['Y'].values[K_T_closest_value_idx]
        return K_T

    def get_hydrodynamic_pitch_angle(self, u, u_P, verbose):
        updated_hydrodynamic_pitch_angle = 0  # initial
        if (u >= 0) and (self.n <= 0):  # 제 2상한 (2nd quadrant)  # it's overlapped with the 4th quadrant at zero tho.
            if self.verbose:
                print("제 2상한 (u >= 0) and (self.n <= 0)") if verbose else None
            updated_hydrodynamic_pitch_angle = np.arctan(u_P / (0.7 * np.pi * self.n * self.D + self.small_val)) * (
                        180 / np.pi) + 180  # [deg.]
        elif (u <= 0) and (self.n <= 0):  # 제 3상한 (3rd quadrant)
            if self.verbose:
                print("제 3상한 (u <= 0) and (self.n <= 0)") if verbose else None
            updated_hydrodynamic_pitch_angle = np.arctan(u_P / (0.7 * np.pi * self.n * self.D + self.small_val)) * (
                        180 / np.pi) + 180  # [deg.]
        elif (u <= 0) and (self.n >= 0):  # 제 4상한 (4th quadrant)
            if self.verbose:
                print("제 4상한 (u <= 0) and (self.n >= 0)") if verbose else None
            updated_hydrodynamic_pitch_angle = np.arctan(u_P / (0.7 * np.pi * self.n * self.D + self.small_val)) * (
                        180 / np.pi) + 180 * 2  # [deg.]
        return updated_hydrodynamic_pitch_angle

    def get_Y_P(self):
        if (self.u >= 0) and (self.n < -0.2):  # 제 2상한 (2nd quadrant)
            J_s = self.u / (self.n * self.P + self.small_val)
            Y_P_star_closest_value_idx = self.find_closest_value_idx(-J_s, self.Y_P_star_curve['X'].values)
            Y_P_star = self.Y_P_star_curve['Y'].values[Y_P_star_closest_value_idx]  # estimated by Fujino(1977)
            Y_P = self.seawater_density * self.n ** 2 * self.D ** 4 * Y_P_star
            if -J_s > np.max(self.Y_P_star_curve['X'].values):
                if self.verbose:
                    print("[Warning] -J_s(={}) exceeds the maximum range of the estimated Y_P_star curve(={})\n".format(
                        -J_s, np.max(self.Y_P_star_curve['X'].values)))
        else:
            Y_P = 0
        return Y_P

    def get_N_P(self):
        if (self.u >= 0) and (self.n < -0.1):  # 제 2상한 (2nd quadrant)
            J_s = self.u / (self.n * self.P + self.small_val)
            N_P_star_closest_value_idx = self.find_closest_value_idx(-J_s, self.N_P_star_curve['X'].values)
            N_P_star = self.N_P_star_curve['Y'].values[N_P_star_closest_value_idx]  # estimated by Fujino(1977)
            N_P = self.seawater_density * self.n ** 2 * self.D ** 4 * self.L * N_P_star
            if -J_s > np.max(self.N_P_star_curve['X'].values):
                if self.verbose:
                    print("[Warning] -J_s(={}) exceeds the maximum range of the estimated N_P_star curve(={})\n".format(
                        -J_s, np.max(self.N_P_star_curve['X'].values)))
        else:
            N_P = 0
        return N_P

    def get_X_R(self, u, w_P, u_P):
        F_N = self.get_F_N_for_X_R(u, w_P, u_P)  # The normal force acting on a rudder  # by Fujii
        X_R = -(1 - self.t_R) * F_N * np.sin(self.rudder_angle)
        return X_R

    def get_Y_R_for_diff_v_star(self, v):
        F_N = self.get_F_N_affected_by_v(v)
        Y_R = -(1 + self.a_H) * F_N * np.cos(self.rudder_angle)
        return Y_R

    def get_Y_R_for_diff_r(self, r):
        F_N = self.get_F_N_affected_by_r(r)
        Y_R = -(1 + self.a_H) * F_N * np.cos(self.rudder_angle)
        return Y_R

    def get_N_R_for_diff_r(self, r):
        F_N = self.get_F_N_affected_by_r(r)
        N_R = -(self.x_R + self.a_H * self.x_H) * F_N * np.cos(self.rudder_angle)
        return N_R

    def get_N_R_for_diff_v_star(self, v):
        F_N = self.get_F_N_affected_by_v(v)  # it's not for Y_R but it's used because it makes use of v
        Y_R = -(self.x_R + self.a_H * self.x_H) * F_N * np.cos(self.rudder_angle)
        return Y_R

    def get_F_N_for_X_R(self, u, w_P, u_P):
        epsilon = -156.2 * (self.C_b * self.B / self.L) ** 2 + 41.6 * (
                    self.C_b * self.B / self.L) - 1.76  # wake fraction ratio by Kijima(1990)
        eta = self.D / self.H_R
        k = 0.6 / epsilon
        s = 1 - (u_P / (self.n * self.P + self.small_val))
        u_R = epsilon * self.n * self.P * np.sqrt(
            1 - 2 * (1 - eta * k) * s + (1 - eta * k * (2 - k)) * s ** 2) + self.small_val

        gamma_R = -22.2 * (self.C_b * self.B / self.L) ** 2 + 0.02 * (
                    self.C_b * self.B / self.L) + 0.68  # flow straightening coefficient by Kijima(1990)
        l_R_prime = -0.9  # non-dimensional x coordinate of a center of a rudder from the origin of the center of a ship (negative value)  # typically, -0.9 ~ -1.0, 부유체운동조종 p.208
        l_R = l_R_prime * self.L
        v_R = - gamma_R * (self.v + l_R * self.r)

        V_R = np.sqrt(u_R ** 2 + v_R ** 2)  # [m/s^2]

        w_R = - epsilon * (1 - w_P) + 1

        if (u >= 0) and (self.n >= 0) and (s < 0):
            u_R = u * (1 - w_R) + self.small_val
        if (u >= 0) and (self.n < 0):
            V_R = 0
        if (u < 0) and (self.n > 0):
            u_R = u + 0.6 * np.sqrt(eta) * self.n * self.P
            if not u_R >= 0:
                if self.verbose:
                    print("[Warning] u_R should be >= 0. Currently, it is {}.\n".format(u_R))

        alpha_R = self.rudder_angle - np.arctan(v_R / u_R)  # [rad.]
        if not np.abs(alpha_R) <= 35 * (np.pi / 180):
            if self.verbose:
                print("[Warning] abs(alpha_R) should be less than 35 degrees. Currently, it is {} degrees.\n".format(
                    np.abs(alpha_R) * (180 / np.pi)))

        if (u < 0) and (self.n <= 0):
            V_R = np.sqrt(u ** 2 + (self.v + l_R * self.r) ** 2)
            alpha_R = - self.rudder_angle + np.arctan((self.v + l_R * self.r) / np.abs(u))
            if not np.abs(alpha_R) <= 35 * (np.pi / 180):
                if self.verbose:
                    print(
                        "[Warning] abs(alpha_R) should be less than 35 degrees. Currently, it is {} degrees.\n".format(
                            np.abs(alpha_R) * (180 / np.pi)))

        f_alpha = (6.13 * self.aspect_ratio_of_rudder) / (self.aspect_ratio_of_rudder + 2.25)
        if not 1.0 <= self.aspect_ratio_of_rudder <= 2.5 and (self.step_count == 0 and self.Global_Ep == 0):
            if self.verbose:
                print(
                    "[Warning] aspect_ratio_of_rudder should be in the range of (1.0 ~ 2.5). Currently, it is {}.\n".format(
                        self.aspect_ratio_of_rudder))

        F_N = (1 / 2) * self.seawater_density * self.A_R * V_R ** 2 * f_alpha * np.sin(alpha_R)

        if self.verbose:
            print("u_R: ", u_R)
            print("v_R: ", v_R)
            print("alpha_R: ", alpha_R)
            print("F_N: ", F_N)
        return F_N

    def get_F_N_affected_by_v(self, v):
        epsilon = -156.2 * (self.C_b * self.B / self.L) ** 2 + 41.6 * (
                    self.C_b * self.B / self.L) - 1.76  # wake fraction ratio by Kijima(1990)
        eta = self.D / self.H_R
        k = 0.6 / epsilon
        s = 1 - (self.u_P / (self.n * self.P + self.small_val))
        u_R = epsilon * self.n * self.P * np.sqrt(
            1 - 2 * (1 - eta * k) * s + (1 - eta * k * (2 - k)) * s ** 2) + self.small_val

        gamma_R = -22.2 * (self.C_b * self.B / self.L) ** 2 + 0.02 * (
                    self.C_b * self.B / self.L) + 0.68  # flow straightening coefficient by Kijima(1990)
        l_R_prime = -0.9  # non-dimensional x coordinate of a center of a rudder from the origin of the center of a ship (negative value)  # typically, -0.9 ~ -1.0, 부유체운동조종 p.208
        l_R = l_R_prime * self.L
        v_R = - gamma_R * (v + l_R * self.r)

        V_R = np.sqrt(u_R ** 2 + v_R ** 2)  # [m/s^2]

        w_R = - epsilon * (1 - self.w_P) + 1

        if (self.u >= 0) and (self.n >= 0) and (s < 0):
            u_R = self.u * (1 - w_R) + self.small_val
        if (self.u >= 0) and (self.n < 0):
            V_R = 0
        if (self.u < 0) and (self.n > 0):
            u_R = self.u + 0.6 * np.sqrt(eta) * self.n * self.P
            if not u_R >= 0:
                if self.verbose:
                    print("[Warning] u_R should be >= 0. Currently, it is {}.\n".format(u_R))

        alpha_R = self.rudder_angle - np.arctan(v_R / u_R)  # [rad.]
        if not np.abs(alpha_R) <= 35 * (np.pi / 180):
            if self.verbose:
                print("[Warning] abs(alpha_R) should be less than 35 degrees. Currently, it is {} degrees.\n".format(
                    np.abs(alpha_R) * (180 / np.pi)))

        if (self.u < 0) and (self.n <= 0):
            V_R = np.sqrt(self.u ** 2 + (v + l_R * self.r) ** 2)
            alpha_R = - self.rudder_angle + np.arctan((v + l_R * self.r) / np.abs(self.u))
            if not np.abs(alpha_R) <= 35 * (np.pi / 180):
                if self.verbose:
                    print(
                        "[Warning] abs(alpha_R) should be less than 35 degrees. Currently, it is {} degrees.\n".format(
                            np.abs(alpha_R) * (180 / np.pi)))

        f_alpha = (6.13 * self.aspect_ratio_of_rudder) / (self.aspect_ratio_of_rudder + 2.25)
        if not 1.0 <= self.aspect_ratio_of_rudder <= 2.5 and (self.step_count == 0 and self.Global_Ep == 0):
            if self.verbose:
                print(
                    "[Warning] aspect_ratio_of_rudder should be in the range of (1.0 ~ 2.5). Currently, it is {}.\n".format(
                        self.aspect_ratio_of_rudder))

        F_N = (1 / 2) * self.seawater_density * self.A_R * V_R ** 2 * f_alpha * np.sin(alpha_R)

        return F_N

    def get_F_N_affected_by_r(self, r):
        epsilon = -156.2 * (self.C_b * self.B / self.L) ** 2 + 41.6 * (
                    self.C_b * self.B / self.L) - 1.76  # wake fraction ratio by Kijima(1990)
        eta = self.D / self.H_R
        k = 0.6 / epsilon
        s = 1 - (self.u_P / (self.n * self.P + self.small_val))
        u_R = epsilon * self.n * self.P * np.sqrt(
            1 - 2 * (1 - eta * k) * s + (1 - eta * k * (2 - k)) * s ** 2) + self.small_val

        gamma_R = -22.2 * (self.C_b * self.B / self.L) ** 2 + 0.02 * (
                    self.C_b * self.B / self.L) + 0.68  # flow straightening coefficient by Kijima(1990)
        l_R_prime = -0.9  # non-dimensional x coordinate of a center of a rudder from the origin of the center of a ship (negative value)  # typically, -0.9 ~ -1.0, 부유체운동조종 p.208
        l_R = l_R_prime * self.L
        v_R = - gamma_R * (self.v + l_R * r)

        V_R = np.sqrt(u_R ** 2 + v_R ** 2)  # [m/s^2]

        w_R = - epsilon * (1 - self.w_P) + 1

        if (self.u >= 0) and (self.n >= 0) and (s < 0):
            u_R = self.u * (1 - w_R) + self.small_val
        if (self.u >= 0) and (self.n < 0):
            V_R = 0
        if (self.u < 0) and (self.n > 0):
            u_R = self.u + 0.6 * np.sqrt(eta) * self.n * self.P
            if not u_R >= 0:
                if self.verbose:
                    print("[Warning] u_R should be >= 0. Currently, it is {}.\n".format(u_R))

        alpha_R = self.rudder_angle - np.arctan(v_R / u_R)  # [rad.]
        if not np.abs(alpha_R) <= 35 * (np.pi / 180):
            if self.verbose:
                print("[Warning] abs(alpha_R) should be less than 35 degrees. Currently, it is {} degrees.\n".format(
                    np.abs(alpha_R) * (180 / np.pi)))

        if (self.u < 0) and (self.n <= 0):
            V_R = np.sqrt(self.u ** 2 + (self.v + l_R * r) ** 2)
            alpha_R = - self.rudder_angle + np.arctan((self.v + l_R * r) / np.abs(self.u))
            if not np.abs(alpha_R) <= 35 * (np.pi / 180):
                if self.verbose:
                    print(
                        "[Warning] abs(alpha_R) should be less than 35 degrees. Currently, it is {} degrees.\n".format(
                            np.abs(alpha_R) * (180 / np.pi)))

        f_alpha = (6.13 * self.aspect_ratio_of_rudder) / (self.aspect_ratio_of_rudder + 2.25)
        if not 1.0 <= self.aspect_ratio_of_rudder <= 2.5 and (self.step_count == 0 and self.Global_Ep == 0):
            if self.verbose:
                print(
                    "[Warning] aspect_ratio_of_rudder should be in the range of (1.0 ~ 2.5). Currently, it is {}.\n".format(
                        self.aspect_ratio_of_rudder))

        F_N = (1 / 2) * self.seawater_density * self.A_R * V_R ** 2 * f_alpha * np.sin(alpha_R)

        return F_N

    def get_diff_u_star(self, u):
        w_P = self.get_w_P(u)
        u_P = self.get_u_P(u, w_P)

        X_H = self.get_X_H(u)
        X_P = self.get_X_P(u, u_P)
        X_R = self.get_X_R(u, w_P, u_P)

        diff_u_star = ((self.m + self.m_y) * self.v * self.r + (
                    self.m * self.x_G + self.m_y * self.alpha) * self.r ** 2 - (
                                   self.m + self.m_x) * self.V_c * self.r * np.sin(
            self.current_angle - self.heading_angle) + X_H + X_P + X_R) / (
                                  self.m + self.m_x)  # acceleration of a ship in the surge direction
        return diff_u_star

    def get_diff_v_star(self, v):
        Y_H = self.get_Y_H_for_diff_v_star(v)
        Y_P = self.get_Y_P()
        Y_R = self.get_Y_R_for_diff_v_star(v)

        N_H = self.get_N_H_for_diff_v_star(v)
        N_P = self.N_P  # N_P is not affected by v at all
        N_R = self.get_N_R_for_diff_v_star(v)

        diff_v_star = (self.I_zz + self.J_zz) * (
                    -(self.m + self.m_x) * self.u * self.r + (self.m + self.m_y) * self.V_c * self.r * np.cos(
                self.current_angle - self.heading_angle) + Y_H + Y_P + Y_R - (
                                self.m * self.x_G + self.m_y * self.alpha) * (-self.m * self.x_G * self.u * self.r + (
                        self.m * self.x_G + self.m_y * self.alpha) * self.V_c * self.r * np.cos(
                self.current_angle - self.heading_angle) + N_H + N_P + N_R) / (self.I_zz + self.J_zz)) / (
                                  (self.m + self.m_y) * (self.I_zz + self.J_zz) - (
                                      self.m * self.x_G + self.m_y * self.alpha) ** 2)  # acceleration of a ship in the sway direction
        return diff_v_star

    def get_diff_r(self, r):
        N_H = self.get_N_H_for_diff_r(r)
        N_P = self.get_N_P()
        N_R = self.get_N_R_for_diff_r(r)

        Y_H = self.get_Y_H_for_diff_r(r)
        Y_P = self.Y_P  # Y_P is not affected by r at all
        Y_R = self.get_Y_R_for_diff_r(r)

        diff_r = (self.m + self.m_y) * (-self.m * self.x_G * self.u * r + (
                    self.m * self.x_G + self.m_y * self.alpha) * self.V_c * r * np.cos(
            self.current_angle - self.heading_angle) + N_H + N_P + N_R - (self.m * self.x_G + self.m_y * self.alpha) * (
                                                    -(self.m + self.m_x) * self.u * r + (
                                                        self.m + self.m_y) * self.V_c * r * np.cos(
                                                self.current_angle - self.heading_angle) + Y_H + Y_P + Y_R) / (
                                                    self.m + self.m_y)) / (
                             (self.I_zz + self.J_zz) * (self.m + self.m_y) - (
                                 self.m * self.x_G + self.m_y * self.alpha) ** 2)  # acceleration of a ship in the yaw direction
        return diff_r

    def fa_u(self, x):
        """
        derivative function
        x:(u, integral_u) and x':(u', u)
        """
        return self.get_diff_u_star(x[0])

    def fb_u(self, x):
        """
        derivative function
        x:(u, integral_u) and x':(u', u)
        """
        return x[0]

    def fa_v(self, x):
        """
        derivative function
        x:(v, integral_v) and x':(v', v)
        """
        return self.get_diff_v_star(x[0])

    def fb_v(self, x):
        """
        derivative function
        x:(v, integral_v) and x':(v', v)
        """
        return x[0]

    def fa_r(self, x):
        """
        derivative function
        x:(r, integral_r) and x':(r', r)
        """
        return self.get_diff_r(x[0])

    def fb_r(self, x):
        """
        derivative function
        x:(r, integral_r) and x':(r', r)
        """
        return x[0]

    def rK4(self, x, fa, fb, hs):
        """
        :param x: x
        :param fa: derivative function of the 1st component of the array x'
        :param fb: derivative function of the 2nd component of the array x'
        :param hs: step size (time step)
        :return: x_i+1 == next state of the input x_i
        """
        a = x[0]
        b = x[1]

        a1 = fa([a, b]) * hs
        b1 = fb([a, b]) * hs

        ak = a + a1 * 0.5  # ak, bk are temporary variables to help calculation
        bk = b + b1 * 0.5

        a2 = fa([ak, bk]) * hs
        b2 = fb([ak, bk]) * hs

        ak = a + a2 * 0.5
        bk = b + b2 * 0.5

        a3 = fa([ak, bk]) * hs
        b3 = fb([ak, bk]) * hs

        ak = a + a3
        bk = b + b3

        a4 = fa([ak, bk]) * hs
        b4 = fb([ak, bk]) * hs

        a = a + (a1 + 2 * (a2 + a3) + a4) / 6
        b = b + (b1 + 2 * (b2 + b3) + b4) / 6

        return tuple([a, b])

    def rKGill(self, x, fa, fb, h):
        """
        :param x: x
        :param fa: derivative function of the 1st component of the array x'
        :param fb: derivative function of the 2nd component of the array x'
        :param hs: step size (time step)
        :return: x_i+1 == next state of the input x_i
        """
        y_0_a = x[0]
        y_0_b = x[1]

        # 제1단계
        k_0_a = h * fa([y_0_a, y_0_b])
        r_1_a = (1 / 2) * k_0_a - self.q_a
        y_1_1_a = y_0_a + r_1_a
        self.q_a = 3 * r_1_a - 1 / 2 * k_0_a + self.q_a

        k_0_b = h * fb([y_0_a, y_0_b])
        r_1_b = (1 / 2) * k_0_b - self.q_b
        y_1_1_b = y_0_b + r_1_b
        self.q_b = 3 * r_1_b - 1 / 2 * k_0_b + self.q_b

        # 제2단계
        k_1_a = h * fa([y_1_1_a, y_1_1_b])
        r_2_a = (1 - 1 / np.sqrt(2)) * (k_1_a - self.q_a)
        y_1_2_a = y_1_1_a + r_2_a
        self.q_a = self.q_a + 3 * r_2_a - (1 - 1 / np.sqrt(2)) * k_1_a

        k_1_b = h * fb([y_1_1_a, y_1_1_b])
        r_2_b = (1 - 1 / np.sqrt(2)) * (k_1_b - self.q_b)
        y_1_2_b = y_1_1_b + r_2_b
        self.q_b = self.q_b + 3 * r_2_b - (1 - 1 / np.sqrt(2)) * k_1_b

        # 제3단계
        k_2_a = h * fa([y_1_2_a, y_1_2_b])
        r_3_a = (1 + 1 / np.sqrt(2)) * (k_2_a - self.q_a)
        y_1_3_a = y_1_2_a + r_3_a
        self.q_a = self.q_a + 3 * r_3_a - (1 + 1 / np.sqrt(2)) * k_2_a

        k_2_b = h * fb([y_1_2_a, y_1_2_b])
        r_3_b = (1 + 1 / np.sqrt(2)) * (k_2_b - self.q_b)
        y_1_3_b = y_1_2_b + r_3_b
        self.q_b = self.q_b + 3 * r_3_b - (1 + 1 / np.sqrt(2)) * k_2_b

        # 제4단계
        k_3_a = h * fa([y_1_3_a, y_1_3_b])
        r_4_a = 1 / 6 * (k_3_a - 2 * self.q_a)
        y_1_4_a = y_1_3_a + r_4_a
        self.q_a = self.q_a + 3 * r_4_a - 1 / 2 * k_3_a

        k_3_b = h * fb([y_1_3_a, y_1_3_b])
        r_4_b = 1 / 6 * (k_3_b - 2 * self.q_b)
        y_1_4_b = y_1_3_b + r_4_b
        self.q_b = self.q_b + 3 * r_4_b - 1 / 2 * k_3_b

        return tuple([y_1_4_a, y_1_4_b])

    def step(self, rudder_angle, n, step_size=1.0, verbose=True):
        """
        :param rudder_angle: [rad.]
        :param n: rps
        :param step_size: == time_step, should be smaller than 1s
        :param verbose:
        :return:

        - Using Runge Kutta 4th
        - (u, v, r), (u_prime, v_prime, r_prime), V, (n, rudder_angle)  :  must be updated at the end.
        - coordinate_x, coordinate_y, heading_angle  :  must be updated after them too.
        - update the hydrodynamic forces at the very end.
        """

        if step_size > 1.0:
            step_size = 1.0
            if self.step_count == 0 and self.Global_Ep == 0:
                if self.verbose:
                    print("[Warning] Maximum step_size is 1.0 [s].")
                    print("The input step_size is changed to 1.0 [s] forcefully.\n")
            n_steps_for_rK4 = round(1 / step_size)
        else:
            n_steps_for_rK4 = round(1 / step_size)
            step_size = 1 / n_steps_for_rK4
            if self.step_count == 0 and self.Global_Ep == 0:
                if self.verbose:
                    print("[Warning] The input step_size is set as {}".format(step_size))
                    print("to make it fit to 1 second. For example, 0.2 -> it fits. 0.3 -> doesn't fit.\n")

        for i in range(n_steps_for_rK4):
            # run rK4
            x_u = [self.u,
                   0.0]  # x:(u, integral_u) and x':(u', u)  # because integral_u is distance in the surge direction, we set it as zero and observe how far it moves by x_u[1] and split it into integral_u_x, integral_u_y (in global coordinate) and use them to update the global coordinate.
            x_v = [self.v,
                   0.0]  # x:(v, integral_v) and x':(v', v)  # because integral_v is distance in the sway direction, we set it as zero and observe how far it moves by x_v[1] and split it into integral_v_x, integral_v_y (in global coordinate) and use them to update the global coordinate.
            x_r = [self.r,
                   0.0]  # x:(r, integral_r) and x':(r', r)  # because integral_r is distance[rad] in the yaw direction, we set it as zero and observe how much it rotates by r_v[1] and use it to update the global heading_angle.

            x_u = self.rKGill(x_u, self.fa_u, self.fb_u, step_size)  # step() for the surge direction
            x_v = self.rKGill(x_v, self.fa_v, self.fb_v, step_size)  # step() for the sway direction
            x_r = self.rKGill(x_r, self.fa_r, self.fb_r, step_size)  # step() for the yaw direction

            # update u, v, r, V, u', v', r' and
            # update n, rudder_angle, coordinate_x, coordinate_y, heading_angle
            # =======================================================
            previous_u = self.u
            previous_v = self.v
            previous_r = self.r

            current_u = x_u[0]
            current_v = x_v[0]
            current_r = x_r[0]

            if self.verbose:
                print("current_u - previous_u : ", current_u - previous_u)
                print("current_v - previous_v : ", current_v - previous_v)
                print("current_r - previous_r : ", current_r - previous_r)
            self.u += current_u - previous_u
            self.v += current_v - previous_v
            self.r += current_r - previous_r

            self.V = np.sqrt(self.u ** 2 + self.v ** 2)

            self.u_prime = self.u / (self.V + self.small_val)
            self.v_prime = self.v / (self.V + self.small_val)
            self.r_prime = self.r * self.L / (self.V + self.small_val)

            self.n = self.update_n(n, n_steps_for_rK4)
            self.rudder_angle = self.update_rudder_angle(target_rudder_angle=rudder_angle,
                                                         n_steps_for_rK4=n_steps_for_rK4)

            integral_u = x_u[1]  # moved distance in local coordinate system in surge direction
            integral_u_x = integral_u * np.sin(
                self.heading_angle)  # moved distance in x axis in global coordinate system
            integral_u_y = integral_u * np.cos(
                self.heading_angle)  # moved distance in y axis in global coordinate system

            integral_v = x_v[1]  # moved distance in local coordinate system in sway direction
            integral_v_x = integral_v * np.sin(
                self.heading_angle + np.pi / 2)  # moved distance in x axis in global coordinate system
            integral_v_y = integral_v * np.cos(
                self.heading_angle + np.pi / 2)  # moved distance in y axis in global coordinate system

            integral_r = x_r[1]  # moved heading angle [rad.]
            if self.verbose:
                print("integral_r : ", integral_r)  # 방향성은 맞는데 magnitude 가 현실성이 없네.

            self.coordinate_x += integral_u_x + integral_v_x  # update the x, y global coordinate
            self.coordinate_y += integral_u_y + integral_v_y  # update the x, y global coordinate
            self.heading_angle += integral_r  # update the heading angle
            # =======================================================

            # update the hydrodynamic forces
            self.update_hydrodynamic_forces()

            # update step_count
            self.step_count += step_size  # [s]

        if self.step_count == self.simulationDuration:
            if self.verbose:
                print('The simulation ended. Please do "reset()"')
                print('Current simulation step:{}[s] || simulationDuration:{}[s]\n'.format(self.step_count,
                                                                                           self.simulationDuration))
            return None

        if verbose:
            self.print_verbose()

        updated_coordinates_velocities = np.array(
            [self.coordinate_x, self.coordinate_y, self.heading_angle, self.u, self.v, self.r])
        self.d1, self.d2 = self.get_d1_d2(pos_xy=(self.coordinate_x, self.coordinate_y),
                                          pos_termination=self.pos_termination)  # update d1, d2
        return updated_coordinates_velocities

    def print_verbose(self):
        if self.verbose:
            print("""
            =====================
            coordinate x: {:0.1f} [m]
            coordinate y: {:0.1f} [m]
            heading_angle: {:0.2f} [deg.]
            
            u: {:0.2f} [m/s]
            v: {:0.2f} [m/s]
            r: {:0.2f} [deg/s]
            
            n: {:0.2f} [rps]
            rudder_angle: {:0.2f} [deg.]
            
            step_count: {} [s]
            =====================
            """.format(self.coordinate_x, self.coordinate_y, self.heading_angle * (180 / np.pi),
                       self.u, self.v, self.r * (180 / np.pi),
                       self.n, self.rudder_angle * (180 / np.pi),
                       self.step_count))

    def update_n(self, n, n_steps_for_rK4, T_n=15):
        """
        from Shon(1992)
        T_n: 15 (default)
        """
        target_n = n
        previous_n = self.n

        min_n = self.min_max_rps[0]
        max_n = self.min_max_rps[1]

        diff_n = ((target_n - previous_n) * n_steps_for_rK4) / T_n  # 초당 rps 변화 속도

        updated_n = previous_n + (diff_n / n_steps_for_rK4)

        if updated_n < min_n:
            updated_n = min_n
        elif updated_n > max_n:
            updated_n = max_n

        return updated_n

    def update_rudder_angle(self, target_rudder_angle, n_steps_for_rK4, T_E=2.5):
        """
        from Shon(1992)
        T_E: 15 (default)
        """
        previous_rudder_angle = self.rudder_angle

        min_rudder_angle = self.min_max_rudder_angle[0]
        max_rudder_angle = self.min_max_rudder_angle[1]

        diff_rudder_angle = ((
                                         target_rudder_angle - previous_rudder_angle) * n_steps_for_rK4) / T_E  # 초당 rudder angle 변화 속도
        if np.abs(diff_rudder_angle) <= self.rudder_rate:
            diff_rudder_angle = diff_rudder_angle
        else:
            rudder_moving_direction = np.sign(target_rudder_angle - previous_rudder_angle)
            diff_rudder_angle = rudder_moving_direction * self.rudder_rate

        updated_rudder_angle = previous_rudder_angle + (diff_rudder_angle / n_steps_for_rK4)

        if updated_rudder_angle < min_rudder_angle:
            updated_rudder_angle = min_rudder_angle
        elif updated_rudder_angle > max_rudder_angle:
            updated_rudder_angle = max_rudder_angle

        return updated_rudder_angle

    def reset(self, init_coordinate_x, init_coordinate_y, init_heading_angle, init_u=1.0, init_n=0.5, init_v=0.0,
              init_r=0.0, init_rudder_angle=0.0):
        """
        :param init_coordinate_x: [m]
        :param init_coordinate_y: [m]
        :param init_heading_angle: [rad.]
        :param init_u:
        :param init_v:
        :param init_r: [rad/s]
        :param init_n: [rps]
        :param init_rudder_angle: [rad]
        :return:

        - reset self.step_count
        - update self.Global_Ep
        """
        self.step_count = 0
        self.Global_Ep += 1

        self.coordinate_x = init_coordinate_x
        self.coordinate_y = init_coordinate_y
        self.heading_angle = init_heading_angle

        self.u = init_u  # [m/s]
        self.v = init_v  # [m/s]
        self.V = np.sqrt(self.u ** 2 + self.v ** 2)  # [m/s]  # defined by Yasukawa and Yoshimura(2015)
        self.r = init_r  # [rad/s]

        self.rudder_angle = init_rudder_angle  # [rad.]

        self.u_prime = self.u / (self.V + self.small_val)
        self.v_prime = self.v / (self.V + self.small_val)
        self.r_prime = self.r * self.L / (self.V + self.small_val)

        self.n = init_n

    def update_hydrodynamic_forces(self):
        """
        Must be done after the step() or at the end of the step()
        """
        self.w_P = self.get_w_P(self.u)
        self.u_P = self.u * (1 - self.w_P)

        self.X_H = self.get_X_H(
            self.u)  # initial hydrodynamic force on hull in the surge direction                             # must be updated after the velocities are updated
        self.X_P = self.get_X_P(self.u,
                                self.u_P)  # initial hydrodynamic force on propeller in the surge direction                      # must be updated after the velocities are updated
        self.X_R = self.get_X_R(self.u, self.w_P,
                                self.u_P)  # initial hydrodynamic force on rudder in the surge direction       # must be updated after the velocities are updated
        self.Y_H = self.get_Y_H_for_diff_v_star(
            self.v)  # initial hydrodynamic force on hull in the sway direction                              # must be updated after the velocities are updated
        self.Y_P = self.get_Y_P()  # initial hydrodynamic force on propeller in the sway direction                               # must be updated after the velocities are updated

        self.Y_R = self.get_Y_R_for_diff_v_star(
            self.v)  # initial hydrodynamic force on rudder in the sway direction                            # must be updated after the velocities are updated
        self.N_H = self.get_N_H_for_diff_r(
            self.r)  # initial hydrodynamic force on hull in the yaw direction                               # must be updated after the velocities are updated
        self.N_P = self.get_N_P()  # initial hydrodynamic force on propeller in the yaw direction                                # must be updated after the velocities are updated
        self.N_R = self.get_N_R_for_diff_r(
            self.r)  # initial hydrodynamic force on rudder in the yaw direction                             # must be updated after the velocities are updated

    def C_b_to_str_C_b(self, C_b):
        if C_b == 0.5:
            C_b = '0.50'
        elif C_b == 0.6:
            C_b = '0.60'
        elif C_b == 0.7:
            C_b = '0.70'
        elif C_b == 0.8:
            C_b = '0.80'
        return C_b

    def find_closest_value_idx(self, val, look_up_list):
        temp1 = []
        for i in look_up_list:
            temp1.append(np.abs(i - val))
        close_val_idx = np.argmin(temp1)
        return close_val_idx

    def get_d1_d2(self, pos_xy, pos_termination):
        x, y = pos_xy
        d = np.sqrt((x - pos_termination[0]) ** 2 + (y - pos_termination[1]) ** 2)
        ang_btn_xy_and_imag_line = np.arctan(
            (y - pos_termination[1]) / (x - pos_termination[0])) - self.imag_line_ang  # [rad]
        d1 = d * np.sin(ang_btn_xy_and_imag_line)
        d2 = d * np.cos(ang_btn_xy_and_imag_line)
        return (d1, d2)
