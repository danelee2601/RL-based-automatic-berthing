import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BerthingTrajectoryDrawer:

    def __init__(self, L, pos_termination, terminal_circle_radius, plot_lim=(-1, 13)):
        self.L = L

        self.xlim = plot_lim
        self.ylim = self.xlim

        self.pos_termination = np.array(pos_termination)
        self.norm_pos_termination = self.pos_termination / self.L

        self.terminal_circle_radius = terminal_circle_radius
        self.norm_terminal_circle_radius = self.terminal_circle_radius / self.L

        # plot init. setting
        self.fig, self.ax = plt.subplots(1,1, figsize=(8,8))

        # draw
        self.draw_a_berthing_facility()
        self.draw_terminal_circle()



    # draw a ship at one time step
    def draw_a_berthing_facility(self, color='blue'):
        left_x = 1.5 - 0.5
        right_x = 1.5 + 0.5

        pos1 = [left_x, 0.]
        pos2 = [left_x, 0.5]
        pos3 = [right_x, 0.5]
        pos4 = [right_x, 0.]

        pos_list = np.array([pos1, pos2, pos3, pos4])
        self.ax.plot(pos_list[:,0], pos_list[:,1], '-', color=color)

    def draw_terminal_circle(self, color="red"):
        circle = plt.Circle(self.norm_pos_termination, self.norm_terminal_circle_radius, color=color, fill=False)
        self.ax.add_artist(circle)

    def draw_a_ship_at_one_timestep(self, x, y, heading_angle, color='blue', linewidth=1):
        # heading_angle must be in radian

        norm_x, norm_y = x / self.L, y / self.L

        # for drawing the ship
        norm_L = 1.
        norm_B = 0.2 #(actual)0.145
        shortening_ratio = 0.2

        middle_pos = np.array([norm_x, norm_y])
        front_pos = np.array([norm_x + (norm_L / 2)*np.sin(heading_angle), norm_y + (norm_L / 2)*np.cos(heading_angle)])
        back_pos = np.array([norm_x + (norm_L / 2)*np.sin(heading_angle + np.pi), norm_y + (norm_L / 2)*np.cos(heading_angle + np.pi)])

        #plt.plot(front_pos[0], front_pos[1], 'o', color='green')
        #plt.plot(middle_pos[0], middle_pos[1], 'o', color='red')
        #plt.plot(back_pos[0], back_pos[1], 'o', color='blue')


        middle_to_front = np.array([(norm_L / 2 - shortening_ratio * norm_L) * np.sin(heading_angle), (norm_L / 2 - shortening_ratio * norm_L) * np.cos(heading_angle)])
        middle_to_back = np.array([(norm_L / 2) * np.sin(heading_angle + np.pi), (norm_L / 2) * np.cos(heading_angle + np.pi)])
        center_left_shifted = np.array([0.0, 0.0])
        center_left_shifted[0] = middle_pos[0] + (norm_B/2)*np.sin(heading_angle - np.pi/2)
        center_left_shifted[1] = middle_pos[1] + (norm_B/2)*np.cos(heading_angle - np.pi/2)

        front_left = center_left_shifted + middle_to_front
        back_left = center_left_shifted + middle_to_back

        #plt.plot(front_left[0], front_left[1], 'o', color='orange')
        #plt.plot(back_left[0], back_left[1], 'o', color='orange')


        center_right_shifted = np.array([0.0, 0.0])
        center_right_shifted[0] = middle_pos[0] + (norm_B/2)*np.sin(heading_angle + np.pi/2)
        center_right_shifted[1] = middle_pos[1] + (norm_B/2)*np.cos(heading_angle + np.pi/2)

        front_right = center_right_shifted + middle_to_front
        back_right = center_right_shifted + middle_to_back

        #plt.plot(front_right[0], front_right[1], 'o', color='purple')
        #plt.plot(back_right[0], back_right[1], 'o', color='purple')

        ship_pos = np.array([back_left, front_left, front_pos, front_right, back_right, back_left])

        self.ax.plot(ship_pos[:, 0], ship_pos[:, 1], '-', color=color, linewidth=linewidth)

    def show_plot(self):
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.show()


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_excel("C:/temp01/trajectory.xlsx")

    x_hist = df["x_hist"].values  # [m]
    y_hist = df["y_hist"].values  # [m]
    heading_angle_hist = df["heading_angle"].values  # [rad.]

    LBP = 175  # [m]
    pos_termination = [1.5 * LBP, 1.5 * LBP]
    terminal_circle_radius = 0.2 * LBP

    # call the class
    drawer = BerthingTrajectoryDrawer(LBP, pos_termination, terminal_circle_radius)

    # draw
    drawing_period = 50
    idx = 0
    for x, y, heading_angle in zip(x_hist, y_hist, heading_angle_hist):
        if idx % drawing_period == 0:
            drawer.draw_a_ship_at_one_timestep(x, y, heading_angle)
        idx +=1

    drawer.show_plot()


